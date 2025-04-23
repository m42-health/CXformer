import torch
import os
import numpy as np
import argparse
import yaml
from transformers import TrainingArguments
from dinov2.models import build_model_from_cfg
from omegaconf import OmegaConf
from dinov2.configs import dinov2_default_config
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb
import logging
from dinov2.models.chexzero_vit import VisualTransformer
from transformers import AutoModel
from open_clip import create_model_from_pretrained
from finetune_utils import get_datasets, LinearClassifier, Dinov2ForClassification, CustomDataCollator, CustomTrainer

def main(args):
    torch.manual_seed(args.seed)
    with open(args.config_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    # cfg = OmegaConf.load(args.cfg_file)
    default_cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.create(cfg)
    cfg = OmegaConf.merge(default_cfg, cfg)

    # make it cmd line args
    cfg.num_epochs = args.num_epochs
    cfg.output_dir = args.output_dir
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.cls_n_layers = args.cls_n_layers 
    cfg.apply_avgpool = args.apply_avgpool
    cfg.backbone_lr = args.backbone_lr
    cfg.clf_lr = args.clf_lr
    cfg.lr = args.lr
    exp_name = args.exp_name if args.exp_name else cfg.output_dir.split('/')[7]
    cfg.exp_name = exp_name
    
    if args.local_rank ==0 :
        run = wandb.init(project="dinov2-finetune", name=exp_name)
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)

    train_dataset, valid_dataset, test_dataset = get_datasets(cfg,args.model_type)

    logger.info("-" * 50)
    logger.info("DATSETS")
    logger.info(f"Train dataset: \n {train_dataset}")
    logger.info(f"Valid dataset: \n {valid_dataset}")
    logger.info(f"Test dataset: \n {test_dataset}")
    logger.info("-" * 50)

    if args.model_type == "raddino":
        logger.info(f"Loading model from HuggingFace: {args.pretrained_weights}")
        model = AutoModel.from_pretrained(args.pretrained_weights)
        model_dim = model.config.hidden_size
    elif args.model_type == "dinov2":
        
        model, model_dim = build_model_from_cfg(cfg, only_teacher=True)
        checkpoint = torch.load(args.pretrained_weights)
        if 'teacher' in checkpoint:
            checkpoint = checkpoint['teacher']
            checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}
        backbone_load_response = model.load_state_dict(checkpoint, strict=False)
        logger.info(f"backbone loaded: {backbone_load_response}")
    elif args.model_type == "chexzero":
        model = VisualTransformer(
            input_resolution=224,
            patch_size=32,
            width=768,
            layers=12,
            heads=8,
            output_dim=512,
            do_interpolate_positional_encoding=True,
        )
        model_dim = 768
        sd = torch.load(args.pretrained_weights,map_location='cpu')
        sd = {k:v for k, v in sd.items() if k.startswith('visual.')}
        checkpoint = {k.replace('visual.',''):v for k,v in sd.items()}

        backbone_load_response = model.load_state_dict(checkpoint, strict=True)
        logger.info(f"backbone loaded: {backbone_load_response}")
    elif args.model_type == 'biomedclip':
        model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        model = model.visual.trunk
        model_dim = model.embed_dim
    else:
        raise ValueError("Please provide a valid model type")

    model.eval()

    num_labels = len(train_dataset.dataset.pathologies)
    temp_inp_dim = model_dim * (cfg.cls_n_layers + 1 if cfg.apply_avgpool else cfg.cls_n_layers)
    logger.info(f"~~~~> Linear classifier inp dim: {temp_inp_dim}")
    linear_clf = LinearClassifier(temp_inp_dim, num_labels=num_labels, n_layers=0)

    cfg.freeze_backbone = False

    if cfg.lr:
        params_to_optimize = list(linear_clf.parameters()) + list(model.parameters())
        optimizer = torch.optim.AdamW(
            params_to_optimize, lr=cfg.lr, weight_decay=0
        )
        logger.info(f"Using same learning rate for backbone and classifier: {cfg.lr}")
    elif args.clf_lr and args.backbone_lr==0:
        cfg.freeze_backbone = True
        params_to_optimize = [
            {'params': linear_clf.parameters(), 'lr': args.clf_lr},
        ]
        optimizer = torch.optim.AdamW(
            params_to_optimize, lr=args.clf_lr, weight_decay=0
        )
        logger.info(f"Freezing backbone!")
    elif args.clf_lr and args.backbone_lr:
        params_to_optimize = [
            {'params': linear_clf.parameters(), 'lr': args.clf_lr},
            {'params': model.parameters(), 'lr': args.backbone_lr}
            ]
        logger.info(f"Using different learning rates for backbone: {args.backbone_lr} and classifier: {args.clf_lr}")
        optimizer = torch.optim.AdamW(params_to_optimize, weight_decay=0)
    else:
        raise ValueError("Please provide learning rate")

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        save_strategy="epoch",
        logging_dir=cfg.output_dir,
        logging_steps=10,
        push_to_hub=False,
        report_to="wandb",
        run_name=exp_name,
        remove_unused_columns=False,  # important!
        dataloader_num_workers=cfg.num_workers,
        metric_for_best_model='macro_auroc',
        load_best_model_at_end=True,
        greater_is_better=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ddp_find_unused_parameters=True,
        deepspeed = args.deepspeed,
        lr_scheduler_type="constant",
    )    

    model = Dinov2ForClassification(
        backbone=model,
        linear_clf=linear_clf,
        cls_n_layers=cfg.cls_n_layers,
        apply_avgpool=cfg.apply_avgpool,
        freeze_backbone=cfg.freeze_backbone
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        optimizers=(optimizer, None),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=CustomDataCollator(),
        compute_metrics=lambda eval_pred: compute_metrics(
            eval_pred
        ),  # calculate val metrics
    )

    trainer.train()
    trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')

def compute_metrics(eval_pred):
    """
    The function `compute_metrics` calculates the macro AUROC and macro AUPRC scores from predicted
    logits and true labels.
    
    Args:
      eval_pred: The `eval_pred` parameter is expected to be a tuple containing two elements:
    
    Returns:
      The `compute_metrics` function returns a dictionary containing two metrics:
    1. "macro_auroc": The average Area Under the Receiver Operating Characteristic Curve (AUROC)
    calculated across all classes.
    2. "macro_auprc": The average Area Under the Precision-Recall Curve (AUPRC) calculated across all
    classes.
    """
    logits, labels = eval_pred
    logits, labels = torch.tensor(logits), torch.tensor(labels)
    preds = torch.nn.functional.sigmoid(torch.tensor(logits))
    auroc = roc_auc_score(labels, preds, average=None)
    auprc = average_precision_score(labels, preds, average=None)
    average_auroc = np.mean(auroc)
    acerage_auprc = np.mean(auprc)
    return {
        "macro_auroc":average_auroc,
        "macro_auprc":acerage_auprc
    }

def load_best_model_auroc(model, linear_clf, cfg):
    """
    The function `load_best_model_auroc` loads a saved linear classifier and backbone model based on
    AUROC performance.
    
    Args:
      model: The `model` parameter is typically a neural network model that consists of a backbone
    network (often a convolutional neural network) followed by a linear classifier (such as a fully
    connected layer) for a specific task like image classification or object detection.
      linear_clf: Linear classifier model
      cfg: The `cfg` parameter is likely a configuration object that contains settings and paths for the
    model training process. It is used to determine the output directory where the best linear model
    with AUROC (Area Under the Receiver Operating Characteristic curve) is saved and loaded from.
    """
    temp_path = os.path.join(cfg.output_dir, "best_linear_model_auroc.pth")
    checkpoint = torch.load(temp_path)
    auroc_load_backbone = model.load_state_dict(checkpoint["backbone"])
    auroc_load_response = linear_clf.load_state_dict(checkpoint["linear_clf"])
    logger.info(f"auroc based linear_clf and backbone loaded: {auroc_load_response} \t {auroc_load_backbone}")

def load_best_model_auprc(model, linear_clf, cfg):
    """
    The function `load_best_model_auprc` loads the best linear classifier and backbone model from a
    checkpoint file.
    
    Args:
      model: The `model` parameter in the `load_best_model_auprc` function is typically a neural network
    model that serves as the backbone for a more complex model. It could be a pre-trained model or a
    custom model architecture that you have defined for a specific task. In this function, the `
      linear_clf: The `linear_clf` parameter in the `load_best_model_auprc` function seems to be an
    instance of a linear classifier model that you are loading the state dictionary into. This function
    is designed to load the best model based on the AUPRC (Area Under the Precision-Recall
      cfg: The `cfg` parameter is likely a configuration object that contains various settings and paths
    for your model training and evaluation process. In this context, it seems to be used to specify the
    output directory where the best linear model AUPRC (Area Under the Precision-Recall Curve)
    checkpoint is saved and
    """
    temp_path = os.path.join(cfg.output_dir, "best_linear_model_auprc.pth")
    checkpoint = torch.load(temp_path)
    auprc_load_response = linear_clf.load_state_dict(checkpoint["linear_clf"])
    auroc_load_backbone = model.load_state_dict(checkpoint["backbone"])
    logger.info(f"auprc based linear_clf and backbone loaded: {auprc_load_response} \t {auroc_load_backbone}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("DINOv2 training")
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="deepspeed config",
    )
    parser.add_argument(
        "--output-dir", type=str, default="", help="path to output dir"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=None, help="number of epochs"
    )    
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size"
    )    
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers"
    )    
    parser.add_argument(
        "--pretrained-weights", type=str, default="", help="path to pretrained teacher weights with heads"
    )    
    parser.add_argument(
        "--exp-name", type=str, default=None, help="Wandb experiment name"
    )    
    parser.add_argument(
        "--cls-n-layers", type=int, default=4, help="Number of layers"
    )    
    parser.add_argument(
        "--apply-avgpool", action='store_true', help="Apply avgpool"
    )
    parser.add_argument(
        "--clf_lr", type=float, default=None, help="classifier layer learning rate"
    )
    parser.add_argument(
        "--model-type", type=str, default="dinov2", help="Model type", choices=["dinov2", "raddino", "chexzero","biomedclip"]
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="accumulate_gradient"
    ) 
    
    parser.add_argument(
        "--backbone_lr", type=float, default=None, help="backbone learning rate"
    )    
    parser.add_argument('--local_rank', type=int, default=0,help='local rank passed from distributed launcher')
    parser.add_argument('--lr', type=float, default=None ,help='learning rate')
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    
    args = parser.parse_args()

    global logger

    logger = logging.getLogger("dinov2-finetune")
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Set the log message format
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'log.txt')),  # Log messages to a file named example.log
            logging.StreamHandler()  # Optionally, log messages to the console
        ]
    )
    main(args)
