import torch
import os
import sys
import numpy as np
import argparse
import yaml
from dinov2.data import (
    collate_data_and_cast,
    collate_data_and_cast_cxr,
    DataAugmentationDINO,
    MaskingGenerator,
)
from functools import partial
from dinov2.data import make_cxr_datasets
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.models import build_model_from_cfg
from omegaconf import OmegaConf
from dinov2.configs import dinov2_default_config
import torch.nn as nn
import torchvision
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb
import torch.distributed as dist
import logging
from dinov2.models.chexzero_vit import VisualTransformer
from transformers import AutoModel
from transformers import AutoImageProcessor
from dinov2.data.transforms import make_classification_eval_transform, make_classification_train_transform
import transformers
from pm_finetune import get_datasets, LinearClassifier, validate

def load_best_model_auroc(model, linear_clf, cfg):
    temp_path = os.path.join(cfg.output_dir, "best_linear_model_auroc.pth")
    checkpoint = torch.load(temp_path)
    auroc_load_backbone = model.load_state_dict(checkpoint["backbone"])
    auroc_load_response = linear_clf.load_state_dict(checkpoint["linear_clf"])
    logger.info(f"auroc based linear_clf and backbone loaded: {auroc_load_response} \t {auroc_load_backbone}")


def load_best_model_auprc(model, linear_clf, cfg):
    temp_path = os.path.join(cfg.output_dir, "best_linear_model_auprc.pth")
    checkpoint = torch.load(temp_path)
    auprc_load_response = linear_clf.load_state_dict(checkpoint["linear_clf"])
    auroc_load_backbone = model.load_state_dict(checkpoint["backbone"])
    logger.info(f"auprc based linear_clf and backbone loaded: {auprc_load_response} \t {auroc_load_backbone}")

def main(args):

    with open(args.config_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    default_cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.create(cfg)
    cfg = OmegaConf.merge(default_cfg, cfg)

    sweep_cfgs = wandb.config
    # save to yaml file

    logger.info("\n---Sweep Configs---")
    logger.info(sweep_cfgs)


    cfg.num_epochs = sweep_cfgs.num_epochs
    cfg.optimizer = sweep_cfgs.optimizer
    cfg.output_dir = args.output_dir
    cfg.batch_size = sweep_cfgs.batch_size
    cfg.num_workers = sweep_cfgs.num_workers
    cfg.cls_n_layers = sweep_cfgs.cls_n_layers 
    cfg.apply_avgpool = sweep_cfgs.apply_avgpool
    cfg.backbone_lr = sweep_cfgs.backbone_lr
    cfg.clf_lr = sweep_cfgs.clf_lr
    cfg.pretrained_weights = sweep_cfgs.pretrained_weights
    cfg.model_type = sweep_cfgs.model_type
    cfg.weight_decay = sweep_cfgs.weight_decay
    cfg.lr = None
    cfg.mlp_n_layers = sweep_cfgs.mlp_n_layers    

    os.makedirs(cfg.output_dir, exist_ok=True)

    with open(os.path.join(cfg.output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    with open(os.path.join(args.output_dir, "sweep_config.yaml"), "w") as f:
        yaml.dump(dict(sweep_cfgs), f, default_flow_style=False)

    train_dataset, valid_dataset, test_dataset = get_datasets(cfg,cfg.model_type)

    logger.info("-" * 50)
    logger.info("DATSETS")
    logger.info(f"Train dataset: \n {train_dataset}")
    logger.info(f"Valid dataset: \n {valid_dataset}")
    logger.info(f"Test dataset: \n {test_dataset}")
    logger.info("-" * 50)

    if cfg.optimizer=='AdamW':
        optimizer = torch.optim.AdamW
    elif cfg.optimizer=='Adam':
        optimizer = torch.optim.Adam
    elif cfg.optimizer=='SGD':
        optimizer = partial(torch.optim.SGD, momentum=0.9)
    elif cfg.optimizer=='RMSprop':
        optimizer = torch.optim.RMSprop

    if cfg.model_type == "raddino":
        logger.info(f"Loading model from HuggingFace: {cfg.pretrained_weights}")
        model = AutoModel.from_pretrained(cfg.pretrained_weights)
        model_dim = model.config.hidden_size
    elif cfg.model_type == "dinov2":
        model, model_dim = build_model_from_cfg(cfg, only_teacher=True)
        checkpoint = torch.load(cfg.pretrained_weights)
        if 'teacher' in checkpoint:
            checkpoint = checkpoint['teacher']
            checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}
        backbone_load_response = model.load_state_dict(checkpoint, strict=False)
        logger.info(f"backbone loaded: {backbone_load_response}")
    elif cfg.model_type == "chexzero":
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
        sd = torch.load(cfg.pretrained_weights,map_location='cpu')
        sd = {k:v for k, v in sd.items() if k.startswith('visual.')}
        checkpoint = {k.replace('visual.',''):v for k,v in sd.items()}
        backbone_load_response = model.load_state_dict(checkpoint, strict=True)
        logger.info(f"backbone loaded: {backbone_load_response}")
    else:
        raise ValueError("Please provide a valid model type")


    model.cuda()
    model.eval()

    num_labels = len(train_dataset.dataset.pathologies)
    temp_inp_dim = model_dim * (cfg.cls_n_layers + 1 if cfg.apply_avgpool else cfg.cls_n_layers)
    logger.info(f"~~~~> Linear classifier inp dim: {temp_inp_dim}")
    linear_clf = LinearClassifier(temp_inp_dim, num_labels=num_labels, n_layers=cfg.mlp_n_layers - 1)
    linear_clf = linear_clf.cuda()
    cfg.freeze_backbone = False
    logger.info(f"Linear classifier:\n {linear_clf}")

    if cfg.lr:
        params_to_optimize = list(linear_clf.parameters()) + list(model.parameters())
        optimizer = optimizer(
            params_to_optimize, lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        logger.info(f"Using same learning rate for backbone and classifier: {cfg.lr}")
    elif cfg.clf_lr and cfg.backbone_lr==0:
        cfg.freeze_backbone = True
        params_to_optimize = [
            {'params': linear_clf.parameters(), 'lr': cfg.clf_lr},
        ]
        optimizer = optimizer(
            params_to_optimize, lr=cfg.clf_lr, weight_decay=cfg.weight_decay
        )
        logger.info(f"Freezing backbone!")
    elif cfg.clf_lr and cfg.backbone_lr:
        params_to_optimize = [
            {'params': linear_clf.parameters(), 'lr': cfg.clf_lr},
            {'params': model.parameters(), 'lr': cfg.backbone_lr}
            ]
        logger.info(f"Using different learning rates for backbone: {cfg.backbone_lr} and classifier: {cfg.clf_lr}")
        optimizer = optimizer(params_to_optimize, weight_decay=cfg.weight_decay)
    else:
        raise ValueError("Please provide learning rate")


    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    train(train_data_loader, valid_data_loader, model, linear_clf, optimizer, cfg)

    load_best_model_auroc(model, linear_clf, cfg)
    test_mean_auroc, test_mean_auprc, _, test_preds = validate(model, linear_clf, test_data_loader, cfg, desc="Test Set Evaluation",return_preds=True)
    np.save(os.path.join(cfg.output_dir,f"preds_test_{test_dataset.dataset_name}_0.npy"), test_preds.cpu().numpy())

    
    logger.info("-" * 50)
    logger.info("AUROC based best model")
    logger.info(f"AUROC: {test_mean_auroc:.3f}\tAUPRC: {test_mean_auprc:.3f}")

    # save test auroc and test auprc in text files
    with open(os.path.join(cfg.output_dir, "best_auroc.txt"), "w") as f:
        f.write(f"Test AUROC: {test_mean_auroc}")
        f.write("\n")
        f.write(f"Test AUPRC: {test_mean_auprc}")
    wandb.log({"test_auroc_best_auroc": test_mean_auroc})
    wandb.log({"test_auprc_best_auroc": test_mean_auprc})
    
    load_best_model_auprc(model, linear_clf, cfg)
    test_mean_auroc, test_mean_auprc, _, test_preds = validate(model, linear_clf, test_data_loader, cfg, desc="Test Set Evaluation")
    logger.info("AUPRC based best model")
    logger.info(f"AUROC: {test_mean_auroc:.3f}\tAUPRC: {test_mean_auprc:.3f}")

    with open(os.path.join(cfg.output_dir, "best_auprc.txt"), "w") as f:
        f.write(f"Test AUROC: {test_mean_auroc}")
        f.write("\n")
        f.write(f"Test AUPRC: {test_mean_auprc}")
    wandb.log({"test_auroc_best_auprc": test_mean_auroc})
    wandb.log({"test_auprc_best_auprc": test_mean_auprc})
    


def train(tr_dataloader, valid_data_loader, model, linear_clf, optimizer, cfg):

    loss_fn = nn.BCEWithLogitsLoss()

    n_epochs = cfg.num_epochs
    total_iters = len(tr_dataloader)

    logger.info(f"Total Iterations: {total_iters}")

    best_val_auroc, best_val_auprc, best_val_epoch_auroc, best_val_epoch_auprc = (
        -np.inf,
        -np.inf,
        0,
        0,
    )
    log_interval = 10
    if total_iters<log_interval:
        log_interval = 1
    batch_iter = 0
    for cur_epoch in tqdm(range(n_epochs), colour='blue', desc="Training Linear classifier model"):
        cur_iter = 0

        for data in tr_dataloader:

            linear_clf.train()
            model.train(mode =not cfg.freeze_backbone)

            batch_iter += 1
            cur_iter += 1

            x, y = data["image"], data["lab"]
            x = x.cuda()
            y = y.cuda()


            # output = model(x)

            with torch.no_grad() if not cfg.freeze_backbone else torch.enable_grad():
                if isinstance(model,transformers.models.dinov2.modeling_dinov2.Dinov2Model):
                    intermediate_output = model(x,output_hidden_states=True)
                    hidden_states = intermediate_output.hidden_states
                    blocks_to_take = range(len(hidden_states) - cfg.cls_n_layers, len(hidden_states))
                    output = [hidden_states[i][:, 0] for i in blocks_to_take]
                    if cfg.apply_avgpool:
                        output.append(torch.mean(hidden_states[-1][:, 1:], dim=1))
                    output = torch.cat(output, dim=-1)
                else:
                    intermediate_output = model.get_intermediate_layers(x, cfg.cls_n_layers)
                    output = [x[:, 0] for x in intermediate_output]
                    if cfg.apply_avgpool:
                        output.append(torch.mean(intermediate_output[-1][:, 1:], dim=1))
                    output = torch.cat(output, dim=-1)
            output = linear_clf(output)

            loss = loss_fn(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_iter % log_interval == 0:
                logger.info(
                    f"Epoch: [{cur_epoch+1}/{n_epochs}] Iteration [{cur_iter}/{total_iters}] Training Loss: {loss.item():.3f}"
                )

                wandb.log({"train_loss": loss},step=batch_iter)
                wandb.log({"lr": optimizer.param_groups[0]["lr"]},step=batch_iter)

            if batch_iter % (total_iters) == 0:
                val_auroc, val_auprc, val_gt, val_preds = validate(model, linear_clf, valid_data_loader, cfg, desc="Valid Set Evaluation", return_preds=True)
                logger.info(
                    f"Epoch: [{cur_epoch+1}/{n_epochs}] Iteration [{cur_iter}/{total_iters}] Validation AUROC: {val_auroc:.3f} (Best: {best_val_auroc:.3f}) Validation AUPRC: {val_auprc:.3f} (Best: {best_val_auprc:.3f})"
                )
                val_loss = nn.functional.binary_cross_entropy(val_preds, val_gt)
                wandb.log({"val_auroc": val_auroc},step=batch_iter)
                wandb.log({"val_loss": val_loss},step=batch_iter)
                wandb.log({"val_auprc": val_auprc},step=batch_iter)
                if val_auroc > best_val_auroc:
                    best_val_auroc = val_auroc
                    best_val_epoch_auroc = cur_epoch
                    logger.info(
                        f"Best model for auroc found at {best_val_auroc:.3f} at epoch {best_val_epoch_auroc}"
                    )
                    torch.save(
                        {
                            "linear_clf": linear_clf.state_dict(),
                            "backbone": model.state_dict(),
                            "auroc": best_val_auroc,
                            "auprc": val_auprc,
                        },
                        os.path.join(cfg.output_dir, "best_linear_model_auroc.pth"),
                    )

                if val_auprc > best_val_auprc:
                    best_val_auprc = val_auprc
                    best_val_epoch_auprc = cur_epoch
                    logger.info(
                        f"Best model for auprc found at {best_val_auprc:.3f} at epoch {best_val_epoch_auprc}"
                    )
                    torch.save(
                        {
                            "linear_clf": linear_clf.state_dict(),
                            "backbone": model.state_dict(),
                            "auroc": val_auroc,
                            "auprc": best_val_auprc,
                        },
                        os.path.join(cfg.output_dir, "best_linear_model_auprc.pth"),
                    )

        logger.info("-" * 50)
        logger.info(f"Epoch {cur_epoch+1} finished !!")
        logger.info("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DINOv2 finetuning sweeps")
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="", help="path to output dir"
    )


    args = parser.parse_args()

    global logger

    run = wandb.init()
    run_id = wandb.run.name
    run_id = f"trial-{run_id}-{wandb.run.id}"
    args.output_dir = os.path.join(args.output_dir, run_id)

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
