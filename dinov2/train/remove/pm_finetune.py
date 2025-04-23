import torch
import os
import sys
import numpy as np
import argparse
# import deepspeed
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
from open_clip import create_model_from_pretrained, get_tokenizer


def get_datasets(cfg, model_type='dinov2'):

    if model_type == "raddino" or model_type=='dinov2':
        inputs_dtype = torch.float32
        img_size = cfg.crops.global_crops_size
        patch_size = cfg.student.patch_size
        n_tokens = (img_size // patch_size) ** 2
        mask_generator = MaskingGenerator(
            input_size=(img_size // patch_size, img_size // patch_size),
            max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
        )
        data_transform = DataAugmentationDINO(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
        )

        # tr_data_transform = torchvision.transforms.Compose(
        #     [
        #         torchvision.transforms.RandomResizedCrop(cfg.crops.global_crops_size),
        #         torchvision.transforms.RandomHorizontalFlip(),
        #         torchvision.transforms.Grayscale(num_output_channels=3),
        #         torchvision.transforms.ToTensor(),
        #         torchvision.transforms.Normalize(
        #             (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        #         ),
        #     ]
        # )

        val_data_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(cfg.crops.global_crops_size, interpolation=3),
                torchvision.transforms.CenterCrop(cfg.crops.global_crops_size),
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        tr_data_transform = make_classification_train_transform(crop_size=cfg.crops.global_crops_size,)

    elif model_type == "chexzero":
        tr_data_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(cfg.crops.global_crops_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                torchvision.transforms.CenterCrop(cfg.crops.global_crops_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.8, 1.2)),
                torchvision.transforms.RandomApply(
                    [torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.5,
                ),
                torchvision.transforms.RandomApply(
                    [torchvision.transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
                    p=0.5,
                ),
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.398, 0.398, 0.398), (0.327, 0.327, 0.327)
                ),
            ]
        )
        val_data_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(cfg.crops.global_crops_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                torchvision.transforms.CenterCrop(cfg.crops.global_crops_size),
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.398, 0.398, 0.398), (0.327, 0.327, 0.327)
                ),
            ]
        )
    elif model_type == "biomedclip":
        tr_data_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(cfg.crops.global_crops_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                torchvision.transforms.CenterCrop(cfg.crops.global_crops_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.8, 1.2)),
                torchvision.transforms.RandomApply(
                    [torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.5,
                ),
                torchvision.transforms.RandomApply(
                    [torchvision.transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
                    p=0.5,
                ),
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                ),
            ]
        )
        val_data_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(cfg.crops.global_crops_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                torchvision.transforms.CenterCrop(cfg.crops.global_crops_size),
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                ),
            ]
        )

    train_dataset = make_cxr_datasets(
        dataset_configs=cfg.train.datasets, dino_transforms=tr_data_transform
    )

    valid_dataset = make_cxr_datasets(
        dataset_configs=cfg.val.datasets, dino_transforms=val_data_transform
    )

    test_dataset = make_cxr_datasets(
        dataset_configs=cfg.test.datasets, dino_transforms=val_data_transform
    )

    return train_dataset, valid_dataset, test_dataset


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, dim, num_labels=1000, n_layers=1):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear_layers = []
        self.n_layers = n_layers - 1
        for _ in range(n_layers):
            temp = nn.Linear(dim, dim)
            temp.weight.data.normal_(mean=0.0, std=0.01)
            temp.bias.data.zero_()
            self.linear_layers.append(temp)
        
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        if self.n_layers:
            for layer in self.linear_layers:
                x = layer(x)
                x = nn.LeakyReLU(inplace=True)(x)

        # linear layer
        return self.linear(x)


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

    run = wandb.init(project="dinov2-finetune", name=exp_name)

    os.makedirs(cfg.output_dir, exist_ok=True)

    with open(os.path.join(cfg.output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    train_dataset, valid_dataset, test_dataset = get_datasets(cfg,args.model_type)

    logger.info("-" * 50)
    logger.info("DATSETS")
    # logger.info(f"Train dataset: \n {train_dataset}")
    # logger.info(f"Valid dataset: \n {valid_dataset}")
    # logger.info(f"Test dataset: \n {test_dataset}")
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


    model.cuda()
    model.eval()

    num_labels = len(train_dataset.dataset.pathologies)
    temp_inp_dim = model_dim * (cfg.cls_n_layers + 1 if cfg.apply_avgpool else cfg.cls_n_layers)
    logger.info(f"~~~~> Linear classifier inp dim: {temp_inp_dim}")
    linear_clf = LinearClassifier(temp_inp_dim, num_labels=num_labels, n_layers=0)
    linear_clf = linear_clf.cuda()

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
    train(train_data_loader, valid_data_loader, model, linear_clf, optimizer, cfg,skip_validation=args.skip_validation)
    if not args.load_last_ckpt:
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
    
    if not args.load_last_ckpt:
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


@torch.no_grad()
def validate(model, linear_clf, valid_data_loader, cfg, desc="Evaluating Model",return_preds=True):
    model.eval()
    linear_clf.eval()
    gts = None
    preds = None
    for data in tqdm(valid_data_loader, colour="green", desc=desc):

        x, y = data["image"], data["lab"]
        x = x.cuda()
        y = y.cuda()

        with torch.no_grad():
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
            output = torch.nn.functional.sigmoid(output)

        if gts is None:
            gts = y
            preds = output
        else:
            gts = torch.cat((gts, y), dim=0)
            preds = torch.cat((preds, output), dim=0)

    auroc = roc_auc_score(gts.cpu(), preds.cpu(), average=None)
    auprc = average_precision_score(gts.cpu(), preds.cpu(), average=None)

    mean_auroc, mean_auprc = np.mean(auroc), np.mean(auprc)
    if return_preds:
        return mean_auroc, mean_auprc, gts, preds
    return mean_auroc, mean_auprc


def train(tr_dataloader, valid_data_loader, model, linear_clf, optimizer, cfg,skip_validation=False):

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
            if not skip_validation:
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
    parser = argparse.ArgumentParser("DINOv2 training")
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
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
        "--model-type", type=str, default="dinov2", help="Model type", choices=["dinov2", "raddino", "chexzero", "biomedclip"]
    )
    parser.add_argument(
        "--backbone_lr", type=float, default=None, help="backbone learning rate"
    )    
    parser.add_argument('--local_rank', type=int, default=-1,help='local rank passed from distributed launcher')
    parser.add_argument('--lr', type=float, default=None ,help='learning rate')
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--load-last-ckpt", action='store_true', help="Load last")
    parser.add_argument("--skip-validation", action='store_true', help="Load last")
    # parser = deepspeed.add_config_arguments(parser)
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
