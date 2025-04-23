import torch
import os
import sys
import numpy as np
import argparse
import deepspeed
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
from torch.profiler import profile, record_function, ProfilerActivity

from dinov2.data.transforms import make_classification_eval_transform, make_classification_train_transform


def get_datasets(cfg):

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

    tr_data_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(cfg.crops.global_crops_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]
    )

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

    # DEBUG! 
    if True:
        tr_data_transform = make_classification_train_transform(crop_size=518,)
    else:
        logger.info("Using default data transforms")

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
    cfg.lr = args.lr
    exp_name = args.exp_name if args.exp_name else cfg.output_dir.split('/')[7]

    run = wandb.init(project="dinov2-finetune", name=exp_name)

    os.makedirs(cfg.output_dir, exist_ok=True)

    train_dataset, valid_dataset, test_dataset = get_datasets(cfg)
    class_counts = train_dataset.dataset.csv['MultiClassEvaluation'].value_counts()
    sample_weights = [1 / class_counts[i] for i in train_dataset.dataset.csv['MultiClassEvaluation'].values]
    sampler = torch.utils.data.WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(train_dataset),
                )
    logger.info("Performing balanced batch sampling")

    logger.info("-" * 50)
    logger.info("DATSETS")
    logger.info(f"Train dataset: \n {train_dataset}")
    logger.info(f"Valid dataset: \n {valid_dataset}")
    logger.info(f"Test dataset: \n {test_dataset}")
    logger.info("-" * 50)

    model, model_dim = build_model_from_cfg(cfg, only_teacher=True)

    checkpoint = torch.load(args.pretrained_weights)

    if 'teacher' in checkpoint:
        checkpoint = checkpoint['teacher']
        checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}
    backbone_load_response = model.load_state_dict(checkpoint, strict=False)
    logger.info(f"backbone loaded: {backbone_load_response}")

    model.cuda()
    model.eval()

    temp_inp_dim = model_dim * (cfg.cls_n_layers + 1 if cfg.apply_avgpool else 0)
    logger.info(f"~~~~> Linear classifier inp dim: {temp_inp_dim}")
    linear_clf = LinearClassifier(temp_inp_dim, num_labels=4, n_layers=0)
    linear_clf = linear_clf.cuda()


    if cfg.lr:
        params_to_optimize = list(linear_clf.parameters()) + list(model.parameters())
        optimizer = torch.optim.AdamW(
            params_to_optimize, lr=cfg.lr, weight_decay=0
        )
        logger.info(f"Using same learning rate for backbone and classifier: {cfg.lr}")
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
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        sampler=sampler
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            val_auroc, val_auprc = validate(model, linear_clf, valid_data_loader, cfg, desc="Valid Set Evaluation")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    logger.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    


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
def validate(model, linear_clf, valid_data_loader, cfg, desc="Evaluating Model"):
    model.eval()
    linear_clf.eval()
    gts = None
    preds = None

    for data in tqdm(valid_data_loader, colour="green", desc=desc):

        x, y = data["image"], data["lab"]
        x = x.cuda()
        y = y.cuda()

        with torch.no_grad():
            # output = model(x)

            intermediate_output = model.get_intermediate_layers(x, cfg.cls_n_layers)
            output = [x[:, 0] for x in intermediate_output]
            if cfg.apply_avgpool:
                output.append(torch.mean(intermediate_output[-1][:, 1:], dim=1))
            output = torch.cat(output, dim=-1)

            output = linear_clf(output)
            output = torch.nn.functional.softmax(output)
            output = output[:,0]
            output = 1 - output
            y = (y>=1).float()

        if gts is None:
            gts = y
            preds = output
        else:
            gts = torch.cat((gts, y), dim=0)
            preds = torch.cat((preds, output), dim=0)
    

    auroc = roc_auc_score(gts.cpu(), preds.cpu(), average=None)
    auprc = average_precision_score(gts.cpu(), preds.cpu(), average=None)

    mean_auroc, mean_auprc = np.mean(auroc), np.mean(auprc)
    return mean_auroc, mean_auprc








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
        "--backbone_lr", type=float, default=None, help="backbone learning rate"
    )    
    parser.add_argument('--local_rank', type=int, default=-1,help='local rank passed from distributed launcher')
    parser.add_argument('--lr', type=float, default=None ,help='learning rate')

    parser = deepspeed.add_config_arguments(parser)
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
