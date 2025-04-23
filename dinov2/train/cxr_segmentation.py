import torch
import os
import sys
import numpy as np
import argparse
from collections import Counter

# import deepspeed
import yaml
from dinov2.data import (
    collate_data_and_cast,
    collate_data_and_cast_cxr,
    DataAugmentationDINO,
    MaskingGenerator,
)
from torchvision.ops import focal_loss
from safetensors.torch import load_file
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
from dinov2.data.transforms import (
    make_classification_eval_transform,
    make_classification_train_transform,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
# from torchmetrics.classification import Dice
import torch.nn.functional as F

from transformers import Trainer, TrainingArguments
import transformers
import torchmetrics
from open_clip import create_model_from_pretrained
import timm

SUPPORTED_CRITERIONS = ['dice_loss', 'cross_entropy','bce','bce_dice_loss','ce_dice_loss','binary_focal_loss','focal_dice_loss']

def is_main_process():
    return dist.get_rank() == 0 

def compute_metrics(eval_pred, num_classes, id2class,ignore_index=None):
    logits, labels = eval_pred
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    if num_classes>1:
        preds = torch.softmax(torch.tensor(logits), dim=1)  # Shape: (B, C+1, H, W)
        preds = torch.argmax(preds, dim=1)
        dice_scores = []
        preds_one_hot = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2)  # Shape: (B, n_classes, H, W)
        label_one_hot = F.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2)  # Shape: (B, n_classes, H, W)
        dice_metric = torchmetrics.classification.Dice(
            ignore_index=ignore_index  # Set ignore_index if needed
        )     
        for c in range(num_classes):
            pred_mask = preds_one_hot[:, c]
            gt_mask = label_one_hot[:, c]
            intersection = torch.sum(pred_mask * gt_mask)
            total_pixels = torch.sum(pred_mask) + torch.sum(gt_mask) 
            _score = (2 * intersection) / total_pixels if total_pixels > 0 else 1.0
            dice_scores.append(_score.item())
        dice_scores = dice_scores[1:]  # ignore background class
    else:
        preds = torch.sigmoid(logits)
        preds = (preds >= 0.5).int()
        dice_scores = torchmetrics.classification.Dice(ignore_index=0)(preds, labels).tolist()
        dice_scores = [dice_scores]

    metrics = {}
    for i in range(len(dice_scores)):
        metrics[f"dice_{id2class[i + 1]}"] = dice_scores[i]
    if num_classes > 1:
        metrics["dice_avg"] = sum(dice_scores) / (num_classes-1) # ignore background class
    else:
        metrics["dice_avg"] = sum(dice_scores)

    return metrics


def compute_metrics_vindr(eval_pred, num_classes, id2class,ignore_index=None):
    logits, labels = eval_pred

    logits = torch.tensor(logits)
    labels = torch.tensor(labels).moveaxis(-1, 1)
    preds = torch.sigmoid(logits)

    preds = (preds >= 0.5).int()
    dice_scores = []
    for c in range(num_classes):
        pred_mask = preds[:, c]
        gt_mask = labels[:, c]
        intersection = torch.sum(pred_mask * gt_mask)
        total_pixels = torch.sum(pred_mask) + torch.sum(gt_mask) 
        _score = (2 * intersection) / total_pixels if total_pixels > 0 else 1.0
        dice_scores.append(_score.item())

    metrics = {}
    for i in range(len(dice_scores)):
        metrics[f"dice_{id2class[i + 1]}"] = dice_scores[i]
        
    metrics["dice_avg"] = sum(dice_scores) / (num_classes-1) # ignore background class


    return metrics


class CustomDataCollator:
    def __call__(self, features):

        images = torch.stack([f["image"] for f in features])
        masks = torch.stack([f["mask"] for f in features])

        return {"pixel_values": images, "labels": masks}


class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1, 1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        return self.classifier(embeddings)

class CustomSIIMTrainer(Trainer):
    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        
        test_dataloader = self.get_test_dataloader(test_dataset)
        eval_loop = self.evaluation_loop
        output = eval_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )        
        logits = output.predictions
        preds = torch.sigmoid(torch.tensor(logits))  # Shape: (B, C+1, H, W)
        preds = (preds >= 0.5).int()
        masks_arr = np.array(preds)
        image_ids = test_dataset.dataset.df.ImageId.tolist()
        return {"image_ids": image_ids, "masks": masks_arr}
    
    def get_train_dataloader(self):
        train_dataset = self.train_dataset

        _df = train_dataset.dataset.df
        _df['binary'] = _df[' EncodedPixels'].apply(lambda x: 0 if x=='-1' else 1)
        labels = _df['binary'].values
        class_counts = Counter(labels)
        class_weights = {cls: len(labels) / count for cls, count in class_counts.items()}

        sample_weights = [class_weights[label] for label in labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

class CustomTrainer(Trainer):
    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        
        test_dataloader = self.get_test_dataloader(test_dataset)
        eval_loop = self.evaluation_loop
        output = eval_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )        

        logits = output.predictions
        if len(logits.shape) == 4 and logits.shape[1] ==1:
            # single class
            preds = torch.sigmoid(torch.tensor(logits))  # Shape: (B, C+1, H, W)
        else:
            preds = torch.softmax(torch.tensor(logits), dim=1)  # Shape: (B, C+1, H, W)
            preds = torch.argmax(preds, dim=1)
        masks_arr = np.array(preds)
        image_ids = test_dataset.dataset.df.dicom_id.tolist()
        return {"image_ids": image_ids, "masks": masks_arr}


class CustomVinDrTrainer(Trainer):
    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        
        test_dataloader = self.get_test_dataloader(test_dataset)
        eval_loop = self.evaluation_loop
        output = eval_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )        

        logits = output.predictions
        preds = torch.sigmoid(torch.tensor(logits))  # Shape: (B, C+1, H, W)
        masks_arr = np.array(preds)
        image_ids = test_dataset.dataset.df.img.tolist()
        image_ids = [id.split('/')[-1].replace('.png','') for id in image_ids]
        return {"image_ids": image_ids, "masks": masks_arr}

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        target = target.unsqueeze(1)
        loss = self.alpha*self.focal(input, target) - torch.log(self.dice_loss(input, target))
        return loss.mean()
    
    def dice_loss(self, input, target):
        input = torch.sigmoid(input)
        smooth = 1.0
        iflat = input.view(-1)
        tflat = target.view(-1)
        #tflat = np.reshape(target,(4, 1, 512, 512))
        intersection = (iflat * tflat).sum()
        return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

class CEDiceLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5,num_classes=None):
        super(CEDiceLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes=num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets) 

        combined_loss = self.ce_weight * ce + self.dice_weight * dice
        return combined_loss

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5,num_classes=1):
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()  
        self.dice_loss = DiceLoss(num_classes=num_classes,multi_label=True)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets) 

        combined_loss = self.bce_weight * bce + self.dice_weight * dice
        return combined_loss

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        targets = targets.unsqueeze(1).float()
        loss = focal_loss.sigmoid_focal_loss(inputs, targets, alpha=self.alpha, gamma=self.gamma, reduction='mean')
        return loss

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6,multi_label=False):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        assert num_classes is not None, "Please provide number of classes"
        self.num_classes = num_classes
        self.multi_label = multi_label
    def forward(self, inputs, targets):
        
        if self.num_classes == 1:
            inputs = torch.sigmoid(inputs)
            targets = targets.unsqueeze(1).float() # Shape: (B, 1, H, W)
        else:
            if self.multi_label:
                inputs = torch.sigmoid(inputs)
                inputs = (inputs >= 0.5).int()
                # targets = torch.tensor(targets).moveaxis(-1, 1)
            else:
                inputs = torch.softmax(inputs, dim=1)  # Shape: (B, C, H, W)        
                targets = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)
                targets = targets.permute(0, 3, 1, 2).float()  # Shape: (B, C, H, W)

        intersection = (inputs * targets).sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth)
        dice_loss = 1 - dice.mean()
        return dice_loss

class Dinov2ForSemanticSegmentation(torch.nn.Module):
    def __init__(
        self, dinov2_backbone, hidden_size=768, tokenW=37, tokenH=37, num_labels=3, loss_fct="dice_loss", backbone_frozen=True
    ):
        super(Dinov2ForSemanticSegmentation, self).__init__()

        self.dinov2 = dinov2_backbone  
        if backbone_frozen:
            self.dinov2.eval()
        self.classifier = LinearClassifier(hidden_size, tokenW, tokenH, num_labels)
        self.num_classes = num_labels

        supported_losses = SUPPORTED_CRITERIONS
        assert loss_fct in supported_losses, f"loss {loss_fct} not supported. Select one of: {supported_losses}"


        if loss_fct=='dice_loss':
            self.loss_fct = DiceLoss(num_classes=self.num_classes)
        elif loss_fct=='cross_entropy':
            assert self.num_classes > 1, "Cross entropy dice loss can only be used for multi-class classification"
            self.loss_fct = torch.nn.CrossEntropyLoss()
        elif loss_fct=='bce':
            # assert self.num_classes == 1, "Binary cross entropy loss can only be used for binary classification"
            self.loss_fct = torch.nn.BCEWithLogitsLoss()
        elif loss_fct=='bce_dice_loss':
            # assert self.num_classes == 1, "Binary cross entropy loss can only be used for binary classification"
            self.loss_fct = BCEDiceLoss(num_classes=self.num_classes)
        elif loss_fct=='ce_dice_loss':
            assert self.num_classes > 1, "Cross entropy dice loss can only be used for multi-class classification"
            self.loss_fct = CEDiceLoss(ce_weight=0.5,dice_weight=.5,num_classes=self.num_classes)
        elif loss_fct=='binary_focal_loss':
            self.loss_fct = BinaryFocalLoss()
        elif loss_fct=='focal_dice_loss':
            self.loss_fct = FocalDiceLoss(alpha=10.0, gamma=2.0)
        else:
            raise ValueError("Please provide a valid loss function")

    def forward(self, pixel_values, labels=None):
        with torch.no_grad():
            if isinstance(self.dinov2,transformers.models.dinov2.modeling_dinov2.Dinov2Model):
                outputs = self.dinov2(pixel_values)
                patch_embeddings = outputs.last_hidden_state[
                    :, 1:, :
                ]  # skip the CLS token
            elif isinstance(self.dinov2, timm.models.vision_transformer.VisionTransformer):
                patch_embeddings = self.dinov2.get_intermediate_layers(x=pixel_values, n=1, return_prefix_tokens=False) # n = 1 for the last layer
                patch_embeddings = patch_embeddings[0]
            else:
                patch_embeddings = self.dinov2.get_intermediate_layers(x=pixel_values, n=1, return_class_token=False) # n = 1 for the last layer
                patch_embeddings = patch_embeddings[0]

        logits = self.classifier(patch_embeddings)
        logits = torch.nn.functional.interpolate(
            logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False
        )

        loss = None
        if labels is not None:
            if (self.loss_fct.__class__.__name__ == "BCEWithLogitsLoss" or self.loss_fct.__class__.__name__ == "BCEDiceLoss") and self.num_classes > 1:
                if labels.shape[-1] == self.num_classes:
                    labels = labels.moveaxis(-1, 1).float()
            loss = self.loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}


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
    cfg.lr = args.lr
    exp_name = args.exp_name if args.exp_name else cfg.output_dir.split("/")[7]
    cfg.exp_name = exp_name
    cfg.criterion = args.criterion
    cfg.gradient_accumulation_steps = args.gradient_accumulation_steps
    cfg.weight_decay = args.weight_decay
    cfg.pretrained_weights = args.pretrained_weights

    if not args.predict_only:
        if args.local_rank==0:
            run = wandb.init(project="dinov2-segmentation", name=exp_name)
    
    if args.local_rank==0:
        os.makedirs(cfg.output_dir, exist_ok=True)

    with open(os.path.join(cfg.output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    logger.info("Preparing datasets. This usually takes some time, since the the CheXmask dataframe is large.")

    if args.predict_only:
        logger.info("Predict only mode. Loading test dataset only.")
        assert "test" in cfg, "Please provide test dataset config"
        train_dataset,valid_dataset = None, None
        test_dataset = make_cxr_datasets(dataset_configs=cfg.test.datasets)
        logger.info("Test dataset loaded!")        
        num_classes = len(test_dataset.dataset.labels)
        ds_labels = test_dataset.dataset.labels
        id2class = {k + 1: v for k, v in enumerate(test_dataset.dataset.labels)}
    else:
        train_dataset = make_cxr_datasets(dataset_configs=cfg.train.datasets)
        logger.info(f"Train dataset loaded with {len(train_dataset)} samples!")
        valid_dataset = make_cxr_datasets(dataset_configs=cfg.val.datasets)
        logger.info(f"Valid dataset loaded with {len(valid_dataset)} samples!")
        if "test" in cfg:
            test_dataset = make_cxr_datasets(dataset_configs=cfg.test.datasets)
        else:
            test_dataset = None
        logger.info(f"Test dataset loaded with {len(test_dataset)} samples!")
        num_classes = len(train_dataset.dataset.labels)
        ds_labels = train_dataset.dataset.labels
        id2class = {k + 1: v for k, v in enumerate(train_dataset.dataset.labels)}

    logger.info("-" * 50)
    logger.info("DATSETS")
    logger.info(f"Train dataset: \n {train_dataset}, len={len(train_dataset) if train_dataset else None}")
    logger.info(f"Valid dataset: \n {valid_dataset}, len={len(valid_dataset) if valid_dataset else None}")
    logger.info(f"Test dataset: \n {test_dataset}, len={len(test_dataset) if test_dataset else None}")
    if num_classes == 1:
        #binary
        pass
    else:
        #multi-class
        if train_dataset and not train_dataset.dataset_name == 'VinDR_RibCXR_Segmentation': 
            num_classes+=1 #
        elif train_dataset is None and not test_dataset.dataset_name == 'VinDR_RibCXR_Segmentation':
            num_classes+=1
    
    logger.info(f"Num classes= {num_classes}: {ds_labels}")
    logger.info("-" * 50)

    trainer_output_dir = os.path.join(args.output_dir, "trainer_output")
    os.makedirs(trainer_output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=trainer_output_dir,
        eval_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        save_strategy="epoch",
        logging_dir=trainer_output_dir,
        logging_steps=10,
        push_to_hub=False,
        report_to="wandb",
        run_name=exp_name,
        remove_unused_columns=False,  # important!
        dataloader_num_workers=args.num_workers,
        lr_scheduler_type="cosine",
        metric_for_best_model='dice_avg',
        load_best_model_at_end=True,
        greater_is_better=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ddp_find_unused_parameters=False,
        deepspeed = args.deepspeed,
    )

    logger.info(f"Initiailizing model for segmentation, criterion = {args.criterion}. Backbone type: {args.model_type}")
    
    if args.model_type == "raddino":
        logger.info(f"Loading model from HuggingFace: {args.pretrained_weights}")
        backbone = AutoModel.from_pretrained(args.pretrained_weights)
        model_dim = backbone.config.hidden_size
        patch_size = 14
    elif args.model_type == "dinov2":
        backbone, model_dim = build_model_from_cfg(cfg, only_teacher=True)
        if args.pretrained_weights:
            checkpoint = torch.load(args.pretrained_weights)
            if "teacher" in checkpoint:
                checkpoint = checkpoint["teacher"]
                checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}
            backbone_load_response = backbone.load_state_dict(checkpoint, strict=False)
            logger.info(f"backbone loaded: {backbone_load_response}")
        else:
            logger.info("No pretrained weights provided. Loading from scratch!")
        patch_size = 14
    elif args.model_type == "chexzero":
        backbone = VisualTransformer(
            input_resolution=224,
            patch_size=32,
            width=768,
            layers=12,
            heads=8,
            output_dim=512,
            do_interpolate_positional_encoding=True,
        )
        patch_size = 32
        model_dim = 768
        sd = torch.load(args.pretrained_weights, map_location="cpu")
        sd = {k: v for k, v in sd.items() if k.startswith("visual.")}
        checkpoint = {k.replace("visual.", ""): v for k, v in sd.items()}

        backbone_load_response = backbone.load_state_dict(checkpoint, strict=True)
        logger.info(f"backbone loaded: {backbone_load_response}")
    elif args.model_type == 'biomedclip':
        backbone, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        backbone = backbone.visual.trunk
        model_dim = backbone.embed_dim
        patch_size = 16
    else:
        raise ValueError("Please provide a valid model type")

    if not args.train_backbone:
        for name, param in backbone.named_parameters():
            param.requires_grad = False

    if train_dataset:
        input_sample = train_dataset[0]["image"]
    else:
        input_sample = test_dataset[0]["image"]

    tokenW, tokenH = (
        input_sample.shape[1] // patch_size,
        input_sample.shape[2] // patch_size,
    )

    model = Dinov2ForSemanticSegmentation(
        dinov2_backbone=backbone,
        hidden_size=model_dim,
        tokenW=tokenW,
        tokenH=tokenH,
        num_labels=num_classes,
        loss_fct=args.criterion,
        backbone_frozen=not args.train_backbone,
    )

    if args.predict_only:
        sd = load_file(args.segmentation_model_path)
        msg = model.load_state_dict(sd)
        logger.info(f"Model for prediction loaded: {msg}")

    if (train_dataset and train_dataset.dataset_name == 'SIIM_ACR_PNX_Segmentation') or (test_dataset and test_dataset.dataset_name == 'SIIM_ACR_PNX_Segmentation'):
        trainer = CustomSIIMTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=CustomDataCollator(),
            compute_metrics=lambda eval_pred: compute_metrics(
                eval_pred, num_classes=num_classes, id2class=id2class,ignore_index=0
            ),  # calculate val metrics
        )
    elif (train_dataset and train_dataset.dataset_name == 'VinDR_RibCXR_Segmentation') or (test_dataset and test_dataset.dataset_name == 'VinDR_RibCXR_Segmentation'):
        trainer = CustomVinDrTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=CustomDataCollator(),
            compute_metrics=lambda eval_pred: compute_metrics_vindr(
                eval_pred, num_classes=num_classes, id2class=id2class
            ),  # calculate val metrics
        )
    else:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=CustomDataCollator(),
            compute_metrics=lambda eval_pred: compute_metrics(
                eval_pred, num_classes=num_classes, id2class=id2class
            ),  # calculate val metrics
        )


    if not args.predict_only:
        trainer.train()
    if test_dataset:
        trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

    try:
        logger.info("Predicting masks on test dataset")
        output = trainer.predict(test_dataset)
        masks = output["masks"]
        image_ids = output["image_ids"]
        masks_output_dir = os.path.join(args.output_dir, 'masks_predictions')
        os.makedirs(masks_output_dir, exist_ok=True)
        for i in tqdm(range(len(image_ids)), desc="Saving masks", unit="mask"):
            mask = masks[i]
            image_id = image_ids[i]            
            filename = os.path.join(masks_output_dir, f"{image_id}.npy")            
            np.save(filename, mask)
        logger.info(f"Mask predictions saved to: {masks_output_dir}")
    except Exception as e:
        logger.error(f"Error in prediction: {e}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser("DINOv2 training")
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument("--output-dir", type=str, default="", help="path to output dir")
    parser.add_argument("--num-epochs", type=int, default=None, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--criterion", type=str, default='dice_loss', help="Criterion to use", choices=SUPPORTED_CRITERIONS)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers")
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        default="",
        help="path to pretrained teacher weights with heads",
    )
    parser.add_argument(
        "--exp-name", type=str, default=None, help="Wandb experiment name"
    )
    parser.add_argument("--lr", type=float, default=None, help="learning rate")
    parser.add_argument("--predict-only", action="store_true", help="predict only")
    parser.add_argument("--train-backbone", action="store_true", help="train vision encoder")
    parser.add_argument("--segmentation-model-path", type=str, default="", help="path to pretrained segmentation model")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="weight decay")
    parser.add_argument(
        "--model-type",
        type=str,
        default="dinov2",
        help="Model type",
        choices=["dinov2", "raddino", "chexzero","biomedclip"],
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="local rank passed from distributed launcher",
    )

    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="deepspeed config",
    )
    args = parser.parse_args()

    global logger

    logger = logging.getLogger("dinov2-segmentation")
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the log message format
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "log.txt")
            ),  # Log messages to a file named example.log
            logging.StreamHandler(),  # Optionally, log messages to the console
        ],
    )
    main(args)
