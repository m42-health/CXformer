import torchvision
from dinov2.data.transforms import make_classification_train_transform
from dinov2.data import make_cxr_datasets
import torch
import torch.nn as nn
from transformers import Trainer
import transformers

###################################
#           DATA
####################################

def get_datasets(cfg, model_type='dinov2'):

    if model_type == "raddino" or model_type=='dinov2':
    
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


###################################################
#               MODEL, Trainer
###################################################

class CustomTrainer(Trainer):
    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        test_dataloader = self.get_test_dataloader(test_dataset)
        eval_loop = self.evaluation_loop
        output = eval_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )        

        logits = output.predictions
        preds = torch.nn.functional.sigmoid(logits)  # Shape: (B, C+1, H, W)
        return {"preds": preds}

class CustomDataCollator:
    def __call__(self, features):
        images = torch.stack([torch.tensor(f["image"]) for f in features])
        labels = torch.stack([torch.tensor(f["lab"]) for f in features])

        return {"pixel_values": images, "labels": labels}

class Dinov2ForClassification(torch.nn.Module):
    def __init__(
        self, backbone, linear_clf, cls_n_layers, apply_avgpool,freeze_backbone=True
    ):
        super(Dinov2ForClassification, self).__init__()

        self.model = backbone  
        self.model.train(not freeze_backbone)
        self.linear_clf = linear_clf
        self.freeze_backbone = freeze_backbone
        self.cls_n_layers = cls_n_layers
        self.apply_avgpool = apply_avgpool

        self.loss_fn = nn.BCEWithLogitsLoss()


    def forward(self, pixel_values, labels=None):

        with torch.no_grad() if not self.freeze_backbone else torch.enable_grad():
            if isinstance(self.model,transformers.models.dinov2.modeling_dinov2.Dinov2Model):
                intermediate_output = self.model(pixel_values,output_hidden_states=True)
                hidden_states = intermediate_output.hidden_states
                blocks_to_take = range(len(hidden_states) - self.cls_n_layers, len(hidden_states))
                output = [hidden_states[i][:, 0] for i in blocks_to_take]
                if self.apply_avgpool:
                    output.append(torch.mean(hidden_states[-1][:, 1:], dim=1))
                output = torch.cat(output, dim=-1)
            else:
                intermediate_output = self.model.get_intermediate_layers(pixel_values, self.cls_n_layers)
                output = [x[:, 0] for x in intermediate_output]
                if self.apply_avgpool:
                    output.append(torch.mean(intermediate_output[-1][:, 1:], dim=1))
                output = torch.cat(output, dim=-1)
        output = self.linear_clf(output)

        loss = self.loss_fn(output, labels)

        return {"loss": loss, "logits": output}


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