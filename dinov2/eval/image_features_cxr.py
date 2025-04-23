# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
from functools import partial
import json
import logging, transformers
import os
import sys
from typing import List, Optional
from omegaconf import OmegaConf
from transformers import AutoImageProcessor
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
from sklearn.metrics import classification_report
from dinov2.data import SamplerType, make_data_loader, make_cxr_datasets
from dinov2.data.transforms import make_classification_eval_transform, make_classification_train_transform
import dinov2.distributed as distributed
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.train.dino_ssl import DINO_SSL
from torchvision import transforms
from dinov2.data.masking import MaskingGenerator
from dinov2.data.transforms import (
    GaussianBlur,
    make_normalize_transform,
)
import random
import pandas as pd

def get_cfg_from_args(args):
    args.output_dir = os.path.abspath(args.output_dir)
    cfg = OmegaConf.load(args.config_file)
    return cfg

logger = logging.getLogger("dinov2")

class DINOEvalAugmentations(object):
    def __init__(
        self,
        global_crops_scale,
        global_crops_size=224,
    ):
        self.global_crops_scale = global_crops_scale
        self.global_crops_size = global_crops_size
        # random resized crop
        self.geometric_augmentation_global = transforms.Compose(
            [   
                transforms.Resize(global_crops_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(global_crops_size),
            ]
        )
        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )
        self.global_transfo1 = transforms.Compose([self.normalize])
        self.global_transfo2 = transforms.Compose([self.normalize])

    def __call__(self, image):
        output = {}

        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)
        output["global_crops"] = [global_crop_1]
        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1]
        output["offsets"] = ()
        return output
    
def collate_data_and_cast_cxr(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    n_global_crops = len(samples_list[0]['image']["global_crops"])
    collated_global_crops = torch.stack([s['image']["global_crops"][i] for i in range(n_global_crops) for s in samples_list])    
    
    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))
    random.shuffle(masks_list)
    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
        "paths": [s['path'] for s in samples_list],
    }

def calculate_batch_entropy(batch_tensor):
    
    sums = batch_tensor.sum(dim=-1)
    assert torch.all(torch.isclose(sums, torch.ones_like(sums), atol=1e-6)), "Must be softmaxed probabilities"

    batch_tensor = batch_tensor + 1e-10
    entropy = -torch.sum(batch_tensor * torch.log(batch_tensor), dim=-1)
    return entropy


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parents = parents or []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch Size (per GPU)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number de Workers",
    )
    parser.add_argument(
        "--dino-center",
        type=str,
        help="Path to DINO center tensor",
        default=None,
    )
    parser.add_argument(
        "--ibot-center",
        type=str,
        help="Path to iBOT center tensor",
        default=None,
    )    
    parser.set_defaults(
        batch_size=128,
        num_workers=8,
        test_metric_types=None,
    )
    return parser


def main(args):
    from dinov2.utils.config import setup
    # rank = torch.distributed.get_rank()
    current_device = torch.cuda.current_device()

    inputs_dtype = torch.float
    cfg = setup(args)
    cfg.pretrained_weights = args.pretrained_weights
    cfg.student.pretrained_weights = args.pretrained_weights
    cfg.teacher.pretrained_weights = args.pretrained_weights
    cfg.crops.global_crops_number = 1
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2

    model = DINO_SSL(cfg)
    
    if args.ibot_center:
        ibot_center = torch.load(args.ibot_center,map_location='cpu') 
        model.ibot_patch_loss.center = ibot_center
    if args.dino_center:
        dino_center = torch.load(args.dino_center, map_location='cpu')    
        model.dino_loss.center = dino_center
    
    model = model.to(torch.device("cuda"))
    model.eval()

    data_transform = DINOEvalAugmentations(
        cfg.crops.global_crops_scale,
        global_crops_size=cfg.crops.global_crops_size,
    )
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )
    collate_fn = partial(
        collate_data_and_cast_cxr,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # setup data loaderxwww
    dataset = make_cxr_datasets(
        dataset_configs=cfg.train.datasets,
        dino_transforms=data_transform
    )
    logger.info(f"Length of dataset: {len(dataset)}")
    sampler_type = SamplerType.DISTRIBUTED

    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        seed=cfg.train.seed,
        sampler_type=sampler_type,
        sampler_advance=0,  
        drop_last=False,
        collate_fn=collate_fn,
    )
    data_keys = [
        "collated_global_crops",
        "collated_masks",
        "mask_indices_list",
        "n_masked_patches",
        "masks_weight",
    ]

    dino_entropies = torch.tensor([])   
    ibot_entropies = torch.tensor([])   
    paths = []
    # iterate over the data loader using tqdm
    for data in tqdm(data_loader,total=len(data_loader)):
        for tempkey in data_keys:
            data[tempkey] = data[tempkey].to(current_device, non_blocking=True)
        data['teacher_temp'] = cfg.teacher.teacher_temp
        with torch.no_grad():
            dino_sm_outputs, ibot_sm_outputs = model.teacher_eval_forward(data)
            dino_sm_outputs = dino_sm_outputs.squeeze(0) # B, D
            assert dino_sm_outputs.shape[0] <= cfg.train.batch_size_per_gpu and dino_sm_outputs.shape[1] == cfg.dino.head_n_prototypes

        dino_entropy = calculate_batch_entropy(dino_sm_outputs).cpu()
        dino_entropies = torch.cat((dino_entropies, dino_entropy))

        ibot_entropy = calculate_batch_entropy(ibot_sm_outputs).cpu()
        ibot_entropies = torch.cat((ibot_entropies, ibot_entropy))
        paths.extend(data['paths'])
    
    dino_entropies = dino_entropies.numpy()
    ibot_entropies = ibot_entropies.numpy()
    ibot_entropies_mean = np.mean(ibot_entropies,axis=-1).tolist()

    ibot_entropies_list = []
    for row in ibot_entropies:
        ibot_entropies_list.append(row.tolist())
    paths = np.array(paths)
    df = pd.DataFrame({"path": paths, "dino_entropy": dino_entropies,"ibot_entropies_mean":ibot_entropies_mean ,"ibot_entropy": ibot_entropies_list, })
    df.to_csv(os.path.join(cfg.train.output_dir, "dino_entropies.csv"), index=False)
    logger.info("DINO entropies saved to %s", os.path.join(cfg.train.output_dir, "dino_entropies.csv"))

    return 0


if __name__ == "__main__":
    description = "DINOv2"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
