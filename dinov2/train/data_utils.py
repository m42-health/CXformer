from dinov2.data import MaskingGenerator
from dinov2.data.augmentations import DataAugmentationDINO
from typing import Any, Callable, List, Optional, TypeVar, Dict
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf
from cxr_data import CXRDataset
from torch.utils.data import ConcatDataset
import torch
from functools import partial
import random

def collate_data_and_cast_cxr(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # dtype = torch.half  # TODO: Remove

    n_global_crops = len(samples_list[0]['image']["global_crops"])
    n_local_crops = len(samples_list[0]['image']["local_crops"])

    collated_global_crops = torch.stack([s['image']["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    collated_local_crops = torch.stack([s['image']["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
  
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
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }

def get_mask_generator(cfg):

    data_dtype = None
    if cfg.compute_precision.policy == "fp32":
        data_dtype = torch.float32
    elif cfg.compute_precision.policy == 'bf16':
        data_dtype = torch.bfloat16
    else:
        raise NotImplementedError
    
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    
    mask_generator = MaskingGenerator(input_size=(img_size//patch_size, img_size//patch_size), max_num_patches=
                                      0.5 * img_size//patch_size * img_size//patch_size)

    collate_fn = partial(
        collate_data_and_cast_cxr,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        # dtype = torch.half,
        dtype = data_dtype
    )

    return collate_fn, mask_generator


def get_data_transformations(cfg):
    data_transformations = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )

    return data_transformations


def make_cxr_datasets(
        dataset_configs: List[Dict],
        dino_transforms: Optional[Callable]=None,
        transforms: Optional[Callable]=None,
):  
    if isinstance(dataset_configs, (DictConfig, ListConfig)):
        dataset_configs = OmegaConf.to_container(dataset_configs)
        if isinstance(dataset_configs, dict):
            dataset_configs = [dataset_configs]

    if dino_transforms:
        # add DINO transforms (global, local crops, augmentations) after any user-defined augmentations
        [dataset_config['transform'].append(dino_transforms) for dataset_config in dataset_configs]

    if transforms:
        [dataset_config['transform'].append(transforms) for dataset_config in dataset_configs]

    if len(dataset_configs) == 1:
        dataset_configs = dataset_configs[0]
        return CXRDataset(**dataset_configs)
    dataset_list = [CXRDataset(**dataset_args) for dataset_args in dataset_configs]

    return ConcatDataset(dataset_list)