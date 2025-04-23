# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from torchvision import transforms

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)


logger = logging.getLogger("dinov2")

class CenterCropByScale:
    def __init__(self, scale):
        self.scale = scale
    
    def __call__(self, image):
        width, height = image.size  
        crop_width = int(self.scale * width)
        crop_height = int(self.scale * height)
        
        return transforms.CenterCrop((crop_height, crop_width))(image)

class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        smart_local=False
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.smart_local = smart_local


        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        if self.smart_local:
            logger.info("Using smart local crops!")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.8, 1.2))
            ]
        )
        if not self.smart_local:
            self.geometric_augmentation_local = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )
        else:
            self.geometric_augmentation_local = transforms.Compose(
                [
                    CenterCropByScale(0.75), # first center crop (to contain lung region),
                    transforms.RandomResizedCrop(
                        local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def toggle_smart_local(self):
        self.smart_local = not self.smart_local
        if self.smart_local:
            logger.info("Switching smart local crops ON!")
            self.geometric_augmentation_local = transforms.Compose(
                [
                    CenterCropByScale(0.75), # first center crop (to contain lung region),
                    transforms.RandomResizedCrop(
                        self.local_crops_size, scale=self.local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )
        else:
            logger.info("Switvhing smart local crops OFF!")
            self.geometric_augmentation_local = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        self.local_crops_size, scale=self.local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
