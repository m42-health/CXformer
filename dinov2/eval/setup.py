# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse, os
from typing import Any, List, Optional, Tuple
from transformers import AutoModel

import torch
import torch.backends.cudnn as cudnn

from dinov2.models import build_model_from_cfg
from dinov2.utils.config import setup
import dinov2.utils.utils as dinov2_utils


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents or [],
        add_help=add_help,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory to write results and logs",
    )
    parser.add_argument(
        "--opts",
        help="Extra configuration options",
        default=[],
        nargs="+",
    )
    return parser


def get_autocast_dtype(config):
    teacher_dtype_str = config.compute_precision.teacher.backbone.mixed_precision.param_dtype
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def build_model_for_eval(config, pretrained_weights):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    model.eval()
    model.cuda()
    return model

def build_model_for_finetune(config, pretrained_weights):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    model.train()
    model.cuda()
    return model


def setup_and_build_model(args) -> Tuple[Any, torch.dtype]:
    cudnn.benchmark = True
    config = setup(args)
    if os.path.exists(args.pretrained_weights):
        model = build_model_for_eval(config, args.pretrained_weights)
    else:
        try:
            print(f"Path to pretrained weights {args.pretrained_weights} does not exist. Attempting to load this from HuggingFace")
            model = AutoModel.from_pretrained(args.pretrained_weights)
            model.eval()
            model.cuda()
        except:
            print(f"Could not load model from HuggingFace. Exiting.")
            exit(1)
    autocast_dtype = get_autocast_dtype(config)
    return model, autocast_dtype

def setup_and_build_model_raddino(args) -> Tuple[Any, torch.dtype]:
    cudnn.benchmark = True
    config = setup(args)

    model = AutoModel.from_pretrained('microsoft/rad-dino')
    msg = model.load_state_dict(torch.load(args.pretrained_weights, map_location='cpu'))
    print(f"rad-dino loaded with message: {msg}")
    model.eval()
    model.cuda()

    autocast_dtype = get_autocast_dtype(config)
    return model, autocast_dtype

def setup_and_build_model_ft(args) -> Tuple[Any, torch.dtype]:
    cudnn.benchmark = True
    config = setup(args)
    if os.path.exists(args.pretrained_weights):
        model = build_model_for_finetune(config, args.pretrained_weights)
    else:
        try:
            print(f"Path to pretrained weights {args.pretrained_weights} does not exist. Attempting to load this from HuggingFace")
            model = AutoModel.from_pretrained(args.pretrained_weights)
            model.train()
            model.cuda()
        except:
            print(f"Could not load model from HuggingFace. Exiting.")
            exit(1)
    autocast_dtype = get_autocast_dtype(config)
    return model, autocast_dtype