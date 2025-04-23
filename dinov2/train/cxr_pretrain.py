# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from functools import partial
import wandb
from fvcore.common.checkpoint import PeriodicCheckpointer
import torch
from omegaconf import OmegaConf
from dinov2.data import SamplerType, make_data_loader, make_dataset, make_cxr_datasets
from dinov2.data import collate_data_and_cast, collate_data_and_cast_cxr,DataAugmentationDINO, MaskingGenerator
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler, StepScheduler
from dinov2.train.ssl_meta_arch import SSLMetaArch
import torch.utils.bottleneck as bottleneck


torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    # parser.add_argument("--local-rank", default=0, type=int, help="Variable for distributed computing.") 

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    logger.info("LR schedule: {}".format(cfg.optim.lr_schedule))
    if cfg.optim.lr_schedule == "cosine":
        lr_schedule = CosineScheduler(**lr)
        last_layer_lr_schedule = CosineScheduler(**lr)

    elif cfg.optim.lr_schedule == "step":
        # update lr dict with gamma, step_size
        lr.update({
            "step_size": cfg.optim["lr_step_size"],
            "gamma": cfg.optim["lr_gamma"]
        
        })
        lr_schedule = StepScheduler(**lr)
        last_layer_lr_schedule = StepScheduler(**lr)
    else:
        raise ValueError(f"Invalid lr_schedule: {cfg.optim.lr_schedule}")

    
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def do_test(cfg, model, iteration):

    new_state_dict = model.teacher.state_dict()
    ibot_center = model.ibot_patch_loss.center
    dino_center = model.dino_loss.center
               
    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)
        torch.save(ibot_center, os.path.join(eval_dir, "ibot_center.pt"))
        torch.save(dino_center, os.path.join(eval_dir, "dino_center.pt"))


def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    # setup optimizer

    print("="*50)
    print(f"Gradient accumulation: {cfg.optim['accumulate_grad_batches']}")
    print("="*50)

    ##@PM: Handle LR when grad_accumulation > 1
    if cfg.optim['accumulate_grad_batches'] > 1:
        cfg.optim["lr"] *= cfg.optim['accumulate_grad_batches']
        cfg.optim["min_lr"] *= cfg.optim['accumulate_grad_batches']

    optimizer = build_optimizer(cfg, model.get_params_groups())

    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)


    # checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
    
    # this will resume, if possible, loading model weights, scheduler, optim, etc.
    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    # Override model weights with provided weights
    # if cfg.MODEL.WEIGHTS_OVERRIDE:
    #     logger.info(f"Overriding model weights with {cfg.MODEL.WEIGHTS_OVERRIDE}")
    #     sd = torch.load(cfg.MODEL.WEIGHTS_OVERRIDE, map_location='cpu')["teacher"]
    #     t = model.teacher.load_state_dict(sd)
    #     s = model.student.load_state_dict(sd)
    #     logger.info(f"Student loaded from {cfg.MODEL.WEIGHTS_OVERRIDE} with status: {s}")
    #     logger.info(f"Teacher loaded from {cfg.MODEL.WEIGHTS_OVERRIDE} with status: {t}")

    if cfg.optim.override_start_iter >= 0:
        start_iter = cfg.optim.override_start_iter
        logger.info(f"Overriding start_iter with {start_iter}")

    torch.cuda.synchronize()

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period= cfg.evaluation.eval_period_iterations, #3 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # setup data preprocessing

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
        smart_local=cfg.crops.smart_local,
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
    logger.info(f"Number of training samples {len(dataset)}")
    # sampler_type = SamplerType.INFINITE
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=cfg.train.seed,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
        batch_balance_ds=cfg.optim.backprop_loss_dataset
    )
    # training loop
    iteration = start_iter
    logger.info(f"Gradient accumulation batches: {cfg.optim.accumulate_grad_batches}")
    logger.info(f"Backpropping highest loss dataset-wise: {cfg.optim.backprop_loss_dataset}")
    
    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):

        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return
        if cfg.optim.early_interrupt_training>0 and iteration >= cfg.optim.early_interrupt_training:
            logger.info("Early interrupting training at iteration {}".format(iteration))
            return
        # apply schedules
        accumulation_steps = cfg.optim.accumulate_grad_batches
        
        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)


        if cfg.crops.toggle_smart_local and iteration >= cfg.crops.toggle_smart_local:
            logger.info("Toggling smart local crops at iteration {}".format(iteration))
            data_loader.dataset.datasets[0].transform.transforms[-1].toggle_smart_local()
            cfg.crops.toggle_smart_local = False

        # compute losses
        if cfg.optim.backprop_loss_dataset: 
            loss_dict,loss_dict_dataset = model.forward_backward_custom(data, teacher_temp=teacher_temp)
        else:
            loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)
            loss_dict_dataset = None
        # clip gradients
        if (iteration + 1) % accumulation_steps == 0:
            if fp16_scaler is not None:
                if cfg.optim.clip_grad:
                    fp16_scaler.unscale_(optimizer)
                    for v in model.student.values():
                        v.clip_grad_norm_(cfg.optim.clip_grad)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                if cfg.optim.clip_grad:
                    for v in model.student.values():
                        v.clip_grad_norm_(cfg.optim.clip_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        # perform teacher EMA update
        if (iteration + 1) % accumulation_steps == 0:
            model.update_teacher(mom)

        # logging
        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
            
            if loss_dict_dataset is not None:
                for v in loss_dict_dataset.values():
                    torch.distributed.all_reduce(v)

        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}
        if loss_dict_dataset is not None:
            loss_dict_dataset_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict_dataset.items()}
        
        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        if distributed.is_main_process():
            wandb.log({"lr": lr},
                        step=iteration) 
            wandb.log({"wd": wd},
                        step=iteration) 
            wandb.log({"mom": mom},
                        step=iteration) 
            wandb.log({"last_layer_lr": last_layer_lr},
                        step=iteration) 
            wandb.log({"total_loss": losses_reduced},
                        step=iteration) 
            wandb.log({k:v for k,v in loss_dict_reduced.items()},step=iteration)
            if loss_dict_dataset is not None:
                wandb.log({f"{k}_total_loss":v for k,v in loss_dict_dataset_reduced.items()},step=iteration)

        if (iteration+1) % 100 == 0:
            try:
                teacher_sd = model.teacher.state_dict()
                alphas = [k for k in teacher_sd.keys() if 'alpha' in k]

                for i in range(len(alphas)):
                    all_alphas = teacher_sd[alphas[i]].cpu().detach()
                    all_alphas = torch.nn.Softmax(dim=0)(all_alphas)
                    layer_num = int(alphas[i].split('.')[-3])
                    for a, alpha in enumerate(all_alphas):
                        wandb.log({f"misc/layer_{layer_num}_alpha_{a}": alpha}, step=iteration)
            except Exception as e:
                pass
        # checkpointing and testing
        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            logger.info("Saving model at iteration {}".format(iteration))
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)
        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)
    # cfg must also have a key "train" for train dataset
    model = SSLMetaArch(cfg)       
    # breakpoint() 
    if model.teacher.backbone.blocks[-1].mlp.__class__.__name__ == 'MixtureActivationMlp':
        logger.info("Using MixtureActivationMlp")
        logger.info("Activations: {}".format(model.teacher.backbone.blocks[-1].mlp.activations))

    model = model.to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    exp_name = args.output_dir.split('/')[-1]
    if distributed.is_main_process():
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(project='dinov2-pretrain', name=exp_name, config=config_dict)
    
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")
    # do_train sets up optimizer, checkpointer, dataset and dataloaders, data augs, lr & wd schedules, 
    # teacher EMA updates, metrics logging and testing after training completes. 
    do_train(cfg, model, resume=not args.no_resume)
    do_test(cfg, model, f"eval_last")
if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    print("ARGS:")
    print(vars(args))
    main(args)
