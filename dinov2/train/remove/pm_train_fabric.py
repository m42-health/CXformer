import sys
import os
import argparse
from dinov2.train.ssl_meta_arch import SSLMetaArch
from omegaconf import OmegaConf
from dinov2.configs import dinov2_default_config
from train_utils import build_optimizer, build_schedulers, count_trainable_params
from tqdm import tqdm
from data_utils import get_mask_generator, get_data_transformations, make_cxr_datasets
from time import time

import torch
from dist_utils import get_rank, ddp_setup, get_policies, wrap_in_fsdp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from dino_ssl import DINO_SSL
from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from io_utils import printit, save_checkpoint, fab_print
from lightning.fabric import Fabric
from lightning.fabric.strategies.deepspeed import DeepSpeedStrategy
import wandb

torch.backends.cudnn.benchmark = True

# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)


def ema_update(student_model, teacher_model, weight):
    student_params = list(student_model.parameters())
    teacher_params = list(teacher_model.parameters())

    for tp, sp in zip(teacher_params, student_params):
        tp.data = torch.lerp(sp.data, tp.data, weight)


def do_train(cfg, model, fabric, resume=False):
    model.train()

    # setup optimizer
    optimizer = build_optimizer(cfg, model.parameters())

    model, optimizer = fabric.setup(model, optimizer)

    # loss function
    dino_loss = DINOLoss(cfg.dino.head_n_prototypes)
    koleo_loss_fn = KoLeoLoss()
    ibot_loss_fn = iBOTPatchLoss(cfg.ibot.head_n_prototypes)
    if cfg.dino.koleo_loss_weight:
        fabric.print("\nApplying KOLEO Regularizer... !!", fabric)

    n_global_crops = cfg.crops.global_crops_number
    n_local_crops = cfg.crops.local_crops_number
    n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
    n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops
    loss_scales = 2  # hardcoded in original code :(
    ibot_loss_scales = 1.0 / n_global_crops

    collate_fn, mask_generator = get_mask_generator(cfg)
    data_transforms = get_data_transformations(cfg)
    dataset = make_cxr_datasets(cfg.train.datasets, data_transforms)
    fabric.print(dataset)

    ckpt_save_fpath = os.path.join(cfg.train.output_dir, "checkpoints")
    os.makedirs(ckpt_save_fpath, exist_ok=True)

    temp_epoch_length = cfg.train.OFFICIAL_EPOCH_LENGTH
    # OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    OFFICIAL_EPOCH_LENGTH = len(dataset) // (cfg.train.batch_size_per_gpu * fabric.world_size)
    cfg.train.OFFICIAL_EPOCH_LENGTH = OFFICIAL_EPOCH_LENGTH
    fabric.print(f"Ignoring epoch length of {temp_epoch_length} mentioned in config, instead setting it to {OFFICIAL_EPOCH_LENGTH}")
    max_iters = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
    fabric.print(f"\nMaximum number of training iterations: {max_iters}")

    # setup schedules
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )

    data_loader = fabric.setup_dataloaders(data_loader)

    data_keys = [
        "collated_global_crops",
        "collated_local_crops",
        "collated_masks",
        "mask_indices_list",
        "n_masked_patches",
        "masks_weight",
    ]

    batch_iterations = 0
    for curr_epoch in tqdm(
        range(1, cfg.optim.epochs),
        desc=f"Training for {cfg.optim.epochs} epochs!!",
        colour="green",
        disable=not fabric.global_rank,
    ):

        for data in tqdm(
            data_loader,
            desc=f"Epoch #{curr_epoch}",
            disable=not fabric.global_rank,
            colour="blue",
        ):

            ### setup model in train mode
            model.train()

            batch_iterations += 1
            teacher_temp = teacher_temp_schedule[batch_iterations]
            data["teacher_temp"] = teacher_temp
            # for tempkey in data_keys:
            #     data[tempkey] = data[tempkey].to(current_device, non_blocking=True)

            output = model(data)

            student_output = output["student_output"]
            teacher_output = output["teacher_output"]

            ####################### DINO LOSS #######################
            # local crops loss
            dino_local_loss = dino_loss(
                student_output_list=student_output["local_output"].chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_output["dino_output"],
            )
            dino_local_loss /= n_global_crops_loss_terms + n_global_crops_loss_terms

            # global crops loss
            dino_global_loss = dino_loss(
                student_output_list=[student_output["global_output"]],
                teacher_out_softmaxed_centered_list=[teacher_output["dino_output"].flatten(0, 1)],
                # these were chunked and stacked in reverse so A is matched to B
            )
            dino_global_loss /= n_global_crops_loss_terms + n_local_crops_loss_terms
            #####################################################################
            #                       IBOT LOSS
            #####################################################################

            ibot_loss = ibot_loss_fn.forward_masked(
                student_output["student_global_masked_patch_tokens_after_head"],
                teacher_output["ibot_output"],
                student_masks_flat=data["collated_masks"],
                n_masked_patches=data["n_masked_patches"].shape[0],
                masks_weight=data["masks_weight"],
            )
            if fabric.is_global_zero:
                wandb.log(
                    {
                        "Iterations": batch_iterations,
                        "dino_local_loss": dino_local_loss.item(),
                    },
                    commit=False,
                    step=batch_iterations,
                )
                wandb.log(
                    {
                        "Iterations": batch_iterations,
                        "dino_global_loss": dino_global_loss.item(),
                    },
                    commit=False,
                    step=batch_iterations,
                )
                wandb.log(
                    {
                        "Iterations": batch_iterations,
                        "dino_loss (local + global)": cfg.dino.loss_weight * dino_local_loss.item() + loss_scales * dino_global_loss.item(),
                    },
                    commit=False,
                    step=batch_iterations,
                )
                wandb.log(
                    {"Iterations": batch_iterations, "ibot_loss": ibot_loss.item()},
                    commit=False,
                    step=batch_iterations,
                )

            total_loss = (cfg.dino.loss_weight * dino_local_loss) + (loss_scales * dino_global_loss) + (ibot_loss_scales * loss_scales * ibot_loss)

            if cfg.dino.koleo_loss_weight:
                koleo_loss = sum(
                    koleo_loss_fn(p) for p in student_output["student_local_cls_tokens"].chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                if fabric.is_global_zero:
                    wandb.log(
                        {
                            "Iterations": batch_iterations,
                            "koleo_loss": koleo_loss.item(),
                        },
                        commit=False,
                        step=batch_iterations,
                    )
                total_loss += cfg.dino.koleo_loss_weight * koleo_loss

            if fabric.is_global_zero:
                wandb.log(
                    {"Iterations": batch_iterations, "Total Loss": total_loss.item()},
                    commit=True,
                    step=batch_iterations,
                )

            optimizer.zero_grad(set_to_none=True)
            # total_loss.backward()
            fabric.backward(total_loss)

            # clip gradients
            # if cfg.optim.clip_grad:
            #     fabric.clip_gradients(model, optimizer, max_norm=cfg.optim.clip_grad, norm_type=2)

            optimizer.step()

            if batch_iterations % 5 == 0:
                fabric.print(f"\nAt iter {batch_iterations}, total_loss: {total_loss.item():.4f}")

            # Update teacher model params
            with torch.no_grad():

                ema_update(
                    model.student_backbone,
                    model.teacher_backbone,
                    momentum_schedule[batch_iterations],
                )
                ema_update(
                    model.student_dino_head,
                    model.teacher_dino_head,
                    momentum_schedule[batch_iterations],
                )
                ema_update(
                    model.student_ibot_head,
                    model.teacher_ibot_head,
                    momentum_schedule[batch_iterations],
                )

            

            with torch.no_grad():
                if fabric.is_global_zero and batch_iterations % cfg.train.saveckp_freq == 0:
                    fabric.print("\nSaving AI model")
                    checkpoint_state = {}
                    checkpoint_state["teacher_model"] = model.teacher_backbone.state_dict()
                    start_time = time()
                    # fabric.save(
                    #     os.path.join(ckpt_save_fpath, f"ckpt_{batch_iterations/1000:.1f}k.pth"),
                    #     checkpoint_state,
                    # )
                    torch.save(checkpoint_state, os.path.join(ckpt_save_fpath, f"ckpt_{batch_iterations/1000:.1f}k.pth"))
                    end_time = time()
                    fabric.print(f"\nTime taken to save model state: {(end_time-start_time)/60:.3f} mins")

            fabric.barrier()

        # loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)


def main():
    parser = argparse.ArgumentParser(description="Train dinov2")
    parser.add_argument("--config_file", type=str, default="dinov2/configs/ssl_default_config.yaml")
    parser.add_argument("--fsdp", action="store_true", help="Run the codebase in Pytorch DDP")
    args = parser.parse_args()

    default_cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg)  # , OmegaConf.from_cli(args.opts))

    cfg.fsdp_model = args.fsdp
    local_rank = None
    rank = None

    # ds_strategy = DeepSpeedStrategy(
    #     stage=2,
    #     pin_memory=True,
    #     config={
    #         "gradient_clipping": 3.0,
    #         "grad_accum_dtype": "fp32",
    #         "bf16": {"enabled": "true"},
    #     },
    #     precision="bf16-true",
    # )
    fabric = Fabric(
        accelerator="cuda",
        devices=2,
        # strategy=ds_strategy,
        strategy="deepspeed_stage_2",
        precision="bf16-true",
    )
    fabric.launch()

    if fabric.is_global_zero:
        log_config = {}
        tempKeys = list(cfg.keys())
        for key in tempKeys:
            log_config[key] = getattr(cfg, key)

        run = wandb.init(
            entity="m42",
            project="dinov2_cxr_pm",  # "llama2-7b-opt",
            name=f"exp_bf16_chexpert",
            config=log_config,
        )

    ## reproducibility stuff
    fabric.seed_everything(cfg.train.seed)

    with fabric.init_module():
        model = DINO_SSL(cfg, rank)
    # model.to(torch.cuda.current_device())
    fabric.print(model)
    fabric.print(f"\nTrainable params: {count_trainable_params(model)/1e6} Million")

    do_train(cfg, model, fabric)

    if fabric.is_global_zero:
        wandb.run.finish()


if __name__ == "__main__":
    main()
