import sys
import os
import argparse
from dinov2.train.ssl_meta_arch import SSLMetaArch
from omegaconf import OmegaConf
from dinov2.configs import dinov2_default_config
from train_utils import build_optimizer, build_schedulers, count_trainable_params
from tqdm import tqdm
from data_utils import get_mask_generator, get_data_transformations, make_cxr_datasets

import torch
from dist_utils import get_rank, ddp_setup, get_policies, wrap_in_fsdp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from dino_ssl import DINO_SSL
from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from io_utils import printit, save_checkpoint

torch.backends.cudnn.benchmark = True

# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)


def ema_update(student_model, teacher_model, weight):
    student_params = list(student_model.parameters())
    teacher_params = list(teacher_model.parameters())

    rank = torch.distributed.get_rank()
    if rank == 0:
        breakpoint()
    else:
        torch.distributed.barrier()

    for tp, sp in zip(teacher_params, student_params):
        tp.data = torch.lerp(sp.data, tp.data, weight)


def do_train(cfg, model, resume=False):
    model.train()

    rank = torch.distributed.get_rank()
    current_device = torch.cuda.current_device()

    # setup optimizer
    optimizer = build_optimizer(cfg, model.parameters())

    # loss function
    dino_loss = DINOLoss(cfg.dino.head_n_prototypes)
    koleo_loss_fn = KoLeoLoss()
    ibot_loss_fn = iBOTPatchLoss(cfg.ibot.head_n_prototypes)
    if cfg.dino.koleo_loss_weight:
        printit("Applying KOLEO Regularizer... !!")

    n_global_crops = cfg.crops.global_crops_number
    n_local_crops = cfg.crops.local_crops_number
    n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
    n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops
    loss_scales = 2  # hardcoded in original code :(
    ibot_loss_scales = 1.0 / n_global_crops

    collate_fn, mask_generator = get_mask_generator(cfg)
    data_transforms = get_data_transformations(cfg)
    dataset = make_cxr_datasets(cfg.train.datasets, data_transforms)
    print(dataset)

    ckpt_save_fpath = os.path.join(cfg.train.output_dir, "checkpoints")
    os.makedirs(ckpt_save_fpath, exist_ok=True)

    temp_epoch_length = cfg.train.OFFICIAL_EPOCH_LENGTH
    # OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    OFFICIAL_EPOCH_LENGTH = len(dataset) // (
        cfg.train.batch_size_per_gpu * torch.distributed.get_world_size()
    )
    cfg.train.OFFICIAL_EPOCH_LENGTH = OFFICIAL_EPOCH_LENGTH
    printit(
        f"Ignoring epoch length of {temp_epoch_length} mentioned in config, instead setting it to {OFFICIAL_EPOCH_LENGTH}"
    )
    max_iters = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
    printit(f"Maximum number of training iterations: {max_iters}")

    # setup schedules
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    dist_sampler = DistributedSampler(
        dataset,
        rank=rank,
        num_replicas=torch.distributed.get_world_size(),
        shuffle=True,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=dist_sampler if dist_sampler else None,
    )

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
        colour='green',
        disable=not rank,
    ):
        dist_sampler.set_epoch(curr_epoch)

        for data in tqdm(data_loader, desc=f"Epoch #{curr_epoch}", disable=not rank, colour='blue'):

            ### setup model in train mode
            model.train()

            batch_iterations += 1
            teacher_temp = teacher_temp_schedule[batch_iterations]
            data["teacher_temp"] = teacher_temp
            for tempkey in data_keys:
                data[tempkey] = data[tempkey].to(current_device, non_blocking=True)

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
                teacher_out_softmaxed_centered_list=[
                    teacher_output["dino_output"].flatten(0, 1)
                ],
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

            total_loss = (
                (cfg.dino.loss_weight * dino_local_loss)
                + (loss_scales * dino_global_loss)
                + (ibot_loss_scales * loss_scales * ibot_loss)
            )
            if cfg.dino.koleo_loss_weight:
                koleo_loss = sum(
                    koleo_loss_fn(p)
                    for p in student_output["student_local_cls_tokens"].chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                total_loss += cfg.dino.koleo_loss_weight * koleo_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()

            # clip gradients
            if cfg.optim.clip_grad:
                if cfg.fsdp_model:
                    grad_norm = model.clip_grad_norm_(cfg.optim.clip_grad).item()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.optim.clip_grad
                    )

            optimizer.step()

            if batch_iterations%5 == 0:
                printit(f"At iter {batch_iterations}, total_loss: {total_loss.item():.4f}", add_lines=False)

            # #Update teacher model params
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

                if batch_iterations % cfg.train.saveckp_freq == 0:
                    save_checkpoint(
                        model.teacher_backbone,
                        rank,
                        os.path.join(
                            ckpt_save_fpath, f"ckpt_{batch_iterations/1000:.1f}k.pth"
                        ),
                    )

            torch.distributed.barrier()
            # if batch_iterations%10==0:
            #     del data, output, student_output, teacher_output, dino_local_loss, dino_global_loss, ibot_loss, total_loss
            #     torch.cuda.empty_cache()

        # loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)


def main():
    parser = argparse.ArgumentParser(description="Train dinov2")
    parser.add_argument(
        "--config_file", type=str, default="dinov2/configs/ssl_default_config.yaml"
    )
    parser.add_argument(
        "--fsdp", action="store_true", help="Run the codebase in Pytorch DDP"
    )
    args = parser.parse_args()

    default_cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg)  # , OmegaConf.from_cli(args.opts))

    cfg.fsdp_model = args.fsdp
    local_rank = None
    rank = None

    ## reproducibility stuff
    torch.cuda.manual_seed_all(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    if args.fsdp:
        # setup PG
        torch.distributed.init_process_group(backend="nccl")

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = torch.distributed.get_rank()
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        # clear gpu cache
        torch.cuda.empty_cache()
        printit(f"FSDP Mode on...!!\nRANK: {rank}")

        print("-" * 50)
        print(f"Setting up FSDP on local_rank: {local_rank}")
        print("-" * 50)

    model = DINO_SSL(cfg, rank)

    printit(model)
    printit(f"Trainable params: {count_trainable_params(model)/1e6} Million")

    if cfg.compute_precision.policy in ["bf16", "bf16_mix"]:
        model = model.to(torch.bfloat16)
    model = wrap_in_fsdp(cfg, model, rank)

    # count trainable params after FSDP
    all_params = list(model.parameters())
    cnt = 0
    for p in all_params:
        if p.requires_grad:
            cnt += len(p)

    print(f"#trainable params on rank {rank}: {cnt/1e6} Million")

    do_train(cfg, model)


if __name__ == "__main__":
    main()
