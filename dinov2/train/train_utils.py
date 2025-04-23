import torch
from dinov2.utils.utils import CosineScheduler

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_checkpoint_statedict(fpath):

    checkpoint = torch.load(fpath)
    if 'model' in checkpoint.keys():
        checkpoint = checkpoint["model"]
    elif 'teacher' in checkpoint.keys():
        checkpoint = checkpoint['teacher']
        #remove ".backbone" prefix from keys
        checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}
    
    return checkpoint
        

def build_optimizer(cfg, param_groups):
    optim = torch.optim.AdamW(
        param_groups, 
        lr=cfg.optim.base_lr,
        betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2),
        weight_decay=cfg.optim.weight_decay,
    )
    return optim

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

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    print("Schedulers ready...!!")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )