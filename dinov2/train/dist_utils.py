import torch
from torch.distributed import init_process_group
from pkg_resources import packaging
from io_utils import printit
from policies import fpSixteen, fpThirtyTwo, bfSixteen, bfSixteen_mixed
import os
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    lambda_auto_wrap_policy,
    _or_policy,
)
from torch.distributed.fsdp import ShardingStrategy
from dinov2.layers.block import NestedTensorBlock
from functools import partial
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)


def ddp_setup():
    init_process_group(backend="nccl")


def get_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def fsdp_auto_wrap_policy():

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transf_wrap_policy = partial(
        transformer_auto_wrap_policy, transformer_layer_cls={NestedTensorBlock}
    )

    auto_wrap_policy = partial(_or_policy, policies=[lambda_policy, transf_wrap_policy])
    return auto_wrap_policy


def get_policies(cfg, rank):

    verify_bfloat_support = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and torch.distributed.is_nccl_available()
        and torch.cuda.nccl.version() >= (2, 10)
    )

    printit(f"Support for bfloat16: {verify_bfloat_support}")

    mixed_precision_policy = None
    assert cfg.compute_precision.policy in [
        "fp16",
        "fp32",
        "bf16",
        "bf16_mix",
    ], f"Wrong value: {cfg.compute_precision.policy} passed. \n Supported values are ['fp16', 'fp32','bf16', 'bf16_mix']. Please use one of these values only."
    if cfg.compute_precision.policy == "fp32":
        mixed_precision_policy = fpThirtyTwo
    elif cfg.compute_precision.policy == "fp16":
        mixed_precision_policy = fpSixteen
    elif cfg.compute_precision.policy == "bf16":
        mixed_precision_policy = bfSixteen
    else:
        mixed_precision_policy = bfSixteen_mixed

    ## wrapping policy
    # model_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={NestedTensorBlock})
    model_wrap_policy = fsdp_auto_wrap_policy()

    ## sharding strategy
    shard_strategy = None
    assert cfg.compute_precision.sharding_strategy in [
        "FULL_SHARD",
        "SHARD_GRAD_OP",
        "NO_SHARD",
    ]

    if cfg.compute_precision.sharding_strategy == "SHARD_GRAD_OP":
        shard_strategy = ShardingStrategy.SHARD_GRAD_OP
    elif cfg.compute_precision.sharding_strategy == "FULL_SHARD":
        shard_strategy = ShardingStrategy.FULL_SHARD
    else:
        raise NotImplementedError

    return mixed_precision_policy, model_wrap_policy, shard_strategy


def wrap_in_fsdp(cfg, model, rank):
    mix_prec_policy, wrap_policy, shard_strategy = get_policies(cfg, rank)


    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mix_prec_policy,
        sharding_strategy=shard_strategy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        sync_module_states=False,
        # use_orig_params=True, 
        # TODO: Add cpu off load params functionality
    )

    return model
