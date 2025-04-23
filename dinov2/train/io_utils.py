import torch
import os
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from time import time

def map_hf_key_to_custom(hf_key):
    """Maps Hugging Face model keys to custom model keys."""
    if hf_key.startswith("embeddings.patch_embeddings.projection."):
        return hf_key.replace("embeddings.patch_embeddings.projection", "patch_embed.proj")
    if hf_key == "embeddings.cls_token":
        return "cls_token"
    if hf_key == "embeddings.mask_token":
        return "mask_token"
    if hf_key == "embeddings.position_embeddings":
        return "pos_embed"
    if hf_key == "layernorm.weight":
        return "norm.weight"
    if hf_key == "layernorm.bias":
        return "norm.bias"

    if hf_key.startswith("encoder.layer."):
        parts = hf_key.split(".")
        layer_num = parts[2]
        subkey = ".".join(parts[3:])

        mapping = {
            "norm1.weight": f"blocks.{layer_num}.norm1.weight",
            "norm1.bias": f"blocks.{layer_num}.norm1.bias",
            "norm2.weight": f"blocks.{layer_num}.norm2.weight",
            "norm2.bias": f"blocks.{layer_num}.norm2.bias",
            "mlp.fc1.weight": f"blocks.{layer_num}.mlp.fc1.weight",
            "mlp.fc1.bias": f"blocks.{layer_num}.mlp.fc1.bias",
            "mlp.fc2.weight": f"blocks.{layer_num}.mlp.fc2.weight",
            "mlp.fc2.bias": f"blocks.{layer_num}.mlp.fc2.bias",
            "layer_scale1.lambda1": f"blocks.{layer_num}.ls1.gamma",
            "layer_scale2.lambda1": f"blocks.{layer_num}.ls2.gamma",
            "attention.output.dense.weight": f"blocks.{layer_num}.attn.proj.weight",
            "attention.output.dense.bias": f"blocks.{layer_num}.attn.proj.bias",
        }

        return mapping.get(subkey)

    return None

def merge_qkv(layer_idx, hf_state):
    prefix = f"encoder.layer.{layer_idx}.attention.attention"
    try:
        qkv_w = torch.cat([
            hf_state[f"{prefix}.query.weight"],
            hf_state[f"{prefix}.key.weight"],
            hf_state[f"{prefix}.value.weight"]
        ], dim=0)
        qkv_b = torch.cat([
            hf_state[f"{prefix}.query.bias"],
            hf_state[f"{prefix}.key.bias"],
            hf_state[f"{prefix}.value.bias"]
        ], dim=0)
        return qkv_w, qkv_b
    except KeyError as e:
        print(f"[!] QKV merge failed for layer {layer_idx}: {e}")
        return None, None

def transfer_weights(hf_model, target_model, num_layers=12):
    """Transfers compatible weights from Hugging Face model to custom model."""
    hf_state = hf_model.state_dict()
    tgt_state = target_model.state_dict()
    new_state = tgt_state.copy()

    matched, skipped = 0, 0

    for hf_key, hf_val in hf_state.items():
        mapped_key = map_hf_key_to_custom(hf_key)
        if mapped_key is None:
            continue  # ignore keys that need special handling
        if mapped_key in tgt_state and hf_val.shape == tgt_state[mapped_key].shape:
            new_state[mapped_key] = hf_val
            matched += 1
        else:
            skipped += 1
            print(f"[!] Skipped: {hf_key} → {mapped_key} (missing or shape mismatch)")

    # Handle QKV block merging
    for i in range(num_layers):
        w_key = f"blocks.{i}.attn.qkv.weight"
        b_key = f"blocks.{i}.attn.qkv.bias"
        if w_key in tgt_state and b_key in tgt_state:
            qkv_w, qkv_b = merge_qkv(i, hf_state)
            if qkv_w is not None and qkv_w.shape == tgt_state[w_key].shape:
                new_state[w_key] = qkv_w
                new_state[b_key] = qkv_b
                matched += 1
                print(f"[✓] Merged QKV for block {i}")
            else:
                skipped += 1
                print(f"[!] Skipped QKV for block {i} (merge fail or shape mismatch)")

    target_model.load_state_dict(new_state)
    print(f"\n Weight Transfer Summary: {matched} matched, {skipped} skipped.")

def fab_print(message, fabric):
    if fabric.global_rank == 0:
        print(message)

def printit(message, add_lines:bool=True):
    rank = 0
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        
    if rank == 0:
        if add_lines:print("-"*50)
        print(message)
        if add_lines:print("-"*50)        

def save_checkpoint(model, rank, save_fpath):
    
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    
    if rank == 0:
        checkpoint_state = {}
        checkpoint_state['teacher_model'] = cpu_state
        start_time = time()
        torch.save(cpu_state, save_fpath)
        end_time = time()
        printit(f"Time taken to save model state: {(end_time-start_time)/60:.3f} mins")