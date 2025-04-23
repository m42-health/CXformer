import torch
import os
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from time import time

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