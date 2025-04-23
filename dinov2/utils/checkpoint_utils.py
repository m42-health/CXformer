import torch
from tqdm import tqdm
import os

def merge_checkpoints(checkpoint_paths:list, save_path:str, ratios:list=None):


    if ratios:
        assert len(checkpoint_paths) == len(ratios)
    else:
        ratios = [1./len(checkpoint_paths) for _ in checkpoint_paths]


    models = [torch.load(ckpt)['teacher'] for ckpt in checkpoint_paths]

    # Combine weights
    combined_state_dict = {}
    combined_state_dict['teacher'] = {}

    for key in tqdm(models[0].keys(), desc="Merging checkpoints"):
        combined_state_dict['teacher'][key] = sum(ratios[i] * models[i][key] for i in range(len(models)))



    os.makedirs(save_path, exist_ok=True)
    ckpt_save_path = os.path.join(save_path, 'merged_ckpt.pth')
    torch.save(combined_state_dict, ckpt_save_path)

    print("-"*50)
    print("Merged checkpoint saved at path: ", ckpt_save_path)
    print("-"*50)


if __name__ == '__main__':
    fpath = "teacher_checkpoint.pth"
    merge_checkpoints([fpath, fpath], save_path="/home/prateek/projects/dinov2/merged_checkpoints/")