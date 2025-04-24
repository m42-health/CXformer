export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 # to reduce fragmentation

# export CUDA_VISIBLE_DEVICES=0,1
export SLURM_JOB_NUM_NODES=1

# cfg_file=dinov2/configs/pretrain/vit_131k.yaml
cfg_file=dinov2/configs/pretrain/cxformer_small.yaml

PYTHONPATH=. python dinov2/train/cxr_pretrain.py \
--config-file $cfg_file \
--output-dir output_ablations_new/pretrain/cxformer_small/