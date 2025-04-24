export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 # to reduce fragmentation

n_nodes=1

# export CUDA_VISIBLE_DEVICES=0
cfg_file=dinov2/configs/pretrain/cxformer_small.yaml

PYTHONPATH=. python dinov2/run/train/train.py \
--nodes $n_nodes \
--nodelist "worker-13" \
--config-file $cfg_file \
--output-dir output_ablations_new/pretrain/cxformer_small_slurm/