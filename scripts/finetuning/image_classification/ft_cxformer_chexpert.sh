export CUDA_VISIBLE_DEVICES=0
n_epochs=5

# pretrained_wt="/home/prateek/projects/scan42/dinov2_cxr/output_ablations_new/pretrain/scan42_small_slurm/eval/eval_last/teacher_checkpoint.pth"
pretrained_wt="/models_vision/scan42/scan42-small/pytorch-model/teacher_checkpoint.pth"

PYTHONPATH=. deepspeed dinov2/train/cxr_finetune.py \
--config-file dinov2/configs/downstream/classification/chexformer_chexpert_small.yaml \
--output-dir output_ablations_new/finetune/chexformer_chexpert \
--exp-name ft_chexformer \
--pretrained-weights $pretrained_wt \
--model-type dinov2 \
--num-epochs $n_epochs \
--batch-size 10 \
--num_workers 1 \
--seed 7479 \
--cls-n-layers 4 \
--apply-avgpool \
--clf_lr 5e-5 \
--backbone_lr 5e-7


# --batch-size 64
# --num_workers 18