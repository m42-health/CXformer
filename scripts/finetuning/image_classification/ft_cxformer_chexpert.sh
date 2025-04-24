export CUDA_VISIBLE_DEVICES=0
n_epochs=5

# pretrained_wt="/models_vision/scan42/scan42-small/pytorch-model/teacher_checkpoint.pth"
pretrained_wt="m42-health/CXFormer-small"

PYTHONPATH=. deepspeed dinov2/train/cxr_finetune.py \
--config-file dinov2/configs/downstream/classification/cxformer_chexpert_small.yaml \
--output-dir output_ablations_new/finetune/cxformer_chexpert \
--exp-name ft_cxformer \
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