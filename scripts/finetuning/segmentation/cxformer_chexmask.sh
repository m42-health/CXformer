# #!/bin/bash

export CUDA_VISIBLE_DEVICES=0

pretrained_wt="/models_vision/scan42/scan42-small/pytorch-model/teacher_checkpoint.pth"

PYTHONPATH=. deepspeed dinov2/train/cxr_segmentation.py \
    --config-file dinov2/configs/downstream/segmentation/chexformer-mimic-chexmask.yaml \
    --output-dir output_ablations_new/finetune_segmentation \
    --exp-name chexformer_seg_ft \
    --num-epochs 5 \
    --batch-size 64 \
    --criterion ce_dice_loss \
    --gradient-accumulation-steps 1 \
    --num-workers 18 \
    --pretrained-weights $pretrained_wt \
    --model-type dinov2 \
    --lr 5e-3