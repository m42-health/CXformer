# CXformer: Scalable Vision Foundation Models for Chest X-rays

![CXformer](figures/overview.jpg) <!-- Placeholder for a banner image -->

**📄 Paper:** [Empirical Analysis of Scaling Vision Foundation Models for Chest X-rays (MIDL 2025)](LINK_TO_PAPER)  
**👨‍⚕️ Authors:** Ahmed Al-Mahrooqi, Prateek Munjal, Ronnie Rajan, Marco AF Pimentel, Praveenkumar Kanithi  
**📍 Affiliation:** M42, Abu Dhabi  
**📦 Models:** [CXformer(S)](https://huggingface.co/m42-health/CXformer-small), [CXformer(B)](https://huggingface.co/m42-health/CXformer-base)  
**🧠 Base Architecture:** Vision Transformers (ViT-S, ViT-B)  
**📊 Tasks:** Image Classification, Semantic Segmentation, Report Generation

---

## Abstract

Recent advancements in multimodal transformers have shown remarkable success in computer vision and natural language tasks, yet their adaptation to the clinical world remains challenging. We introduce CXformer, a vision transformer adapted for chest X-ray analysis, through systematic investigation of architectural choices and training modifications from DINOv2. Our empirical results show that using registers in ViT training, centering the teacher model's softmax outputs, and optimizing the number of heads leads to better performance. The small version of CXformer(S) (22M parameters) achieves 83.28\% mean AUROC on CheXpert test set, surpassing the baseline of 80.46\% achieved with vanilla DINOv2 settings. Contrary to common assumptions, our larger model CXformer(B) with 87M parameters shows similar performance at 84\% mean AUROC on CheXpert, suggesting that training optimizations matter more than model size. Furthermore compared to the current state-of-the-art RAD-DINO, our CXformer(B), with 46\% reduced pretraining compute (in FLOPs) achieves an average AUROC of 87.93\% (vs 87.32\% by RAD-DINO) on pathology image classification task evaluated across three widely used CXR datasets i.e. CheXpert, RSNA Pneumonia, and NIH CXR8. Beyond classification, CXformer also delivers competitive, and occasionally superior, performance in semantic segmentation and radiology report generation, underscoring its versatility. By open-sourcing our model checkpoints, we aim to promote reproducibility, reduce resource barriers, and advance scalable solutions for medical imaging research.

Key Contributions:
- Register-enhanced ViT with fewer prototype heads
- Self-supervised pretraining on 600K+ CXRs from 5 global datasets
- Strong generalization across 3 core tasks: classification, segmentation, and report generation
- Lightweight CXformer(S) matches RAD-DINO with 7× less compute
- Released on HuggingFace and open-sourced for reproducibility

---

## Model Checkpoints

| Model      | Params | Pretrain Compute (FLOPs) | Mean AUROC | HuggingFace Model Card |
|------------|--------|---------------------------|-------------|-------------------------|
| CXformer(S)  | 22M    | 3.63 ExaFLOPs             | 86.05%      | [Huggingface Link](https://huggingface.co/m42-health/CXformer-small)|
| CXformer(B)  | 87M    | 14.42 ExaFLOPs            | **87.93%**  | [Huggingface Link](https://huggingface.co/m42-health/CXformer-base) |

> 📌 Note: Both models are trained solely on image data—no text supervision.

---

## Datasets

### Pretraining
- CheXpert
- MIMIC-CXR
- PadChest
- NIH-CXR8
- BRAX

### Downstream Tasks
| Task                | Dataset(s)                        |
|---------------------|-----------------------------------|
| Image Classification| CheXpert, NIH-CXR8, RSNA, VinDr   |
| Segmentation        | CheXmask                          |
| Report Generation   | MIMIC-CXR, IU-Xray                |

---

## Getting Started

### Installation

```bash
git clone https://github.com/m42-health/CXformer.git
cd CXformer
pip install -r requirements.txt
```

---

### Continual Pretraining from DINOv2

```bash
sh scripts/pretrain/cxformer_slurm_submit.sh
```
<details>
  <summary><i>This script internally runs the following:</i></summary>

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 # to reduce fragmentation

n_nodes=1

cfg_file=dinov2/configs/pretrain/cxformer_small.yaml

PYTHONPATH=. python dinov2/run/train/train.py \
--nodes $n_nodes \
--nodelist "worker-13" \
--config-file $cfg_file \
--output-dir output_ablations_new/pretrain/cxformer_small_slurm/
```
</details>

---

### Fine-Tuning on Downstream Tasks

You can fine-tune **CXformer** on downstream image classification tasks such as **CheXpert** using the provided shell script.

#### Run Image classification finetuning (CheXpert Dataset)

```bash
sh scripts/finetuning/image_classification/ft_cxformer_chexpert.sh
```
<details>
  <summary><i>This script internally runs the following:</i></summary>

```bash
export CUDA_VISIBLE_DEVICES=0
n_epochs=100
pretrained_wt="m42-health/CXformer-small"

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
```

📁 Output
The fine-tuned model, logs, and metrics will be saved in:
```
output_ablations_new/finetune/cxformer_chexpert/
```
</details>

#### Run Image Segmentation finetuning (Mimic ChexMask Dataset)

```bash
sh scripts/finetuning/segmentation/cxformer_chexmask.sh
```

#### For radiology report generation
We refer interested readers to llava repo as we straightaway used it in our work.

<!-- ```bash
# # Report generation
# python train_finetune.py --task report_generation --config configs/mimic_cxr_report.yaml
``` -->

---

## Results Summary

### Classification (AUROC)
| Model       | CheXpert | RSNA | NIH-CXR8 | VinDr | Avg. |
|-------------|----------|------|----------|-------|------|
| CXformer(S)   | 83.34    | 91.13| 83.68    | 46.03 (AUPRC) | 86.05 |
| **CXformer(B)** | **86.80** | **91.71** | **85.28** | **48.02 (AUPRC)** | **87.93** |

### Segmentation (Dice Score)
| Model       | Lungs | Heart | Avg. |
|-------------|-------|-------|------|
| CXformer(S)   | 91.69 | 89.35 | 90.52 |
| **CXformer(B)** | 91.94 | 89.94 | 90.94 |

### Report Generation (MIMIC-CXR)
| Model       | ROUGE-L | BLEU-4 | RGER | F1-14 | Avg. |
|-------------|----------|--------|------|--------|-------|
| **CXformer(S)** | **25.25** | **9.11** | **23.06** | 33.85 | 27.51 |
| CXformer(B)   | 24.93   | 9.03   | 22.94 | 33.45 | 27.16 |

---

## Repo Structure

```
CXformer/
├── configs/                              # YAML configuration files
├── models/                               # Vision backbone encoders and heads
├── dinov2/
│   ├── data/                             # Data loading utilities
│   ├── cxr_data/                         # Custom dataset classes and preprocessing logic for chest X-rays
│   └── train/
│       ├── cxr_pretrain.py              # Pretraining script
│       ├── cxr_finetune.py              # Finetuning script for image classification
│       └── cxr_segmentation.py          # Finetuning script for image segmentation
└── README.md                            # Project overview
```

---

## License

This project is licensed under CC BY-NC-4.0 - see the [LICENSE.md](./LICENSE.md) file for details.


## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Citation

```bibtex
@inproceedings{CXformer_2025,
  title={Empirical Analysis of Scaling Vision Foundation Models for Chest X-rays},
  author={Al-Mahrooqi, Ahmed and Munjal, Prateek and Rajan, Ronnie and Pimentel, Marco AF and Kanithi, Praveenkumar},
  booktitle={Medical Imaging with Deep Learning (MIDL)},
  year={2025}
}
```


