# CXformer: Scalable Vision Foundation Models for Chest X-rays

CXformer is a vision transformer tailored for chest X-ray analysis, adapted from DINOv2 with clinically motivated training modifications. This repository provides code for pretraining CXformer using our optimized pipeline, as well as scripts for finetuning on downstream tasks like classification, segmentation, and report generation.

Key highlights:
- Improved training with register tokens, teacher centering, and optimized attention heads.
- Self-supervised pretraining on 600K+ CXRs from 5 global datasets.
- Strong generalization across 3 core tasks: classification, segmentation, and report generation.
- CXformer(S) matches the performance of RAD-DINO with 7× less training compute (in FLOPS).
- Models are available on Hugging Face 🤗: [CXformer(B)](https://huggingface.co/m42-health/CXformer-base), [CXformer(S)](https://huggingface.co/m42-health/CXformer-small)

---

![CXformer](figures/overview.jpg) <!-- Placeholder for a banner image -->
<!-- TODO update paper url-->
**📄 Paper:** [Empirical Analysis of Scaling Vision Foundation Models for Chest X-rays (MIDL 2025)](https://openreview.net/forum?id=6fqfInxqG1#discussion)  
**👨‍⚕️ Authors:** Ahmed Al-Mahrooqi, Prateek Munjal, Ronnie Rajan, Marco AF Pimentel, Praveenkumar Kanithi  
**📍 Affiliation:** M42, Abu Dhabi  
**📦 Models:** [CXformer(S)](https://huggingface.co/m42-health/CXformer-small), [CXformer(B)](https://huggingface.co/m42-health/CXformer-base)  
**🧠 Base Architecture:** Vision Transformers (ViT-S, ViT-B)  
**📊 Tasks:** Image Classification, Semantic Segmentation, Report Generation

---


## Model Checkpoints

| Model      | Params | Pretrain Compute (FLOPs) | Mean AUROC | HuggingFace Model Card |
|------------|--------|---------------------------|-------------|-------------------------|
| CXformer(S)  | 22M    | 3.63 ExaFLOPs             | 86.05%      | [Huggingface Link](https://huggingface.co/m42-health/CXformer-small)|
| CXformer(B)  | 87M    | 14.42 ExaFLOPs            | **87.93%**  | [Huggingface Link](https://huggingface.co/m42-health/CXformer-base) |
| RAD-DINO     | 87M    | 26.71 ExaFLOPs            | 87.32       | [Huggingface Link](https://huggingface.co/microsoft/rad-dino) |
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


