# Scan42: Scalable Vision Foundation Models for Chest X-rays

![Scan42](assets/scan42_banner.png) <!-- Placeholder for a banner image -->

**📄 Paper:** [Empirical Analysis of Scaling Vision Foundation Models for Chest X-rays (MIDL 2025)](LINK_TO_PAPER)  
**👨‍⚕️ Authors:** Ahmed Al-Mahrooqi, Prateek Munjal, Ronnie Rajan, Marco AF Pimentel, Praveenkumar Kanithi  
**📍 Affiliation:** M42, Abu Dhabi  
**📦 Models:** Scan42(S), Scan42(B)  
**🧠 Base Architecture:** Vision Transformers (ViT-S, ViT-B)  
**📊 Tasks:** Image Classification, Semantic Segmentation, Report Generation

---

## 🔬 Overview

**Scan42** is a self-supervised foundation model family tailored for **Chest X-ray (CXR)** analysis. Built on top of DINOv2 and adapted with domain-specific optimizations, Scan42 delivers **SOTA** performance on multiple medical imaging tasks while being compute-efficient.

Key Contributions:
- Register-enhanced ViT with fewer prototype heads
- Self-supervised pretraining on 600K+ CXRs from 5 global datasets
- Strong generalization across 3 core tasks: classification, segmentation, and report generation
- Lightweight Scan42(S) matches RAD-DINO with 7× less compute
- Released on HuggingFace and open-sourced for reproducibility

---

## 📦 Model Checkpoints

| Model      | Params | Pretrain Compute (FLOPs) | Mean AUROC | HuggingFace Model Card |
|------------|--------|---------------------------|-------------|-------------------------|
| Scan42(S)  | 22M    | 3.63 ExaFLOPs             | 86.05%      | `[link-to-scan42-s]`    |
| Scan42(B)  | 87M    | 14.42 ExaFLOPs            | **87.93%**  | `[link-to-scan42-b]`    |

> 📌 Note: Both models are trained solely on image data—no text supervision.

---

## 🧪 Datasets

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

## 🚀 Getting Started

### 🔧 Installation

```bash
git clone https://github.com/YOUR_ORG/scan42.git
cd scan42
pip install -r requirements.txt
```

> You will need access to NVIDIA H100/A100 GPUs for optimal training.

---

### 🏋️‍♀️ Pretraining from DINOv2

```bash
python train_pretrain.py --config configs/scan42_pretrain.yaml
```

---

### 🎯 Finetuning on Downstream Tasks

```bash
# Image classification
sh scripts/finetuning/ft_scan42_chexpert.sh
```
```bash
# Segmentation
# python train_finetune.py --task segmentation --config configs/chexmask.yaml
```

```bash
# # Report generation
# python train_finetune.py --task report_generation --config configs/mimic_cxr_report.yaml
```

---

## 📈 Results Summary

### 🩻 Classification (AUROC)
| Model       | CheXpert | RSNA | NIH-CXR8 | VinDr | Avg. |
|-------------|----------|------|----------|-------|------|
| Scan42(S)   | 83.34    | 91.13| 83.68    | 46.03 (AUPRC) | 86.05 |
| **Scan42(B)** | **86.80** | **91.71** | **85.28** | **48.02 (AUPRC)** | **87.93** |

### 🫁 Segmentation (Dice Score)
| Model       | Lungs | Heart | Avg. |
|-------------|-------|-------|------|
| Scan42(S)   | 91.69 | 89.35 | 90.52 |
| **Scan42(B)** | 91.94 | 89.94 | 90.94 |

### 📄 Report Generation (MIMIC-CXR)
| Model       | ROUGE-L | BLEU-4 | RGER | F1-14 | Avg. |
|-------------|----------|--------|------|--------|-------|
| **Scan42(S)** | **25.25** | **9.11** | **23.06** | 33.85 | 27.51 |
| Scan42(B)   | 24.93   | 9.03   | 22.94 | 33.45 | 27.16 |

---

## 🗂 Repo Structure

```
scan42/
├── configs/                     # YAML configs for training
├── models/                      # Vision encoders & heads
├── data/                        # Dataset preparation scripts
├── train/train_cxr.py            # Pretraining script
├── train/scan42_finetune.py            # Finetuning script
├── eval/                        # Evaluation tools per task
└── README.md
```

---

## 📜 Citation

```bibtex
@inproceedings{scan42_2025,
  title={Empirical Analysis of Scaling Vision Foundation Models for Chest X-rays},
  author={Al-Mahrooqi, Ahmed and Munjal, Prateek and Rajan, Ronnie and Pimentel, Marco AF and Kanithi, Praveenkumar},
  booktitle={Medical Imaging with Deep Learning (MIDL)},
  year={2025}
}
```

---

## 📬 Contact

For questions/suggetions, feel free to reach out at [pmunjal@m42.ae](mailto:pmunjal@m42.ae) or [prateekmunjal31@gmail.com](mailto:prateekmunjal31@gmail.com)

---

Built with ❤️ by the M42 AI Research team.