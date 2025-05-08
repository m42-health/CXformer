# Model Card for CXFormer

![CXformer Architecture](https://raw.githubusercontent.com/m42-health/CXformer/refs/heads/main/figures/overview.jpg?token=GHSAT0AAAAAAC3YAMZYT6BPW5QYQWJ6OBM62A4TUWQ)

CXformer is a vision transformer tailored for chest X-ray analysis, adapted from DINOv2 with clinically motivated training modifications. This repository provides code for pretraining CXformer using our optimized pipeline, as well as scripts for finetuning on downstream tasks like classification, segmentation, and report generation.
For more details on pre-training, please checkout [our paper](link_to_paper).

- **Finetuned from model**: [```facebook/dinov2-with-registers-small```](https://huggingface.co/facebook/dinov2-with-registers-small)
- **License**: [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): This work is licensed under CC BY-NC. Additionally, use is limited to research purposes only.

## Key highlights:
- Improved training with register tokens, teacher centering, and optimized attention heads.
- Self-supervised pretraining on 600K+ CXRs from 5 global datasets.
- Strong generalization across 3 core tasks: classification, segmentation, and report generation.
- CXformer(S) matches the performance of RAD-DINO with 7× less training compute (in FLOPS).
- Models are available on Hugging Face 🤗: [CXformer(B)](https://huggingface.co/m42-health/CXformer-base), [CXformer(S)](https://huggingface.co/m42-health/CXformer-small)


## Pretrain Dataset
CXformer was pretrained on publicly available datasets, focusing on frontal views of chest X-rays (PA/AP):

- CheXpert
- MIMIC-CXR
- PadChest
- NIH-CXR8
- BRAX

The official training splits were used for CheXpert, MIMIC and NIH, and all available samples in BRAX and PadChest were used in pretraining.

## Downstream Tasks
| Task                | Dataset(s)                        |
|---------------------|-----------------------------------|
| Image Classification| CheXpert, NIH-CXR8, RSNA, VinDr   |
| Segmentation        | CheXmask                          |
| Report Generation   | MIMIC-CXR, IU-Xray                |


## Usage

```python
from transformers import AutoModel, AutoImageProcessor
from PIL import Image

model_name = 'm42-health/CXformer-base' # or "m42-health/CXformer-small"

image_processor = AutoImageProcessor.from_pretrained(model_name,trust_remote_code=True)
model = AutoModel.from_pretrained(model_name)

model.eval()

image = Image.open('sample_cxr.png')

image = image_processor(image, return_tensors='pt')
print(image['pixel_values'].shape) # [1,3,518,518]

print("Doing forward...!!")
output = model(**image).last_hidden_state  # [1, 1374, 768]

```


<!-- ## Evaluation Results -->
<!-- We report downstream finetuning performance on three tasks: 1) image classification; 2) semantic anatomy segmentation, and 3) image-to-text radiology report generation. We report the median and 95% confidence intervals over 500 bootstrapped samples.

### Image Classification

| Model        | CheXpert       | RSNA          | NIH CXR8       | Agg.  |
|--------------|----------------|---------------|----------------|-------|
| [DINOv2](https://huggingface.co/facebook/dinov2-base)       | 78.53 [78.25,78.53] | 84.83 [84.74,84.89] | 74.85 [74.67,74.91] | 79.40 |
| [CheXzero](https://github.com/rajpurkarlab/CheXzero)     | 82.48 [82.31,82.54] | 89.18 [89.05,89.17] | 77.51 [77.37,77.57] | 83.06 |
| [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)   | 83.18 [83.04,83.27] | 89.54 [89.46,89.58] | 79.30 [79.10,79.31] | 84.01 |
| [RAD-DINO](https://huggingface.co/microsoft/rad-dino)     | 85.06 [84.88,85.07] | **92.19 [92.17,92.26]** | 84.73 [84.53,84.71] | 87.32 |
| CXformer (S)   | 83.34 [83.17,83.39] | 91.13 [91.03,91.13] | 83.68 [83.51,83.68] | 86.05 |
| CXformer (B)   | **86.80 [86.67,86.85]** | 91.71 [91.59,91.70] | **85.28 [85.17,85.32]** | **87.93** |

*AUROC is reported (or macro AUROC for multi-label classification tasks).*

### Semantic Segmentation
Anatomy segmentation results on [CheXmask database](https://physionet.org/content/chexmask-cxr-segmentation-data/1.0.0/) (MIMIC-CXR subset).
| Model        | Lung                | Heart               | Average |
|--------------|---------------------|---------------------|---------|
| DINOv2       | 91.44 [89.87,90.60] | 85.96 [84.83,85.61] | 88.70   |
| CheXzero     | 84.20 [82.90,83.64] | 91.24 [89.70,90.50] | 87.72   |
| BiomedCLIP   | 90.56 [89.11,89.82] | 88.38 [87.03,87.78] | 89.47   |
| RAD-DINO     | **93.28 [91.84,92.54]** | **91.24 [89.70,90.50]** | **92.26** |
| CXformer (S)   | 91.69 [90.16,90.90] | 89.35 [87.62,88.49] | 90.52   |
| CXformer (B)   | 91.94 [90.32,91.10] | 89.94 [87.96,88.85] | 90.94   |

*Dice score is reported.*

### Radiology Report Generation
Reported on MIMIC-CXR "findings" section
| Model        | ROUGE-L       | BLEU-4       | RG_ER         | CheXbert F1-14 | CheXbert F1-5 | Average |
|--------------|---------------|--------------|--------------|----------------|---------------|---------|
| DINOv2       | 24.24 [24.21, 24.25] | 8.51 [8.49, 8.52] | 21.43 [21.42, 21.46] | 28.62 [28.59, 28.69] | 42.09 [42.00, 42.15] | 24.98   |
| CheXzero     | 23.36 [23.34, 23.38] | 7.95 [7.93, 7.96] | 20.95 [20.93, 20.98] | 29.06 [29.04, 29.13] | 44.22 [44.13, 44.27] | 25.11   |
| BiomedCLIP   | 23.35 [23.33, 23.36] | 7.71 [7.70, 7.73] | 20.47 [20.45, 20.49] | 28.77 [28.73, 28.83] | 42.84 [42.73, 42.88] | 24.63   |
| RAD-DINO     | 24.91 [24.89, 24.92] | 8.82 [8.82, 8.85] | 22.92 [22.91, 22.96] | **35.40 [35.39, 35.52]** | **47.54 [47.46, 47.61]** | **27.92** |
| CXformer (S)   | **25.25 [25.23, 25.27]** | **9.11 [9.09, 9.12]** | **23.06 [23.04, 23.08]** | 33.85 [33.83, 33.94] | 46.28 [46.14, 46.28] | 27.51   |
| CXformer (B)   | 24.93 [24.90, 24.94] | 9.03 [9.01, 9.05] | 22.94 [22.93, 22.98] | 33.45 [33.39, 33.50] | 45.45 [45.36, 45.49] | 27.16   |

*[RG_ER](https://pypi.org/project/radgraph/):RadGraph-F1 score. [CheXbert F1](https://pypi.org/project/f1chexbert/): CheXpert 14 and 5 findings F1 score using the [CheXbert](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1204/reports/custom/report32.pdf) model. -->

## Results Summary

### Classification (AUROC)
| Model       | CheXpert | RSNA | NIH-CXR8 | Avg. |
|-------------|----------|------|----------|------|
| CXformer(S)   | 83.34    | 91.13| 83.68  | 86.05 |
| **CXformer(B)** | **86.80** | **91.71** | **85.28** | **87.93** |

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

# Disclaimer
*CXformer is intended exclusively for research purposes. It is not validated for clinical decision-making, nor is it approved for use in healthcare environments. The model should not be used for any diagnostic or therapeutic applications in a clinical setting.*

# License
This project is licensed under CC BY-NC-4.0

# Citation

```bibtex
@inproceedings{CXformer_2025,
  title={Empirical Analysis of Scaling Vision Foundation Models for Chest X-rays},
  author={Al-Mahrooqi, Ahmed and Munjal, Prateek and Rajan, Ronnie and Pimentel, Marco AF and Kanithi, Praveenkumar},
  booktitle={Medical Imaging with Deep Learning (MIDL)},
  year={2025}
}
```