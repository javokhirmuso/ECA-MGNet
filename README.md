# ECA-MGNet: An Efficient Multi-Scale Ghost Network with Dual Attention for Lightweight Image Classification

Official PyTorch implementation for the paper:

> **ECA-MGNet: An Efficient Multi-Scale Ghost Network with Dual Attention for Lightweight Image Classification**
> *Expert Systems with Applications, 2026*

---

## Overview

ECA-MGNet is a lightweight convolutional neural network (**2.50M parameters, 0.16 GFLOPs**) designed for resource-constrained image classification. It augments a pretrained GhostNet-1.0x backbone with two novel components:

1. **Multi-Scale Refinement Block (MSRB)** — four parallel branches (1×1, 3×3 DW, 5×5 DW, global average pooling) that expand the backbone's 160-channel output to a rich 960-channel representation, capturing complementary spatial scales.
2. **Dual Attention Module** — sequential application of Efficient Channel Attention (ECA) followed by Spatial Attention, providing explicit channel- and spatial-selection capability with negligible parameter overhead (~0.1K params).

---

## Results

All models are fine-tuned from ImageNet-pretrained weights. ECA-MGNet uses a two-phase transfer learning schedule; baselines use a single-phase protocol. All experiments use 10-class subsets with fixed seed (42) for reproducibility.

### Classification Performance

| Model | Params (M) | FLOPs (G) | Flowers102 | EuroSAT | Avg Acc | F1 (macro) | κ | MCC |
|-------|-----------|-----------|------------|---------|---------|------------|------|------|
| **ECA-MGNet (ours)** | **2.50** | **0.16** | **98.94** | **96.67** | **97.81** | **97.71** | **0.976** | **0.976** |
| ResNet-18 | 11.18 | 1.82 | 95.04 | 84.22 | 89.63 | 89.39 | 0.885 | 0.886 |
| ShuffleNetV2 | 1.26 | 0.15 | 93.62 | 86.89 | 90.26 | 90.15 | 0.891 | 0.892 |
| EfficientNet-B0 | 4.02 | 0.41 | 93.26 | 85.78 | 89.52 | 89.69 | 0.884 | 0.884 |
| MobileNetV2 | 2.24 | 0.33 | 91.13 | 86.00 | 88.57 | 88.09 | 0.873 | 0.873 |

All improvements are statistically significant (McNemar's test, p < 0.01). Bootstrap 95% CI: Flowers102 [97.5%, 100.0%], EuroSAT [94.9%, 98.2%].

### Inference Efficiency

| Model | Latency (ms) | Throughput (img/s) | Size (MB) | GPU Mem (MB) |
|-------|-------------|-------------------|-----------|-------------|
| **ECA-MGNet** | 7.20 | 139 | **9.8** | **25.2** |
| ResNet-18 | **2.25** | **444** | 42.7 | 61.6 |
| ShuffleNetV2 | 5.00 | 200 | 5.0 | 16.6 |
| EfficientNet-B0 | 6.54 | 153 | 15.6 | 34.9 |
| MobileNetV2 | 4.06 | 246 | 8.8 | 28.0 |

Measured on NVIDIA TITAN Xp, batch size 1, 100 runs after 20 warmup iterations.

### Ablation Study

| Variant | Params (M) | Flowers102 (%) | EuroSAT (%) | Avg (%) |
|---------|-----------|---------------|------------|---------|
| Backbone only | 0.96 | 98.23 | 95.11 | 96.67 |
| + MSRB | 2.50 | 98.58 | 95.56 | 97.07 |
| + Dual Attention | 0.96 | 97.16 | 96.89 | 97.03 |
| + MSRB + ECA only | 2.50 | 99.29 | 94.89 | 97.09 |
| + MSRB + Spatial only | 2.50 | 98.58 | 94.67 | 96.63 |
| **Full ECA-MGNet** | **2.50** | **99.65** | **96.44** | **98.05** |

---

## Architecture

```
Input (224×224×3)
    │
  Pretrained Stem (Conv 3×3, s=2, BN, Act)  [3 → 16]
    │
  Pretrained GhostNet Backbone (blocks 0–7)  [16 → 160]
    │                                          0.95M params
  Multi-Scale Refinement Block (MSRB)
    ├── Branch 1: Conv 1×1
    ├── Branch 2: Conv 1×1 → DWConv 3×3
    ├── Branch 3: Conv 1×1 → DWConv 5×5       1.09M params
    ├── Branch 4: GAP → Conv 1×1
    └── Concat → Fusion Conv 1×1              [160 → 960]
    │
  Dual Attention
    ├── ECA (channel attention)                ~0.1K params
    └── Spatial Attention (7×7 conv)
    │
  GAP → FC(960,480) → ReLU → Dropout(0.2) → FC(480,N)
    │                                          0.46M params
  Output (N classes)
```

**Total: 2.50M parameters, 0.16 GFLOPs at 224×224 resolution.**

---

## Repository Structure

```
.
├── src/
│   ├── models.py               # ECA-MGNet and all baseline model definitions
│   ├── dataset.py              # Generic image classification dataset loader
│   ├── train.py                # Training engine (single-phase + two-phase)
│   ├── run_training.py         # Batch runner with resume support
│   ├── run_all_experiments.py  # Train all models on all datasets
│   ├── generate_figures.py     # Publication-quality figure generation
│   └── gradcam.py              # Grad-CAM visualization
├── scripts/
│   └── download_datasets.py    # Download and prepare benchmark datasets
├── weights/
│   ├── flowers102/             # Pretrained model weights (Flowers102)
│   └── eurosat/                # Pretrained model weights (EuroSAT)
├── results/
│   ├── ablation_study_results.json
│   └── comprehensive_analysis/ # All analysis JSONs (metrics, ROC, McNemar, etc.)
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/javokhirmuso/ECA-MGNet.git
cd ECA-MGNet
pip install -r requirements.txt
```

### Download Datasets

```bash
python scripts/download_datasets.py
```

### Train ECA-MGNet

```bash
python src/train.py \
    --data_dir datasets/flowers102 \
    --model ecamgnet \
    --save_dir results/flowers102
```

### Evaluate with Pretrained Weights

```python
import torch
from src.models import get_model

model = get_model('ecamgnet', num_classes=10, pretrained=False)
state = torch.load('weights/flowers102/ecamgnet_best.pth', map_location='cpu', weights_only=True)
model.load_state_dict(state, strict=False)
model.eval()

# Inference
x = torch.randn(1, 3, 224, 224)
pred = model(x).argmax(dim=1)
```

---

## Training Details

| Setting | Baselines (Single-Phase) | ECA-MGNet (Two-Phase) |
|---|---|---|
| Optimizer | AdamW | AdamW |
| Weight decay | 1e-4 | 1e-4 |
| Learning rate | 1e-3 (cosine annealing) | Phase 1: 3e-3, Phase 2: 1e-3 |
| Label smoothing | 0.1 | 0.1 |
| Early stopping | 15 epochs patience | Phase 2: 20 epochs patience |
| Max epochs | 50 | Phase 1: 5, Phase 2: 60 |
| Backbone | ImageNet-pretrained | ImageNet-pretrained (frozen in Phase 1) |
| Batch size | 32 | 32 |
| Image size | 224×224 | 224×224 |
| Augmentation | HFlip, VFlip, Rotation ±15°, ColorJitter, RandomErasing | Same |

---

## Datasets

| Dataset | Description | Classes | Train / Val / Test |
|---|---|---|---|
| Flowers102 | Oxford 102 Flowers (subset) | 10 | ~1,311 / 281 / 281 |
| EuroSAT | Satellite imagery (RGB) | 10 | ~2,100 / 450 / 450 |

Each dataset uses a fixed 70/15/15 split with seed=42.

---

## Citation

```bibtex
@article{ecamgnet2026,
  title   = {ECA-MGNet: An Efficient Multi-Scale Ghost Network with Dual Attention
             for Lightweight Image Classification},
  author  = {Musaev, Javokhir},
  journal = {Expert Systems with Applications},
  year    = {2026},
}
```

---

## License

This project is released under the MIT License.

---

## Acknowledgements

- **GhostNet**: Han et al. (CVPR 2020)
- **ECA-Net**: Wang et al. (CVPR 2020)
- **CBAM**: Woo et al. (ECCV 2018)
- **timm**: Wightman, R. — [pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
