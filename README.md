# ECA-MGNet: An Efficient Multi-Scale Ghost Network with Dual Attention

Official PyTorch implementation for the paper:

> **ECA-MGNet: An Efficient Multi-Scale Ghost Network with Dual Attention for Lightweight Image Classification**
> *IEEE Access, 2026*

---

## Overview

ECA-MGNet is a lightweight convolutional neural network (**2.50M parameters, 0.16 GFLOPs**) designed for resource-constrained image classification. It augments a pretrained GhostNet-1.0x backbone with two novel components:

1. **Multi-Scale Refinement Block (MSRB)** — four parallel branches (1x1, 3x3 DW, 5x5 DW, global average pooling) that expand the backbone's 160-channel output to a rich 960-channel representation, capturing complementary spatial scales.
2. **Dual Attention Module** — sequential application of Efficient Channel Attention (ECA) followed by Spatial Attention, providing explicit channel- and spatial-selection capability with negligible parameter overhead (~0.1K params).

The model uses a **two-phase transfer learning** strategy and is evaluated against MobileNetV2, EfficientNet-B0, ShuffleNetV2, and ResNet-18 on four diverse benchmark datasets. ECA-MGNet achieves the **highest average accuracy (90.79%)** across all benchmarks, outperforming the best baseline by over 16 percentage points.

---

## Results

All baselines are fine-tuned from ImageNet-pretrained weights with a single-phase training protocol (AdamW, cosine annealing, 50 epochs, batch size 32). ECA-MGNet uses a two-phase transfer learning schedule (see Training section below).

| Model | Params (M) | FLOPs (G) | Flowers102 | DTD | Food101 | EuroSAT | **Average** |
|-------|-----------|-----------|------------|-----|---------|---------|---------|
| **ECA-MGNet (ours)** | **2.50** | **0.16** | **98.94** | **83.33** | **84.22** | **96.67** | **90.79** |
| ResNet-18 | 11.18 | 1.82 | 95.04 | 51.11 | 68.22 | 84.22 | 74.65 |
| ShuffleNetV2 | 1.26 | 0.15 | 93.62 | 53.33 | 62.89 | 86.89 | 74.18 |
| EfficientNet-B0 | 4.02 | 0.41 | 93.26 | 52.78 | 64.44 | 85.78 | 74.07 |
| MobileNetV2 | 2.24 | 0.33 | 91.13 | 55.56 | 58.22 | 86.00 | 72.73 |

**Bold** indicates best result per column.

---

## Architecture

```
Input (224x224x3)
    |
  Pretrained Stem (Conv 3x3, s=2, BN, Act)  [3 -> 16]
    |
  Pretrained GhostNet Backbone (blocks 0-7)  [16 -> 160]
    |                                          0.95M params
  Multi-Scale Refinement Block (MSRB)
    |-- Branch 1: Conv 1x1
    |-- Branch 2: Conv 1x1 -> DWConv 3x3
    |-- Branch 3: Conv 1x1 -> DWConv 5x5      1.09M params
    |-- Branch 4: GAP -> Conv 1x1
    |-- Concat -> Fusion Conv 1x1              [160 -> 960]
    |
  Dual Attention
    |-- ECA (channel attention)                ~0.1K params
    |-- Spatial Attention (7x7 conv)
    |
  Global Average Pool -> FC(960,480) -> ReLU -> Dropout(0.2) -> FC(480,N)
    |                                          0.46M params
  Output (N classes)
```

**Total: 2.50M parameters, 0.16 GFLOPs at 224x224 resolution.**

---

## Repository Structure

```
.
+-- src/
|   +-- __init__.py             # Package init
|   +-- models.py               # ECA-MGNet and all baseline model definitions
|   +-- dataset.py              # Generic image classification dataset loader
|   +-- train.py                # Training engine (single-phase + two-phase)
|   +-- run_training.py         # Batch runner with resume support
|   +-- run_all_experiments.py  # Train all models on all datasets
|   +-- generate_figures.py     # Publication-quality figure generation
|   +-- gradcam.py              # Grad-CAM visualization for attention analysis
+-- scripts/
|   +-- download_datasets.py    # Download and prepare benchmark datasets
+-- requirements.txt
+-- README.md
+-- .gitignore
+-- LICENSE
```

---

## Requirements

- Python >= 3.8
- PyTorch >= 2.0 with CUDA support (GPU strongly recommended)
- [timm](https://github.com/huggingface/pytorch-image-models) >= 0.9.0 (for pretrained GhostNet backbone)

### Installation

```bash
pip install -r requirements.txt
```

---

## Datasets

### Download Datasets

```bash
python scripts/download_datasets.py
```

Datasets are organized as:

```
datasets/
+-- flowers102/
|   +-- class_a/  (*.jpg, *.png, ...)
|   +-- class_b/
+-- dtd/
+-- food101/
+-- eurosat/
```

Each dataset is split automatically into train / validation / test sets (70% / 15% / 15%) using a fixed random seed (42) for reproducibility.

| Dataset     | Description                        | Classes | Split Sizes |
|-------------|------------------------------------|---------|-------------|
| Flowers102  | Oxford 102 Flowers (subset)        | 10      | ~1,311 / 281 / 281 |
| DTD         | Describable Textures Dataset       | 10      | ~840 / 180 / 180 |
| Food101     | Food-101 (subset)                  | 10      | ~2,100 / 450 / 450 |
| EuroSAT     | Satellite imagery (RGB)            | 10      | ~2,100 / 450 / 450 |

---

## Training

### Train ECA-MGNet (Two-Phase Transfer Learning)

ECA-MGNet uses a two-phase transfer learning strategy:

- **Phase 1** (5 epochs): Freeze pretrained backbone, train custom MSRB + attention + classifier head. LR=3e-3 with cosine annealing.
- **Phase 2** (up to 60 epochs): Unfreeze all parameters for end-to-end fine-tuning. LR=1e-3 with cosine annealing, early stopping (patience 20).

```bash
python src/train.py \
    --data_dir datasets/flowers102 \
    --model ecamgnet \
    --save_dir results/flowers102
```

### Train Baseline Models (Single-Phase)

Baselines are fine-tuned from ImageNet-pretrained weights with a single-phase protocol:

```bash
python src/train.py \
    --data_dir datasets/flowers102 \
    --model resnet18 \
    --save_dir results/flowers102/resnet18 \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --patience 15
```

Available `--model` choices:

| Name              | Description                    | Params (M) |
|-------------------|--------------------------------|------------|
| `ecamgnet`        | Proposed ECA-MGNet             | 2.50       |
| `mobilenetv2`     | MobileNetV2                    | 2.24       |
| `efficientnet_b0` | EfficientNet-B0                | 4.02       |
| `shufflenetv2`    | ShuffleNetV2 x1.0              | 1.26       |
| `resnet18`        | ResNet-18                      | 11.18      |

### Train All Models on All Datasets

```bash
python src/run_all_experiments.py \
    --datasets_dir datasets \
    --results_dir results
```

Results are saved to `results/all_results.json` plus per-model files under `results/<dataset>/<model>/`.

---

## Grad-CAM Visualization

Generate Grad-CAM heatmaps comparing ECA-MGNet with a baseline model:

```bash
python src/gradcam.py \
    --model_path results/flowers102/ecamgnet_best.pth \
    --baseline_path results/flowers102/mobilenetv2/mobilenetv2_best.pth \
    --data_dir datasets/flowers102 \
    --num_classes 10 \
    --output_dir figures/gradcam
```

---

## Generate Figures

```bash
python src/generate_figures.py --results_dir results --figures_dir figures
```

Figures generated include:
- Architecture block diagram
- Dual Attention module diagram
- Accuracy comparison bar chart
- Training curves
- Confusion matrices
- Grad-CAM comparison grids

---

## Training Details

| Setting              | Baselines (Single-Phase)      | ECA-MGNet (Two-Phase)         |
|----------------------|-------------------------------|-------------------------------|
| Optimizer            | AdamW                         | AdamW                         |
| Weight decay         | 1e-4                          | 1e-4                          |
| Learning rate        | 1e-3 (cosine annealing)       | Phase 1: 3e-3, Phase 2: 1e-3 |
| Min LR               | 1e-6                          | Phase 1: 1e-4, Phase 2: 1e-6 |
| Label smoothing      | 0.1                           | 0.1                           |
| Gradient clipping    | max norm 1.0                  | max norm 1.0                  |
| Early stopping       | 15 epochs patience            | Phase 2: 20 epochs patience   |
| Max epochs           | 50                            | Phase 1: 5, Phase 2: 60      |
| Backbone             | ImageNet-pretrained            | ImageNet-pretrained (frozen in Phase 1) |
| Data augmentation    | RandomCrop 224, HFlip, VFlip, Rotation ±15°, ColorJitter, Affine, RandomErasing | Same |
| Normalization        | ImageNet mean/std             | Same                          |
| Batch size           | 32                            | 32                            |

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{ecamgnet2026,
  title   = {ECA-MGNet: An Efficient Multi-Scale Ghost Network with Dual Attention
             for Lightweight Image Classification},
  journal = {IEEE Access},
  year    = {2026},
}
```

---

## License

This project is released under the MIT License.

---

## Acknowledgements

This work builds on:

- **GhostNet**: Han, K. et al. (CVPR 2020). "GhostNet: More Features from Cheap Operations."
- **ECA-Net**: Wang, Q. et al. (CVPR 2020). "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks."
- **CBAM**: Woo, S. et al. (ECCV 2018). "CBAM: Convolutional Block Attention Module."
- **timm**: Wightman, R. "PyTorch Image Models." https://github.com/huggingface/pytorch-image-models
