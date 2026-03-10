# Box-Supervised Instance Segmentation Core Algorithms

This repository contains core algorithmic innovations for box-supervised instance segmentation, specifically designed for challenging underwater environments.

## Core Innovations

### 1. MILA (Multi-Instance Learning-driven Masked Attention)
- Iterative mask refinement via masked multi-head attention
- Bag-level MIL loss with stochastic box expansion (r ~ U(0.8, 1.6))
- Explicitly suppresses background leakage in weak box supervision
- Key files: `mila/masked_attention.py`, `mila/mil_loss.py`

### 2. PCR (Perturbation Consistency Regularization)
- Dual-view training with strong/weak photometric augmentation
- Same-instance alignment across views (weak indices + strong params)
- Enhances robustness to underwater image degradations (color cast, blur, noise)
- Key files: `pcr/pcr_detector.py`, `pcr/photometric_aug.py`

### 3. Quality-aware Classification
- Quality Focal Loss (QFL) aligning confidence with localization quality
- CIoU regression with centerness-on-reg
- Soft-NMS for inference
- Key files: `quality_head/quality_focal_loss.py`

## Quick Start

```python
# 1. Use MILA head for mask refinement
from mila import MILMaskedAttentionCondInstMaskHead

# 2. Use PCR for photometric consistency
from pcr import PCRDetector

# 3. Use Quality Focal Loss
from quality_head import QualityFocalLoss
```

## Architecture Overview

```
Input Image
    │
    ▼
┌─────────────────────────────────────┐
│  Swin Transformer + PAFPN Backbone  │
└─────────────────────────────────────┘
    │
    ├──────────────────────┬──────────────────────┐
    ▼                      ▼                      ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐
│ Detection   │    │ Mask Branch │    │ Dual-View PCR Branch│
│ Head (QFL)  │    │             │    │ (Strong Augmentation│
└─────────────┘    └─────────────┘    └─────────────────────┘
    │                      │                      │
    └──────────────────────┼──────────────────────┘
                           ▼
              ┌────────────────────────┐
              │  MILA Mask Refinement  │
              │ (Masked Attention +    │
              │  MIL Bag-level Loss)   │
              └────────────────────────┘
                           │
                           ▼
                   Instance Masks
```

## Citation

```bibtex
@article{seaboxinst2026,
  title={Box-Supervised Underwater Instance Segmentation with MIL-Driven Mask Refinement and Photometric Consistency},
  author={},
  journal={},
  year={2026}
}
```

## License

For research use only. Full code will be released after paper publication.
