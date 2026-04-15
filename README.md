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





## License

For research use only. Full code will be released after paper publication.
