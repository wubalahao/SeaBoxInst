"""
Photometric Augmentation for PCR

This module implements differentiable photometric augmentations for generating
strong and weak views in the Perturbation Consistency Regularization framework.

Reference: SeaBoxInst - Photometric Consistency Regularization (Section 3.3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhotometricAugmentation(nn.Module):
    """Differentiable photometric augmentation for PCR.

    Generates strong photometric views from input images with:
    - Brightness adjustment
    - Contrast adjustment
    - Per-channel color scaling
    - Blur augmentation
    - Gaussian noise
    """

    def __init__(self,
                 bright_range=0.2,
                 contrast_range=0.2,
                 color_range=0.2,
                 blur_prob=0.2,
                 noise_std=0.02):
        super().__init__()
        self.bright_range = bright_range
        self.contrast_range = contrast_range
        self.color_range = color_range
        self.blur_prob = blur_prob
        self.noise_std = noise_std

    def forward(self, img):
        """Apply photometric augmentation.

        Args:
            img: [B, C, H, W] normalized image tensor

        Returns:
            Augmented image tensor
        """
        b, c, h, w = img.shape
        device = img.device
        dtype = img.dtype
        out = img.clone()

        # Brightness: random delta in [-bright_range, bright_range]
        b_delta = (torch.rand(b, 1, 1, 1, device=device, dtype=dtype) - 0.5) * 2 * self.bright_range

        # Contrast: random scale in [1-contrast_range, 1+contrast_range]
        c_alpha = 1.0 + (torch.rand(b, 1, 1, 1, device=device, dtype=dtype) - 0.5) * 2 * self.contrast_range

        # Per-channel color scale
        ch_scale = 1.0 + (torch.rand(b, c, 1, 1, device=device, dtype=dtype) - 0.5) * 2 * self.color_range

        # Apply adjustments
        out = out * c_alpha + b_delta
        out = out * ch_scale

        # Blur augmentation
        if self.blur_prob > 0:
            do_blur = torch.rand(b, device=device) < self.blur_prob
            if do_blur.any():
                k = 3
                pad = k // 2
                blurred = F.avg_pool2d(out, kernel_size=k, stride=1, padding=pad)
                blend_alpha = 0.5
                out[do_blur] = blend_alpha * blurred[do_blur] + (1.0 - blend_alpha) * out[do_blur]

        # Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(out) * self.noise_std
            out = out + noise

        return out


def create_weak_strong_augment(aug_config=None):
    """Create weak and strong augmentation for PCR.

    Args:
        aug_config: dict with augmentation parameters

    Returns:
        weak_aug, strong_aug: augmentation modules
    """
    if aug_config is None:
        aug_config = {}

    # Weak view: minimal augmentation
    weak_aug = None  # Use original image

    # Strong view: heavy augmentation
    strong_aug = PhotometricAugmentation(
        bright_range=aug_config.get('bright', 0.2),
        contrast_range=aug_config.get('contrast', 0.2),
        color_range=aug_config.get('sat', 0.2),
        blur_prob=aug_config.get('blur_prob', 0.2),
        noise_std=aug_config.get('noise_std', 0.02)
    )

    return weak_aug, strong_aug
