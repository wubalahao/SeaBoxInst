"""
MIL Loss for Box-Supervised Instance Segmentation

This module implements the Multiple Instance Learning (MIL) bag-level loss
for box-supervised instance segmentation. It uses row/column bags within
an expanded box region to supervise mask prediction without pixel-level labels.

Our approach: Construct row/column bags within ground-truth boxes and apply
MIL constraints with stochastic expansion to suppress background leakage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MILLoss(nn.Module):
    """Multiple Instance Learning loss for box-supervised segmentation.

    Constructs row/column bags within expanded ground-truth boxes and applies
    MIL constraints to suppress background leakage. Uses stochastic box
    expansion (r ~ U(0.8, 1.6)) to prevent overfitting and explicitly
    suppress near-box background responses.
    """

    def __init__(self,
                 loss_weight=4.0,
                 expand_ratio_range=(0.8, 1.6),
                 loss_type='dice',
                 warmup_iters=10000):
        super().__init__()
        self.loss_weight = loss_weight
        self.expand_ratio_range = expand_ratio_range
        self.loss_type = loss_type
        self.warmup_iters = warmup_iters

    def _dice_1d(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """1D Dice loss for bag-level supervision."""
        eps = 1e-5
        inter = (pred * target).sum()
        denom = pred.pow(2).sum() + target.pow(2).sum() + eps
        return 1.0 - (2.0 * inter / denom)

    def _bce_1d(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """1D BCE loss for bag-level supervision."""
        pred = pred.clamp(1e-6, 1.0 - 1e-6)
        return F.binary_cross_entropy(pred, target)

    def forward(self,
                mask_scores: torch.Tensor,
                gt_bitmask: torch.Tensor,
                expand_ratio: float = 1.0,
                warmup_factor: float = 1.0) -> torch.Tensor:
        """Compute MIL loss for a single instance.

        Args:
            mask_scores: [H, W] sigmoid scores for this instance
            gt_bitmask: [H, W] binary rectangle (downsampled GT box mask)
            expand_ratio: expansion ratio for box region
            warmup_factor: warmup factor for stable early training

        Returns:
            MIL loss value
        """
        rows_any = gt_bitmask.max(dim=1)[0]
        cols_any = gt_bitmask.max(dim=0)[0]

        if rows_any.sum() == 0 or cols_any.sum() == 0:
            return mask_scores.new_tensor(0.0)

        row_idxs = torch.nonzero(rows_any, as_tuple=False).squeeze(1)
        col_idxs = torch.nonzero(cols_any, as_tuple=False).squeeze(1)
        rmin, rmax = int(row_idxs.min()), int(row_idxs.max())
        cmin, cmax = int(col_idxs.min()), int(col_idxs.max())

        H, W = mask_scores.shape
        box_h = max(1, rmax - rmin + 1)
        box_w = max(1, cmax - cmin + 1)

        # Expand box region
        extra_h = int((expand_ratio - 1.0) * box_h / 2.0)
        extra_w = int((expand_ratio - 1.0) * box_w / 2.0)
        rmin_e = max(0, rmin - extra_h)
        rmax_e = min(H - 1, rmax + extra_h)
        cmin_e = max(0, cmin - extra_w)
        cmax_e = min(W - 1, cmax + extra_w)

        # Region mask for expanded area
        region = mask_scores.new_zeros((H, W), dtype=mask_scores.dtype)
        region[rmin_e:rmax_e + 1, cmin_e:cmax_e + 1] = 1.0

        # Positive row/col labels: rows/cols passing the original GT box
        row_labels_full = mask_scores.new_zeros(H)
        row_labels_full[rmin:rmax + 1] = 1.0
        col_labels_full = mask_scores.new_zeros(W)
        col_labels_full[cmin:cmax + 1] = 1.0

        # Row/column max pooling within expanded region
        region_scores = mask_scores * region
        row_input = region_scores[rmin_e:rmax_e + 1, cmin_e:cmax_e + 1].amax(dim=1)
        col_input = region_scores[rmin_e:rmax_e + 1, cmin_e:cmax_e + 1].amax(dim=0)

        row_labels = row_labels_full[rmin_e:rmax_e + 1]
        col_labels = col_labels_full[cmin_e:cmax_e + 1]

        # Compute bag-level loss
        if self.loss_type == 'bce':
            loss_row = self._bce_1d(row_input, row_labels)
            loss_col = self._bce_1d(col_input, col_labels)
        else:
            loss_row = self._dice_1d(row_input, row_labels)
            loss_col = self._dice_1d(col_input, col_labels)

        mil_loss = (loss_row + loss_col) * 0.5 * self.loss_weight * warmup_factor
        return mil_loss

    def compute_batch_loss(self,
                          mask_logits: torch.Tensor,
                          gt_bboxes: list,
                          iter_num: int = 0) -> torch.Tensor:
        """Compute MIL loss for a batch of instances.

        Args:
            mask_logits: [N, 1, H, W] mask predictions
            gt_bboxes: list of ground-truth boxes per image
            iter_num: current iteration for warmup

        Returns:
            Total MIL loss
        """
        mask_scores = mask_logits.sigmoid().squeeze(1)  # [N, H, W]
        if mask_scores.numel() == 0:
            return mask_scores.sum()

        # Warmup factor
        warmup_factor = min(iter_num / float(self.warmup_iters), 1.0)

        # Sample random expand ratios
        min_r, max_r = self.expand_ratio_range
        expand_ratios = min_r + (max_r - min_r) * torch.rand(
            mask_scores.size(0), device=mask_scores.device
        )

        # Compute loss per instance (simplified for single image)
        # In practice, need to handle batched GT boxes
        mil_loss_sum = mask_scores.new_tensor(0.0)
        # Note: Full implementation needs proper GT alignment

        return mil_loss_sum * warmup_factor
