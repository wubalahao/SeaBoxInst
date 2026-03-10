"""
Masked Attention Refinement for Instance Segmentation

This module implements the masked multi-head attention mechanism for iterative
mask refinement in box-supervised instance segmentation. The attention is conditioned
on the predicted mask probability to dynamically gate feature aggregation.

Our approach: Uses predicted mask probabilities as attention masks to gate
feature aggregation, enabling progressive boundary refinement without pixel-level
mask supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedAttentionRefinement(nn.Module):
    """Lightweight masked attention refinement for mask logits.

    Uses predicted mask probabilities as attention masks to gate feature
    aggregation, enabling progressive boundary refinement.
    """

    def __init__(self,
                 in_channels,
                 param_dim=8,
                 attn_channels=64,
                 num_heads=4,
                 num_refine_layers=2,
                 attn_dropout=0.0,
                 attn_mask_threshold=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.param_dim = param_dim
        self.attn_channels = attn_channels
        self.num_heads = num_heads
        self.num_refine_layers = num_refine_layers
        self.attn_mask_threshold = attn_mask_threshold

        assert attn_channels % num_heads == 0, \
            'attn_channels must be divisible by num_heads'

        self.feat_proj = nn.Conv2d(in_channels, attn_channels, kernel_size=1)
        self.param_embed = nn.Linear(param_dim, attn_channels)

        self.attn_layers = nn.ModuleList()
        self.refine_ffns = nn.ModuleList()
        self.refine_convs = nn.ModuleList()

        for _ in range(num_refine_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=attn_channels,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    batch_first=False
                )
            )
            self.refine_ffns.append(
                nn.Sequential(
                    nn.LayerNorm(attn_channels),
                    nn.Linear(attn_channels, attn_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(attn_channels, attn_channels)
                )
            )
            self.refine_convs.append(
                nn.Conv2d(attn_channels, 1, kernel_size=3, padding=1)
            )

    def forward(self, mask_logits, instance_params, base_feat):
        """Refine mask logits using masked attention.

        Args:
            mask_logits: [N, 1, H, W] initial mask logits
            instance_params: [N, param_dim] instance-specific parameters
            base_feat: [N, C, H, W] feature maps

        Returns:
            refined_logits: [N, 1, H, W] refined mask logits
        """
        num_insts, _, height, width = mask_logits.size()
        proj_feat = self.feat_proj(base_feat)
        key_value = proj_feat.flatten(2).permute(2, 0, 1)  # [HW, N, C]

        query = self.param_embed(instance_params).unsqueeze(0)  # [1, N, C]
        refined_logits = mask_logits

        for layer_idx in range(self.num_refine_layers):
            attn_map = torch.sigmoid(refined_logits).reshape(num_insts, -1)

            # Create padding mask from predicted mask
            if self.attn_mask_threshold is not None:
                padding_mask = attn_map <= self.attn_mask_threshold
                # Avoid empty mask for MultiheadAttention
                if padding_mask.any():
                    all_true = padding_mask.all(dim=1, keepdim=True)
                    padding_mask = padding_mask & ~all_true
            else:
                padding_mask = None

            # Multi-head attention with masked feature aggregation
            attn_output, _ = self.attn_layers[layer_idx](
                query,
                key_value,
                key_value,
                key_padding_mask=padding_mask
            )

            # Scale-modulated feature enhancement
            context = attn_output.squeeze(0)  # [N, C]
            scale = torch.sigmoid(self.refine_ffns[layer_idx](context))
            scale = scale.view(num_insts, self.attn_channels, 1, 1)

            modulated_feat = proj_feat * scale
            delta_mask = self.refine_convs[layer_idx](modulated_feat)

            # Residual update
            refined_logits = refined_logits + delta_mask
            query = attn_output  # Update query for next layer

        return refined_logits


def aligned_bilinear(tensor, factor):
    """Align and upsample tensor by factor.

    Args:
        tensor: [N, C, H, W]
        factor: upsampling factor

    Returns:
        aligned tensor [N, C, H*factor, W*factor]
    """
    n, c, h, w = tensor.shape
    tensor = F.interpolate(
        tensor, scale_factor=factor, mode='bilinear', align_corners=False
    )
    return tensor
