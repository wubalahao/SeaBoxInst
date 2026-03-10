"""
Core Configuration for Box-Supervised Instance Segmentation

This configuration file demonstrates how to integrate the core algorithms:
- MILA (Multi-Instance Learning-driven Masked Attention)
- PCR (Perturbation Consistency Regularization)
- Quality-aware Classification (Quality Focal Loss)
"""

# Model configuration
model_config = {
    # Backbone: Swin Transformer
    'backbone': {
        'type': 'SwinTransformer',
        'pretrain_img_size': 224,
        'embed_dims': 96,
        'patch_size': 4,
        'window_size': 7,
        'mlp_ratio': 4,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
    },

    # Neck: PAFPN
    'neck': {
        'type': 'PAFPN',
        'in_channels': [96, 192, 384, 768],
        'out_channels': 256,
        'num_outs': 5,
    },

    # Detection Head with Quality Focal Loss
    'bbox_head': {
        'type': 'BoxInstBoxHead',  # Generic name
        'num_classes': 7,
        'in_channels': 256,
        'loss_cls': {
            'type': 'QualityFocalLoss',
            'use_sigmoid': True,
            'beta': 2.0,
            'loss_weight': 1.0,
        },
        'loss_bbox': {
            'type': 'CIoULoss',
            'loss_weight': 1.0,
        },
        'loss_centerness': {
            'type': 'CrossEntropyLoss',
            'use_sigmoid': True,
            'loss_weight': 1.0,
        },
        'centerness_on_reg': True,
    },

    # Mask Branch
    'mask_branch': {
        'type': 'BoxInstMaskBranch',  # Generic name
        'in_channels': 256,
        'branch_convs': 4,
        'branch_channels': 128,
        'branch_out_channels': 16,
    },

    # MILA Head (Core Innovation 1)
    'mask_head': {
        'type': 'MILMaskedAttentionBoxInstMaskHead',  # Generic name
        'in_channels': 16,

        # Masked Attention parameters
        'num_refine_layers': 2,
        'attn_channels': 64,
        'num_heads': 4,
        'attn_dropout': 0.0,
        'attn_mask_threshold': 0.2,

        # MIL parameters (Core)
        'mil_enable': True,
        'mil_loss_weight': 4.0,
        'mil_expand_ratio_range': (0.8, 1.6),
        'mil_loss_type': 'dice',
        'mil_warmup_iters': 10000,

        # Mask supervision parameters
        'bottom_pixels_removed': 10,
        'pairwise_size': 3,
        'pairwise_dilation': 2,
        'pairwise_color_thresh': 0.3,
        'pairwise_warmup': 10000,
    },
}

# PCR Configuration (Core Innovation 2)
train_config = {
    # PCR: Perturbation Consistency Regularization
    'consistency': {
        'enable': True,
        'mask_weight': 0.4,
        'cls_weight': 0.0,
        'ctr_weight': 0.0,
        'warmup_iters': 5000,
        'conf_thr': 0.3,
        'every_k_iters': 1,
        'same_instance': True,  # Key: weak indices + strong params

        # Strong photometric augmentation
        'aug': {
            'flip_prob': 0.0,
            'rotate_deg': 0,
            'scale_range': (1.0, 1.0),
            'color_jitter': {
                'bright': 0.2,
                'contrast': 0.2,
                'sat': 0.2,
                'hue': 0.1,
            },
            'blur_prob': 0.2,
            'jpeg_prob': 0.0,
        },
    },
}

# Inference Configuration
test_config = {
    'nms_pre': 2000,
    'score_thr': 0.05,
    'nms': {
        'type': 'soft_nms',
        'iou_threshold': 0.5,
        'min_score': 0.001,
    },
    'max_per_img': 2000,
}


def get_core_config():
    """Get complete configuration for box-supervised instance segmentation."""
    return {
        'model': model_config,
        'train': train_config,
        'test': test_config,
    }
