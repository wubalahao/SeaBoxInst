"""
Quality Focal Loss for Quality-Aware Classification

This module implements Quality Focal Loss (QFL) which aligns classification
confidence with localization quality (IoU) for box-supervised instance segmentation.

Our approach: Use IoU as continuous supervision signal to align classification
confidence with localization quality, particularly beneficial for box-supervised
learning where proposal quality is critical.

Reference: Generalized Focal Loss https://arxiv.org/abs/2006.04388
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def quality_focal_loss(pred, target, beta=2.0):
    r"""Quality Focal Loss (QFL).

    QFL is used to jointly learn classification and quality (IoU) estimation.
    Unlike standard cross-entropy, QFL treats the IoU as a continuous supervision
    signal and assigns larger gradients to high-quality samples.

    Args:
        pred: Predicted joint representation of classification and quality
              with shape (N, C), where C is the number of classes.
        target: Tuple of (label, score) where:
                - label: category id with shape (N,)
                - score: quality (IoU) score with shape (N,)
        beta: The beta parameter for calculating the modulating factor.
              Defaults to 2.0.

    Returns:
        Loss tensor with shape (N,).
    """
    assert len(target) == 2, "target for QFL must be a tuple of (label, score)"

    label, score = target

    # Negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()

    # Positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos],
        reduction='none') * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=False)
    return loss


class QualityFocalLoss(nn.Module):
    r"""Quality Focal Loss (QFL) module.

    A variant of Generalized Focal Loss that aligns classification confidence
    with localization quality. Particularly useful for box-supervised learning
    where proposal quality is critical.

    Args:
        use_sigmoid: Whether sigmoid operation is conducted in QFL.
                     Defaults to True.
        beta: The beta parameter for calculating the modulating factor.
              Defaults to 2.0.
        reduction: Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight: Loss weight of current loss. Defaults to 1.0.
        activated: Whether the input is activated. If True, input is treated
                   as probabilities. Defaults to False.
    """

    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False):
        super().__init__()
        assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred: Predicted joint representation of classification and quality
                  with shape (N, C), C is the number of classes.
            target: Tuple of (label, score) where:
                    - label: category id with shape (N,)
                    - score: quality (IoU) score with shape (N,)
            weight: The weight of loss for each prediction. Defaults to None.
            avg_factor: Average factor for averaging loss. Defaults to None.
            reduction_override: Override the reduction method. Defaults to None.

        Returns:
            Loss tensor
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.use_sigmoid:
            if self.activated:
                # Input is already probabilities
                calculate_loss_func = quality_focal_loss_with_prob
            else:
                # Input is logits
                calculate_loss_func = quality_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError

        return loss_cls


def quality_focal_loss_with_prob(pred, target, beta=2.0):
    r"""Quality Focal Loss with probability input.

    Similar to quality_focal_loss but accepts probability as input instead of logits.

    Args:
        pred: Predicted probabilities with shape (N, C)
        target: Tuple of (label, score)
        beta: Beta parameter for modulating factor

    Returns:
        Loss tensor with shape (N,)
    """
    assert len(target) == 2, "target for QFL must be a tuple of (label, score)"

    label, score = target

    # Negatives supervised by 0 quality score
    pred_sigmoid = pred
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()

    # Positives supervised by quality score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy(
        pred[pos, pos_label], score[pos],
        reduction='none') * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=False)
    return loss


def compute_iouQuality_score(pred_boxes, target_boxes):
    """Compute IoU-based quality scores for QFL.

    Args:
        pred_boxes: Predicted boxes [N, 4] (x1, y1, x2, y2)
        target_boxes: Target boxes [N, 4]

    Returns:
        IoU scores [N,]
    """
    # Compute intersection
    lt = torch.max(pred_boxes[:, :2], target_boxes[:, :2])
    rb = torch.min(pred_boxes[:, 2:], target_boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    # Compute union
    pred_wh = pred_boxes[:, 2:] - pred_boxes[:, :2]
    target_wh = target_boxes[:, 2:] - target_boxes[:, :2]
    pred_area = pred_wh[:, 0] * pred_wh[:, 1]
    target_area = target_wh[:, 0] * target_wh[:, 1]
    union = pred_area + target_area - inter

    # IoU
    iou = inter / (union + 1e-7)

    return iou
