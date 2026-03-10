"""
PCR: Perturbation Consistency Regularization for Box-Supervised Segmentation

This module implements the dual-view training framework with photometric consistency
regularization to enhance robustness against underwater image degradations.

Our approach:
- Dual-view training: weak photometric view + strong photometric view
- Same-instance alignment: reuse weak-view indices with strong-view parameters
- Stop-gradient on weak view to prevent unstable bidirectional pulling
- MSE loss on mask predictions within GT box regions
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PCRDetector(nn.Module):
    """Detector with Perturbation Consistency Regularization.

    Implements dual-view training for robust mask prediction under severe
    photometric variations (color cast, blur, noise) common in underwater imagery.
    """

    def __init__(self,
                 base_detector,
                 pcr_config=None):
        super().__init__()
        self.base_detector = base_detector
        self.pcr_cfg = pcr_config or {}

        # PCR parameters
        self.pcr_enable = self.pcr_cfg.get('enable', True)
        self.mask_weight = self.pcr_cfg.get('mask_weight', 0.4)
        self.warmup_iters = self.pcr_cfg.get('warmup_iters', 5000)
        self.conf_thr = self.pcr_cfg.get('conf_thr', 0.3)

        # Same-instance alignment: reuse weak indices, use strong params
        self.same_instance = self.pcr_cfg.get('same_instance', True)

        # Iteration counter for warmup
        self.register_buffer('_pcr_iter', torch.zeros(1))

    def forward_train(self,
                     img,
                     img_metas,
                     gt_bboxes,
                     gt_labels,
                     gt_bboxes_ignore=None,
                     gt_masks=None):
        """Forward training with PCR.

        Args:
            img: [B, C, H, W] input images
            img_metas: image metadata
            gt_bboxes: ground-truth boxes
            gt_labels: ground-truth labels
            gt_bboxes_ignore: boxes to ignore
            gt_masks: ground-truth masks (not used in box-supervised)

        Returns:
            Dictionary of losses
        """
        # ==================== Weak View (Standard Training) ====================
        feats_w = self.base_detector.extract_feat(img)
        cls_score_w, bbox_pred_w, centerness_w, param_pred_w = \
            self.base_detector.bbox_head(feats_w, self.base_detector.mask_head.param_conv)

        # Detection losses
        bbox_loss_w, coors_w, level_inds_w, img_inds_w, gt_inds_w = \
            self.base_detector.bbox_head.loss(
                cls_score_w, bbox_pred_w, centerness_w,
                gt_bboxes, gt_labels, img_metas,
                gt_bboxes_ignore=gt_bboxes_ignore
            )

        # Mask branch
        mask_feat_w = self.base_detector.mask_branch(feats_w)
        inputs = (cls_score_w, centerness_w, param_pred_w,
                  coors_w, level_inds_w, img_inds_w, gt_inds_w)
        param_w, coors_w, level_inds_w, img_inds_w, gt_inds_w = \
            self.base_detector.mask_head.training_sample(*inputs)
        mask_logits_w = self.base_detector.mask_head(
            mask_feat_w, param_w, coors_w, level_inds_w, img_inds_w
        )
        mask_loss_w = self.base_detector.mask_head.loss(
            img, img_metas, mask_logits_w, gt_inds_w, gt_bboxes,
            gt_masks, gt_labels
        )
        bbox_loss_w.update(mask_loss_w)

        # ==================== PCR Branch (Strong View) ====================
        pcr_loss_dict = {}
        if self.pcr_enable:
            self._pcr_iter += 1

            # Generate strong photometric view
            img_s = self._photometric_augment(img)

            # Forward strong view
            feats_s = self.base_detector.extract_feat(img_s)
            cls_score_s, bbox_pred_s, centerness_s, param_pred_s = \
                self.base_detector.bbox_head(feats_s, self.base_detector.mask_head.param_conv)

            mask_feat_s = self.base_detector.mask_branch(feats_s)

            # Same-instance alignment: reuse weak indices but use strong parameters
            if self.same_instance:
                mask_logits_s = self._forward_same_instance(
                    mask_feat_s, param_pred_s, param_pred_w,
                    coors_w, level_inds_w, img_inds_w, img
                )
                img_inds_s = img_inds_w
                gt_inds_s = gt_inds_w
            else:
                # Independent sampling
                _, coors_s, level_inds_s, img_inds_s, gt_inds_s = \
                    self.base_detector.bbox_head.loss(
                        cls_score_s, bbox_pred_s, centerness_s,
                        gt_bboxes, gt_labels, img_metas,
                        gt_bboxes_ignore=gt_bboxes_ignore
                    )
                inputs_s = (cls_score_s, centerness_s, param_pred_s,
                            coors_s, level_inds_s, img_inds_s, gt_inds_s)
                param_s, coors_s, level_inds_s, img_inds_s, gt_inds_s = \
                    self.base_detector.mask_head.training_sample(*inputs_s)
                mask_logits_s = self.base_detector.mask_head(
                    mask_feat_s, param_s, coors_s, level_inds_s, img_inds_s
                )

            # Compute PCR loss
            loss_pcr = self._compute_mask_pcr(
                mask_logits_w, img_inds_w, gt_inds_w,
                mask_logits_s, img_inds_s, gt_inds_s,
                gt_bboxes, out_stride=4
            )
            if loss_pcr is not None:
                pcr_loss_dict['loss_pcr_mask'] = loss_pcr

        if pcr_loss_dict:
            bbox_loss_w.update(pcr_loss_dict)

        return bbox_loss_w

    def _photometric_augment(self, img):
        """Generate strong photometric augmentation.

        Applies color jitter, blur, and noise to create heavily augmented view.

        Args:
            img: [B, C, H, W] normalized image tensor

        Returns:
            Augmented image tensor
        """
        b, c, h, w = img.shape
        device = img.device
        dtype = img.dtype
        out = img.clone()

        # Brightness: [-0.2, 0.2]
        b_delta = (torch.rand(b, 1, 1, 1, device=device, dtype=dtype) - 0.5) * 0.4
        # Contrast: [0.8, 1.2]
        c_alpha = 0.8 + torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * 0.4
        # Per-channel color scale: [0.8, 1.2]
        ch_scale = 0.8 + torch.rand(b, c, 1, 1, device=device, dtype=dtype) * 0.4

        out = out * c_alpha + b_delta
        out = out * ch_scale

        # Blur augmentation
        blur_prob = 0.2
        if blur_prob > 0:
            do_blur = torch.rand(b, device=device) < blur_prob
            if do_blur.any():
                k = 3
                pad = k // 2
                blurred = F.avg_pool2d(out, kernel_size=k, stride=1, padding=pad)
                blend_alpha = 0.5
                out[do_blur] = blend_alpha * blurred[do_blur] + (1.0 - blend_alpha) * out[do_blur]

        # Gaussian noise
        noise_std = 0.02
        noise = torch.randn_like(out) * noise_std
        out = out + noise

        return out

    def _forward_same_instance(self, mask_feat_s, param_pred_s, param_pred_w,
                               coors_w, level_inds_w, img_inds_w, img):
        """Forward strong view with same-instance alignment.

        Reuses weak-view spatial indices but fetches dynamic parameters
        from strong-view prediction at corresponding locations.

        Args:
            mask_feat_s: strong view mask features
            param_pred_s: strong view dynamic parameters
            param_pred_w: weak view dynamic parameters (unused in this path)
            coors_w, level_inds_w, img_inds_w: weak view indices
            img: original image for stride info

        Returns:
            mask_logits_s: strong view mask logits
        """
        strides_tensor = torch.tensor(
            self.base_detector.bbox_head.strides,
            device=img.device, dtype=coors_w.dtype
        )
        params_s_list = []

        for k in range(coors_w.size(0)):
            im = int(img_inds_w[k].item())
            lvl = int(level_inds_w[k].item())
            stride = float(strides_tensor[lvl].item())
            x = float(coors_w[k, 0].item())
            y = float(coors_w[k, 1].item())

            # Map to feature grid
            u = int(round((x - stride / 2.0) / stride))
            v = int(round((y - stride / 2.0) / stride))

            H_l = int(param_pred_s[lvl].shape[2])
            W_l = int(param_pred_s[lvl].shape[3])
            if H_l <= 0 or W_l <= 0:
                continue
            u = max(0, min(W_l - 1, u))
            v = max(0, min(H_l - 1, v))

            # Get strong-view parameters at weak-view location
            params_vec = param_pred_s[lvl][im, :, v, u]
            params_s_list.append(params_vec)

        if len(params_s_list) == 0:
            # Fallback: empty
            return torch.zeros_like(mask_feat_s)

        param_s = torch.stack(params_s_list, dim=0)
        mask_logits_s = self.base_detector.mask_head(
            mask_feat_s, param_s, coors_w, level_inds_w, img_inds_w
        )

        return mask_logits_s

    def _compute_mask_pcr(self,
                          mask_logits_w, img_inds_w, gt_inds_w,
                          mask_logits_s, img_inds_s, gt_inds_s,
                          gt_bboxes,
                          out_stride=4):
        """Compute mask consistency loss between views.

        Enforces MSE consistency on mask probabilities within GT box regions.

        Args:
            mask_logits_w: weak view mask logits
            img_inds_w, gt_inds_w: weak view indices
            mask_logits_s: strong view mask logits
            img_inds_s, gt_inds_s: strong view indices
            gt_bboxes: ground-truth boxes
            out_stride: output stride of mask

        Returns:
            PCR loss value
        """
        if mask_logits_w.numel() == 0 or mask_logits_s.numel() == 0:
            return None

        # Stop gradients on weak view (teacher detach)
        prob_w = mask_logits_w.sigmoid().detach()
        prob_s = mask_logits_s.sigmoid()

        # Prepare GT offsets
        num_imgs = len(gt_bboxes)
        gt_counts = [len(b) for b in gt_bboxes]
        gt_offsets = [0]
        for i in range(1, num_imgs):
            gt_offsets.append(gt_offsets[i - 1] + gt_counts[i - 1])

        start = out_stride // 2
        h_low = prob_w.shape[-2]
        w_low = prob_w.shape[-1]

        # Build strong view index map
        from collections import defaultdict
        strong_map = defaultdict(list)
        for j in range(gt_inds_s.numel()):
            gi = int(gt_inds_s[j].item())
            if gi < 0:
                continue
            im = int(img_inds_s[j].item())
            strong_map[(im, gi)].append(j)

        loss_sum = mask_logits_w.new_tensor(0.0)
        count = 0

        # Compute MSE loss per instance
        for i in range(gt_inds_w.numel()):
            gi = int(gt_inds_w[i].item())
            if gi < 0:
                continue
            im = int(img_inds_w[i].item())
            key = (im, gi)
            cand = strong_map.get(key, None)
            if not cand:
                continue
            j = cand.pop(0)

            # Get local GT index
            local_base = gt_offsets[im]
            local_idx = gi - local_base
            if local_idx < 0 or local_idx >= gt_counts[im]:
                continue
            box = gt_bboxes[im][local_idx]
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

            # Convert to low-res coordinates
            x0i = int(max(0, math.floor((x1 - start) / out_stride)))
            y0i = int(max(0, math.floor((y1 - start) / out_stride)))
            x1i = int(min(w_low - 1, math.ceil((x2 - start) / out_stride)))
            y1i = int(min(h_low - 1, math.ceil((y2 - start) / out_stride)))
            if x1i <= x0i or y1i <= y0i:
                continue

            pw = prob_w[i, 0, y0i:y1i + 1, x0i:x1i + 1]
            ps = prob_s[j, 0, y0i:y1i + 1, x0i:x1i + 1]
            if pw.numel() == 0 or ps.numel() == 0:
                continue

            # MSE on probabilities
            loss_sum = loss_sum + F.mse_loss(pw, ps, reduction='mean')
            count += 1

        if count == 0:
            return None

        loss = loss_sum / float(count)

        # Warmup ramp
        if self.warmup_iters > 0:
            ramp = min(self._pcr_iter.item() / float(self.warmup_iters), 1.0)
            loss = loss * ramp

        loss = loss * float(self.mask_weight)
        return loss
