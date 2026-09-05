"""CTTA loss functions used across AMROD / CoTTA / CTCMT-MTL.

Every implementation here is a direct port of the corresponding reference in
``CTCMT/detectron2/detectron2/modeling/meta_arch/ctcmt_mtl.py`` and
``.../amrod.py``; behaviour is intentionally identical for reproducibility.

Cityscapes taxonomy mapping used by the cross-task losses:
    detection classes 0..7 = (person, rider, car, truck, bus, train,
                              motorcycle, bicycle)
    seg trainIds  = (11, 12, 13, 14, 15, 16, 17, 18)
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DET_TO_SEG_CLASS_CITYSCAPES: Tuple[int, ...] = (11, 12, 13, 14, 15, 16, 17, 18)


# --------------------------------------------------------------------------- #
# SupCon (Khosla et al. 2020) — building block for CT-CL and Moraiti-CL.
# --------------------------------------------------------------------------- #
def supcon_loss(
    features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """features: (N, D) L2-normalized. labels: (N,) int."""
    N = features.size(0)
    if N < 2:
        return features.new_zeros(())
    device = features.device
    logits = torch.matmul(features, features.t()) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    labels = labels.view(-1, 1)
    mask_pos = torch.eq(labels, labels.t()).float().to(device)
    diag = torch.eye(N, device=device)
    mask_pos = mask_pos - diag
    exp_logits = torch.exp(logits) * (1.0 - diag)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    n_pos = mask_pos.sum(dim=1)
    valid = n_pos > 0
    if not valid.any():
        return features.new_zeros(())
    mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1)[valid] / n_pos[valid]
    return -mean_log_prob_pos.mean()


# --------------------------------------------------------------------------- #
# Cross-Task Contrastive Learning (CT-CL) — student FPN features.
# --------------------------------------------------------------------------- #
class CrossTaskContrastive(nn.Module):
    """Match ``CTCMT_MTL._ctcl_loss``. Uses the deepest FPN feature level
    referenced by the ROI head."""

    def __init__(
        self,
        temperature: float,
        roi_output: Tuple[int, int],
        include_seg_view: bool,
        det_to_seg_cls: Sequence[int] = DET_TO_SEG_CLASS_CITYSCAPES,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.roi_output = tuple(int(x) for x in roi_output)
        self.include_seg_view = bool(include_seg_view)
        self.det_to_seg_cls = tuple(int(x) for x in det_to_seg_cls)

    def forward(
        self,
        feat: torch.Tensor,
        boxes_feat_coords: torch.Tensor,
        classes: torch.Tensor,
        teacher_sem_probs_feat: Optional[torch.Tensor],
        num_seg_classes: int,
    ) -> torch.Tensor:
        """feat: (1, C, H, W) — a single FPN level in feature coordinates.

        boxes_feat_coords: (N, 4) already divided by FPN stride.
        classes: (N,) detection class ids.
        teacher_sem_probs_feat: (1, K, H, W) probabilities at the same
            resolution as ``feat`` (interpolated by the caller), or None.
        """
        from torchvision.ops import roi_align

        N = boxes_feat_coords.size(0)
        if N == 0:
            return feat.new_zeros(())

        H, W = feat.shape[-2:]
        rois = torch.cat(
            [torch.zeros((N, 1), device=boxes_feat_coords.device), boxes_feat_coords], dim=1
        )
        pooled = roi_align(
            feat, rois, output_size=self.roi_output, spatial_scale=1.0, aligned=True
        )
        z_det = F.normalize(pooled.mean(dim=(2, 3)), dim=1)

        views = [z_det]
        labels = [classes]

        if self.include_seg_view and teacher_sem_probs_feat is not None:
            z_seg_rows = []
            box_list = boxes_feat_coords.detach().cpu().tolist()
            cls_list = classes.detach().cpu().tolist()
            for b, c in zip(box_list, cls_list):
                if not (0 <= c < len(self.det_to_seg_cls)):
                    continue
                seg_c = self.det_to_seg_cls[c]
                if seg_c >= num_seg_classes or seg_c >= teacher_sem_probs_feat.shape[1]:
                    continue
                x1, y1, x2, y2 = [max(int(round(v)), 0) for v in b]
                x2 = min(x2, W)
                y2 = min(y2, H)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop_f = feat[0:1, :, y1:y2, x1:x2]
                crop_p = teacher_sem_probs_feat[0:1, seg_c:seg_c + 1, y1:y2, x1:x2]
                w_sum = crop_p.sum().clamp(min=1e-6)
                pooled_seg = (crop_f * crop_p).sum(dim=(2, 3)) / w_sum
                z_seg_rows.append((pooled_seg.squeeze(0), c))

            if z_seg_rows:
                z_seg = F.normalize(
                    torch.stack([z for z, _ in z_seg_rows], dim=0), dim=1
                )
                seg_classes = torch.tensor(
                    [c for _, c in z_seg_rows],
                    device=z_seg.device,
                    dtype=classes.dtype,
                )
                views.append(z_seg)
                labels.append(seg_classes)

        return supcon_loss(
            torch.cat(views, dim=0),
            torch.cat(labels, dim=0),
            temperature=self.temperature,
        )


# --------------------------------------------------------------------------- #
# Cross-Task Consistency Regularizer (CT-CR).
# --------------------------------------------------------------------------- #
class CrossTaskConsistency(nn.Module):
    """Match ``CTCMT_MTL._ctcr_loss``. Assigns entire box interior to the
    box's seg-taxonomy class in the target, then CE against the student
    seg logits with 255 as ignore index. Returns None when no valid pixels.
    """

    def __init__(self, det_to_seg_cls: Sequence[int] = DET_TO_SEG_CLASS_CITYSCAPES):
        super().__init__()
        self.det_to_seg_cls = tuple(int(x) for x in det_to_seg_cls)

    def forward(
        self,
        student_seg_logits: torch.Tensor,
        boxes_img_coords: torch.Tensor,
        classes: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        if boxes_img_coords.numel() == 0:
            return None
        B, K, H, W = student_seg_logits.shape
        img_h, img_w = image_hw
        sx = W / max(img_w, 1)
        sy = H / max(img_h, 1)
        target = torch.full(
            (B, H, W), 255, dtype=torch.long, device=student_seg_logits.device
        )
        boxes = boxes_img_coords.detach()
        cls_list = classes.detach().long().tolist()
        for j, (x1, y1, x2, y2) in enumerate(boxes.tolist()):
            c = cls_list[j]
            if not (0 <= c < len(self.det_to_seg_cls)):
                continue
            seg_c = self.det_to_seg_cls[c]
            if seg_c >= K:
                continue
            x1i = max(int(round(x1 * sx)), 0)
            y1i = max(int(round(y1 * sy)), 0)
            x2i = min(int(round(x2 * sx)), W)
            y2i = min(int(round(y2 * sy)), H)
            if x2i <= x1i or y2i <= y1i:
                continue
            target[0, y1i:y2i, x1i:x2i] = seg_c
        if (target != 255).sum() == 0:
            return None
        return F.cross_entropy(student_seg_logits, target, ignore_index=255)


# --------------------------------------------------------------------------- #
# CoTTA-style soft-CE segmentation consistency.
# --------------------------------------------------------------------------- #
class SoftSegConsistency(nn.Module):
    """- ∑_k teacher_p_k · log_softmax(student_logits)_k, mean over pixels."""

    def forward(
        self,
        student_seg_logits: torch.Tensor,
        teacher_seg_probs: torch.Tensor,
    ) -> torch.Tensor:
        # Match head resolutions if caller didn't already interpolate.
        if teacher_seg_probs.shape[-2:] != student_seg_logits.shape[-2:]:
            teacher_seg_probs = F.interpolate(
                teacher_seg_probs,
                size=student_seg_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        student_log_probs = F.log_softmax(student_seg_logits.float(), dim=1)
        per_pixel = -(teacher_seg_probs.detach() * student_log_probs).sum(dim=1)
        return per_pixel.mean()


# --------------------------------------------------------------------------- #
# Moraiti object-level contrastive (single-task CT-CMT, YOLOX reference).
# Kept for the AMROD baseline. Same math as SupCon on RoI-pooled student vs.
# teacher features labelled by pseudo class, matching
# ``mean_teacher_yolox_adapter_contrastive.py``.
# --------------------------------------------------------------------------- #
class MoraitiObjectContrastive(nn.Module):
    """Object-level CL between student and teacher RoI features per pseudo box."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        student_roi_feats: torch.Tensor,
        teacher_roi_feats: torch.Tensor,
        classes: torch.Tensor,
    ) -> torch.Tensor:
        z_s = F.normalize(student_roi_feats, dim=1)
        z_t = F.normalize(teacher_roi_feats, dim=1)
        feats = torch.cat([z_s, z_t], dim=0)
        labels = torch.cat([classes, classes], dim=0)
        return supcon_loss(feats, labels, temperature=self.temperature)
