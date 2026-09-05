"""CoTTA-style multi-scale augmentation-averaged teacher seg probabilities.

Direct port of ``CTCMT_MTL._teacher_aug_avg_seg_probs`` and the CoTTA-SemSeg
version in ``CTCMT/detectron2/detectron2/modeling/meta_arch/cotta_semseg.py``.

Given the teacher wrapper and a preprocessed image tensor at (B=1, 3, H, W),
returns (B, K, H, W) probabilities averaged over all (scale, flip) pairs.
Each pass is snapped to the backbone's size-divisibility.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def _snap(v: int, divisor: int) -> int:
    v = max(int(round(v)), divisor)
    return ((v + divisor - 1) // divisor) * divisor


@torch.no_grad()
def aug_averaged_teacher_seg(
    teacher_wrapper,
    image_tensor: torch.Tensor,
    scales: Sequence[float],
    flips: Sequence[bool],
) -> torch.Tensor:
    H, W = image_tensor.shape[-2:]
    divisor = teacher_wrapper.size_divisibility()

    accum = None
    n = 0
    for s in scales:
        Hs, Ws = _snap(int(H * s), divisor), _snap(int(W * s), divisor)
        for flip in flips:
            x = F.interpolate(image_tensor, size=(Hs, Ws), mode="bilinear", align_corners=False)
            if flip:
                x = torch.flip(x, dims=[-1])
            feats = teacher_wrapper.backbone_features(x)
            logits = teacher_wrapper.sem_seg_logits(feats)
            logits = F.interpolate(logits, size=(Hs, Ws), mode="bilinear", align_corners=False)
            probs = logits.float().softmax(dim=1)
            if flip:
                probs = torch.flip(probs, dims=[-1])
            probs = F.interpolate(probs, size=(H, W), mode="bilinear", align_corners=False)
            accum = probs if accum is None else accum + probs
            n += 1
    return accum / float(max(n, 1))
