"""AMROD-style dynamic per-class thresholds and pseudo-label conversion.

Both pieces are direct ports of ``dynamic_threshold`` / ``process_pseudo_label``
in ``CTCMT/detectron2/detectron2/modeling/meta_arch/amrod.py``, and of the
``_dyn_thresholds`` helper in ``ctcmt_mtl.py``.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch


def update_dynamic_thresholds(
    prev: Sequence[float],
    per_class_mean_scores: Sequence[float],
    alpha: float,
    gamma: float,
    lo: float,
    hi: float,
) -> List[float]:
    """Per-class τ update: τ ← γ·τ + (1-γ)·α·√score, clipped to [lo, hi]."""
    out: List[float] = []
    for th, mean in zip(prev, per_class_mean_scores):
        if mean > 0:
            th = gamma * th + (1.0 - gamma) * alpha * math.sqrt(mean)
        out.append(max(min(th, hi), lo))
    return out


def per_class_mean_scores(
    scores: torch.Tensor,
    pred_classes: torch.Tensor,
    num_classes: int,
    score_floor: float = 0.1,
) -> List[float]:
    """Mean confidence per detection class, over boxes whose score > floor.

    Classes with no surviving boxes get 0.0 (so ``update_dynamic_thresholds``
    leaves the threshold untouched for that class, matching AMROD).
    """
    if scores.numel() == 0:
        return [0.0] * num_classes
    valid = scores > score_floor
    if not valid.any():
        return [0.0] * num_classes
    s = scores[valid]
    c = pred_classes[valid].long()
    out = [0.0] * num_classes
    for k in range(num_classes):
        idx = c == k
        if int(idx.sum()) > 0:
            out[k] = float(s[idx].mean().item())
    return out


class DynamicThresholdFilter:
    """Stateful per-class threshold. Call ``update`` after teacher inference,
    then ``filter`` to keep boxes whose score exceeds their class threshold.
    """

    def __init__(
        self,
        num_classes: int,
        init: float,
        lo: float,
        hi: float,
        alpha: float,
        gamma: float,
    ):
        self.num_classes = int(num_classes)
        self.lo = float(lo)
        self.hi = float(hi)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.thresholds: List[float] = [float(init)] * self.num_classes

    def update(self, scores: torch.Tensor, pred_classes: torch.Tensor) -> List[float]:
        means = per_class_mean_scores(scores, pred_classes, self.num_classes)
        self.thresholds = update_dynamic_thresholds(
            self.thresholds, means, self.alpha, self.gamma, self.lo, self.hi
        )
        return list(self.thresholds)

    def keep_mask(self, scores: torch.Tensor, pred_classes: torch.Tensor) -> torch.Tensor:
        thr = torch.tensor(self.thresholds, device=scores.device, dtype=scores.dtype)
        return scores >= thr[pred_classes.long()]


# --------------------------------------------------------------------------- #
# Detectron2 helpers (imported lazily to keep this module import-safe without
# detectron2 installed — the math helpers above are tested standalone).
# --------------------------------------------------------------------------- #
def filter_instances(instances, keep: torch.Tensor):
    """Return a copy of ``Instances`` with ``keep`` applied to every field."""
    from detectron2.structures import Boxes, Instances

    new_inst = Instances(instances.image_size)
    new_inst.pred_boxes = Boxes(instances.pred_boxes.tensor[keep])
    new_inst.pred_classes = instances.pred_classes[keep]
    new_inst.scores = instances.scores[keep]
    for k in instances.get_fields():
        if k not in ("pred_boxes", "pred_classes", "scores"):
            new_inst.set(k, instances.get(k)[keep])
    return new_inst


def to_gt_instances(instances_list):
    """Convert filtered teacher ``Instances`` (``pred_boxes`` / ``pred_classes``)
    into student-style ``Instances`` (``gt_boxes`` / ``gt_classes``).
    """
    from detectron2.structures import Boxes, Instances

    out = []
    for inst in instances_list:
        g = Instances(inst.image_size)
        g.gt_boxes = Boxes(inst.pred_boxes.tensor.clone())
        g.gt_classes = inst.pred_classes.long().clone()
        out.append(g)
    return out
