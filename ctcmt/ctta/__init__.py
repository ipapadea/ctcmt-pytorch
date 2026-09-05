"""Continual test-time adaptation components (framework-agnostic where possible).

Modules:
  ema.py             - EMAUpdater (mean-teacher update)
  pseudo_labels.py   - DynamicThresholdFilter, filter_instances, to_gt_instances
  gate.py            - ScoreEMGate
  restore.py         - StochasticRestore (with V2 shared-trunk factor)
  ctpv.py            - CTPVFilter (hard-argmax agreement)
  losses.py          - supcon_loss, CrossTaskContrastive, CrossTaskConsistency,
                       SoftSegConsistency, MoraitiObjectContrastive
  seg_aug.py         - aug_averaged_teacher_seg (CoTTA-style multi-scale)
  prototypes.py      - PrototypeAnchor (V4 seg feature-prototype loss)
  adapt_step.py      - CTCMTAdaptStep: MTL orchestrator (PanopticFPN)
  seg_adapt_step.py  - CTCMTSegAdaptStep: seg-only orchestrator (SemanticSegmentor)
"""
from .adapt_step import CTCMTAdaptStep, CTCMTHyperParams
from .ctpv import CTPVFilter
from .ema import EMAUpdater
from .losses import (
    DET_TO_SEG_CLASS_CITYSCAPES,
    CrossTaskConsistency,
    CrossTaskContrastive,
    MoraitiObjectContrastive,
    SoftSegConsistency,
    supcon_loss,
)
from .prototypes import PrototypeAnchor
from .pseudo_labels import (
    DynamicThresholdFilter,
    filter_instances,
    per_class_mean_scores,
    to_gt_instances,
    update_dynamic_thresholds,
)
from .restore import StochasticRestore
from .gate import ScoreEMGate
from .seg_adapt_step import CTCMTSegAdaptStep, CTCMTSegHyperParams
from .seg_aug import aug_averaged_teacher_seg

__all__ = [
    "CTCMTAdaptStep",
    "CTCMTHyperParams",
    "CTCMTSegAdaptStep",
    "CTCMTSegHyperParams",
    "CTPVFilter",
    "EMAUpdater",
    "ScoreEMGate",
    "DET_TO_SEG_CLASS_CITYSCAPES",
    "CrossTaskConsistency",
    "CrossTaskContrastive",
    "MoraitiObjectContrastive",
    "SoftSegConsistency",
    "PrototypeAnchor",
    "supcon_loss",
    "DynamicThresholdFilter",
    "filter_instances",
    "per_class_mean_scores",
    "to_gt_instances",
    "update_dynamic_thresholds",
    "StochasticRestore",
    "aug_averaged_teacher_seg",
]
