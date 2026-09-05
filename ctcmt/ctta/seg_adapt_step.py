"""CT-CMT-Seg adapt step: seg-only CTTA on a Semantic-FPN source.

Mirrors ``CTCMT/detectron2/detectron2/modeling/meta_arch/ctcmt_seg.py``
line-for-line, split into small named components:

  teacher soft-CE consistency  -> :class:`SoftSegConsistency`
  optional aug-averaging       -> :func:`aug_averaged_teacher_seg`
  V4 prototype anchor          -> :class:`PrototypeAnchor`
  EMA teacher update           -> :class:`EMAUpdater`
  backbone-protected restore   -> :class:`StochasticRestore` (V2 factor)

Teacher predictions in evaluator schema come from
``teacher.sem_seg_predict(batched_inputs)`` (which forces ``eval()``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from detectron2.structures import ImageList

from ..d2.model_adapter import Detectron2ModelAdapter, Triplet
from .ema import EMAUpdater
from .losses import SoftSegConsistency
from .prototypes import PrototypeAnchor
from .restore import StochasticRestore
from .seg_aug import aug_averaged_teacher_seg


@dataclass
class CTCMTSegHyperParams:
    """Config surface for CT-CMT-Seg. Reads the same SOLVER keys the
    reference detectron2 ``CTCMT_Seg`` meta-arch reads."""

    ema_decay: float
    rst_prob: float
    backbone_rst_factor: float
    num_classes: int
    proto_enabled: bool
    proto_weight: float
    proto_ema: float
    proto_conf_thresh: float
    source_proto_init: bool
    aug_enabled: bool
    aug_scales: tuple
    aug_flips: tuple
    aug_conf_thresh: float

    @classmethod
    def from_cfg(cls, cfg) -> "CTCMTSegHyperParams":
        s = cfg.SOLVER
        return cls(
            ema_decay=float(getattr(s, "COTTA_EMA_DECAY", getattr(s, "MT", 0.999))),
            rst_prob=float(getattr(s, "COTTA_RESTORE_PROB", getattr(s, "RST_M", 0.01))),
            backbone_rst_factor=float(getattr(s, "CTCMT_BACKBONE_RST_FACTOR", 1.0)),
            num_classes=int(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES),
            proto_enabled=bool(getattr(s, "CTCMT_PROTO_ANCHOR", False)),
            proto_weight=float(getattr(s, "CTCMT_PROTO_WEIGHT", 0.0)),
            proto_ema=float(getattr(s, "CTCMT_PROTO_EMA", 0.999)),
            proto_conf_thresh=float(getattr(s, "CTCMT_SEG_PROTO_CONF_THRESH", 0.9)),
            source_proto_init=bool(getattr(s, "CTCMT_SEG_SOURCE_PROTO", False)),
            aug_enabled=bool(getattr(s, "CTCMT_SEG_AUG_ENABLED", False)),
            aug_scales=tuple(getattr(s, "CTCMT_SEG_AUG_SCALES", (1.0,))),
            aug_flips=tuple(getattr(s, "CTCMT_SEG_AUG_FLIPS", (False,))),
            aug_conf_thresh=float(getattr(s, "CTCMT_SEG_AUG_CONF_THRESH", 0.9)),
        )


class CTCMTSegAdaptStep:
    """One CT-CMT-Seg adaptation step on a Semantic-FPN source."""

    def __init__(
        self,
        triplet: Triplet,
        optimizer: torch.optim.Optimizer,
        hp: CTCMTSegHyperParams,
    ):
        self.triplet = triplet
        self.optimizer = optimizer
        self.hp = hp

        self.ema = EMAUpdater(decay=hp.ema_decay)
        # V2: shared-trunk restore factor only meaningful when < 1.
        self.restore = StochasticRestore(
            anchor_named_wb=self._anchor_snapshot(triplet.anchor),
            rst_prob=hp.rst_prob,
            cross_task_fisher=(hp.backbone_rst_factor < 1.0),
            backbone_rst_factor=hp.backbone_rst_factor,
        )
        self.soft_seg = SoftSegConsistency()
        self.proto = (
            PrototypeAnchor(
                num_classes=hp.num_classes,
                proto_ema=hp.proto_ema,
                proto_conf_thresh=hp.proto_conf_thresh,
                source_init=hp.source_proto_init,
            )
            if hp.proto_enabled and hp.proto_weight > 0.0
            else None
        )
        self.iter = 0

    @staticmethod
    def _anchor_snapshot(anchor: Detectron2ModelAdapter):
        for nm, m in anchor.model.named_modules():
            for p_name, p in m.named_parameters(recurse=False):
                if p_name in ("weight", "bias"):
                    yield f"{nm}.{p_name}", p

    def _preprocess(self, batched_inputs: List[Dict[str, Any]]):
        """Match ``SemanticSegmentor.forward``'s preprocessing exactly so
        our internal image tensor matches what the model would produce."""
        m = self.triplet.student.model
        images = [x["image"].to(m.device).float() for x in batched_inputs]
        images = [(x - m.pixel_mean) / m.pixel_std for x in images]
        return ImageList.from_tensors(
            images,
            m.backbone.size_divisibility,
            padding_constraints=m.backbone.padding_constraints,
        )

    @staticmethod
    def _sem_seg_logits(model: torch.nn.Module, image_tensor: torch.Tensor):
        """Deterministic head call in eval() so ``None`` targets don't crash."""
        features = model.backbone(image_tensor)
        was_training = model.sem_seg_head.training
        model.sem_seg_head.eval()
        logits, _ = model.sem_seg_head(features, None)
        if was_training:
            model.sem_seg_head.train()
        return logits, features

    def step(self, batched_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.iter += 1
        hp = self.hp
        student = self.triplet.student
        teacher = self.triplet.teacher
        anchor = self.triplet.anchor

        # Strip any target keys the DatasetMapper may pass through.
        clean_inputs = [
            {k: v for k, v in d.items() if k not in ("instances", "sem_seg")}
            for d in batched_inputs
        ]

        student.model.train()
        student.model.sem_seg_head.eval()

        images = self._preprocess(clean_inputs)
        image_tensor = images.tensor
        target_hw = image_tensor.shape[-2:]

        # Teacher pseudo-labels (raw, then optional aug-averaged).
        with torch.no_grad():
            t_logits, _ = self._sem_seg_logits(teacher.model, image_tensor)
            teacher_probs = F.interpolate(
                t_logits, size=target_hw, mode="bilinear", align_corners=False,
            ).float().softmax(dim=1)

            if hp.aug_enabled:
                a_logits, _ = self._sem_seg_logits(anchor.model, image_tensor)
                anchor_conf = a_logits.float().softmax(dim=1).max(dim=1)[0].mean()
                if float(anchor_conf.item()) < hp.aug_conf_thresh:
                    aug_probs = aug_averaged_teacher_seg(
                        teacher, image_tensor, hp.aug_scales, hp.aug_flips,
                    )
                    teacher_probs = F.interpolate(
                        aug_probs, size=target_hw, mode="bilinear", align_corners=False,
                    )

        # Student forward (backbone kept in train mode; head stays in eval).
        s_feats = student.model.backbone(image_tensor)
        s_logits, _ = student.model.sem_seg_head(s_feats, None)
        s_logits_full = F.interpolate(
            s_logits, size=target_hw, mode="bilinear", align_corners=False,
        )

        # Core seg loss: soft-CE consistency.
        loss = self.soft_seg(s_logits_full, teacher_probs)

        # V4: prototype anchor.
        if self.proto is not None:
            feat_key = student.deepest_feature_key()
            if hp.source_proto_init:
                with torch.no_grad():
                    a_logits, a_feats = self._sem_seg_logits(anchor.model, image_tensor)
                    a_probs_full = F.interpolate(
                        a_logits, size=target_hw, mode="bilinear", align_corners=False,
                    ).float().softmax(dim=1)
                self.proto.maybe_init(a_probs_full, a_feats[feat_key])
            proto_loss = self.proto(s_feats[feat_key], teacher_probs.detach())
            loss = loss + hp.proto_weight * proto_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.ema.update(teacher.model, student.model)
        self.restore.apply(student.named_trainable_wb())

        if self.iter % 50 == 0:
            print(f"[CTCMT-Seg] iter={self.iter} loss={float(loss.detach()):.4f}")

        return teacher.sem_seg_predict(clean_inputs)
