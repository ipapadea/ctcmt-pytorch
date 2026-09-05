"""Detectron2 PanopticFPN wrapper: preprocessing, features, det, seg, losses.

This is the *only* place the CTTA layer touches detectron2. Everything else in
``ctcmt.ctta`` operates on plain tensors and detectron2 ``Instances`` returned
by this wrapper, so the algorithmic code stays framework-agnostic.

The wrapper never rebuilds the model, never reimplements preprocessing, and
never redefines postprocessing — it only exposes handles onto the existing
``PanopticFPN`` submodules with a stable surface used by the adapt step.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Core wrapper
# --------------------------------------------------------------------------- #
class Detectron2ModelAdapter:
    """Thin, stable surface over a detectron2 ``PanopticFPN`` instance.

    Attributes:
        model: the underlying ``PanopticFPN`` (or, for legacy det-only source
            checkpoints, a ``GeneralizedRCNN`` — the ``sem_seg_head`` calls
            simply return ``None`` in that case).
    """

    def __init__(self, model: nn.Module):
        self.model = model

    # ---- device / mode ---------------------------------------------------- #
    @property
    def device(self) -> torch.device:
        return self.model.pixel_mean.device

    def student_train_mode(self) -> None:
        """PanopticFPN convention used by the reference adapter: put the
        model in train() so RPN/ROI emit losses, but keep the sem_seg head
        in eval() because we consume raw logits, not the head's own losses.
        """
        self.model.train()
        if hasattr(self.model, "sem_seg_head"):
            self.model.sem_seg_head.eval()

    def eval_mode(self) -> None:
        self.model.eval()

    def freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)

    def disable_mask_head(self) -> None:
        """No pseudo-masks available at test time — skip the mask head."""
        if hasattr(self.model, "roi_heads") and hasattr(self.model.roi_heads, "mask_on"):
            self.model.roi_heads.mask_on = False

    # ---- preprocessing / features ---------------------------------------- #
    def preprocess(self, batched_inputs: List[Dict[str, Any]]):
        """Detectron2 ``ImageList`` with padding and normalization applied."""
        return self.model.preprocess_image(batched_inputs)

    def backbone_features(self, image_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model.backbone(image_tensor)

    def sem_seg_logits(
        self,
        features: Dict[str, torch.Tensor],
        gt_sem_seg: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Raw semantic-segmentation logits at head resolution.

        Returns ``None`` on models without ``sem_seg_head`` (det-only).
        """
        if not hasattr(self.model, "sem_seg_head"):
            return None
        logits, _ = self.model.sem_seg_head(features, gt_sem_seg)
        return logits

    # ---- detection: raw predictions (resized-input coords) --------------- #
    @torch.no_grad()
    def detect_raw(
        self, batched_inputs: List[Dict[str, Any]]
    ) -> Tuple[list, Optional[torch.Tensor]]:
        """Teacher-style inference: raw ``Instances`` in the model's input
        frame plus raw seg logits (or None). Matches
        ``PanopticFPN.inference(..., do_postprocess=False)``.
        """
        out = self.model.inference(batched_inputs, do_postprocess=False)
        if isinstance(out, tuple) and len(out) == 2:
            det_results, sem_seg_results = out
        else:
            det_results, sem_seg_results = out, None
        return det_results, sem_seg_results

    # ---- detection: postprocessed for evaluators ------------------------- #
    @torch.no_grad()
    def detect(self, batched_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluator-facing output. Delegates to ``PanopticFPN.inference``
        with post-processing so the returned dicts are compatible with
        ``CityscapesInstanceEvaluator`` / ``SemSegEvaluator`` / ``COCOEvaluator``.
        """
        return self.model.inference(batched_inputs, do_postprocess=True)

    # ---- semantic-segmentation-only: evaluator-facing output ------------ #
    @torch.no_grad()
    def sem_seg_predict(self, batched_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Postprocessed segmentation output for ``SemanticSegmentor``-style
        models. Forces ``eval()`` because ``SemanticSegmentor.forward``
        returns losses in training mode (and crashes on ``None`` targets).
        """
        was_training = self.model.training
        self.model.eval()
        out = self.model(batched_inputs)
        if was_training:
            self.model.train()
        return out

    # ---- student losses --------------------------------------------------- #
    def det_pseudo_losses(
        self,
        images,
        features: Dict[str, torch.Tensor],
        gt_instances: list,
    ) -> Dict[str, torch.Tensor]:
        """Standard Faster R-CNN losses using pseudo ``gt_instances``.

        Runs the student's ``proposal_generator`` + ``roi_heads`` in
        train() mode with pseudo-GT already converted via
        :func:`ctcmt.ctta.pseudo_labels.to_gt_instances`.
        """
        from detectron2.utils.events import EventStorage

        with EventStorage(0):
            proposals, prop_losses = self.model.proposal_generator(
                images, features, gt_instances
            )
            _, det_losses = self.model.roi_heads(
                images, features, proposals, gt_instances
            )
        merged: Dict[str, torch.Tensor] = {}
        merged.update({f"rpn/{k}": v for k, v in prop_losses.items()})
        merged.update({f"roi/{k}": v for k, v in det_losses.items()})
        return merged

    # ---- introspection used by the CTTA layer --------------------------- #
    @property
    def num_classes(self) -> int:
        """Detection class count. 0 for models without ROI heads."""
        if not hasattr(self.model, "roi_heads"):
            return 0
        return int(self.model.roi_heads.num_classes)

    @property
    def num_seg_classes(self) -> int:
        if not hasattr(self.model, "sem_seg_head"):
            return 0
        # SemSegFPNHead doesn't expose num_classes directly; read it off the
        # final 1x1 predictor conv.
        return int(self.model.sem_seg_head.predictor.out_channels)

    @property
    def roi_box_in_features(self) -> List[str]:
        if not hasattr(self.model, "roi_heads"):
            return []
        return list(self.model.roi_heads.box_in_features)

    def fpn_stride_of(self, feat_key: str) -> int:
        return int(self.model.backbone.output_shape()[feat_key].stride)

    def deepest_feature_key(self) -> str:
        """Deepest FPN key. Prefers the deepest ROI-input feature when the
        model has ROI heads; otherwise falls back to the last key of the
        backbone output. Used by CT-CL (MTL) and V4 prototypes (Seg).
        """
        if hasattr(self.model, "roi_heads"):
            return list(self.model.roi_heads.box_in_features)[-1]
        return list(self.model.backbone.output_shape().keys())[-1]

    def size_divisibility(self) -> int:
        return int(getattr(self.model.backbone, "size_divisibility", 32) or 32)

    # ---- state helpers used by EMA / restore ---------------------------- #
    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    def named_trainable_wb(self):
        """Iterate over ``(fully.qualified.name, param)`` for weight/bias
        params that carry gradient (used by stochastic restore)."""
        for nm, m in self.model.named_modules():
            for p_name, p in m.named_parameters(recurse=False):
                if p_name in ("weight", "bias") and p.requires_grad:
                    yield f"{nm}.{p_name}", p


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
@dataclass
class Triplet:
    """Student/teacher/anchor triplet, each already loaded from the same
    checkpoint. Convention:
      - ``student`` is trainable and in train() at adapt time.
      - ``teacher`` is frozen but kept in train() so BN keeps eval semantics
        set by the reference (matches the detectron2 CTCMT_MTL adapter).
      - ``anchor`` is frozen and in eval() — the immutable source snapshot.
    """
    student: Detectron2ModelAdapter
    teacher: Detectron2ModelAdapter
    anchor: Detectron2ModelAdapter


def _build_and_load(cfg, *, train_mode: bool, freeze: bool, disable_mask_head: bool):
    """Mirrors ``CTCMT_MTL._build_and_load`` from the reference meta-arch."""
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import META_ARCH_REGISTRY

    arch_name = getattr(cfg.MODEL, "CTCMT_STUDENT_META_ARCH", None) or cfg.MODEL.META_ARCHITECTURE
    # Never construct the wrapping CTCMT_MTL meta-arch here — we replace it.
    if arch_name == "CTCMT_MTL":
        arch_name = "PanopticFPN"

    model_cls = META_ARCH_REGISTRY.get(arch_name)
    m = model_cls(cfg)
    DetectionCheckpointer(m).load(cfg.MODEL.WEIGHTS)
    m.to(torch.device(cfg.MODEL.DEVICE))
    if train_mode:
        m.train()
    else:
        m.eval()
    if freeze:
        for p in m.parameters():
            p.requires_grad_(False)
    adapter = Detectron2ModelAdapter(m)
    if disable_mask_head:
        adapter.disable_mask_head()
    return adapter


def build_source_only(cfg) -> Detectron2ModelAdapter:
    """Single frozen, eval-mode PanopticFPN loaded from ``cfg.MODEL.WEIGHTS``.

    Use this for the source-only equivalence test and for any pre-adaptation
    evaluation baseline.
    """
    return _build_and_load(cfg, train_mode=False, freeze=True, disable_mask_head=False)


def build_triplet(cfg) -> Triplet:
    """Load student/teacher/anchor from the same source checkpoint.

    Teacher and anchor are kept in ``eval()`` so their internal
    ``sem_seg_head.forward(features, None)`` returns logits instead of
    trying to compute cross-entropy against a ``None`` target. The
    reference detectron2 CTCMT_MTL relies on ``Trainer.test()`` doing that
    flip implicitly; we do it explicitly at construction time.
    """
    student = _build_and_load(cfg, train_mode=True, freeze=False, disable_mask_head=True)
    teacher = _build_and_load(cfg, train_mode=False, freeze=True, disable_mask_head=False)
    anchor = _build_and_load(cfg, train_mode=False, freeze=True, disable_mask_head=True)
    return Triplet(student=student, teacher=teacher, anchor=anchor)


def build_seg_triplet(cfg) -> Triplet:
    """Same triplet layout as ``build_triplet`` but forces the student
    meta-arch to ``SemanticSegmentor`` — used by CT-CMT-Seg with a
    dedicated Semantic-FPN source checkpoint.
    """
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import META_ARCH_REGISTRY

    def _build(train_mode: bool, freeze: bool):
        m = META_ARCH_REGISTRY.get("SemanticSegmentor")(cfg)
        DetectionCheckpointer(m).load(cfg.MODEL.WEIGHTS)
        m.to(torch.device(cfg.MODEL.DEVICE))
        if train_mode:
            m.train()
        else:
            m.eval()
        if freeze:
            for p in m.parameters():
                p.requires_grad_(False)
        return Detectron2ModelAdapter(m)

    student = _build(train_mode=True, freeze=False)
    teacher = _build(train_mode=False, freeze=True)
    anchor = _build(train_mode=False, freeze=True)
    return Triplet(student=student, teacher=teacher, anchor=anchor)
