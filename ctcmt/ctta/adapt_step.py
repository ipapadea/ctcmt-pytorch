"""CTCMT-MTL adapt step: one readable orchestration of every component.

Flow (per input image / batch):

    1. Teacher raw prediction (no grad).
    2. Score-EMA gate; if we skip, still run EMA + restore, then return.
    3. Dynamic per-class threshold update + box filter.
    4. Optional CTPV filter (teacher seg vs. det agreement).
    5. Student full forward: backbone -> sem_seg logits + det losses on
       pseudo-GT.
    6. Cross-task losses: CT-CL (contrastive) + CT-CR (consistency).
    7. Backward + optimizer step.
    8. EMA update, stochastic restore.
    9. Return teacher predictions in evaluator format.

The step never touches target GT: only ``image`` / ``height`` / ``width`` in
``batched_inputs`` are read.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from ..d2.model_adapter import Detectron2ModelAdapter, Triplet
from .ctpv import CTPVFilter
from .ema import EMAUpdater
from .gate import ScoreEMGate
from .losses import (
    DET_TO_SEG_CLASS_CITYSCAPES,
    CrossTaskConsistency,
    CrossTaskContrastive,
    SoftSegConsistency,
)
from .pseudo_labels import DynamicThresholdFilter, filter_instances, to_gt_instances
from .restore import StochasticRestore
from .seg_aug import aug_averaged_teacher_seg


# --------------------------------------------------------------------------- #
# Config plucked out of the detectron2 CfgNode. Keeps the step signature clean.
# --------------------------------------------------------------------------- #
@dataclass
class CTCMTHyperParams:
    # EMA + restore
    ema_decay: float
    rst_prob: float
    cross_task_fisher: bool
    backbone_rst_factor: float
    # Dynamic thresholds
    threshold_init: float
    threshold_max: float
    threshold_mini: float
    alpha_dt: float
    gamma_dt: float
    # Score-EMA gate
    score_em: float
    score_gamma: float
    score_thresh: float
    skip_score_em_gate: bool
    # Loss weights
    weight_det: float
    weight_seg: float
    weight_ctcl: float
    weight_ctcr: float
    # CT-CL
    ctcl_enabled: bool
    ctcl_include_seg_view: bool
    ctcl_temperature: float
    ctcl_roi_output: tuple
    # Seg aug-averaging
    seg_aug_enabled: bool
    seg_aug_conf_thresh: float
    seg_aug_scales: tuple
    seg_aug_flips: tuple
    # CTPV
    ctpv_enabled: bool
    ctpv_thresh: float
    # Single-task ablation switches
    det_only: bool
    seg_only: bool

    @classmethod
    def from_cfg(cls, cfg) -> "CTCMTHyperParams":
        s = cfg.SOLVER
        return cls(
            ema_decay=float(s.MT),
            rst_prob=float(s.RST_M),
            cross_task_fisher=bool(getattr(s, "CTCMT_CROSS_TASK_FISHER", False)),
            backbone_rst_factor=float(getattr(s, "CTCMT_BACKBONE_RST_FACTOR", 1.0)),
            threshold_init=float(s.THRESHOLD_INIT),
            threshold_max=float(s.THRESHOLD_MAX),
            threshold_mini=float(s.THRESHOLD_MINI),
            alpha_dt=float(s.ALPHA_DT),
            gamma_dt=float(s.GAMMA_DT),
            score_em=float(s.SCORE_EM),
            score_gamma=float(s.SCORE_GAMMA),
            score_thresh=float(s.SCORE_THRESH),
            skip_score_em_gate=bool(getattr(s, "CTCMT_SKIP_SCORE_EM_GATE", False)),
            weight_det=float(s.CTCMT_WEIGHT_DET),
            weight_seg=float(s.CTCMT_WEIGHT_SEG),
            weight_ctcl=float(s.CTCMT_WEIGHT_CTCL),
            weight_ctcr=float(getattr(s, "CTCMT_WEIGHT_CTCR", 0.0)),
            ctcl_enabled=bool(s.CTCMT_CTCL_ENABLED),
            ctcl_include_seg_view=bool(s.CTCMT_CTCL_SEG_VIEW),
            ctcl_temperature=float(s.CTCMT_CTCL_TEMPERATURE),
            ctcl_roi_output=tuple(s.CTCMT_CTCL_ROI_OUTPUT),
            seg_aug_enabled=bool(getattr(s, "CTCMT_SEG_AUG_ENABLED", False)),
            seg_aug_conf_thresh=float(getattr(s, "CTCMT_SEG_AUG_CONF_THRESH", 0.9)),
            seg_aug_scales=tuple(getattr(s, "CTCMT_SEG_AUG_SCALES", (1.0,))),
            seg_aug_flips=tuple(getattr(s, "CTCMT_SEG_AUG_FLIPS", (False,))),
            ctpv_enabled=bool(getattr(s, "CTCMT_CTPV_ENABLED", False)),
            ctpv_thresh=float(getattr(s, "CTCMT_CTPV_THRESH", 0.3)),
            det_only=bool(getattr(s, "CTCMT_DET_ONLY", False)),
            seg_only=bool(getattr(s, "CTCMT_SEG_ONLY", False)),
        )


# --------------------------------------------------------------------------- #
# The orchestrator.
# --------------------------------------------------------------------------- #
class CTCMTAdaptStep:
    """One CTCMT-MTL adaptation step.

    Parameters
    ----------
    triplet: student/teacher/anchor wrappers built by ``build_triplet(cfg)``.
    optimizer: torch optimizer over ``triplet.student.model.parameters()``.
    hp: hyperparameters parsed out of the detectron2 CfgNode.
    det_to_seg_cls: taxonomy mapping. Default: the Detectron2 Cityscapes order
        ``(11, 12, 13, 14, 15, 16, 17, 18)``. Do not change without also
        changing the source-trained model.
    """

    def __init__(
        self,
        triplet: Triplet,
        optimizer: torch.optim.Optimizer,
        hp: CTCMTHyperParams,
        det_to_seg_cls=DET_TO_SEG_CLASS_CITYSCAPES,
    ):
        self.triplet = triplet
        self.optimizer = optimizer
        self.hp = hp
        self.det_to_seg_cls = tuple(int(x) for x in det_to_seg_cls)

        student = triplet.student
        self.threshold_filter = DynamicThresholdFilter(
            num_classes=student.num_classes,
            init=hp.threshold_init,
            lo=hp.threshold_mini,
            hi=hp.threshold_max,
            alpha=hp.alpha_dt,
            gamma=hp.gamma_dt,
        )
        self.gate = ScoreEMGate(
            init=hp.score_em,
            gamma=hp.score_gamma,
            thresh=hp.score_thresh,
            disabled=hp.skip_score_em_gate,
        )
        self.ema = EMAUpdater(decay=hp.ema_decay)
        # Snapshot every weight/bias on the anchor — anchor params have
        # requires_grad=False so we can't use named_trainable_wb() here.
        self.restore = StochasticRestore(
            anchor_named_wb=self._anchor_snapshot(triplet.anchor),
            rst_prob=hp.rst_prob,
            cross_task_fisher=hp.cross_task_fisher,
            backbone_rst_factor=hp.backbone_rst_factor,
        )
        self.ctpv = (
            CTPVFilter(det_to_seg_cls=self.det_to_seg_cls, thresh=hp.ctpv_thresh)
            if hp.ctpv_enabled else None
        )
        self.ct_cl = CrossTaskContrastive(
            temperature=hp.ctcl_temperature,
            roi_output=hp.ctcl_roi_output,
            include_seg_view=hp.ctcl_include_seg_view,
            det_to_seg_cls=self.det_to_seg_cls,
        ) if hp.ctcl_enabled else None
        self.ct_cr = CrossTaskConsistency(det_to_seg_cls=self.det_to_seg_cls)
        self.soft_seg = SoftSegConsistency()
        self.iter = 0

    # ---- source snapshot: iterate over ALL weight/bias params on the anchor,
    #      not only those with requires_grad (the anchor is frozen). ------- #
    @staticmethod
    def _anchor_snapshot(anchor: Detectron2ModelAdapter):
        for nm, m in anchor.model.named_modules():
            for p_name, p in m.named_parameters(recurse=False):
                if p_name in ("weight", "bias"):
                    yield f"{nm}.{p_name}", p

    # ---- one CTTA step -------------------------------------------------- #
    def step(self, batched_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.iter += 1
        hp = self.hp
        student = self.triplet.student
        teacher = self.triplet.teacher

        # Adaptation is unsupervised. Strip any GT keys defensively so we
        # never accidentally hit a supervised code path in PanopticFPN.
        clean_inputs = [
            {k: v for k, v in d.items() if k not in ("instances", "sem_seg")}
            for d in batched_inputs
        ]

        # Student needs train() so RPN/ROI emit losses; sem_seg head in eval().
        student.student_train_mode()

        # 1. Teacher raw pseudo-labels + sem-seg logits.
        with torch.no_grad():
            t_det_raw, t_seg_raw = teacher.detect_raw(clean_inputs)

        inst = t_det_raw[0]

        # 2. Score-EMA gate: decides whether the DETECTION branch runs.
        # Matching the reference, thresholds + filtered inst are always
        # produced, and the seg / CT-CR branches keep running on gate=False.
        keep_step_det = self.gate.step(inst.scores) if len(inst) > 0 else False

        # 3. Dynamic thresholds + filter (independent of keep_step).
        if len(inst) > 0:
            self.threshold_filter.update(inst.scores, inst.pred_classes)
            keep_mask = self.threshold_filter.keep_mask(inst.scores, inst.pred_classes)
            pseudo_inst = filter_instances(inst, keep_mask)
        else:
            pseudo_inst = inst

        # 4. CTPV filter (optional).
        if self.ctpv is not None and t_seg_raw is not None and len(pseudo_inst) > 0:
            t_probs_ctpv = t_seg_raw.float().softmax(dim=1)
            pseudo_inst = self.ctpv.filter(pseudo_inst, t_probs_ctpv)

        # 5. Student forward.
        images = student.preprocess(clean_inputs)
        features = student.backbone_features(images.tensor)

        losses: Dict[str, torch.Tensor] = {}

        # Detection consistency (gated by score-EMA + non-empty pseudo set).
        if (not hp.seg_only) and keep_step_det and len(pseudo_inst) > 0:
            gt = to_gt_instances([pseudo_inst])
            det_losses = student.det_pseudo_losses(images, features, gt)
            for k, v in det_losses.items():
                losses[f"det/{k}"] = hp.weight_det * v

        # Segmentation soft-CE (runs even when det branch is gated off,
        # unless det_only). Also required to compute CT-CR.
        s_seg_logits = None
        if (not hp.det_only) and student.num_seg_classes > 0:
            s_seg_logits = student.sem_seg_logits(features)
        if (not hp.det_only) and s_seg_logits is not None and t_seg_raw is not None:
            t_probs = F.interpolate(
                t_seg_raw.float(),
                size=s_seg_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).softmax(dim=1)

            # CoTTA aug-averaging trigger: anchor confidence below threshold.
            if hp.seg_aug_enabled:
                with torch.no_grad():
                    anchor_feats = self.triplet.anchor.backbone_features(images.tensor)
                    anchor_logits = self.triplet.anchor.sem_seg_logits(anchor_feats)
                    anchor_conf = anchor_logits.float().softmax(dim=1).max(dim=1)[0].mean()
                    if float(anchor_conf.item()) < hp.seg_aug_conf_thresh:
                        aug_probs = aug_averaged_teacher_seg(
                            teacher,
                            images.tensor,
                            hp.seg_aug_scales,
                            hp.seg_aug_flips,
                        )
                        t_probs = F.interpolate(
                            aug_probs,
                            size=s_seg_logits.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )

            loss_seg = self.soft_seg(s_seg_logits, t_probs)
            losses["seg/soft_ce"] = hp.weight_seg * loss_seg

        # CT-CL requires the det gate (only meaningful when det branch runs).
        if (self.ct_cl is not None and not hp.det_only and not hp.seg_only
                and keep_step_det and len(pseudo_inst) > 0 and t_seg_raw is not None):
            feat_key = student.roi_box_in_features[-1]
            feat = features[feat_key]
            stride = student.fpn_stride_of(feat_key)
            boxes_feat = pseudo_inst.pred_boxes.tensor / float(stride)
            t_probs_feat = F.interpolate(
                t_seg_raw.float(), size=feat.shape[-2:],
                mode="bilinear", align_corners=False,
            ).softmax(dim=1)
            loss_ctcl = self.ct_cl(
                feat=feat,
                boxes_feat_coords=boxes_feat,
                classes=pseudo_inst.pred_classes.long(),
                teacher_sem_probs_feat=t_probs_feat,
                num_seg_classes=student.num_seg_classes,
            )
            losses["ctcl"] = hp.weight_ctcl * loss_ctcl

        # CT-CR runs whenever there's a pseudo set and student seg is on
        # (matches reference: gated by det_gate OR seg_gate; seg_gate is
        # True by default in the default non-per-task config).
        if (hp.weight_ctcr > 0 and not hp.det_only and not hp.seg_only
                and len(pseudo_inst) > 0 and s_seg_logits is not None):
            loss_ctcr = self.ct_cr(
                student_seg_logits=s_seg_logits,
                boxes_img_coords=pseudo_inst.pred_boxes.tensor,
                classes=pseudo_inst.pred_classes.long(),
                image_hw=pseudo_inst.image_size,
            )
            if loss_ctcr is not None:
                losses["ctcr"] = hp.weight_ctcr * loss_ctcr

        # 7. Backward + step.
        if losses:
            total = sum(losses.values())
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            self.optimizer.step()

        # 8. EMA + restore (unconditional, matching the reference).
        self.ema.update(teacher.model, student.model)
        self.restore.apply(student.named_trainable_wb())

        # 9. Report teacher predictions for the evaluator.
        with torch.no_grad():
            t_det_final, t_seg_final = teacher.detect_raw(clean_inputs)
        return self._build_predictions(clean_inputs, t_det_final, t_seg_final)

    # ---- postprocess teacher outputs into the evaluator schema ---------- #
    def _build_predictions(self, batched_inputs, det_raw, sem_raw):
        from detectron2.modeling.postprocessing import detector_postprocess, sem_seg_postprocess

        images = self.triplet.teacher.preprocess(batched_inputs)
        processed = []
        for i, (inp, image_size) in enumerate(zip(batched_inputs, images.image_sizes)):
            H = inp.get("height", image_size[0])
            W = inp.get("width", image_size[1])
            det_r = detector_postprocess(det_raw[i], H, W)
            out_i: Dict[str, Any] = {"instances": det_r}
            if sem_raw is not None:
                out_i["sem_seg"] = sem_seg_postprocess(sem_raw[i], image_size, H, W)
            processed.append(out_i)
        return processed
