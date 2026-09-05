"""Seg source-equivalence + one-step-adapt tests for CT-CMT-Seg.

Same pattern as ``test_source_equivalence.py`` / ``test_adapt_step.py`` but
targeting the ``SemanticSegmentor`` source used by CT-CMT-Seg.

Skips cleanly when detectron2 or the semantic-FPN checkpoint is not on disk.
Override the paths via ``CTCMT_SEG_CONFIG`` / ``CTCMT_SEG_WEIGHTS``.
"""
from __future__ import annotations

import os

import pytest
import torch


DEFAULT_CONFIG = "/workspace/CTCMT/detectron2/configs/Cityscapes/ctcmt_seg_semfpn_R_50_ACDC.yaml"
DEFAULT_WEIGHTS = "/workspace/panoptic_fpn/output/semantic_R50_cityscapes/model_final.pth"


def _resolve():
    try:
        import detectron2  # noqa: F401
    except Exception as e:
        pytest.skip(f"detectron2 not importable: {e}")
    cfg_path = os.environ.get("CTCMT_SEG_CONFIG", DEFAULT_CONFIG)
    weights = os.environ.get("CTCMT_SEG_WEIGHTS", DEFAULT_WEIGHTS)
    if not os.path.isfile(cfg_path):
        pytest.skip(f"config not available: {cfg_path}")
    if not os.path.isfile(weights):
        pytest.skip(f"checkpoint not available: {weights}")
    return cfg_path, weights


def _synthetic_batch(cfg):
    H = int(cfg.INPUT.MIN_SIZE_TEST)
    W = min(int(cfg.INPUT.MAX_SIZE_TEST), 2 * H)
    torch.manual_seed(1234)
    img = (torch.rand(3, H, W) * 255).clamp(0, 255)
    return [{"image": img, "height": H, "width": W}]


def test_seg_wrapper_matches_raw_semantic_segmentor():
    cfg_path, weights = _resolve()

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import META_ARCH_REGISTRY

    from ctcmt.d2 import build_seg_triplet, setup_cfg

    cfg = setup_cfg(cfg_path, weights=weights, freeze=True)

    raw = META_ARCH_REGISTRY.get("SemanticSegmentor")(cfg)
    DetectionCheckpointer(raw).load(cfg.MODEL.WEIGHTS)
    raw.to(torch.device(cfg.MODEL.DEVICE)).eval()

    triplet = build_seg_triplet(cfg)
    wrapped = triplet.teacher

    batch = _synthetic_batch(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    for d in batch:
        d["image"] = d["image"].to(device)

    with torch.no_grad():
        out_raw = raw(batch)
        out_wrp = wrapped.sem_seg_predict(batch)

    assert len(out_raw) == len(out_wrp) == 1
    a = out_raw[0]["sem_seg"]
    b = out_wrp[0]["sem_seg"]
    assert a.shape == b.shape
    assert torch.allclose(a, b, atol=1e-5), "sem_seg logits diverge"


def _snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if v.dtype.is_floating_point}


def _fraction_changed(before, after, atol=1e-7, rtol=1e-5):
    total = 0
    changed = 0
    for k, v0 in before.items():
        v1 = after[k]
        total += 1
        if not torch.allclose(v0, v1.detach().to(v0.device), atol=atol, rtol=rtol):
            changed += 1
    return changed / max(total, 1)


def _fraction_bytewise_changed(before, after):
    """Byte-exact 'did any value change' test — catches EMA moves that
    ``torch.allclose`` swallows via ``rtol``."""
    total = 0
    changed = 0
    for k, v0 in before.items():
        v1 = after[k].detach().to(v0.device)
        total += 1
        if not torch.equal(v0, v1):
            changed += 1
    return changed / max(total, 1)


def test_seg_adapt_step_moves_student_and_teacher_but_not_anchor():
    cfg_path, weights = _resolve()

    from ctcmt.ctta import CTCMTSegAdaptStep, CTCMTSegHyperParams
    from ctcmt.d2 import build_seg_triplet, setup_cfg

    cfg = setup_cfg(cfg_path, weights=weights, freeze=False)
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 1e-2
    cfg.SOLVER.COTTA_RESTORE_PROB = 0.05
    cfg.SOLVER.COTTA_EMA_DECAY = 0.5  # strong EMA drift so 1-step deltas clear atol
    cfg.freeze()

    triplet = build_seg_triplet(cfg)
    from detectron2.solver import build_optimizer
    opt = build_optimizer(cfg, triplet.student.model)
    hp = CTCMTSegHyperParams.from_cfg(cfg)
    step = CTCMTSegAdaptStep(triplet, opt, hp)

    s0 = _snapshot(triplet.student.model)
    t0 = _snapshot(triplet.teacher.model)
    a0 = _snapshot(triplet.anchor.model)

    device = torch.device(cfg.MODEL.DEVICE)
    batch = _synthetic_batch(cfg)
    for d in batch:
        d["image"] = d["image"].to(device)

    outputs = step.step(batch)

    s1 = _snapshot(triplet.student.model)
    t1 = _snapshot(triplet.teacher.model)
    a1 = _snapshot(triplet.anchor.model)

    assert _fraction_changed(a0, a1, atol=1e-9) == 0.0, "anchor must be frozen"
    assert _fraction_bytewise_changed(s0, s1) > 0.0, "student did not move"
    assert _fraction_bytewise_changed(t0, t1) > 0.0, "teacher EMA never fired"
    assert isinstance(outputs, list) and outputs and "sem_seg" in outputs[0]


def test_seg_adapt_step_ignores_gt_keys():
    cfg_path, weights = _resolve()

    from ctcmt.ctta import CTCMTSegAdaptStep, CTCMTSegHyperParams
    from ctcmt.d2 import build_seg_triplet, setup_cfg

    cfg = setup_cfg(cfg_path, weights=weights, freeze=False)
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 1e-4
    cfg.freeze()

    triplet = build_seg_triplet(cfg)
    from detectron2.solver import build_optimizer
    opt = build_optimizer(cfg, triplet.student.model)
    hp = CTCMTSegHyperParams.from_cfg(cfg)
    step = CTCMTSegAdaptStep(triplet, opt, hp)

    device = torch.device(cfg.MODEL.DEVICE)
    H = int(cfg.INPUT.MIN_SIZE_TEST)
    W = min(int(cfg.INPUT.MAX_SIZE_TEST), 2 * H)
    batch = [{
        "image": (torch.rand(3, H, W) * 255).to(device),
        "height": H,
        "width": W,
        "sem_seg": torch.full((H, W), 255, dtype=torch.long),
    }]
    _ = step.step(batch)
    assert "sem_seg" in batch[0], "step must not mutate caller's dict"
