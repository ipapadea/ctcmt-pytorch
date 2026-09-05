"""Framework-agnostic tests for CTTA math (no detectron2 required)."""
from __future__ import annotations

import math

import pytest
import torch

from ctcmt.ctta import (
    DynamicThresholdFilter,
    EMAUpdater,
    ScoreEMGate,
    SoftSegConsistency,
    StochasticRestore,
    per_class_mean_scores,
    supcon_loss,
    update_dynamic_thresholds,
)
from ctcmt.ctta.losses import (
    CrossTaskConsistency,
    CrossTaskContrastive,
    DET_TO_SEG_CLASS_CITYSCAPES,
)


# ---- Dynamic thresholds ------------------------------------------------- #
def test_update_dynamic_thresholds_matches_reference():
    prev = [0.8] * 8
    means = [0.9, 0.0, 0.5, 0.0, 0.7, 0.6, 0.4, 0.85]
    alpha, gamma, lo, hi = 1.3, 0.95, 0.7, 0.9
    got = update_dynamic_thresholds(prev, means, alpha, gamma, lo, hi)
    # Reference formula: gamma*th + (1-gamma)*alpha*sqrt(mean), clipped.
    exp = []
    for th, m in zip(prev, means):
        v = gamma * th + (1 - gamma) * alpha * math.sqrt(m) if m > 0 else th
        exp.append(max(min(v, hi), lo))
    assert got == exp


def test_dynamic_threshold_filter_keep_mask_uses_per_class_thresh():
    f = DynamicThresholdFilter(num_classes=3, init=0.5, lo=0.1, hi=0.9, alpha=1.0, gamma=0.9)
    f.thresholds = [0.4, 0.6, 0.8]
    scores = torch.tensor([0.5, 0.55, 0.85, 0.79])
    cls = torch.tensor([0, 1, 2, 2])
    keep = f.keep_mask(scores, cls)
    # cls=0 uses 0.4 -> 0.5 kept; cls=1 uses 0.6 -> 0.55 dropped
    # cls=2 uses 0.8 -> 0.85 kept, 0.79 dropped
    assert keep.tolist() == [True, False, True, False]


def test_per_class_mean_ignores_scores_below_floor():
    scores = torch.tensor([0.05, 0.5, 0.9, 0.15])
    cls = torch.tensor([0, 0, 1, 1])
    means = per_class_mean_scores(scores, cls, num_classes=2, score_floor=0.1)
    assert means[0] == pytest.approx(0.5)
    assert means[1] == pytest.approx((0.9 + 0.15) / 2)


# ---- Score-EMA gate ----------------------------------------------------- #
def test_score_em_gate_updates_ema_and_skips_when_stable():
    gate = ScoreEMGate(init=0.5, gamma=0.7, thresh=1.4)
    # Very high mean -> ratio > thresh -> skip, but EMA still updated.
    keep = gate.step(torch.tensor([0.9, 0.9, 0.9]))
    assert keep is False
    assert 0.5 < gate.score_em < 0.9

    gate2 = ScoreEMGate(init=0.5, gamma=0.7, thresh=1.4)
    # Mean 0.6: ratio 1.2 within [1/1.4, 1.4] -> accept.
    keep2 = gate2.step(torch.tensor([0.6, 0.55, 0.65]))
    assert keep2 is True


def test_score_em_gate_empty_input_returns_false_without_updating():
    gate = ScoreEMGate(init=0.5, gamma=0.7, thresh=1.4)
    assert gate.step(torch.empty(0)) is False
    assert gate.score_em == pytest.approx(0.5)


# ---- EMA updater -------------------------------------------------------- #
def test_ema_moves_teacher_toward_student():
    torch.manual_seed(0)
    student = torch.nn.Linear(4, 4)
    teacher = torch.nn.Linear(4, 4)
    ema = EMAUpdater(decay=0.9)
    t0 = teacher.weight.detach().clone()
    s0 = student.weight.detach().clone()
    ema.update(teacher, student)
    expected = 0.9 * t0 + 0.1 * s0
    assert torch.allclose(teacher.weight.detach(), expected, atol=1e-6)


# ---- Stochastic restore ------------------------------------------------- #
def test_stochastic_restore_zero_prob_is_noop():
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 4)
    anchor_wb = list(_named_wb(m))
    student = torch.nn.Linear(4, 4)
    with torch.no_grad():
        for (n, p_a), (_, p_s) in zip(anchor_wb, _named_wb(student)):
            p_s.copy_(p_a + 1.0)
    r = StochasticRestore(anchor_named_wb=anchor_wb, rst_prob=0.0)
    before = {n: p.detach().clone() for n, p in _named_wb(student)}
    r.apply(_named_wb(student))
    for n, p in _named_wb(student):
        assert torch.equal(p, before[n])


def test_stochastic_restore_probability_pushes_toward_source():
    torch.manual_seed(0)
    m_anchor = torch.nn.Linear(64, 64)
    student = torch.nn.Linear(64, 64)
    with torch.no_grad():
        for (n_a, p_a), (n_s, p_s) in zip(_named_wb(m_anchor), _named_wb(student)):
            p_s.copy_(p_a + 10.0)
    r = StochasticRestore(anchor_named_wb=_named_wb(m_anchor), rst_prob=0.5)
    r.apply(_named_wb(student))
    # After restore with p=0.5, roughly half the entries should match anchor.
    total, matched = 0, 0
    for (_, p_a), (_, p_s) in zip(_named_wb(m_anchor), _named_wb(student)):
        total += p_a.numel()
        matched += int(torch.isclose(p_a, p_s, atol=1e-6).sum().item())
    frac = matched / total
    assert 0.4 < frac < 0.6, f"expected ~0.5 restore rate, got {frac:.3f}"


def _named_wb(module):
    for nm, m in module.named_modules():
        for p_name, p in m.named_parameters(recurse=False):
            if p_name in ("weight", "bias"):
                yield f"{nm}.{p_name}", p


# ---- SupCon ------------------------------------------------------------- #
def test_supcon_zero_when_all_same_class():
    torch.manual_seed(0)
    feats = torch.randn(8, 32)
    feats = torch.nn.functional.normalize(feats, dim=1)
    labels = torch.zeros(8, dtype=torch.long)
    # All positives; loss can be nonzero but must be finite and >= 0.
    loss = supcon_loss(feats, labels, temperature=0.07)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_supcon_singleton_returns_zero():
    feats = torch.nn.functional.normalize(torch.randn(1, 8), dim=1)
    labels = torch.zeros(1, dtype=torch.long)
    assert supcon_loss(feats, labels).item() == 0.0


# ---- SoftSegConsistency ------------------------------------------------- #
def test_soft_seg_matches_ce_at_matching_shapes():
    torch.manual_seed(0)
    student = torch.randn(1, 5, 8, 8)
    teacher_probs = torch.softmax(torch.randn(1, 5, 8, 8), dim=1)
    loss = SoftSegConsistency()(student, teacher_probs)
    assert torch.isfinite(loss)


# ---- CT-CR -------------------------------------------------------------- #
def test_ct_cr_returns_none_when_no_boxes():
    logits = torch.randn(1, 20, 16, 16)
    boxes = torch.empty(0, 4)
    classes = torch.empty(0, dtype=torch.long)
    assert CrossTaskConsistency()(logits, boxes, classes, image_hw=(64, 64)) is None


def test_ct_cr_places_target_at_correct_seg_class():
    # 20 seg classes -> DET_TO_SEG_CLASS_CITYSCAPES all fit.
    logits = torch.randn(1, 20, 16, 16)
    boxes = torch.tensor([[0.0, 0.0, 64.0, 64.0]])
    classes = torch.tensor([2])  # car -> seg 13
    loss = CrossTaskConsistency()(logits, boxes, classes, image_hw=(64, 64))
    assert loss is not None and torch.isfinite(loss)


# ---- CT-CL -------------------------------------------------------------- #
def test_ctcl_runs_without_seg_view():
    torch.manual_seed(0)
    feat = torch.randn(1, 32, 16, 16)
    boxes = torch.tensor([[0.0, 0.0, 5.0, 5.0], [5.0, 5.0, 10.0, 10.0]])
    classes = torch.tensor([0, 1])
    ct = CrossTaskContrastive(
        temperature=0.07,
        roi_output=(7, 7),
        include_seg_view=False,
        det_to_seg_cls=DET_TO_SEG_CLASS_CITYSCAPES,
    )
    loss = ct(feat, boxes, classes, teacher_sem_probs_feat=None, num_seg_classes=19)
    assert torch.isfinite(loss)


def test_ctcl_runs_with_seg_view():
    torch.manual_seed(0)
    feat = torch.randn(1, 32, 16, 16)
    boxes = torch.tensor([[0.0, 0.0, 5.0, 5.0], [5.0, 5.0, 10.0, 10.0]])
    classes = torch.tensor([0, 2])  # person, car -> seg 11, 13
    probs = torch.softmax(torch.randn(1, 19, 16, 16), dim=1)
    ct = CrossTaskContrastive(
        temperature=0.07,
        roi_output=(7, 7),
        include_seg_view=True,
        det_to_seg_cls=DET_TO_SEG_CLASS_CITYSCAPES,
    )
    loss = ct(feat, boxes, classes, teacher_sem_probs_feat=probs, num_seg_classes=19)
    assert torch.isfinite(loss)


def test_det_to_seg_mapping_is_detectron2_cityscapes_order():
    # Guard: user-confirmed correction. Do NOT reorder.
    assert DET_TO_SEG_CLASS_CITYSCAPES == (11, 12, 13, 14, 15, 16, 17, 18)
