"""Adaptation integration test.

Verifies, on one real synthetic Detectron2 batch:
  1. The student parameters actually change after one adapt step
     (i.e. gradients flowed and the optimizer stepped).
  2. The teacher parameters shift toward the student (EMA fired).
  3. The anchor parameters do NOT change.
  4. Adapt step is truly unsupervised: it drops ``instances`` and
     ``sem_seg`` from ``batched_inputs`` even if a caller passes them.

Skips cleanly when detectron2 or the source checkpoint is unavailable.
"""
from __future__ import annotations

from copy import deepcopy
import os
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch


DEFAULT_CONFIG = "/home/ilias/CTCMT/detectron2/configs/Cityscapes/ctcmt_mtl_panoptic_fpn_R_50_ACDC.yaml"
DEFAULT_WEIGHTS = "/workspace/output/panoptic_fpn_R50_cityscapes/model_final.pth"


def _resolve():
    try:
        import detectron2  # noqa: F401
    except Exception as e:
        pytest.skip(f"detectron2 not importable: {e}")
    cfg_path = os.environ.get("CTCMT_CONFIG", DEFAULT_CONFIG)
    weights = os.environ.get("CTCMT_WEIGHTS", DEFAULT_WEIGHTS)
    if not os.path.isfile(cfg_path):
        pytest.skip(f"config not available: {cfg_path}")
    if not os.path.isfile(weights):
        pytest.skip(f"checkpoint not available: {weights}")
    return cfg_path, weights


def _synthetic_batch(cfg, include_gt: bool = False):
    """Realistic detectron2 batched_inputs entry."""
    from detectron2.structures import Boxes, Instances

    H, W = int(cfg.INPUT.MIN_SIZE_TEST), int(min(cfg.INPUT.MAX_SIZE_TEST, 2 * cfg.INPUT.MIN_SIZE_TEST))
    torch.manual_seed(999)
    img = (torch.rand(3, H, W) * 255).clamp(0, 255)
    d = {"image": img, "height": H, "width": W}
    if include_gt:
        gt = Instances((H, W))
        gt.gt_boxes = Boxes(torch.tensor([[10.0, 10.0, 100.0, 100.0]]))
        gt.gt_classes = torch.tensor([0])
        d["instances"] = gt
        d["sem_seg"] = torch.full((H, W), 255, dtype=torch.long)
    return [d]


def _snapshot(model):
    # Keep full states (including buffers) on CPU between the two GT runs.
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _trainable_snapshot(model):
    return {k: p.detach().cpu().clone() for k, p in model.named_parameters()
            if p.requires_grad}


def _assert_finite(state, label):
    assert state, f"{label}: empty state"
    for name, value in state.items():
        assert torch.isfinite(value).all(), f"{label}: non-finite values in {name}"


def _assert_states_close(a, b, *, label, atol, rtol):
    assert a.keys() == b.keys(), f"{label}: state keys differ"
    _assert_finite(a, f"{label} first")
    _assert_finite(b, f"{label} second")
    for name, value in a.items():
        other = b[name]
        assert value.shape == other.shape, f"{label}: shape mismatch in {name}"
        assert value.dtype == other.dtype, f"{label}: dtype mismatch in {name}"
        if value.dtype.is_floating_point:
            assert torch.allclose(value, other, atol=atol, rtol=rtol, equal_nan=False), (
                f"{label}: {name} differs (atol={atol}, rtol={rtol})"
            )
        else:
            assert torch.equal(value, other), f"{label}: buffer {name} differs"


def _fraction_bytewise_changed(before, after):
    """Byte-exact 'did any value change' — catches sub-atol moves that
    ``torch.allclose``'s rtol swallows."""
    total = 0
    changed = 0
    for k, v0 in before.items():
        v1 = after[k].detach().to(v0.device)
        total += 1
        if not torch.equal(v0, v1):
            changed += 1
    return changed / max(total, 1)


def _step_with_optimizer_check(step, batch):
    """Observe the real optimizer call, excluding forward buffers and restore."""
    model = step.triplet.student.model
    real_optimizer_step = step.optimizer.step

    def checked_optimizer_step(*args, **kwargs):
        before = _trainable_snapshot(model)
        _assert_finite(before, "student before optimizer.step")
        result = real_optimizer_step(*args, **kwargs)
        after = _trainable_snapshot(model)
        _assert_finite(after, "student after optimizer.step")
        assert before.keys() == after.keys(), "trainable student parameters changed"
        assert _fraction_bytewise_changed(before, after) > 0.0, (
            "optimizer.step did not change any trainable student parameter"
        )
        return result

    with patch.object(step.optimizer, "step", side_effect=checked_optimizer_step) as optimizer_step:
        outputs = step.step(batch)
    optimizer_step.assert_called_once_with()
    return outputs


def test_one_adapt_step_moves_student_and_teacher_but_not_anchor():
    cfg_path, weights = _resolve()

    from ctcmt.ctta import CTCMTAdaptStep, CTCMTHyperParams
    from ctcmt.d2 import build_triplet, setup_cfg

    cfg = setup_cfg(cfg_path, weights=weights, freeze=False)
    # Isolate optimizer updates from stochastic restoration.
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 1e-3
    cfg.SOLVER.RST_M = 0.0
    cfg.SOLVER.MT = 0.99  # strong EMA drift so first step is clearly visible
    cfg.SOLVER.CTCMT_SKIP_SCORE_EM_GATE = True  # force the step to actually run
    cfg.freeze()

    triplet = build_triplet(cfg)
    from detectron2.solver import build_optimizer
    opt = build_optimizer(cfg, triplet.student.model)

    hp = CTCMTHyperParams.from_cfg(cfg)
    step = CTCMTAdaptStep(triplet, opt, hp)

    t_before = _snapshot(triplet.teacher.model)
    a_before = _snapshot(triplet.anchor.model)

    device = torch.device(cfg.MODEL.DEVICE)
    batch = _synthetic_batch(cfg, include_gt=False)
    for d in batch:
        d["image"] = d["image"].to(device)

    outputs = _step_with_optimizer_check(step, batch)

    s_after = _trainable_snapshot(triplet.student.model)
    t_after = _snapshot(triplet.teacher.model)
    a_after = _snapshot(triplet.anchor.model)

    _assert_finite(s_after, "student after adaptation")
    _assert_finite(t_before, "teacher before adaptation")
    _assert_finite(t_after, "teacher after adaptation")
    _assert_states_close(a_before, a_after, label="anchor", atol=0.0, rtol=0.0)
    assert _fraction_bytewise_changed(t_before, t_after) > 0.0, (
        "teacher did not change: EMA never fired"
    )

    # The step must return the evaluator schema.
    assert isinstance(outputs, list) and outputs and "instances" in outputs[0]


def _seed_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_one_seeded_step(cfg, batch):
    """Build a fresh adapter/optimizer and require a real, finite update."""
    from ctcmt.ctta import CTCMTAdaptStep, CTCMTHyperParams
    from ctcmt.d2 import build_triplet
    from detectron2.solver import build_optimizer

    seed = 4242
    _seed_rng(seed)
    triplet = build_triplet(cfg)
    opt = build_optimizer(cfg, triplet.student.model)
    hp = CTCMTHyperParams.from_cfg(cfg)
    step = CTCMTAdaptStep(triplet, opt, hp)
    initial = {name: _snapshot(getattr(triplet, name).model)
               for name in ("student", "teacher", "anchor")}
    for name, state in initial.items():
        _assert_finite(state, f"initial {name}")

    # Model construction and proposal sampling must consume identical RNG.
    _seed_rng(seed + 1)
    _step_with_optimizer_check(step, batch)
    final = {name: _snapshot(getattr(triplet, name).model)
             for name in ("student", "teacher")}
    for name, state in final.items():
        _assert_finite(state, f"adapted {name}")
    return initial, final


@pytest.mark.parametrize("gt_variant", ["different-labels", "no-gt"])
def test_adaptation_ignores_target_ground_truth(gt_variant):
    """Target GT presence and label values must not affect either adapted model."""
    cfg_path, weights = _resolve()

    from ctcmt.d2 import setup_cfg

    cfg = setup_cfg(cfg_path, weights=weights, freeze=False)
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 1e-3
    cfg.SOLVER.RST_M = 0.0
    cfg.SOLVER.MT = 0.9
    cfg.SOLVER.CTCMT_SKIP_SCORE_EM_GATE = True
    cfg.freeze()

    batch_a = _synthetic_batch(cfg, include_gt=True)
    batch_a[0]["image"] = batch_a[0]["image"].to(torch.device(cfg.MODEL.DEVICE))
    batch_a[0]["sem_seg"].fill_(11)  # Cityscapes person; gt_classes is 0.
    batch_b = deepcopy(batch_a)
    if gt_variant == "no-gt":
        del batch_b[0]["instances"]
        del batch_b[0]["sem_seg"]
    else:
        batch_b[0]["instances"].gt_classes.fill_(1)  # Cityscapes rider.
        batch_b[0]["sem_seg"].fill_(12)
        assert not torch.equal(batch_a[0]["instances"].gt_classes,
                               batch_b[0]["instances"].gt_classes)
        assert not torch.equal(batch_a[0]["sem_seg"], batch_b[0]["sem_seg"])
    assert torch.equal(batch_a[0]["image"], batch_b[0]["image"])

    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=range(torch.cuda.device_count())), \
                torch.backends.cudnn.flags(deterministic=True, benchmark=False):
            initial_a, final_a = _run_one_seeded_step(cfg, batch_a)
            initial_b, final_b = _run_one_seeded_step(cfg, batch_b)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)

    # Defensive stripping must also leave the caller's dictionaries intact.
    for key in ("instances", "sem_seg"):
        assert key in batch_a[0]
        assert (key in batch_b[0]) == (gt_variant != "no-gt")

    # Verify identical starting states rather than relying on the seed alone.
    for name in ("student", "teacher", "anchor"):
        _assert_states_close(initial_a[name], initial_b[name],
                             label=f"initial {name}", atol=0.0, rtol=0.0)
    for name in ("student", "teacher"):
        _assert_states_close(final_a[name], final_b[name],
                             label=f"GT independence ({name})", atol=1e-6, rtol=1e-5)


def test_adapt_step_rejects_multi_image_batch():
    """B=1 is a hard requirement; the step must fail loudly, not silently
    mis-adapt (only image 0 pseudo-labels would be honored)."""
    cfg_path, weights = _resolve()

    from ctcmt.ctta import CTCMTAdaptStep, CTCMTHyperParams
    from ctcmt.d2 import build_triplet, setup_cfg

    cfg = setup_cfg(cfg_path, weights=weights, freeze=False)
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 1e-4
    cfg.freeze()

    triplet = build_triplet(cfg)
    from detectron2.solver import build_optimizer
    opt = build_optimizer(cfg, triplet.student.model)
    hp = CTCMTHyperParams.from_cfg(cfg)
    step = CTCMTAdaptStep(triplet, opt, hp)

    device = torch.device(cfg.MODEL.DEVICE)
    b1 = _synthetic_batch(cfg, include_gt=False)[0]
    b2 = _synthetic_batch(cfg, include_gt=False)[0]
    for d in (b1, b2):
        d["image"] = d["image"].to(device)
    with pytest.raises(ValueError, match="batch size 1"):
        step.step([b1, b2])


def test_hyperparams_rejects_unsupported_ablation_flags():
    """The MTL adapter does not implement V1/V4/E2/E3/E4/E5 branches.
    Enabling any of them in the CfgNode must raise ``NotImplementedError``
    with a clear message, instead of silently running the default variant.
    """
    cfg_path, weights = _resolve()

    from ctcmt.ctta import CTCMTHyperParams
    from ctcmt.d2 import setup_cfg

    unsupported = [
        "CTCMT_PER_TASK_GATE",
        "CTCMT_PROTO_ANCHOR",
        "CTCMT_ENTROPY_WEIGHTED_CE",
        "CTCMT_AUG_TRIGGER_TEACHER_ENTROPY",
        "CTCMT_DIRECTIONAL_GATE",
        "CTCMT_ADAPTIVE_STR",
    ]
    for flag in unsupported:
        cfg = setup_cfg(cfg_path, weights=weights, freeze=False)
        setattr(cfg.SOLVER, flag, True)
        cfg.freeze()
        with pytest.raises(NotImplementedError, match=flag):
            CTCMTHyperParams.from_cfg(cfg)
