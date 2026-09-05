"""Source equivalence: the ``Detectron2ModelAdapter`` wrapper must produce
predictions identical to a raw ``PanopticFPN`` loaded from the same weights.

This is the first thing that must pass before any CTTA code is trusted. If
this test fails, the wrapper is silently changing behavior and every
downstream comparison against the reference is meaningless.

The test needs:
  - ``detectron2`` installed on PYTHONPATH,
  - a real source-trained checkpoint (defaults to the ACDC config's
    ``MODEL.WEIGHTS``; override with env var ``CTCMT_WEIGHTS``),
  - a real config file (override with ``CTCMT_CONFIG``).

If either dependency is missing, the test skips cleanly.
"""
from __future__ import annotations

import os

import pytest
import torch


DEFAULT_CONFIG = "/home/ilias/CTCMT/detectron2/configs/Cityscapes/ctcmt_mtl_panoptic_fpn_R_50_ACDC.yaml"
DEFAULT_WEIGHTS = "/workspace/output/panoptic_fpn_R50_cityscapes/model_final.pth"


def _resolve_config():
    p = os.environ.get("CTCMT_CONFIG", DEFAULT_CONFIG)
    if not os.path.isfile(p):
        pytest.skip(f"config not available: {p}")
    return p


def _resolve_weights():
    p = os.environ.get("CTCMT_WEIGHTS", DEFAULT_WEIGHTS)
    if not os.path.isfile(p):
        pytest.skip(f"source checkpoint not available: {p}")
    return p


def _require_detectron2():
    try:
        import detectron2  # noqa: F401
    except Exception as e:
        pytest.skip(f"detectron2 not importable: {e}")


def _synthetic_batch(cfg):
    """Build a valid detectron2 batched_inputs entry from a random tensor.

    Uses the config's INPUT.MIN_SIZE_TEST as the short side, aspect 2:1
    (Cityscapes/ACDC). PanopticFPN.preprocess_image handles normalization.
    """
    min_size = int(cfg.INPUT.MIN_SIZE_TEST)
    max_size = int(cfg.INPUT.MAX_SIZE_TEST)
    H, W = min_size, min(max_size, 2 * min_size)
    torch.manual_seed(1234)
    # Uint8 image in (C, H, W) — detectron2 mappers hand these back.
    img = (torch.rand(3, H, W) * 255).clamp(0, 255)
    return [{"image": img, "height": H, "width": W}]


def _assert_instances_equal(a, b, rtol=0.0, atol=1e-4):
    assert len(a) == len(b), f"instance count differs: {len(a)} vs {len(b)}"
    if len(a) == 0:
        return
    ta = a.pred_boxes.tensor
    tb = b.pred_boxes.tensor
    assert torch.allclose(ta, tb, rtol=rtol, atol=atol), "pred_boxes mismatch"
    assert torch.equal(a.pred_classes, b.pred_classes), "pred_classes mismatch"
    assert torch.allclose(a.scores, b.scores, rtol=rtol, atol=atol), "scores mismatch"


def test_wrapper_matches_raw_panoptic_fpn():
    _require_detectron2()
    cfg_path = _resolve_config()
    weights = _resolve_weights()

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import META_ARCH_REGISTRY

    from ctcmt.d2 import Detectron2ModelAdapter, setup_cfg

    cfg = setup_cfg(cfg_path, weights=weights, freeze=True)

    # 1. Raw PanopticFPN loaded exactly like the reference CTCMT_MTL does.
    arch = getattr(cfg.MODEL, "CTCMT_STUDENT_META_ARCH", "PanopticFPN") or cfg.MODEL.META_ARCHITECTURE
    if arch == "CTCMT_MTL":
        arch = "PanopticFPN"
    raw = META_ARCH_REGISTRY.get(arch)(cfg)
    DetectionCheckpointer(raw).load(cfg.MODEL.WEIGHTS)
    raw.to(torch.device(cfg.MODEL.DEVICE)).eval()

    # 2. Same, wrapped.
    from ctcmt.d2.model_adapter import build_source_only
    wrapped = build_source_only(cfg)

    # 3. Same input.
    batch = _synthetic_batch(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    for d in batch:
        d["image"] = d["image"].to(device)

    with torch.no_grad():
        out_raw = raw.inference(batch, do_postprocess=True)
        out_wrapped = wrapped.detect(batch)

    # 4. Compare.
    assert len(out_raw) == len(out_wrapped) == 1
    _assert_instances_equal(out_raw[0]["instances"], out_wrapped[0]["instances"])
    assert "sem_seg" in out_raw[0] and "sem_seg" in out_wrapped[0], "missing sem_seg"
    a = out_raw[0]["sem_seg"]
    b = out_wrapped[0]["sem_seg"]
    assert a.shape == b.shape, f"sem_seg shape mismatch: {a.shape} vs {b.shape}"
    assert torch.allclose(a, b, atol=1e-5), "sem_seg values differ"


def test_wrapper_backbone_and_seg_head_match_raw():
    """Second, finer-grained check: same features from same tensor."""
    _require_detectron2()
    cfg_path = _resolve_config()
    weights = _resolve_weights()

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import META_ARCH_REGISTRY

    from ctcmt.d2 import Detectron2ModelAdapter, setup_cfg

    cfg = setup_cfg(cfg_path, weights=weights, freeze=True)
    arch = getattr(cfg.MODEL, "CTCMT_STUDENT_META_ARCH", "PanopticFPN") or cfg.MODEL.META_ARCHITECTURE
    if arch == "CTCMT_MTL":
        arch = "PanopticFPN"
    raw = META_ARCH_REGISTRY.get(arch)(cfg)
    DetectionCheckpointer(raw).load(cfg.MODEL.WEIGHTS)
    raw.to(torch.device(cfg.MODEL.DEVICE)).eval()

    from ctcmt.d2.model_adapter import build_source_only
    wrapped = build_source_only(cfg)

    batch = _synthetic_batch(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    for d in batch:
        d["image"] = d["image"].to(device)

    with torch.no_grad():
        img_raw = raw.preprocess_image(batch)
        img_wrp = wrapped.preprocess(batch)
        assert torch.equal(img_raw.tensor, img_wrp.tensor)

        f_raw = raw.backbone(img_raw.tensor)
        f_wrp = wrapped.backbone_features(img_wrp.tensor)
        assert set(f_raw.keys()) == set(f_wrp.keys())
        for k in f_raw:
            assert torch.allclose(f_raw[k], f_wrp[k], atol=1e-5), f"feat {k} differs"

        if hasattr(raw, "sem_seg_head"):
            l_raw, _ = raw.sem_seg_head(f_raw, None)
            l_wrp = wrapped.sem_seg_logits(f_wrp)
            assert torch.allclose(l_raw, l_wrp, atol=1e-5)
