#!/usr/bin/env python3
"""Diagnose reference vs clean MTL on a short, ordered real-image sequence.

Use --comparison clean-clean --rng-mode continuous to diagnose repeatability.
Both passes run sequentially in one process using sorted files, not the production
dataloader. This does not test cross-process ordering or worker RNG.

Runs each implementation independently, with matching construction/per-image
seeds, and compares initial states, gates, thresholds, pseudo-labels, weighted
losses, teacher posteriors, model states, and net trainable parameter updates.
No adaptation source is rewritten. Restoration defaults to zero to isolate
optimizer/EMA behavior; use --restore-prob for a subsequent restoration check.
This is a diagnostic sequence, not a benchmark or proof of general equivalence.

Exit codes: 0 = compared values match and active loss branches were exercised;
1 = mismatch; 2 = setup/runtime failure or insufficient branch coverage.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import pickle
import os
from pathlib import Path
import random
import sys
import tempfile
from unittest.mock import patch


REFERENCE_SHA256 = "3502bcf0fdae4cdc2027d875777939039721e0fdac7fbcc05082401b64df7517"


def finite_nonnegative(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return result


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", choices=("reference-clean", "clean-clean"),
                        default="reference-clean")
    parser.add_argument("--rng-mode", choices=("per-image", "continuous"),
                        default="per-image",
                        help="continuous seeds once before construction, without per-image reseeding")
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--pattern", default="*_rgb_anon.png")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--atol", type=finite_nonnegative, default=1e-6)
    parser.add_argument("--rtol", type=finite_nonnegative, default=1e-5)
    parser.add_argument("--restore-prob", type=finite_nonnegative, default=0.0)
    parser.add_argument("--report", type=Path, default=Path("reference_steps_report.json"))
    args = parser.parse_args()
    if args.steps < 1 or args.restore_prob > 1:
        parser.error("steps must be positive and restore-prob must be in [0, 1]")
    return args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_fingerprint():
    def digest(value):
        return hashlib.sha256(value).hexdigest()
    return {
        "python": digest(pickle.dumps(random.getstate())),
        "numpy": digest(pickle.dumps(np.random.get_state())),
        "torch_cpu": digest(torch.get_rng_state().cpu().numpy().tobytes()),
        "torch_cuda": [digest(v.cpu().numpy().tobytes())
                       for v in torch.cuda.get_rng_state_all()]
                       if torch.cuda.is_available() else [],
    }


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def instances_snapshot(inst):
    return {"image_size": list(inst.image_size),
            "boxes": inst.pred_boxes.tensor.detach().cpu().clone(),
            "classes": inst.pred_classes.detach().cpu().clone(),
            "scores": inst.scores.detach().cpu().clone()}


def report_scalar(value):
    number = float(value)
    return number if math.isfinite(number) else str(number)


def capture_call(call, obj, code, batch, kind):
    """Read the actual return-frame locals; do not recompute any loss."""
    result = {}
    seen = []

    def profiler(frame, event, arg):
        if event != "return":
            return
        local = frame.f_locals
        if local.get("self") is not obj:
            return
        if len(seen) < 10:
            seen.append({"function": frame.f_code.co_name,
                         "locals": sorted(k for k in local if k != "self")})
        if kind == "reference":
            required = ("loss_dict", "pseudo_inst", "det_gate", "teacher_seg_probs_full")
        else:
            required = ("losses", "pseudo_inst", "keep_step_det", "t_probs_full")
        if any(key not in local for key in required):
            return  # Exception or incompatible implementation: fail after call.
        loss_dict = local[required[0]]
        normalized = {}
        for key, value in loss_dict.items():
            name = key.replace("det/rpn/", "det/").replace("det/roi/", "det/")
            if name in normalized:
                raise RuntimeError(f"loss-name collision: {name}")
            normalized[name] = value.detach().cpu().clone()
        pseudo = local["pseudo_inst"]
        if kind == "reference":
            pseudo = pseudo[0]
        probs = local[required[3]]
        result.update({"gate": bool(local[required[2]]),
                       "pseudo": instances_snapshot(pseudo),
                       "losses": normalized,
                       "teacher_probs": None if probs is None else probs.detach().cpu().clone()})

    if sys.gettrace() is not None or sys.getprofile() is not None:
        raise RuntimeError("An existing profiler/tracer is active; run in a fresh Python process")

    # ``sys.setprofile`` is not guaranteed to expose the Python return frame
    # when the call crosses ``nn.Module.__call__`` and C-extension frames.
    # A line tracer sees the bound implementation frame reliably, while this
    # callback only inspects its final return event.
    def tracer(frame, event, arg):
        if event == "return":
            profiler(frame, event, arg)
        return tracer

    sys.settrace(tracer)
    try:
        call(batch)
    finally:
        sys.settrace(None)
    if not result:
        raise RuntimeError(
            f"Could not capture {kind} forward locals; source differs from reviewed layout. "
            f"Observed target-frame locals: {seen}"
        )
    return result


def compare(a, b, atol, rtol):
    """Finite checks and full tensor comparisons, with bounded diagnostics."""
    errors = []
    count = 0

    def fail(path, detail):
        nonlocal count
        count += 1
        if len(errors) < 20:
            errors.append({"path": path, "detail": detail})

    def walk(x, y, path):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            if x.shape != y.shape or x.dtype != y.dtype:
                fail(path, f"shape/dtype: {tuple(x.shape)}/{x.dtype} vs {tuple(y.shape)}/{y.dtype}")
            elif not torch.isfinite(x).all() or not torch.isfinite(y).all():
                fail(path, "non-finite tensor")
            elif x.is_floating_point():
                if not torch.allclose(y, x, atol=atol, rtol=rtol, equal_nan=False):
                    delta = float((y.double() - x.double()).abs().max()) if x.numel() else 0.0
                    fail(path, f"max_abs_diff={delta:.9g}")
            elif not torch.equal(x, y):
                fail(path, "non-floating tensor differs")
        elif isinstance(x, dict) and isinstance(y, dict):
            if x.keys() != y.keys():
                fail(path, f"keys differ: reference-only={list(x.keys()-y.keys())}, clean-only={list(y.keys()-x.keys())}")
            for key in x:
                if key in y:
                    walk(x[key], y[key], f"{path}/{key}")
        elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            if len(x) != len(y):
                fail(path, "length differs")
            for i, (xx, yy) in enumerate(zip(x, y)):
                walk(xx, yy, f"{path}/{i}")
        elif isinstance(x, float) and isinstance(y, float):
            if not (math.isfinite(x) and math.isfinite(y)) or abs(y-x) > atol + rtol*abs(x):
                fail(path, f"reference={x}, clean={y}")
        elif type(x) is not type(y) or x != y:
            fail(path, f"reference={x}, clean={y}")

    walk(a, b, "")
    return {"passed": count == 0, "mismatch_count": count, "examples": errors}


def run(args, report):
    global torch, np
    import numpy as np
    import torch
    import detectron2
    from detectron2.solver import build_optimizer
    from detectron2.data import detection_utils as utils, transforms as T
    from ctcmt.d2 import setup_cfg, build_triplet
    from ctcmt.ctta import CTCMTAdaptStep, CTCMTHyperParams

    if args.comparison == "reference-clean":
        from detectron2.modeling.meta_arch.ctcmt_mtl import CTCMT_MTL
        reference_path = Path(inspect.getsourcefile(CTCMT_MTL))
        reference_hash = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        report["reference"] = {"file": str(reference_path), "sha256": reference_hash}
        if reference_hash != REFERENCE_SHA256:
            raise RuntimeError("Installed reference differs from uploaded/reviewed source; send this report before continuing")
    report["environment"] = {"python": sys.version, "torch": torch.__version__,
                             "detectron2": detectron2.__file__}
    for name, cls in (("clean_adapter", CTCMTAdaptStep),):
        source = Path(inspect.getsourcefile(cls))
        report[name] = {"file": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    images = sorted(p for p in args.image_root.rglob(args.pattern) if p.is_file())[:args.steps]
    if len(images) != args.steps:
        raise RuntimeError(f"Need {args.steps} images matching {args.pattern}; found {len(images)} under {args.image_root}")
    report["images"] = [str(p) for p in images]
    cfg = setup_cfg(args.config, weights=args.weights, freeze=False)
    original_rst = float(cfg.SOLVER.RST_M)
    cfg.SOLVER.RST_M = args.restore_prob
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.freeze()
    hp = CTCMTHyperParams.from_cfg(cfg)  # Fail on unsupported reference flags.
    if hp.det_only or hp.seg_only:
        raise RuntimeError("Use a joint MTL config for this diagnostic")
    report["config"] = cfg.dump()
    report["restoration"] = {"configured": original_rst, "diagnostic": args.restore_prob}
    report["weights"] = {"path": args.weights, "size_bytes": Path(args.weights).stat().st_size}
    print(f"Restoration: config={original_rst}, diagnostic={args.restore_prob}", flush=True)
    for p in images:
        print(f"Image: {p}", flush=True)
    resize = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
                                 cfg.INPUT.MAX_SIZE_TEST)

    def batch_for(path):
        image = utils.read_image(str(path), format=cfg.INPUT.FORMAT)
        h, w = image.shape[:2]
        image = resize.get_transform(image).apply_image(image)
        return [{"image": torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))),
                 "height": h, "width": w}]

    def models_of(obj, kind):
        if kind == "reference":
            return {n: getattr(obj, n) for n in ("student", "teacher", "anchor")}
        return {n: getattr(obj.triplet, n).model for n in ("student", "teacher", "anchor")}

    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Run without strict deterministic algorithms: ROIAlign fallback may exhaust memory")
    labels = ("reference", "clean") if args.comparison == "reference-clean" else ("clean_first", "clean_second")
    report["comparison"] = args.comparison
    report["rng_mode"] = args.rng_mode
    report["labels"] = list(labels)
    report["environment"].update({"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                  "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                                  "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    report["steps"] = []
    coverage = {kind: {"optimizer_calls": 0, "changed_steps": 0, "losses": set()}
                for kind in labels}
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available() and args.comparison == "reference-clean":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    report["backend"] = {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                         "cudnn_deterministic": torch.backends.cudnn.deterministic,
                         "cudnn_benchmark": torch.backends.cudnn.benchmark,
                         "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                         "cudnn_tf32": torch.backends.cudnn.allow_tf32}

    # A single triplet stays on GPU. Reference states are temporary CPU files.
    with tempfile.TemporaryDirectory(prefix="ctcmt-compare-") as tmp:
        tmp = Path(tmp)
        for pass_index, label in enumerate(labels):
            kind = "reference" if label == "reference" else "clean"
            seed_all(args.seed)
            if kind == "reference":
                obj = CTCMT_MTL(cfg)
                obj.eval()  # Match inference_on_dataset / Trainer.test.
                call, code = obj, CTCMT_MTL.forward.__code__
            else:
                triplet = build_triplet(cfg)
                obj = CTCMTAdaptStep(triplet, build_optimizer(cfg, triplet.student.model), hp)
                del triplet
                call, code = obj.step, CTCMTAdaptStep.step.__code__
            models = models_of(obj, kind)
            initial = {n: snapshot(m) for n, m in models.items()}
            construction_rng = rng_fingerprint()
            if pass_index == 0:
                first_construction_rng = construction_rng
                torch.save(initial, tmp / "initial.pt")
            else:
                expected = torch.load(tmp / "initial.pt", map_location="cpu", weights_only=False)
                report["construction_rng"] = compare(first_construction_rng, construction_rng, 0, 0)
                report["initial"] = compare(expected, initial, 0.0, 0.0)
                del expected
                if not report["initial"]["passed"]:
                    report["status"] = "MISMATCH_INITIAL_STATE"
                    return 1
            del initial
            for i, path in enumerate(images):
                print(f"{label}: step {i+1}/{len(images)}", flush=True)
                before = {n: p.detach().cpu().clone() for n, p in models["student"].named_parameters()
                          if p.requires_grad}
                if args.rng_mode == "per-image":
                    seed_all(args.seed + 1000 + i)
                rng_before_input = rng_fingerprint()
                batch = batch_for(path)
                rng_before_step = rng_fingerprint()
                input_hash = hashlib.sha256(batch[0]["image"].numpy().tobytes()).hexdigest()
                real_step = obj.optimizer.step
                with patch.object(obj.optimizer, "step", wraps=real_step) as step_spy:
                    trace = capture_call(call, obj, code, batch, kind)
                    calls = step_spy.call_count
                trace["rng_before_input"] = rng_before_input
                trace["rng_before_step"] = rng_before_step
                trace["rng_after_step"] = rng_fingerprint()
                trace["input_sha256"] = input_hash
                del real_step, step_spy, batch
                states = {n: snapshot(m) for n, m in models.items()}
                updates = {n: states["student"][n] - p for n, p in before.items()}
                changed = any(bool(torch.count_nonzero(v)) for v in updates.values())
                del before
                coverage[label]["optimizer_calls"] += calls
                coverage[label]["changed_steps"] += int(changed)
                coverage[label]["losses"].update(trace["losses"])
                trace["thresholds"] = list(obj.thresholds if kind == "reference" else obj.threshold_filter.thresholds)
                trace["score_ema"] = float(obj.score_em if kind == "reference" else obj.gate.score_em)
                trace["optimizer_calls"] = calls
                trace["modes"] = {n: {k: m.training for k, m in model.named_modules()} for n, model in models.items()}
                trace["states"] = states
                trace["net_student_updates"] = updates
                # Include momentum/optimizer state, so equal weights cannot mask divergence.
                def cpu_tree(x):
                    if isinstance(x, torch.Tensor):
                        return x.detach().cpu().clone()
                    if isinstance(x, dict):
                        return {k: cpu_tree(v) for k, v in x.items()}
                    if isinstance(x, (list, tuple)):
                        return type(x)(cpu_tree(v) for v in x)
                    return x
                trace["optimizer"] = cpu_tree(obj.optimizer.state_dict())
                if pass_index == 0:
                    torch.save(trace, tmp / f"step-{i}.pt")
                else:
                    expected = torch.load(tmp / f"step-{i}.pt", map_location="cpu", weights_only=False)
                    # ``Optimizer.state_dict()`` contains implementation-local
                    # parameter IDs and group bookkeeping. Two independently
                    # constructed models can have different IDs even when the
                    # optimizer tensors and model updates are identical. Keep
                    # it in the report for inspection, but exclude it from the
                    # equivalence verdict; ``net_student_updates`` and the
                    # finite optimizer-call checks cover effective updates.
                    comparisons = {
                        k: compare(expected[k], trace[k], args.atol, args.rtol)
                        for k in expected if k != "optimizer"
                    }
                    if "optimizer" in expected:
                        comparisons["optimizer_info"] = compare(
                            expected["optimizer"], trace["optimizer"], args.atol, args.rtol
                        )
                    # Same builder/order in clean-clean: optimizer state is part of verdict.
                    if args.comparison == "clean-clean":
                        comparisons["optimizer"] = comparisons["optimizer_info"]
                    summary = {"step": i+1, "image": str(path), "comparisons": comparisons,
                               "first_losses": {k: report_scalar(v) for k, v in expected["losses"].items()},
                               "second_losses": {k: report_scalar(v) for k, v in trace["losses"].items()},
                               "first_gate": expected["gate"], "second_gate": trace["gate"],
                               "first_boxes": len(expected["pseudo"]["scores"]),
                               "second_boxes": len(trace["pseudo"]["scores"])}
                    summary["rng"] = {"first": {k: v for k, v in expected.items() if k.startswith("rng_")},
                                      "second": {k: v for k, v in trace.items() if k.startswith("rng_")}}
                    summary["input_sha256"] = {"first": expected["input_sha256"], "second": trace["input_sha256"]}
                    if args.comparison == "reference-clean":
                        for field in ("losses", "gate", "boxes"):
                            summary["reference_" + field] = summary["first_" + field]
                            summary["clean_" + field] = summary["second_" + field]
                    report["steps"].append(summary)
                    failed = [k for k, v in comparisons.items()
                              if k != "optimizer_info" and not v["passed"]]
                    print(f"  {'MISMATCH: ' + ', '.join(failed) if failed else 'MATCH'}", flush=True)
                    del expected
                del trace, states, updates
            del call, obj, models
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    report["coverage"] = {k: {**v, "losses": sorted(v["losses"])} for k, v in coverage.items()}
    mismatched = any(
        not c["passed"]
        for s in report["steps"]
        for name, c in s["comparisons"].items()
        if name != "optimizer_info"
    )
    missing = []
    for kind, data in coverage.items():
        if data["optimizer_calls"] == 0 or data["changed_steps"] == 0:
            missing.append(f"{kind}: no effective adaptation update")
        if hp.weight_det > 0 and not any(k.startswith("det/") for k in data["losses"]):
            missing.append(f"{kind}: detection losses not exercised")
        for enabled, name in ((hp.weight_seg > 0, "seg/soft_ce"),
                              (hp.ctcl_enabled and hp.weight_ctcl > 0, "ctcl"),
                              (hp.weight_ctcr > 0, "ctcr")):
            if enabled and name not in data["losses"]:
                missing.append(f"{kind}: {name} not exercised")
    report["first_mismatch_step"] = next((s["step"] for s in report["steps"]
        if any(not c["passed"] for name, c in s["comparisons"].items() if name != "optimizer_info")), None)
    report["coverage_gaps"] = missing
    report["status"] = "MISMATCH" if mismatched else ("INCOMPLETE_COVERAGE" if missing else "MATCH_TESTED_SEQUENCE")
    return 1 if mismatched else (2 if missing else 0)


def main():
    args = cli()
    report = {"status": "STARTED", "atol": args.atol, "rtol": args.rtol,
              "seed": args.seed, "scope": "short real-image diagnostic; losses name-normalized; no ground truth"}
    try:
        status = run(args, report)
    except Exception as exc:
        import traceback
        report["status"] = "ERROR"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr)
        status = 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['status']}: {args.report}", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
