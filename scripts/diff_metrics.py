"""Compute the delta between the clean CTTA runner and the reference
detectron2 ``CTCMT_MTL`` on the same source checkpoint + dataset + seed.

Reads two JSON files (each an ``OrderedDict`` produced by extracting
metrics from the runner logs) and prints a side-by-side table with
signed differences. A metric passes when
``abs(clean - ref) <= atol + rtol * abs(ref)``. Missing or non-finite
metrics, empty comparisons, non-finite differences, and differences
beyond tolerance fail validation (exit status 1).

Usage
-----
    python scripts/diff_metrics.py clean.json reference.json
    python scripts/diff_metrics.py clean.json reference.json --atol 1e-4 --rtol 0

Metrics absent from both files are optional, but every domain must have
at least one comparable metric from ``KEY_FIELDS``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, Dict


KEY_FIELDS = [
    ("bbox", "AP"), ("bbox", "AP50"), ("bbox", "AP75"),
    ("segm", "AP"), ("segm", "AP50"),
    ("sem_seg", "mIoU"), ("sem_seg", "fwIoU"), ("sem_seg", "pACC"),
]


def _get(d: Dict[str, Any], head: str, key: str):
    if head not in d:
        return None
    if not isinstance(d[head], dict):
        raise ValueError(f"{head} must be a JSON object")
    if key not in d[head]:
        return None
    value = d[head][key]
    if isinstance(value, bool):
        raise ValueError(f"{head}.{key} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{head}.{key} must be numeric") from exc


def _fmt(v):
    return "-" if v is None else f"{v:7.3f}"


def _tolerance(value):
    try:
        tolerance = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tolerance must be numeric") from exc
    if not math.isfinite(tolerance) or tolerance < 0:
        raise argparse.ArgumentTypeError("tolerance must be finite and non-negative")
    return tolerance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clean", help="clean runner metrics JSON")
    parser.add_argument("reference", help="reference runner metrics JSON")
    parser.add_argument("--atol", type=_tolerance, default=1e-6,
                        help="absolute tolerance (default: %(default)s)")
    parser.add_argument("--rtol", type=_tolerance, default=1e-5,
                        help="relative tolerance scaled by abs(ref) (default: %(default)s)")
    args = parser.parse_args(argv)
    try:
        with open(args.clean) as f:
            clean = json.load(f)
        with open(args.reference) as f:
            ref = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not isinstance(clean, dict) or not isinstance(ref, dict):
        print("error: metrics files must contain JSON objects", file=sys.stderr)
        return 1

    domains = sorted(set(clean) | set(ref))
    errors = []
    if not domains:
        errors.append("no metrics to compare")
    print(f"{'domain':30s} {'metric':16s} {'clean':>8s} {'ref':>8s} {'Δ':>8s}")
    print("-" * 74)
    for dom in domains:
        c = clean.get(dom, {})
        r = ref.get(dom, {})
        if not isinstance(c, dict) or not isinstance(r, dict):
            errors.append(f"{dom}: domain metrics must be JSON objects")
            continue
        compared = 0
        for head, key in KEY_FIELDS:
            metric = head + "." + key
            try:
                vc = _get(c, head, key)
                vr = _get(r, head, key)
            except ValueError as exc:
                errors.append(f"{dom} {metric}: {exc}")
                continue
            if vc is None and vr is None:
                continue
            if vc is None or vr is None:
                delta = "-"
                missing = "clean" if vc is None else "reference"
                errors.append(f"{dom} {metric}: missing metric in {missing}")
            else:
                delta = f"{vc - vr:+7.3f}"
                if not math.isfinite(vc) or not math.isfinite(vr):
                    errors.append(f"{dom} {metric}: non-finite metric (clean={vc}, ref={vr})")
                else:
                    compared += 1
                    difference = abs(vc - vr)
                    limit = args.atol + args.rtol * abs(vr)
                    if not math.isfinite(difference):
                        errors.append(f"{dom} {metric}: non-finite difference")
                    elif difference > limit:
                        errors.append(
                            f"{dom} {metric}: difference {difference:.6g} exceeds "
                            f"tolerance {limit:.6g} (atol={args.atol:g}, rtol={args.rtol:g})"
                        )
            print(f"{dom:30s} {head+'.'+key:16s} {_fmt(vc):>8s} {_fmt(vr):>8s} {delta:>8s}")
        if not compared:
            errors.append(f"{dom}: no comparable metrics")

    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
