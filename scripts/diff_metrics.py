"""Compute the delta between the clean CTTA runner and the reference
detectron2 ``CTCMT_MTL`` on the same source checkpoint + dataset + seed.

Reads two JSON files (each an ``OrderedDict`` produced by extracting
metrics from the runner logs) and prints a side-by-side table with
absolute differences. Non-zero deltas beyond a small tolerance are the
signal that clean and reference diverge — otherwise the port is
faithful.

Usage
-----
    python scripts/diff_metrics.py clean.json reference.json
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict


KEY_FIELDS = [
    ("bbox", "AP"), ("bbox", "AP50"), ("bbox", "AP75"),
    ("segm", "AP"), ("segm", "AP50"),
    ("sem_seg", "mIoU"), ("sem_seg", "fwIoU"), ("sem_seg", "pACC"),
]


def _get(d: Dict[str, Any], head: str, key: str):
    if head in d and isinstance(d[head], dict) and key in d[head]:
        return float(d[head][key])
    return None


def _fmt(v):
    return "-" if v is None else f"{v:7.3f}"


def main():
    if len(sys.argv) != 3:
        print("usage: diff_metrics.py clean.json reference.json", file=sys.stderr)
        sys.exit(2)
    with open(sys.argv[1]) as f:
        clean = json.load(f)
    with open(sys.argv[2]) as f:
        ref = json.load(f)

    domains = sorted(set(clean) | set(ref))
    print(f"{'domain':30s} {'metric':16s} {'clean':>8s} {'ref':>8s} {'Δ':>8s}")
    print("-" * 74)
    for dom in domains:
        c = clean.get(dom, {})
        r = ref.get(dom, {})
        for head, key in KEY_FIELDS:
            vc = _get(c, head, key)
            vr = _get(r, head, key)
            if vc is None and vr is None:
                continue
            if vc is None or vr is None:
                delta = "-"
            else:
                delta = f"{vc - vr:+7.3f}"
            print(f"{dom:30s} {head+'.'+key:16s} {_fmt(vc):>8s} {_fmt(vr):>8s} {delta:>8s}")


if __name__ == "__main__":
    main()
