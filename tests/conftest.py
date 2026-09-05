"""Pytest conftest: pre-empt CTCMT/detectron2's dataset-root crash.

The CTCMT fork of detectron2 has a bare ``raise`` in its builtin dataset
registration when ``$DETECTRON2_DATASETS`` is not a real directory. That
runs at *import time* of ``detectron2.data.datasets.builtin``, which is
often pulled in transitively by ``detectron2.checkpoint`` / ``.modeling``
before any test code executes. Set a valid fallback root here so
detectron2 can be imported cleanly by every test.

Real dataloader calls (in ``ctcmt/main.py``) still expect the caller to
point ``$DETECTRON2_DATASETS`` at a real directory containing the
Cityscapes / ACDC / Cityscapes-C trees.
"""
from __future__ import annotations

import os


def _ensure_valid_datasets_root() -> None:
    root = os.path.expanduser(os.environ.get("DETECTRON2_DATASETS", "~/AMROD/datasets"))
    if os.path.isdir(root):
        return
    fallback = "/tmp/ctcmt_d2_datasets_empty"
    os.makedirs(fallback, exist_ok=True)
    os.environ["DETECTRON2_DATASETS"] = fallback


_ensure_valid_datasets_root()
