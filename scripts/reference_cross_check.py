"""Reference cross-check runner: exact detectron2 ``CTCMT_MTL`` meta-arch.

Mirrors ``CTCMT/detectron2/tools/adapt.py::main`` but exposes ``--rounds``
so we can request a single round for cross-checking against
``ctcmt.main``. Uses the same seeding scheme as the clean runner so
side-by-side comparison is meaningful.

Usage
-----
    python scripts/reference_cross_check.py \\
        --config-file /workspace/CTCMT/detectron2/configs/Cityscapes/ctcmt_mtl_panoptic_fpn_R_50_ACDC.yaml \\
        --rounds 1 --seed 42 \\
        MODEL.WEIGHTS /workspace/panoptic_fpn/output/panoptic_fpn_R50_cityscapes/model_final.pth \\
        DATASETS.TEST '("acdc_fog_mtl",)' \\
        OUTPUT_DIR /tmp/ctcmt_ref_fog
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import OrderedDict

import numpy as np
import torch

# Make sure detectron2's builtin dataset registration doesn't crash if the
# default DETECTRON2_DATASETS root is missing.
from ctcmt.d2.setup import register_all_builtin_datasets  # noqa: E402


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("ctcmt.ref")


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser(description="CTCMT_MTL reference cross-check.")
    p.add_argument("--config-file", required=True)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("opts", nargs=argparse.REMAINDER)
    return p.parse_args()


def get_evaluator(cfg, dataset_name, output_folder):
    from ctcmt.d2.evaluators import get_evaluator as _get_eval
    return _get_eval(cfg, dataset_name, output_folder)


def adapt_and_eval(model, data_loader, evaluator):
    """CT-CMT test-time adaptation loop, one epoch over the target dataset.

    ``CTCMT_MTL.forward`` was written for the ``Trainer.test`` harness which
    flips every submodule to ``eval()`` before running inference. Without
    that flip, teacher/anchor sem_seg heads try to compute CE against
    ``None`` targets and crash. Emulate the flip here.
    """
    from detectron2.utils.events import EventStorage

    model.eval()
    if hasattr(model, "student"):
        model.student.eval()
        if hasattr(model.student, "sem_seg_head"):
            model.student.sem_seg_head.eval()
    if hasattr(model, "teacher"):
        model.teacher.eval()
    if hasattr(model, "anchor"):
        model.anchor.eval()

    evaluator.reset()
    with EventStorage():
        for _, data in enumerate(data_loader):
            outputs = model(data)
            evaluator.process(data, outputs)
    return evaluator.evaluate()


def main():
    args = parse_args()

    register_all_builtin_datasets(strict=True)

    from detectron2.config import get_cfg
    from detectron2.data import build_detection_test_loader
    from detectron2.modeling import build_model

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(list(args.opts))
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.freeze()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if args.seed >= 0:
        logger.info("Seeding with %d", args.seed)
        _seed_all(args.seed)

    logger.info("META_ARCH=%s WEIGHTS=%s", cfg.MODEL.META_ARCHITECTURE, cfg.MODEL.WEIGHTS)
    model = build_model(cfg)

    data_loaders = [
        build_detection_test_loader(cfg, name) for name in cfg.DATASETS.TEST
    ]
    evaluators = [
        get_evaluator(cfg, name, os.path.join(cfg.OUTPUT_DIR, "inference", name))
        for name in cfg.DATASETS.TEST
    ]

    all_results = OrderedDict()
    for round_idx in range(args.rounds):
        for name, loader, evalr in zip(cfg.DATASETS.TEST, data_loaders, evaluators):
            key = f"round{round_idx}/{name}"
            logger.info("[round %d] adapt+eval on %s", round_idx, name)
            r = adapt_and_eval(model, loader, evalr)
            all_results[key] = r
            logger.info("[round %d] %s -> %s", round_idx, name, r)

    logger.info("All reference results:\n%s", all_results)
    metrics_path = os.path.join(cfg.OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Wrote reference metrics -> %s", metrics_path)


if __name__ == "__main__":
    main()
