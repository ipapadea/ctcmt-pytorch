"""Entry point for CTCMT-MTL continual test-time adaptation.

Everything heavy is delegated:
  - config          -> detectron2 CfgNode via ``ctcmt.d2.setup_cfg``
  - model + weights -> ``ctcmt.d2.build_triplet``
  - datasets        -> detectron2's builtin registrations
  - dataloader      -> ``detectron2.data.build_detection_test_loader``
  - evaluator       -> ``ctcmt.d2.get_evaluator``
  - CTTA math       -> ``ctcmt.ctta.CTCMTAdaptStep``

Usage:
    python -m ctcmt.main \\
        --config-file /home/ilias/CTCMT/detectron2/configs/Cityscapes/ctcmt_mtl_panoptic_fpn_R_50_ACDC.yaml \\
        MODEL.WEIGHTS /workspace/output/panoptic_fpn_R50_cityscapes/model_final.pth
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections import OrderedDict

import torch

from ctcmt.ctta import (
    CTCMTAdaptStep,
    CTCMTHyperParams,
    CTCMTSegAdaptStep,
    CTCMTSegHyperParams,
)
from ctcmt.d2 import build_seg_triplet, build_triplet, get_evaluator, setup_cfg


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("ctcmt")


def parse_args():
    p = argparse.ArgumentParser(description="CTCMT-MTL CTTA runner")
    p.add_argument("--config-file", required=True, help="Detectron2 YAML config.")
    p.add_argument("--output-dir", default=None, help="Overrides cfg.OUTPUT_DIR.")
    p.add_argument("--rounds", type=int, default=1, help="Continual rounds over TEST domains.")
    p.add_argument("--seed", type=int, default=None,
                   help="Global seed (torch + python). Overrides cfg.SEED. -1 = do not seed.")
    p.add_argument("opts", nargs=argparse.REMAINDER,
                   help="Detectron2 opts (KEY VALUE pairs).")
    return p.parse_args()


def _seed_all(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(cfg, model):
    from detectron2.solver import build_optimizer as d2_build_optimizer
    return d2_build_optimizer(cfg, model)


def run_on_dataset(step: CTCMTAdaptStep, cfg, dataset_name: str, output_dir: str):
    from detectron2.data import build_detection_test_loader

    dataloader = build_detection_test_loader(cfg, dataset_name)
    evaluator = get_evaluator(cfg, dataset_name, output_dir)
    evaluator.reset()

    n = 0
    for batch in dataloader:
        outputs = step.step(batch)
        evaluator.process(batch, outputs)
        n += 1
        if n % 50 == 0:
            logger.info("  %s: processed %d images", dataset_name, n)

    return evaluator.evaluate()


def main():
    args = parse_args()
    cfg = setup_cfg(
        args.config_file,
        opts=args.opts if args.opts else None,
        output_dir=args.output_dir,
        freeze=True,
        register_datasets=True,
        strict_registration=True,
    )
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Seed precedence: --seed > cfg.SEED > unseeded.
    seed = args.seed if args.seed is not None else int(cfg.SEED)
    if seed >= 0:
        logger.info("Seeding torch/numpy/python with %d", seed)
        _seed_all(seed)
    else:
        logger.info("Not seeding (SEED=%d); results will vary across runs.", seed)

    logger.info("Building student / teacher / anchor from %s", cfg.MODEL.WEIGHTS)
    arch = cfg.MODEL.META_ARCHITECTURE
    if arch == "CTCMT_Seg":
        triplet = build_seg_triplet(cfg)
        hp = CTCMTSegHyperParams.from_cfg(cfg)
        optimizer = build_optimizer(cfg, triplet.student.model)
        step = CTCMTSegAdaptStep(triplet, optimizer, hp)
    elif arch in ("CTCMT_MTL", "PanopticFPN", "GeneralizedRCNN"):
        triplet = build_triplet(cfg)
        hp = CTCMTHyperParams.from_cfg(cfg)
        optimizer = build_optimizer(cfg, triplet.student.model)
        step = CTCMTAdaptStep(triplet, optimizer, hp)
    else:
        raise ValueError(
            f"Unsupported META_ARCHITECTURE {arch!r}. "
            "Set to CTCMT_MTL (panoptic MTL), CTCMT_Seg (semantic FPN), "
            "or use CTCMT_STUDENT_META_ARCH to override."
        )

    all_results = OrderedDict()
    for round_idx in range(args.rounds):
        for dataset_name in cfg.DATASETS.TEST:
            output_dir = os.path.join(
                cfg.OUTPUT_DIR, "inference", f"round{round_idx}_{dataset_name}"
            )
            os.makedirs(output_dir, exist_ok=True)
            logger.info("[round %d] adapt+eval on %s", round_idx, dataset_name)
            r = run_on_dataset(step, cfg, dataset_name, output_dir)
            all_results[f"round{round_idx}/{dataset_name}"] = r
            logger.info("[round %d] %s -> %s", round_idx, dataset_name, r)

    logger.info("All results:\n%s", all_results)
    metrics_path = os.path.join(cfg.OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Wrote metrics -> %s", metrics_path)


if __name__ == "__main__":
    main()
