"""Delegate evaluator construction to detectron2's builtin evaluators.

Same logic as ``CTCMT/detectron2/tools/adapt.py::get_evaluator``, kept in
one place so the clean CTTA runner can construct evaluators without any
metric reimplementation.
"""
from __future__ import annotations

import os
from typing import Optional


def get_evaluator(cfg, dataset_name: str, output_folder: Optional[str] = None):
    """Return a detectron2 evaluator (or DatasetEvaluators) for a dataset."""
    from detectron2.data import MetadataCatalog
    from detectron2.evaluation import (
        CityscapesInstanceEvaluator,
        CityscapesSemSegEvaluator,
        COCOEvaluator,
        COCOPanopticEvaluator,
        DatasetEvaluators,
        LVISEvaluator,
        PascalVOCDetectionEvaluator,
        SemSegEvaluator,
    )

    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)

    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

    if evaluator_type in ("sem_seg", "coco_panoptic_seg", "coco_sem_seg"):
        evaluator_list.append(
            SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder)
        )
    if evaluator_type in ("coco", "coco_panoptic_seg", "coco_sem_seg"):
        evaluator_list.append(
            COCOEvaluator(dataset_name, output_dir=output_folder, allow_cached_coco=False)
        )
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)

    if not evaluator_list:
        raise NotImplementedError(
            f"No evaluator for dataset {dataset_name} (type={evaluator_type})"
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)
