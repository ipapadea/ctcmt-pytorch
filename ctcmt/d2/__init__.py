"""Detectron2 bridge: PanopticFPN wrapper, config loading, evaluators."""
from .setup import setup_cfg, register_all_builtin_datasets
from .model_adapter import (
    Detectron2ModelAdapter,
    Triplet,
    build_seg_triplet,
    build_source_only,
    build_triplet,
)
from .evaluators import get_evaluator

__all__ = [
    "setup_cfg",
    "register_all_builtin_datasets",
    "Detectron2ModelAdapter",
    "Triplet",
    "build_source_only",
    "build_triplet",
    "build_seg_triplet",
    "get_evaluator",
]
