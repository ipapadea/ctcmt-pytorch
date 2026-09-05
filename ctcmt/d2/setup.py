"""Detectron2 config loading + dataset registration.

Reuses ``CTCMT/detectron2`` for the exact configs and dataset registrations
that produced the published CT-CMT numbers. Importing detectron2's builtin
module has the side effect of registering ``cityscapes_fine_mtl_*``,
``acdc_*``, ``acdc_*_semseg``, ``acdc_*_mtl``, and ``cityscapes_c_*``.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable, Optional


logger = logging.getLogger(__name__)

_BUILTIN_REGISTERED = False


def register_all_builtin_datasets(strict: bool = False) -> bool:
    """Idempotent: trigger detectron2's builtin dataset registrations.

    The CTCMT fork of detectron2 raises at module-import time if its
    dataset root (``$DETECTRON2_DATASETS``, default ``~/AMROD/datasets``)
    does not exist, so we swallow that failure by default: tests that
    only need the model surface (source equivalence, one-step adapt)
    do not touch datasets. Callers that need real dataloaders should
    pass ``strict=True`` and ensure ``DETECTRON2_DATASETS`` points at a
    real directory.
    """
    global _BUILTIN_REGISTERED
    if _BUILTIN_REGISTERED:
        return True

    # The CTCMT fork of detectron2 does a bare ``raise`` in builtin.py when
    # $DETECTRON2_DATASETS is missing. Silence that by pointing at an empty
    # temp dir when the caller has not set anything usable. Real dataloader
    # calls will then fail with a clear detectron2 error naming the dataset.
    default_root = os.path.expanduser(os.environ.get("DETECTRON2_DATASETS", "~/AMROD/datasets"))
    if not os.path.isdir(default_root):
        fallback = "/tmp/ctcmt_d2_datasets_empty"
        os.makedirs(fallback, exist_ok=True)
        os.environ["DETECTRON2_DATASETS"] = fallback

    try:
        import detectron2.data.datasets.builtin  # noqa: F401
        _BUILTIN_REGISTERED = True
        return True
    except Exception as e:
        msg = (
            "detectron2 builtin dataset registration failed "
            f"(DETECTRON2_DATASETS={os.environ.get('DETECTRON2_DATASETS')}): {e!r}"
        )
        if strict:
            raise
        logger.warning("%s -- continuing without registered datasets.", msg)
        return False


def setup_cfg(
    config_file: str,
    weights: Optional[str] = None,
    opts: Optional[Iterable[str]] = None,
    output_dir: Optional[str] = None,
    freeze: bool = True,
    register_datasets: bool = True,
    strict_registration: bool = False,
):
    """Load a detectron2 yaml the same way the CTCMT training scripts do.

    Parameters mirror ``tools/adapt.py``: an optional ``weights`` overrides
    ``MODEL.WEIGHTS`` and ``opts`` is a flat list of key/value pairs applied
    on top (e.g. ``["SOLVER.BASE_LR", "1e-4"]``).
    """
    from detectron2.config import get_cfg

    if register_datasets:
        register_all_builtin_datasets(strict=strict_registration)

    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    if opts:
        cfg.merge_from_list(list(opts))
    if weights is not None:
        cfg.MODEL.WEIGHTS = weights
    if output_dir is not None:
        cfg.OUTPUT_DIR = output_dir
    if os.environ.get("CTCMT_FORCE_CPU") == "1":
        cfg.MODEL.DEVICE = "cpu"
    if freeze:
        cfg.freeze()
    return cfg
