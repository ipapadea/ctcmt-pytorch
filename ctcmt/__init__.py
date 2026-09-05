"""Clean-PyTorch CTTA layer for the CTCMT PanopticFPN experiments.

The model, checkpoints, datasets, and evaluators are reused verbatim from
the existing detectron2 project at ``CTCMT/detectron2``. This package only
rewrites the adaptation layer (mean teacher, pseudo-labels, gates,
restoration, cross-task losses, adapt step).
"""
__version__ = "0.2.0"
