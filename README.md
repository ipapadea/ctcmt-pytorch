# ctcmt-pytorch-clean

Clean-PyTorch **CTTA layer** for the CT-CMT experiments.

The point of this package is deliberately narrow:

* Reuse the exact detectron2 **PanopticFPN R50** architecture, source
  checkpoint, dataset registrations (`cityscapes_fine_mtl_*`, `acdc_*_mtl`,
  `cityscapes_c_*`), preprocessing, and evaluators (`CityscapesInstanceEvaluator`,
  `SemSegEvaluator`, `COCOEvaluator`) that live in
  [`CTCMT/detectron2`](../CTCMT/detectron2).
* Rewrite only the **adaptation layer** (mean teacher, pseudo-labels,
  score-EMA gate, stochastic restoration, cross-task losses, adapt step)
  into small, readable, framework-agnostic modules.

Nothing in this package reimplements a detector, a dataset, or an
evaluator. The reference detectron2 meta-arch
[`CTCMT_MTL`](../CTCMT/detectron2/detectron2/modeling/meta_arch/ctcmt_mtl.py)
remains the ground truth for algorithmic behavior; the modules here are
1:1 ports of the pieces inside it.

## Layout

```
ctcmt/
├── d2/                  <- Detectron2 bridge (the only place we touch d2)
│   ├── setup.py         <- setup_cfg, register_all_builtin_datasets
│   ├── model_adapter.py <- Detectron2ModelAdapter, build_source_only, build_triplet
│   └── evaluators.py    <- get_evaluator (delegates to detectron2)
├── ctta/                <- Framework-agnostic adaptation layer
│   ├── ema.py           <- EMAUpdater
│   ├── pseudo_labels.py <- DynamicThresholdFilter, filter_instances, to_gt_instances
│   ├── gate.py          <- ScoreEMGate
│   ├── restore.py       <- StochasticRestore (V2 shared-trunk factor)
│   ├── ctpv.py          <- CTPVFilter (hard-argmax agreement)
│   ├── losses.py        <- supcon_loss, CrossTaskContrastive (CT-CL),
│   │                       CrossTaskConsistency (CT-CR), SoftSegConsistency,
│   │                       MoraitiObjectContrastive
│   ├── seg_aug.py       <- aug_averaged_teacher_seg (CoTTA-style)
│   └── adapt_step.py    <- CTCMTAdaptStep: the single readable adapt step
└── main.py              <- CLI: setup_cfg → build_triplet → dataloader loop
tests/
├── test_ctta_math.py           <- framework-agnostic (no detectron2 needed)
├── test_source_equivalence.py  <- wrapper == raw PanopticFPN on same weights
└── test_adapt_step.py          <- one real adaptation step, weights change,
                                    anchor frozen, no GT ever leaks in
```

## Cityscapes taxonomy — verified

Detectron2 order is:
`(person, rider, car, truck, bus, train, motorcycle, bicycle)`. The
corresponding semantic-seg trainIds are
`(11, 12, 13, 14, 15, 16, 17, 18)`. This linear ordering is used everywhere
in `losses.py`, `ctpv.py`, and `adapt_step.py`. Do **not** reorder without
also re-training the source model.

## Running

Inside a container / venv that has detectron2 available and the
`CTCMT/detectron2` package installed:

```bash
pip install -e /home/ilias/ctcmt-pytorch-clean

# Standalone math tests (no detectron2 required)
pytest -q tests/test_ctta_math.py

# Integration tests (need detectron2 + a real source checkpoint)
CTCMT_CONFIG=/home/ilias/CTCMT/detectron2/configs/Cityscapes/ctcmt_mtl_panoptic_fpn_R_50_ACDC.yaml \
CTCMT_WEIGHTS=/path/to/panoptic_fpn_R50_cityscapes/model_final.pth \
    pytest -q tests/

# Full CTTA run (same yaml as the reference project)
python -m ctcmt.main \
    --config-file /home/ilias/CTCMT/detectron2/configs/Cityscapes/ctcmt_mtl_panoptic_fpn_R_50_ACDC.yaml \
    MODEL.WEIGHTS /path/to/panoptic_fpn_R50_cityscapes/model_final.pth
```

The runner produces evaluator output identical in schema to the reference
`tools/adapt.py`, so downstream tables and aggregation scripts still work.
