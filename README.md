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
is the reference for the supported adaptation behavior. The goal is to
preserve that behavior, but reference-only ablations are not all ported,
and per-image numerical equivalence has not yet been established.

## Layout

```
ctcmt/
├── d2/                  <- Detectron2 bridge
│   ├── setup.py         <- setup_cfg, register_all_builtin_datasets
│   ├── model_adapter.py <- Detectron2ModelAdapter, build_source_only, build_triplet
│   └── evaluators.py    <- get_evaluator (delegates to detectron2)
├── ctta/                <- PyTorch math modules and adaptation orchestration
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
├── test_adapt_step.py          <- optimizer update, teacher change, anchor frozen,
│                                GT independence on controlled synthetic inputs
└── test_diff_metrics.py        <- CLI validation of aggregate metric comparisons
```

The math components can be tested separately, but the MTL orchestration
still uses the Detectron2 bridge, structures, and postprocessing.

## Supported vs. unsupported reference flags

This policy applies to `CTCMTAdaptStep` (MTL), not to every adapter in the
repository. The following `SOLVER` switches are read by `CTCMTHyperParams`:

| Switches | MTL behavior |
| --- | --- |
| `CTCMT_SKIP_SCORE_EM_GATE` | Bypass the ratio test; empty / all-below-floor scores still close the gate. |
| `CTCMT_CROSS_TASK_FISHER`, `CTCMT_BACKBONE_RST_FACTOR` | Configure the V2 shared-backbone restoration factor. |
| `CTCMT_CTPV_ENABLED`, `CTCMT_CTPV_THRESH` | Enable and configure box filtering by teacher semantic agreement. |
| `CTCMT_CTCL_ENABLED`, `CTCMT_CTCL_SEG_VIEW` | Enable CT-CL and its semantic feature view. |
| `CTCMT_SEG_AUG_ENABLED`, `CTCMT_SEG_AUG_CONF_THRESH`, `CTCMT_SEG_AUG_SCALES`, `CTCMT_SEG_AUG_FLIPS` | Configure teacher segmentation augmentation averaging, triggered by low anchor confidence. |
| `CTCMT_DET_ONLY`, `CTCMT_SEG_ONLY` | Select a single-task loss path; either switch disables cross-task losses. Use at most one. |

Loss weights (`CTCMT_WEIGHT_DET`, `CTCMT_WEIGHT_SEG`, `CTCMT_WEIGHT_CTCL`,
`CTCMT_WEIGHT_CTCR`), CT-CL temperature/ROI settings, dynamic thresholds,
score EMA, `MT`, and `RST_M` are also read by `CTCMTHyperParams.from_cfg`.

Enabling any of the following six flags raises `NotImplementedError`
during MTL hyperparameter construction:

| Unsupported flag | Reference-only option |
| --- | --- |
| `CTCMT_PER_TASK_GATE` | Separate task gates |
| `CTCMT_PROTO_ANCHOR` | MTL prototype anchoring |
| `CTCMT_ENTROPY_WEIGHTED_CE` | Entropy-weighted segmentation CE |
| `CTCMT_AUG_TRIGGER_TEACHER_ENTROPY` | Teacher-entropy augmentation trigger |
| `CTCMT_DIRECTIONAL_GATE` | Directional gating |
| `CTCMT_ADAPTIVE_STR` | Adaptive stochastic restoration |

Keep these flags false or use the reference implementation for those
ablations. This guard covers the six flags listed above; it is not a
general validator for every option a reference YAML might contain.

## Adaptation and evaluation behavior

The MTL adapter requires batch size 1 and strips `instances` and `sem_seg`
from copies of the input dictionaries before model calls.

The score gate compares the mean of teacher scores above its score floor
with the previous score EMA. For a positive EMA and enabled ratio test,
it accepts ratios in `[1 / SCORE_THRESH, SCORE_THRESH]`. It rejects large
relative changes, not stable confidence. Valid scores update the score EMA
even when rejected; empty / all-below-floor scores do not update it.

A closed gate disables detection and CT-CL. Threshold updates and box
filtering remain independent of that decision. Segmentation soft-CE and
CT-CR may continue when their inputs and task settings permit them. CT-CL
uses the final teacher probabilities shared with segmentation soft-CE,
including augmentation averaging when triggered.

The optimizer runs when the loss dictionary is nonempty. On a normally
completed step, teacher EMA and student restoration follow regardless of
the detection gate. Returned predictions come from the updated teacher
on the same image: **adapt-then-predict**. Baseline comparisons must state
and align this evaluation protocol.

CT-CL retains two student cross-task feature views with teacher-provided
pseudo-labels; it is not the original SHIFT-TTA teacher/student object
contrastive implementation. CT-CR retains full-box class supervision.
These are methodological choices. CTPV/CT-CR retain reference coordinate
scaling; padding-aware box-to-segmentation alignment remains a known
limitation for inputs whose logit grid includes padded regions.

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

## Validation scope

Run the independent metric CLI tests with:

```bash
python -m pytest -q tests/test_diff_metrics.py
```

`scripts/diff_metrics.py clean.json reference.json --atol 1e-6 --rtol 1e-5`
checks selected metrics using `abs(clean - ref) <= atol + rtol * abs(ref)`.
These are also the default tolerances. It returns status 1 for missing or
non-finite selected metrics, empty comparisons, non-finite differences,
or tolerance violations. Metrics absent from both files are optional,
but each domain must contain at least one comparable selected metric.

The adaptation tests require PyTorch, the compatible Detectron2 fork,
config, and source checkpoint. Set `CTCMT_CONFIG` and `CTCMT_WEIGHTS` to
paths visible inside the environment that runs pytest. Use `-rs` to see
skip reasons; skipped integration tests do not establish correctness.

The reported validation at commit `68d167a` on 2026-09-05 was 70 metric
tests passed and 5 adaptation cases passed (no skips). The adaptation run
used Python 3.11.15, PyTorch 2.1.2+cu121, the installed `/opt/AMROD_d2`
fork, `ctcmt_mtl_panoptic_fpn_R_50_ACDC.yaml`, and the
`panoptic_fpn_R50_cityscapes/model_final.pth` checkpoint. These results
cover the tested configuration and synthetic inputs, not all ablations.

The adaptation checks observe a real optimizer parameter update with
restoration disabled, a teacher state change, an unchanged anchor, and
finite student/teacher states independent of GT presence or label values
under controlled RNG and explicit tolerances. They also reject multi-image
batches and the six unsupported flags. A teacher state change alone does
not verify the exact EMA equation.

Neither these tests nor agreement of aggregate metrics establishes
reference-vs-clean adaptation equivalence. A controlled short sequence
comparison of gates, thresholds, pseudo-labels, individual losses, and
parameter updates remains pending.
