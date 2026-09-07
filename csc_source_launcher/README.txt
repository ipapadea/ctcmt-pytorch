Frozen PanopticFPN source-only baseline

Put this directory under /home/ilias/ctcmt-pytorch-clean on gpu1.
Start on the host:
    bash csc_source_launcher/start_source.sh
Monitor:
    docker logs -f ctcmt-csc-source-seed0

The container runs detached on host GPU 2 only. It reuses all mounts from
ctcmt-csc-v2-seed0-v3, which may be stopped but must still exist, and uses
the same snapshot image. It refuses to overwrite the output directory.

Evaluation: seed 0, TF32 off, one pass through fog, motion_blur, snow,
brightness, defocus_blur (500 images each). The source model is built by
ctcmt.d2.model_adapter.build_source_only. No adaptation step or optimizer
is instantiated. A state hash checks that parameters and buffers remain
unchanged after each domain. Ground truth is passed only to the evaluator.

The historical registration function and full config are read from the
completed adaptation run's saved copies. Loader, model preprocessing,
postprocessing and evaluators come from the same clean/Detectron2 code.

Host outputs:
/home/ilias/ctcmt-pytorch-clean/outputs/source_panoptic_cs_c_seed0_no_tf32/
  metrics.json                Five completed source evaluations
  source_vs_adapted.csv        Source vs each of the 50 adaptation evaluations
  metrics.partial.json        Progress retained after every domain
  launch_metadata.json        Hashes and completion/state-invariance status
  config.yaml                 Resolved evaluation config

A single source evaluation per domain is used as the fixed baseline for
all adaptation rounds. The comparison reports arithmetic means of domain
metrics, not pooled dataset metrics. It is not a ten-round source run.

Validation performed by assistant: Python and Bash syntax, plus comparison
alignment on the actual 50-evaluation adaptation JSON and missing-data
rejection. GPU/Detectron2 evaluation must run on your server.
