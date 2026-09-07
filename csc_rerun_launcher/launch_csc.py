"""Run clean historical MTL+V2 with process-local dataset registration."""
import ast
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

import torch
import detectron2.data.datasets.builtin as builtin
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.cityscapes import load_cityscapes_semantic
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances

out = Path("/workspace/ctcmt-pytorch-clean/outputs/clean_mtl_v2_cs_c_lt_seed0_no_tf32")
if out.exists():
    raise SystemExit(f"Refusing existing output directory: {out}")

source = Path("/workspace/CTCMT/detectron2/detectron2/data/datasets/builtin.py")
text = source.read_text()
nodes = [n for n in ast.parse(text).body
         if isinstance(n, ast.FunctionDef) and n.name == "register_cityscapes_c"]
assert len(nodes) == 1, "Expected one historical registration function"
node = nodes[0]
corruptions = ("gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
              "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
              "brightness", "contrast", "elastic_transform", "pixelate",
              "jpeg_compression")
for corruption in corruptions:
    for suffix in ("", "_semseg", "_mtl"):
        name = corruption + suffix
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
        if name in MetadataCatalog.list():
            MetadataCatalog.remove(name)

namespace = dict(vars(builtin))
namespace.update(os=os, DatasetCatalog=DatasetCatalog,
                 MetadataCatalog=MetadataCatalog,
                 _get_builtin_metadata=_get_builtin_metadata,
                 load_cityscapes_semantic=load_cityscapes_semantic,
                 load_coco_json=load_coco_json,
                 register_coco_instances=register_coco_instances)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
namespace["register_cityscapes_c"]("/workspace/datasets_cs_c")

datasets = ("fog_mtl", "motion_blur_mtl", "snow_mtl",
            "brightness_mtl", "defocus_blur_mtl")
for name in datasets:
    records = DatasetCatalog.get(name)
    assert len(records) == 500, (name, len(records))
    for record in records:
        for key in ("file_name", "sem_seg_file_name"):
            assert Path(record[key]).is_file(), (name, key, record[key])
    print(f"READY: {name}: 500 images + 500 semantic labels", flush=True)

config = Path("/workspace/panoptic_fpn/output/ft_historical/ctcmt-mtl-v2-ft__cs-c-lt__seed0/config.yaml")
weights = Path("/workspace/panoptic_fpn/output/panoptic_fpn_R50_cityscapes/model_final.pth")
assert weights.is_file(), weights
from ctcmt.d2 import setup_cfg
from ctcmt.ctta import CTCMTHyperParams
cfg = setup_cfg(str(config), weights=str(weights), freeze=False)
hp = CTCMTHyperParams.from_cfg(cfg)
assert hp.cross_task_fisher and hp.backbone_rst_factor == 0.1
assert hp.ctcl_enabled and not hp.ctpv_enabled
print("Hyperparameters:", hp, flush=True)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
assert torch.cuda.device_count() == 1, "Expected one visible GPU"
print("GPU:", torch.cuda.get_device_name(0), flush=True)

out.mkdir(parents=True, exist_ok=False)
(out / "historical_registration.py").write_text(ast.get_source_segment(text, node) + "\n")
(out / "launcher.py").write_text(Path(__file__).read_text())
(out / "historical_config.yaml").write_bytes(config.read_bytes())
(out / "launch_metadata.json").write_text(json.dumps({
    "seed": 0, "rounds": 10, "datasets": datasets, "tf32": False,
    "host_gpu_requested": 2, "torch": torch.__version__,
    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "registration_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
}, indent=2))

sys.argv = ["ctcmt.main", "--config-file", str(config), "--output-dir", str(out),
            "--rounds", "10", "--seed", "0", "MODEL.WEIGHTS", str(weights),
            "MODEL.DEVICE", "cuda", "SOLVER.IMS_PER_BATCH", "1",
            "DATASETS.TEST", repr(datasets)]
runpy.run_module("ctcmt.main", run_name="__main__")
