"""Frozen PanopticFPN baseline for the completed clean MTL+V2 experiment.

Uses the saved historical registration/config, existing source builder,
Detectron2 loader and clean evaluator factory. No CTTA step is constructed.
"""
import ast
import csv
import hashlib
import json
import logging
import math
import os
import sys
from pathlib import Path
from statistics import mean

ROOT = Path('/workspace/ctcmt-pytorch-clean')
PREVIOUS = ROOT / 'outputs/clean_mtl_v2_cs_c_lt_seed0_no_tf32'
OUT = ROOT / 'outputs/source_panoptic_cs_c_seed0_no_tf32'
DOMAINS = ('fog_mtl', 'motion_blur_mtl', 'snow_mtl',
           'brightness_mtl', 'defocus_blur_mtl')
FIELDS = ('bbox.AP', 'bbox.AP50', 'bbox.AP75', 'segm.AP', 'sem_seg.mIoU')
WEIGHTS = Path('/workspace/panoptic_fpn/output/panoptic_fpn_R50_cityscapes/model_final.pth')
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
LOG = logging.getLogger('source_eval')


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def state_hash(model):
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str((value.shape, value.dtype)).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def metric(record, field):
    task, key = field.split('.', 1)
    return float(record[task][key])


def compare(source, adapted):
    expected = {f'round{r}/{d}' for r in range(10) for d in DOMAINS}
    if set(adapted) != expected:
        raise ValueError('Adapted metrics do not contain exactly 50 expected evaluations')
    rows = []
    for r in range(10):
        for d in DOMAINS:
            row = {'round': r+1, 'dataset': d}
            for field in FIELDS:
                a = metric(source[f'round0/{d}'], field)
                b = metric(adapted[f'round{r}/{d}'], field)
                if not math.isfinite(a) or not math.isfinite(b):
                    raise ValueError(f'Non-finite {field}: round {r}, {d}')
                row.update({f'source.{field}': a, f'adapted.{field}': b,
                            f'delta.{field}': b-a})
            rows.append(row)
    return rows


def main():
    import torch
    import detectron2
    import detectron2.data.datasets.builtin as builtin
    from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
    from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
    from detectron2.data.datasets.cityscapes import load_cityscapes_semantic
    from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
    from ctcmt.d2 import setup_cfg, get_evaluator
    from ctcmt.d2.model_adapter import build_source_only
    from ctcmt.main import _seed_all

    if OUT.exists():
        raise SystemExit(f'Refusing existing output directory: {OUT}')
    config_file = PREVIOUS / 'historical_config.yaml'
    registration_file = PREVIOUS / 'historical_registration.py'
    adapted_file = PREVIOUS / 'metrics.json'
    for path in (config_file, registration_file, adapted_file, WEIGHTS):
        if not path.is_file():
            raise FileNotFoundError(path)
    adapted = json.loads(adapted_file.read_text())
    if set(adapted) != {f'round{r}/{d}' for r in range(10) for d in DOMAINS}:
        raise ValueError('Expected the completed 50-evaluation clean run')

    # Use the exact function saved by the successful adaptation launcher.
    nodes = [n for n in ast.parse(registration_file.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name == 'register_cityscapes_c']
    if len(nodes) != 1:
        raise ValueError('Expected one saved registration function')
    corruptions = ('gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                   'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                   'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression')
    for corruption in corruptions:
        for suffix in ('', '_semseg', '_mtl'):
            name = corruption + suffix
            if name in DatasetCatalog.list():
                DatasetCatalog.remove(name)
            if name in MetadataCatalog.list():
                MetadataCatalog.remove(name)
    namespace = dict(vars(builtin))
    namespace.update(os=os, DatasetCatalog=DatasetCatalog, MetadataCatalog=MetadataCatalog,
                     _get_builtin_metadata=_get_builtin_metadata,
                     load_cityscapes_semantic=load_cityscapes_semantic,
                     load_coco_json=load_coco_json, register_coco_instances=register_coco_instances)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(registration_file), 'exec'), namespace)
    namespace['register_cityscapes_c']('/workspace/datasets_cs_c')

    for domain in DOMAINS:
        records = DatasetCatalog.get(domain)
        if len(records) != 500:
            raise ValueError(f'{domain}: expected 500 images, got {len(records)}')
        for record in records:
            for key in ('file_name', 'sem_seg_file_name'):
                if not Path(record[key]).is_file():
                    raise FileNotFoundError(record[key])
        LOG.info('READY: %s, 500 images + semantic labels', domain)

    cfg = setup_cfg(str(config_file), weights=str(WEIGHTS), output_dir=str(OUT), freeze=False)
    cfg.DATASETS.TEST = DOMAINS
    cfg.MODEL.DEVICE = 'cuda'
    cfg.SEED = 0
    cfg.CUDNN_BENCHMARK = False
    cfg.freeze()
    _seed_all(0)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible GPU')
    source = build_source_only(cfg)
    if source.model.training or any(p.requires_grad for p in source.model.parameters()):
        raise RuntimeError('Source model must be frozen and in eval mode')
    before = state_hash(source.model)
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'config.yaml').write_text(cfg.dump())
    (OUT/'evaluate_source.py').write_text(Path(__file__).read_text())
    (OUT/'historical_registration.py').write_bytes(registration_file.read_bytes())
    provenance = {
        'mode': 'source_only', 'rounds': 1, 'seed': 0, 'datasets': DOMAINS,
        'tf32': False, 'gpu': torch.cuda.get_device_name(0), 'host_gpu_requested': 2,
        'torch': torch.__version__, 'detectron2_path': detectron2.__file__,
        'weights_sha256': file_hash(WEIGHTS), 'initial_state_sha256': before,
        'historical_config_sha256': file_hash(config_file),
        'registration_sha256': file_hash(registration_file),
        'adapted_metrics_sha256': file_hash(adapted_file),
        'completed': False,
    }
    (OUT/'launch_metadata.json').write_text(json.dumps(provenance, indent=2))
    results = {}
    for domain in DOMAINS:
        output_folder = OUT/'inference'/f'round0_{domain}'
        output_folder.mkdir(parents=True)
        loader = build_detection_test_loader(cfg, domain)
        evaluator = get_evaluator(cfg, domain, str(output_folder))
        evaluator.reset()
        LOG.info('Source-only evaluation: %s', domain)
        count = 0
        with torch.no_grad():
            for batch in loader:
                # Labels are used only by the evaluator.
                inputs = [{k: v for k, v in record.items() if k not in ('instances', 'sem_seg')}
                          for record in batch]
                outputs = source.detect(inputs)
                evaluator.process(batch, outputs)
                count += len(batch)
                if count % 50 == 0:
                    LOG.info('%s: processed %d images', domain, count)
        if count != 500:
            raise RuntimeError(f'{domain}: processed {count}, expected 500')
        r = evaluator.evaluate()
        for field in FIELDS:
            if not math.isfinite(metric(r, field)):
                raise ValueError(f'{domain}: invalid {field}')
        if state_hash(source.model) != before:
            raise RuntimeError('Source model state changed during evaluation')
        results[f'round0/{domain}'] = r
        (OUT/'metrics.partial.json').write_text(json.dumps(results, indent=2, allow_nan=False))
        LOG.info('%s: AP50=%.3f, mIoU=%.3f', domain, r['bbox']['AP50'], r['sem_seg']['mIoU'])

    (OUT/'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    rows = compare(results, adapted)
    with (OUT/'source_vs_adapted.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    LOG.info('SOURCE MEAN: AP50=%.3f, mIoU=%.3f',
             mean(r['bbox']['AP50'] for r in results.values()),
             mean(r['sem_seg']['mIoU'] for r in results.values()))
    for round_number in (1,2,10):
        subset = [r for r in rows if r['round'] == round_number]
        LOG.info('Adapted round %d minus source: AP50=%+.3f, mIoU=%+.3f',
                 round_number, mean(r['delta.bbox.AP50'] for r in subset),
                 mean(r['delta.sem_seg.mIoU'] for r in subset))
    provenance.update(completed=True, model_state_unchanged=True)
    (OUT/'launch_metadata.json').write_text(json.dumps(provenance, indent=2))
    LOG.info('DONE: %s', OUT/'metrics.json')


if __name__ == '__main__':
    main()
