#!/usr/bin/env bash
set -euo pipefail
cd /home/ilias/ctcmt-pytorch-clean
test -f csc_source_launcher/evaluate_source.py
docker run -d --name ctcmt-csc-source-seed0 --init \
  --gpus '"device=2"' --shm-size=16g \
  --volumes-from ctcmt-csc-v2-seed0-v3 \
  -e CUDA_VISIBLE_DEVICES=0 -e PYTHONHASHSEED=0 \
  -e DETECTRON2_DATASETS=/workspace/datasets_cs_c \
  -w /workspace/ctcmt-pytorch-clean --entrypoint python \
  sha256:0e01b5628e4eeae7664747eb9b745cdb646f3a4efedd23342eec581646a59582 \
  -u csc_source_launcher/evaluate_source.py
