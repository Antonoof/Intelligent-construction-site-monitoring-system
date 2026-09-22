#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:-}"
python -m pip install -r requirements.txt

if [ -n "$DATASET_ROOT" ]; then
  python train.py --dataset-root "$DATASET_ROOT"
else
  python train.py
fi
