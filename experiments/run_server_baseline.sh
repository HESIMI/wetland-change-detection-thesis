#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate wetland_cd

cd "$PROJECT_ROOT"

python "$PROJECT_ROOT/training/inspect_dataset.py"
python "$PROJECT_ROOT/training/train_baseline.py" \
  --epochs 5 \
  --batch-size 4 \
  --outdir "$PROJECT_ROOT/runs/siamese_unet_server"
