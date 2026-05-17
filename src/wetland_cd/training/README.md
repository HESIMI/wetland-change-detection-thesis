# Training Module

This directory is the unified experiment framework for binary wetland change detection and public benchmark validation. New models should be added to `models.py` and trained through `train.py`, rather than creating a separate repository or a separate training loop.

## Unified Items

| Item | Unified Location |
| --- | --- |
| Data reading | `data_module.py`, `dataset.py`, `public_datasets.py` |
| Training epochs | `configs/training/*.json -> training.epochs` |
| Input size | `configs/training/*.json -> data.image_size` |
| Learning rate strategy | `configs/training/*.json -> training.optimizer / training.scheduler` |
| Metrics | `metrics.py` |
| Loss | `losses.py` |
| Model registry | `models.py -> MODEL_REGISTRY` |
| Result format | `train.py -> runs/<experiment>/` |

## Result Format

Each run writes the same files:

```text
runs/<experiment>/
  config_resolved.json
  history.jsonl
  history.csv
  metrics.json
  checkpoints/
    best.pt
    last.pt
```

The final `metrics.json` always contains dataset name, model name, input size, epoch count, optimizer, scheduler, best validation epoch, train/val history, test metrics, and checkpoint paths.

## Run Wetland Baseline

```powershell
python src/wetland_cd/training/train.py `
  --config configs/training/wetland_siamese_unet.json
```

Quick smoke test:

```powershell
python src/wetland_cd/training/train.py `
  --config configs/training/wetland_siamese_unet.json `
  --epochs 1 `
  --batch-size 1 `
  --limit-train-batches 1 `
  --limit-val-batches 1 `
  --limit-test-batches 1 `
  --outdir runs/smoke_unified
```

## Run Public Benchmarks

SECOND:

```powershell
python src/wetland_cd/training/train.py `
  --config configs/training/second_siamese_unet.json
```

HRSCD sample:

```powershell
python src/wetland_cd/training/train.py `
  --config configs/training/hrscd_sample_siamese_unet.json
```

## Adding A Model

1. Implement the model in `models.py` or import it there.
2. Register it in `MODEL_REGISTRY`.
3. Create a config file under `configs/training/`.
4. Run it with `train.py`.

The model forward signature should be:

```python
logits = model(t1, t2)
```

where `t1` and `t2` are tensors shaped `B x C x H x W`, and `logits` is `B x 1 x H x W`.
