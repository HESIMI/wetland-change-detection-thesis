# HRSCD-Clean Balanced Sample Subset

HRSCD-Clean 官方压缩包约 60.3 GB。为降低本机存储与训练成本，当前阶段先构建了一个本地均衡子集，用于数据读取验证、模型调试和 HRSCD 迁移实验的前期验证。

## Local Path

```text
D:/桌面/文献/论文/公开数据集/HRSCD_sample_balanced
```

## Current Size

- train: 220 complete samples
- val: 40 complete samples
- test: 40 complete samples
- image size: 256 x 256
- sampling target: about 50% changed patches in each split

## Directory Format

```text
HRSCD_sample_balanced/
  train/
    images1/
    images2/
    labels/
    labels_map/
    landcovers1/
    landcovers2/
  val/
  test/
```

## Field Mapping

- `images1`: first temporal image
- `images2`: second temporal image
- `labels`: binary change mask
- `labels_map`: transition map provided by HRSCD-Clean
- `landcovers1`: first temporal land cover map
- `landcovers2`: second temporal land cover map

## Reader Check

```bash
python src/wetland_cd/training/inspect_public_datasets.py --dataset hrscd --root D:/桌面/文献/论文/公开数据集/HRSCD_sample_balanced --split train
```

The unified reader returns:

- `t1`: `3 x 256 x 256`
- `t2`: `3 x 256 x 256`
- `image`: `6 x 256 x 256`
- `binary_mask`: `1 x 256 x 256`
- `semantic_t1`: `256 x 256`
- `semantic_t2`: `256 x 256`

## Sample Quality Summary

| split | patch count | changed patch count | mean change ratio | empty label patch count |
| --- | ---: | ---: | ---: | ---: |
| train | 220 | 110 | 0.079254 | 110 |
| val | 40 | 20 | 0.114799 | 20 |
| test | 40 | 20 | 0.093776 | 20 |

The full machine-readable table is saved at `docs/dataset/hrscd_balanced_sample_stats.csv`.

## Scope

This subset is sufficient for local dataloader verification, single-batch smoke tests, short HRSCD migration trials, and visualization debugging. It is not a full official HRSCD benchmark run. Full-scale HRSCD experiments should still use a larger subset or the complete dataset on the server.

The current balanced subset was built from the complete samples already downloaded from the official train split, then re-stratified locally into train/val/test. This keeps the local workflow practical while guaranteeing changed patches in every split. For final paper-scale HRSCD comparison, prefer downloading the full archive on the server and sampling from the official train/val/test splits.

## Local Rebalance Command

```bash
python scripts/data_preparation/rebalance_hrscd_sample.py ^
  --source-root D:/桌面/文献/论文/公开数据集/HRSCD_sample_balanced_remote_partial_20260518 ^
  --output-root D:/桌面/文献/论文/公开数据集/HRSCD_sample_balanced ^
  --source-splits train ^
  --train 220 --val 40 --test 40 ^
  --changed-fraction 0.5 ^
  --seed 2026 ^
  --overwrite
```

## Thesis-Scale Sampling Target

To make HRSCD comparable to SECOND in sample count, use the following target:

- train: 2500 samples
- val: 400 samples
- test: 1600 samples

Recommended workflow:

1. Download the full official archive to an English path, for example `D:/datasets_public/HRSCD_clean/HRSCD_Clean.zip`.
2. Extract a thesis-scale sampled subset from the local archive.

```bash
python scripts/data_preparation/sample_hrscd_clean.py ^
  --archive D:/datasets_public/HRSCD_clean/HRSCD_Clean.zip ^
  --output-root D:/桌面/文献/论文/公开数据集/HRSCD_sample_large ^
  --train 2500 --val 400 --test 1600 ^
  --balanced --changed-fraction 0.5
```

Remote sampling without a local archive is supported, but it is slow and unstable for thousands of samples because each patch requires multiple range requests to the 60 GB ZIP file.
