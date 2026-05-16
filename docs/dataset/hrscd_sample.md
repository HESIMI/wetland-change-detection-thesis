# HRSCD-Clean Sample Subset

HRSCD-Clean 官方压缩包约 60.3 GB。为降低本机存储与训练成本，当前阶段从官方压缩包中按需抽取了一个本地子集，用于数据读取验证、模型调试和论文中的补充性高分辨率实验。

## Local Path

```text
D:/桌面/文献/论文/公开数据集/HRSCD_sample
```

## Current Size

- train: 30 complete samples
- val: 5 complete samples
- test: 5 complete samples
- image size: 256 x 256
- local storage: about 8.78 MB

## Directory Format

```text
HRSCD_sample/
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
python src/wetland_cd/training/inspect_public_datasets.py --dataset hrscd --root D:/桌面/文献/论文/公开数据集/HRSCD_sample --split train
```

The unified reader returns:

- `t1`: `3 x 256 x 256`
- `t2`: `3 x 256 x 256`
- `image`: `6 x 256 x 256`
- `binary_mask`: `1 x 256 x 256`
- `semantic_t1`: `256 x 256`
- `semantic_t2`: `256 x 256`

## Scope

This subset is sufficient for local dataloader verification and lightweight model debugging. Full-scale HRSCD experiments should be run on a larger subset or the complete dataset on the server.

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
  --train 2500 --val 400 --test 1600
```

Remote sampling without a local archive is supported, but it is slow and unstable for thousands of samples because each patch requires multiple range requests to the 60 GB ZIP file.
