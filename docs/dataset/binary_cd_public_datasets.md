# Binary Change Detection Public Datasets

This note tracks the local preparation of LEVIR-CD, WHU-CD, and SYSU-CD for
shared binary change detection dataloaders.

## Target Layout

All three datasets should be normalized to:

```text
datasets/
  <DATASET>/
    train/
      A/
      B/
      label/
    val/
      A/
      B/
      label/
    test/
      A/
      B/
      label/
```

Label convention:

- `changed = 1`
- `unchanged = 0`

Stored label PNGs use `255` for changed pixels and `0` for unchanged pixels so
that common binary CD code can read them with `mask > 0`.

## Local Paths

Raw downloads are kept under:

```text
D:\桌面\文献\论文\公开数据集\LEVIR-CD
D:\桌面\文献\论文\公开数据集\WHU-CD
D:\桌面\文献\论文\公开数据集\SYSU-CD
```

Unified dataloader-ready data is written to:

```text
D:\桌面\文献\论文\公开数据集\datasets
```

## Preparation Script

```powershell
python scripts\data_preparation\organize_binary_cd_datasets.py `
  --public-root D:\桌面\文献\论文\公开数据集 `
  --dataset all `
  --extract-zip
```

The script creates:

- `manifest.csv`
- `summary.json`
- normalized `A/B/label` split folders

## Input Settings

Use the same preprocessing for all models:

- patch size: `256`
- image channels: `3`
- normalization: either `[0, 1]` or ImageNet mean/std, but the chosen setting
  must be shared by all compared models.
