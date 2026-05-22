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

Raw downloads are kept under the public dataset root, for example:

```text
<PUBLIC_DATA_ROOT>/LEVIR-CD
<PUBLIC_DATA_ROOT>/WHU-CD
<PUBLIC_DATA_ROOT>/SYSU-CD
```

Unified dataloader-ready data is written to:

```text
<PUBLIC_DATA_ROOT>/datasets
```

On the current local workstation, set this variable in PowerShell before
running the commands:

```powershell
$env:PUBLIC_DATA_ROOT = "<local public dataset root>"
```

## Current Status

| Dataset | Source | Unified layout | Dataloader smoke | Single-batch train | Current conclusion |
|---|---|---:|---:|---:|---|
| LEVIR-CD | HuggingFace mirror of official split archives | Done | Passed | Passed | Ready for local baseline checks |
| SYSU-CD | HuggingFace `ericyu/SYSU_CD` parquet mirror | Done | Passed | Passed | Ready for local baseline checks |
| WHU-CD | Official WHU archive | Done | Passed | Passed | Ready for local baseline checks |

## Preparation Commands

LEVIR-CD and WHU-CD archives can be normalized with:

```powershell
python scripts\data_preparation\organize_binary_cd_datasets.py `
  --public-root "$env:PUBLIC_DATA_ROOT" `
  --dataset LEVIR-CD `
  --extract-zip `
  --patch-size 256
```

SYSU-CD parquet files can be exported with:

```powershell
python scripts\data_preparation\export_sysu_hf_parquet.py `
  --parquet-root "$env:PUBLIC_DATA_ROOT\SYSU-CD\huggingface_ericyu_SYSU_CD\data" `
  --out-root "$env:PUBLIC_DATA_ROOT\datasets\SYSU-CD" `
  --overwrite
```

WHU-CD official archive can be exported with:

```powershell
python scripts\data_preparation\export_whu_cd_official.py `
  --raw-root "$env:PUBLIC_DATA_ROOT\WHU-CD" `
  --out-root "$env:PUBLIC_DATA_ROOT\datasets\WHU-CD" `
  --patch-size 256 `
  --val-ratio 0.1 `
  --overwrite
```

The scripts create:

- `manifest.csv`
- `summary.json`
- normalized `A/B/label` split folders

## Prepared Dataset Statistics

| Dataset | Split | Patch count | Changed patches | Empty label patches | Mean changed-pixel ratio |
|---|---:|---:|---:|---:|---:|
| LEVIR-CD | train | 7120 | 3167 | 3953 | 0.0459 |
| LEVIR-CD | val | 1024 | 436 | 588 | 0.0420 |
| LEVIR-CD | test | 2048 | 935 | 1113 | 0.0509 |
| SYSU-CD | train | 12000 | 12000 | 0 | 0.2134 |
| SYSU-CD | val | 4000 | 4000 | 0 | 0.2153 |
| SYSU-CD | test | 4000 | 4000 | 0 | 0.2358 |
| WHU-CD | train | 4504 | 3214 | 1290 | 0.0635 |
| WHU-CD | val | 536 | 399 | 137 | 0.0708 |
| WHU-CD | test | 2760 | 1852 | 908 | 0.0511 |

## Input Settings

Use the same preprocessing for all models:

- patch size: `256`
- image channels: `3`
- normalization: either `[0, 1]` or ImageNet mean/std, but the chosen setting
  must be shared by all compared models.

## Role in Thesis Experiments

These public binary change detection datasets are not the wetland weak-label
main dataset. Their role is to check whether baseline model implementations can
learn standard binary change detection under a unified dataloader, training
schedule, metric code, and result format before being transferred to HRSCD,
SECOND, and the wetland weak-supervision experiments.

## CDMamba Official Layout

The official CDMamba repository expects a flat dataset layout:

```text
<CDMAMBA_DATA_ROOT>/
  A/
  B/
  label/
  list/
    train.txt
    val.txt
    test.txt
```

The normalized split layout can be converted with:

```powershell
python scripts\data_preparation\export_cdmamba_layout.py `
  --src-root "$env:PUBLIC_DATA_ROOT\datasets\LEVIR-CD" `
  --out-root "$env:PUBLIC_DATA_ROOT\cdmamba_format\LEVIR-CD" `
  --mode hardlink

python scripts\data_preparation\export_cdmamba_layout.py `
  --src-root "$env:PUBLIC_DATA_ROOT\datasets\WHU-CD" `
  --out-root "$env:PUBLIC_DATA_ROOT\cdmamba_format\WHU-CD" `
  --mode hardlink
```

The exporter prefixes every filename with its split name before writing the
flat CDMamba layout. This avoids filename collisions in datasets such as WHU-CD,
where the official train and test tiles may share the same original names.
