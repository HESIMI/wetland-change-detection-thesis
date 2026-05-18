# HRSCD Balanced Sample Acceptance

## Purpose

The original local HRSCD sample had too few changed patches, and val/test were effectively empty for change detection. The current balanced sample is built to support local migration checks before running larger HRSCD experiments on the server.

## Location

```text
D:/桌面/文献/论文/公开数据集/HRSCD_sample_balanced
```

## Statistics

| split | patch count | changed patch count | mean change ratio | empty label patch count |
| --- | ---: | ---: | ---: | ---: |
| train | 220 | 110 | 0.079254 | 110 |
| val | 40 | 20 | 0.114799 | 20 |
| test | 40 | 20 | 0.093776 | 20 |

Supporting files in the sample root:

- `sample_manifest.csv`
- `sample_change_records.csv`
- `sample_quality_summary.csv`
- `sample_quality_summary.json`

The repository copy of the summary table is `docs/dataset/hrscd_balanced_sample_stats.csv`.

## Acceptance

| Check | Result |
| --- | --- |
| Patch quantity | Passed; 300 complete patches, larger than the previous 30/5/5 debug sample |
| Changed patch coverage | Passed; each split has 50% changed patches |
| Empty label patches | Controlled; each split keeps empty labels for unchanged/background cases |
| Unified reader | Passed; train/val/test all return `t1`, `t2`, `image`, `binary_mask`, `semantic_t1`, `semantic_t2` |
| Training role | Suitable for local HRSCD transfer smoke tests and visualization checks |
| Thesis-scale role | Not sufficient as a final large-scale HRSCD benchmark; use server/full archive for that |

## Commands

Reader checks:

```bash
python src/wetland_cd/training/inspect_public_datasets.py --dataset hrscd --root D:/桌面/文献/论文/公开数据集/HRSCD_sample_balanced --split train --batch-size 2
python src/wetland_cd/training/inspect_public_datasets.py --dataset hrscd --root D:/桌面/文献/论文/公开数据集/HRSCD_sample_balanced --split val --batch-size 2
python src/wetland_cd/training/inspect_public_datasets.py --dataset hrscd --root D:/桌面/文献/论文/公开数据集/HRSCD_sample_balanced --split test --batch-size 2
```

Local rebalance:

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

Server-scale official split sampling:

```bash
python scripts/data_preparation/sample_hrscd_clean.py ^
  --archive D:/datasets_public/HRSCD_clean/HRSCD_Clean.zip ^
  --output-root D:/桌面/文献/论文/公开数据集/HRSCD_sample_large ^
  --train 2500 --val 400 --test 1600 ^
  --balanced --changed-fraction 0.5
```
