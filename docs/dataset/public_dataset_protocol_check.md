# Public Dataset Protocol Check

This note verifies whether the local LEVIR-CD, WHU-CD, and SYSU-CD data can be
directly compared with common public change detection benchmark settings.

## Checked Items

1. Whether the train/val/test split matches common paper settings.
2. Whether patch size matches common paper settings.
3. Whether the evaluation metrics match common binary change detection metrics.

## Local Data Check

Local root:

```text
D:/桌面/文献/论文/公开数据集/datasets
```

| Dataset | Split | Local A/B/label count | Patch size | Label values | Local conclusion |
|---|---:|---:|---:|---:|---|
| LEVIR-CD | train | 7120/7120/7120 | 256 x 256 | 0/255 | Matched |
| LEVIR-CD | val | 1024/1024/1024 | 256 x 256 | 0/255 | Matched |
| LEVIR-CD | test | 2048/2048/2048 | 256 x 256 | 0/255 | Matched |
| WHU-CD | train | 4504/4504/4504 | 256 x 256 | 0/255 | Split differs from some papers |
| WHU-CD | val | 536/536/536 | 256 x 256 | 0/255 | Split differs from some papers |
| WHU-CD | test | 2760/2760/2760 | 256 x 256 | 0/255 | Split differs from some papers |
| SYSU-CD | train | 12000/12000/12000 | 256 x 256 | 0/255 | Matched |
| SYSU-CD | val | 4000/4000/4000 | 256 x 256 | 0/255 | Matched |
| SYSU-CD | test | 4000/4000/4000 | 256 x 256 | 0/255 | Matched |

All local training configs for these three datasets use:

```text
image_size = 256
```

The dataloader reads label PNGs with `mask > 0`, so stored `255` values are
converted to binary changed pixels.

## Protocol Match Assessment

| Dataset | Split protocol | Patch protocol | Metric protocol | Can directly compare with common paper table? |
|---|---|---|---|---|
| LEVIR-CD | Yes. Local split is 7120/1024/2048. | Yes. 256 x 256 non-overlap patches. | Yes. P/R/F1/IoU/OA from binary confusion matrix. | Yes, if training schedule and model implementation are also official/comparable. |
| SYSU-CD | Yes. Local split is 12000/4000/4000. | Yes. Original samples are 256 x 256. | Yes. P/R/F1/IoU/OA from binary confusion matrix. | Basically yes, but note the local copy was exported from a HuggingFace parquet mirror rather than downloaded directly from the original OneDrive/Baidu source. |
| WHU-CD | Not always. Local split is 4504/536/2760, produced from the official WHU train/test archive with a local 10% validation split. Some papers use 6096/762/762 random split. | Yes. 256 x 256 patches. | Yes. P/R/F1/IoU/OA from binary confusion matrix. | Not directly comparable to papers using 6096/762/762; comparable only under the same split protocol or as a unified local setting. |

## Metric Implementation

The current code in `src/wetland_cd/training/metrics.py` computes:

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
IoU       = TP / (TP + FP + FN)
OA        = (TP + TN) / (TP + FP + FN + TN)
```

The positive class is the changed class. This matches the common binary change
detection evaluation protocol for LEVIR-CD, WHU-CD, and SYSU-CD.

## Final Judgment

- LEVIR-CD: dataset protocol is suitable for direct benchmark comparison.
- SYSU-CD: dataset protocol is suitable for benchmark comparison after noting
  that the local source is a parquet mirror.
- WHU-CD: current data is valid for unified local experiments, but not safe for
  direct comparison with papers that use the 6096/762/762 random split. If WHU
  is used in the final paper table against published numbers, prepare an
  additional WHU export matching the target paper's split.

One more constraint remains independent of the dataset protocol: the current
local runs are reduced runs and several models are lightweight reimplementations.
Those results should not be described as official full reproductions.
