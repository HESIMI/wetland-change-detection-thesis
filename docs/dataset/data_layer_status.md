# Data Layer Status

## Completion Checklist

| Item | Status | Local Path |
| --- | --- | --- |
| 湿地训练集 | Completed | `D:/桌面/毕业论文/项目/data/processed/train` |
| 湿地验证集 | Completed | `D:/桌面/毕业论文/项目/data/processed/val` |
| 湿地测试集 | Completed | `D:/桌面/毕业论文/项目/data/processed/test` |
| 数据统计表 | Completed | `D:/桌面/毕业论文/项目/data/processed/dataset_manifest.csv` |
| 标签构建流程图 | Completed | `docs/dataset/label_construction_flow.md` |

## Dataset Split Statistics

| Split | Patch Count | Area Coverage |
| --- | ---: | --- |
| train | 2762 | `hangzhou_xixi`, `dongting_lake`, `yellow_river_delta` |
| val | 1036 | `qiantang_estuary`, `chongming_dongtan` |
| test | 427 | `poyang_lake` |

## Area Statistics

| Area ID | Patch Count |
| --- | ---: |
| `chongming_dongtan` | 856 |
| `dongting_lake` | 2492 |
| `hangzhou_xixi` | 81 |
| `poyang_lake` | 427 |
| `qiantang_estuary` | 180 |
| `yellow_river_delta` | 189 |

## Supporting Tables

| File | Description |
| --- | --- |
| `data/raw/sentinel_download_summary.json` | Sentinel-2 影像检索与合成记录 |
| `data/glc_subsets/clip_summary.json` | GLC_FCS30D 研究区裁剪统计 |
| `data/change_labels/change_summary.json` | 原始变化、伪变化与语义变化统计 |
| `data/weak_labels/initial_change/initial_weak_label_summary.json` | 初始弱监督变化标签统计 |
| `data/weak_labels/confidence/weak_label_confidence_summary.json` | 高低置信样本统计 |

