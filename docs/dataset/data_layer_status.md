# 数据层产出状态

## 完成清单

| Item | Status | Local Path |
| --- | --- | --- |
| 湿地训练集 | Completed | `D:/桌面/毕业论文/项目/data/processed/train` |
| 湿地验证集 | Completed | `D:/桌面/毕业论文/项目/data/processed/val` |
| 湿地测试集 | Completed | `D:/桌面/毕业论文/项目/data/processed/test` |
| 数据统计表 | Completed | `D:/桌面/毕业论文/项目/data/processed/dataset_manifest.csv` |
| 初始弱监督变化标签 | Completed | `D:/桌面/毕业论文/项目/data/weak_labels/initial_change` |
| ESA WorldCover 对齐与一致性图层 | Completed | `D:/桌面/毕业论文/项目/data/esa_worldcover` |
| ESA 融合后的弱标签置信度图层 | Completed | `D:/桌面/毕业论文/项目/data/weak_labels/confidence` |
| 标签构建流程说明 | Completed | `docs/dataset/label_construction_flow.md` |

## 数据集切片统计

| Split | Patch Count | Area Coverage |
| --- | ---: | --- |
| train | 2762 | `hangzhou_xixi`, `dongting_lake`, `yellow_river_delta` |
| val | 1036 | `qiantang_estuary`, `chongming_dongtan` |
| test | 427 | `poyang_lake` |

## 研究区切片统计

| Area ID | Patch Count |
| --- | ---: |
| `chongming_dongtan` | 856 |
| `dongting_lake` | 2492 |
| `hangzhou_xixi` | 81 |
| `poyang_lake` | 427 |
| `qiantang_estuary` | 180 |
| `yellow_river_delta` | 189 |

## ESA WorldCover 一致性统计

| Area ID | ESA-GLC T1 Ratio | ESA-GLC T2 Ratio | Any Consistent Pixels |
| --- | ---: | ---: | ---: |
| `chongming_dongtan` | 0.1735 | 0.1724 | 1290214 |
| `dongting_lake` | 0.1821 | 0.1812 | 2703148 |
| `hangzhou_xixi` | 0.4560 | 0.4559 | 170308 |
| `poyang_lake` | 0.4699 | 0.4689 | 1175959 |
| `qiantang_estuary` | 0.0770 | 0.0781 | 119553 |
| `yellow_river_delta` | 0.1836 | 0.1863 | 1965886 |

## ESA 融合后的置信样本统计

| Area ID | High-confidence Change | High-confidence Unchanged | Low-confidence |
| --- | ---: | ---: | ---: |
| `chongming_dongtan` | 46183 | 407458 | 6097369 |
| `dongting_lake` | 56855 | 1051048 | 12067369 |
| `hangzhou_xixi` | 4143 | 93533 | 201031 |
| `poyang_lake` | 32580 | 288771 | 1813394 |
| `qiantang_estuary` | 7835 | 63618 | 1356471 |
| `yellow_river_delta` | 49998 | 1229012 | 9139286 |

## 支撑文件

| File | Description |
| --- | --- |
| `data/raw/sentinel_download_summary.json` | Sentinel-2 影像检索与合成记录 |
| `data/glc_subsets/clip_summary.json` | GLC_FCS30D 研究区裁剪统计 |
| `data/change_labels/change_summary.json` | 原始变化、伪变化与语义变化统计 |
| `data/weak_labels/initial_change/initial_weak_label_summary.json` | 初始弱监督变化标签统计 |
| `data/esa_worldcover/esa_worldcover_consistency_summary.json` | ESA WorldCover 对齐与一致性统计 |
| `data/weak_labels/confidence/weak_label_confidence_summary.json` | 高低置信样本统计 |
