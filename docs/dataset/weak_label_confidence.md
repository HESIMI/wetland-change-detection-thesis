# Weak Label Confidence Screening

本阶段在初始弱监督变化图基础上进行标签质量筛选，目标是区分高置信样本、低置信样本和不确定区域，为后续噪声鲁棒训练与伪变化抑制提供数据基础。

## Inputs

输入数据包括：

- 初始弱监督变化标签：`data/weak_labels/initial_change/<area>/initial_change.tif`
- T1 土地覆盖图：`lc_t1_2018.tif`
- T2 土地覆盖图：`lc_t2_2022.tif`
- Sentinel-2 T1 影像：`data/raw/<area>/sentinel2_2018.tif`
- Sentinel-2 T2 影像：`data/raw/<area>/sentinel2_2022.tif`
- 伪变化候选图：`data/change_labels/<area>/pseudo_change_mask.tif`

## Multi-source Consistency

当前多源一致性由两类证据构成：

- 土地覆盖变化证据：由 GLC_FCS30D 两期土地覆盖差分得到。
- 光学影像变化证据：由 Sentinel-2 双时相影像计算得到。

Sentinel-2 光谱变化证据基于 NDVI、NDWI 与亮度差异构建：

```text
spectral_change = sqrt(delta_NDVI^2 + delta_NDWI^2 + delta_brightness^2)
```

其中 Sentinel-2 波段顺序为：

```text
B02, B03, B04, B08
```

光谱变化得分会被重采样到土地覆盖标签网格，并进行鲁棒归一化。若土地覆盖变化与较强光谱变化一致，或土地覆盖未变化与较弱光谱变化一致，则认为该像素具有多源一致性。

## Temporal Consistency

当前数据为 2018 与 2022 双时相设置，尚未引入第三时相。因此本阶段采用双时相同季节光谱稳定性作为时序一致性的代理指标：

- 土地覆盖发生变化，且 Sentinel-2 光谱变化显著，同时不属于伪变化候选，则判定为时序一致变化。
- 土地覆盖未变化，且 Sentinel-2 光谱变化较弱，则判定为时序一致未变化。

后续若补充第三时相或 Dynamic World 时间序列，可将该模块扩展为真正的多时相稳定性筛选。

## Confidence Masks

输出目录：

```text
D:/桌面/毕业论文/项目/data/weak_labels/confidence/<area>/
```

输出文件：

- `spectral_change_score.tif`: Sentinel-2 光谱变化得分，范围缩放为 `0-10000`。
- `multi_source_consistency.tif`: 多源一致性掩码。
- `temporal_consistency.tif`: 双时相时序一致性代理掩码。
- `high_confidence_mask.tif`: 高置信样本掩码。
- `low_confidence_mask.tif`: 低置信样本掩码。
- `confidence_score.tif`: 置信等级图，`0` 表示未选中，`1` 表示低置信，`2` 表示高置信。

高置信样本定义：

```text
变化样本：initial_change = 1, spectral_change high, pseudo_change = 0
未变化样本：initial_change = 0, spectral_change low
```

低置信样本定义：

```text
pseudo_change = 1
或 initial_change = 1 但 spectral_change low
或 initial_change = 0 但 spectral_change high
```

## Current Statistics

| Area ID | Initial Change Pixels | Pseudo-change Pixels | High-confidence Pixels | Low-confidence Pixels |
| --- | ---: | ---: | ---: | ---: |
| `chongming_dongtan` | 569271 | 181629 | 2652026 | 1979894 |
| `dongting_lake` | 1281122 | 446585 | 5106735 | 4212337 |
| `hangzhou_xixi` | 38450 | 1713 | 128381 | 97537 |
| `poyang_lake` | 433749 | 244256 | 862468 | 843036 |
| `qiantang_estuary` | 172990 | 78356 | 554228 | 404222 |
| `yellow_river_delta` | 1098496 | 440474 | 4945577 | 3399685 |

## Reproduction

```bash
python scripts/data_preparation/build_weak_label_confidence.py ^
  --initial-root D:/桌面/毕业论文/项目/data/weak_labels/initial_change ^
  --raw-root D:/桌面/毕业论文/项目/data/raw ^
  --change-label-root D:/桌面/毕业论文/项目/data/change_labels ^
  --output-root D:/桌面/毕业论文/项目/data/weak_labels/confidence
```

