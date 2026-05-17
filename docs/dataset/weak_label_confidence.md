# 弱监督标签置信度筛选

## 目标

当前湿地变化标签不是人工精标，而是由 GLC_FCS30D 两期土地覆盖产品差分推导得到，因此属于弱监督标签。置信度筛选的目标是把初始弱标签进一步划分为：

- 高置信变化样本
- 高置信未变化样本
- 低置信或不确定样本

该步骤用于缓解土地覆盖产品分类误差、湿地季节性水位波动、潮滩/水体边界变化和破碎斑块带来的伪变化问题。

## 输入数据

```text
data/weak_labels/initial_change/<area>/
  initial_change.tif
  lc_t1_2018.tif
  lc_t2_2022.tif

data/raw/<area>/
  sentinel2_2018.tif
  sentinel2_2022.tif

data/change_labels/<area>/
  pseudo_change_mask.tif

data/esa_worldcover/<area>/
  esa_worldcover_2021_semantic.tif
  esa_glc_any_consistency.tif
```

其中 ESA WorldCover 为可选输入。若未提供 `--esa-root`，脚本仍按 GLC_FCS30D + Sentinel-2 光谱变化证据运行；若提供 ESA 结果，则将 ESA-GLC 一致性作为额外多源证据。

## 筛选逻辑

Sentinel-2 双时相影像用于计算光谱变化强度：

```text
spectral_change = sqrt(delta_NDVI^2 + delta_NDWI^2 + delta_brightness^2)
```

GLC_FCS30D 差分提供初始变化/未变化判断，Sentinel-2 提供同季节光谱变化证据，ESA WorldCover 2021 提供独立土地覆盖产品的一致性证据。

高置信变化样本定义为：

```text
initial_change = 1
spectral_change = high
pseudo_change = 0
ESA 与 GLC_T1 或 GLC_T2 至少一期语义一致
```

高置信未变化样本定义为：

```text
initial_change = 0
spectral_change = low
ESA 与 GLC_T1 或 GLC_T2 至少一期语义一致
```

低置信样本包括：

```text
pseudo_change = 1
或 initial_change = 1 但 spectral_change = low
或 initial_change = 0 但 spectral_change = high
或 ESA 有效但与 GLC_T1/GLC_T2 均不一致
```

## 输出文件

```text
D:/桌面/毕业论文/项目/data/weak_labels/confidence/<area>/
```

| File | Description |
| --- | --- |
| `spectral_change_score.tif` | Sentinel-2 光谱变化得分，缩放至 `0-10000` |
| `multi_source_consistency.tif` | GLC、Sentinel-2 与 ESA 融合后的一致性掩码 |
| `temporal_consistency.tif` | 双时相同季节光谱稳定性/变化代理指标 |
| `high_confidence_mask.tif` | 高置信样本总掩码 |
| `high_confidence_change.tif` | 高置信变化样本 |
| `high_confidence_unchanged.tif` | 高置信未变化样本 |
| `low_confidence_mask.tif` | 低置信或不确定样本 |
| `confidence_score.tif` | 置信等级图，`0` 未选中，`1` 低置信，`2` 高置信 |

## 当前统计

| Area ID | High-confidence Change | High-confidence Unchanged | Low-confidence | ESA Used |
| --- | ---: | ---: | ---: | --- |
| `chongming_dongtan` | 46183 | 407458 | 6097369 | yes |
| `dongting_lake` | 56855 | 1051048 | 12067369 | yes |
| `hangzhou_xixi` | 4143 | 93533 | 201031 | yes |
| `poyang_lake` | 32580 | 288771 | 1813394 | yes |
| `qiantang_estuary` | 7835 | 63618 | 1356471 | yes |
| `yellow_river_delta` | 49998 | 1229012 | 9139286 | yes |

ESA 融合后，高置信样本更保守，低置信区域显著增加。该结果符合当前论文设定：弱标签更适合通过置信样本筛选、噪声鲁棒训练或样本加权使用，而不应被直接视为人工精标。

## 复现命令

```powershell
python scripts/data_preparation/build_weak_label_confidence.py `
  --initial-root D:/桌面/毕业论文/项目/data/weak_labels/initial_change `
  --raw-root D:/桌面/毕业论文/项目/data/raw `
  --change-label-root D:/桌面/毕业论文/项目/data/change_labels `
  --output-root D:/桌面/毕业论文/项目/data/weak_labels/confidence `
  --esa-root D:/桌面/毕业论文/项目/data/esa_worldcover
```
