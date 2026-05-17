# Wetland Data Sources

本文自建湿地弱监督变化检测数据集以 Sentinel-2 双时相影像和 GLC_FCS30D 年度土地覆盖产品为基础，当前已覆盖 6 个典型湿地区域，并完成原始数据整理、变化标签构建与 patch 切片。

## Study Areas

| Area ID | 中文名称 | 湿地类型 | 实验定位 |
| --- | --- | --- | --- |
| `hangzhou_xixi` | 杭州西溪湿地 | 城市湿地 / 河网湿地 | 小尺度湿地斑块与城市背景干扰 |
| `qiantang_estuary` | 钱塘江口 | 河口湿地 / 潮滩湿地 | 水体、潮滩与湿地之间的伪变化分析 |
| `poyang_lake` | 鄱阳湖 | 湖泊湿地 | 跨区域测试与季节水位波动分析 |
| `dongting_lake` | 洞庭湖 | 湖泊湿地 | 大型湖泊湿地训练样本补充 |
| `yellow_river_delta` | 黄河三角洲 | 滨海湿地 / 三角洲湿地 | 滨海湿地与土地开发变化分析 |
| `chongming_dongtan` | 崇明东滩 | 滨海湿地 / 滩涂湿地 | 河口滩涂与滨海湿地样本补充 |

当前区域组合覆盖湖泊湿地、河口湿地、城市湿地、滨海湿地和滩涂湿地，满足开题阶段 3 到 6 个研究区的设计要求。

## Image List

每个研究区当前均包含 2018 年与 2022 年两个时相：

| Area ID | Sentinel-2 T1 | Sentinel-2 T2 | GLC_FCS30D T1 | GLC_FCS30D T2 |
| --- | --- | --- | --- | --- |
| `hangzhou_xixi` | `sentinel2_2018.tif` | `sentinel2_2022.tif` | `glc_2018.tif` | `glc_2022.tif` |
| `qiantang_estuary` | `sentinel2_2018.tif` | `sentinel2_2022.tif` | `glc_2018.tif` | `glc_2022.tif` |
| `poyang_lake` | `sentinel2_2018.tif` | `sentinel2_2022.tif` | `glc_2018.tif` | `glc_2022.tif` |
| `dongting_lake` | `sentinel2_2018.tif` | `sentinel2_2022.tif` | `glc_2018.tif` | `glc_2022.tif` |
| `yellow_river_delta` | `sentinel2_2018.tif` | `sentinel2_2022.tif` | `glc_2018.tif` | `glc_2022.tif` |
| `chongming_dongtan` | `sentinel2_2018.tif` | `sentinel2_2022.tif` | `glc_2018.tif` | `glc_2022.tif` |

Sentinel-2 影像采用生长季低云量影像合成结果，GLC_FCS30D 标签从年度土地覆盖瓦片中裁剪得到。当前数据为双时相设置，后续若需要强化时序一致性筛选，可继续补充中间年份或第三时相影像。

## GLC_FCS30D Tiles

本地已下载以下年度土地覆盖瓦片：

| File | 用途 |
| --- | --- |
| `GLC_FCS30D_20002022_E120N35_Annual.tif` | 杭州西溪湿地、钱塘江口、崇明东滩相关区域 |
| `GLC_FCS30D_20002022_E115N30_Annual.tif` | 鄱阳湖相关区域 |
| `GLC_FCS30D_20002022_E110N30_Annual.tif` | 洞庭湖相关区域 |
| `GLC_FCS30D_20002022_E115N40_Annual.tif` | 黄河三角洲相关区域 |

## Current Folder Structure

本地湿地数据主目录：

```text
D:/桌面/毕业论文/项目/data/
  raw_glc_fcs30d/
    GLC_FCS30D_20002022_*.tif
  raw/
    <area>/
      sentinel2_2018.tif
      sentinel2_2022.tif
      glc_2018.tif
      glc_2022.tif
  glc_subsets/
    <area>/
      glc_2018.tif
      glc_2022.tif
  change_labels/
    <area>/
      binary_change_raw.tif
      binary_change_final.tif
      pseudo_change_mask.tif
      semantic_change.tif
      glc_2018_semantic.tif
      glc_2022_semantic.tif
      wetland_mask_2018.tif
      wetland_mask_2022.tif
  processed/
    train/
    val/
    test/
    dataset_manifest.csv
```

## Processed Patch Statistics

| Split | Patch Count |
| --- | ---: |
| train | 2762 |
| val | 1036 |
| test | 427 |

| Area ID | Patch Count |
| --- | ---: |
| `dongting_lake` | 2492 |
| `chongming_dongtan` | 856 |
| `poyang_lake` | 427 |
| `yellow_river_delta` | 189 |
| `qiantang_estuary` | 180 |
| `hangzhou_xixi` | 81 |

## Optional Auxiliary Products

ESA WorldCover 与 Dynamic World 尚未纳入当前本地数据目录。两者适合作为后续弱监督标签置信度筛选的辅助来源：

- ESA WorldCover: 可用于与 GLC_FCS30D 进行多源一致性检查。
- Dynamic World: 可用于时间序列一致性分析与高低置信样本划分。

当前阶段已经完成基于 Sentinel-2 与 GLC_FCS30D 的基础湿地数据源构建。辅助产品属于下一阶段弱监督标签质量提升内容。

