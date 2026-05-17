# ESA WorldCover 2021 多源一致性验证

## 数据定位

ESA WorldCover 2021 v200 用于作为 GLC_FCS30D 之外的独立土地覆盖产品，辅助评估弱监督变化标签的可信度。该数据不作为人工精标真值，而是作为多源一致性证据，用于识别更可信的变化/未变化样本，以及提示可能由分类体系差异、湿地水位波动或潮滩边界造成的不确定区域。

官方说明显示，WorldCover 2021 v200 为 10 m 全球土地覆盖产品，采用 EPSG:4326，经 3° x 3° Cloud Optimized GeoTIFF 瓦片发布，可通过公开 S3 bucket 获取。

## 本地目录

```text
D:/桌面/毕业论文/项目/data/
  raw_esa_worldcover/
    tiles/
      ESA_WorldCover_10m_2021_v200_N27E111_Map.tif
      ESA_WorldCover_10m_2021_v200_N27E114_Map.tif
      ESA_WorldCover_10m_2021_v200_N30E120_Map.tif
      ESA_WorldCover_10m_2021_v200_N36E117_Map.tif
  esa_worldcover/
    <area>/
      esa_worldcover_2021_raw_aligned.tif
      esa_worldcover_2021_semantic.tif
      glc_2018_semantic_for_esa.tif
      glc_2022_semantic_for_esa.tif
      esa_glc_t1_consistency.tif
      esa_glc_t2_consistency.tif
      esa_glc_any_consistency.tif
      esa_consistency_score.tif
    esa_worldcover_consistency_summary.json
```

## 类别重编码

为与当前 GLC_FCS30D 弱标签体系对齐，ESA WorldCover 原始类别被重编码为统一语义类别：

| ESA WorldCover | 统一类别 |
| ---: | --- |
| 10 Tree cover | forest |
| 20 Shrubland | shrubland |
| 30 Grassland | grassland |
| 40 Cropland | cropland |
| 50 Built-up | built_up |
| 60 Bare/sparse vegetation | bare_land |
| 70 Snow and ice | snow_ice |
| 80 Permanent water bodies | water |
| 90 Herbaceous wetland | wetland |
| 95 Mangroves | wetland |
| 100 Moss and lichen | grassland |

GLC_FCS30D 类别同步重编码到 `cropland / forest / shrubland / grassland / water / wetland / built_up / bare_land / snow_ice`，之后逐像素生成 ESA 与 GLC_T1、GLC_T2 的一致性图层。

## 一致性统计

| Area ID | ESA Tile | ESA-GLC T1 Ratio | ESA-GLC T2 Ratio | Any Consistent Pixels |
| --- | --- | ---: | ---: | ---: |
| `chongming_dongtan` | `N30E120` | 0.1735 | 0.1724 | 1290214 |
| `dongting_lake` | `N27E111` | 0.1821 | 0.1812 | 2703148 |
| `hangzhou_xixi` | `N30E120` | 0.4560 | 0.4559 | 170308 |
| `poyang_lake` | `N27E114` | 0.4699 | 0.4689 | 1175959 |
| `qiantang_estuary` | `N30E120` | 0.0770 | 0.0781 | 119553 |
| `yellow_river_delta` | `N36E117` | 0.1836 | 0.1863 | 1965886 |

河口、滨海和大湖湿地区域的一致性比例偏低，说明 ESA 与 GLC 在水体、湿地、潮滩、草地/裸地等边界类别上存在明显体系差异。因此本文将 ESA 作为弱监督置信度的辅助筛选证据，而不是直接替代 GLC_FCS30D 或人工标注。

## 复现命令

```powershell
python scripts/data_preparation/prepare_esa_worldcover.py `
  --raw-root D:/桌面/毕业论文/项目/data/raw `
  --tile-dir D:/桌面/毕业论文/项目/data/raw_esa_worldcover/tiles `
  --output-root D:/桌面/毕业论文/项目/data/esa_worldcover
```

随后在弱标签置信度构建中加入 ESA 证据：

```powershell
python scripts/data_preparation/build_weak_label_confidence.py `
  --initial-root D:/桌面/毕业论文/项目/data/weak_labels/initial_change `
  --raw-root D:/桌面/毕业论文/项目/data/raw `
  --change-label-root D:/桌面/毕业论文/项目/data/change_labels `
  --output-root D:/桌面/毕业论文/项目/data/weak_labels/confidence `
  --esa-root D:/桌面/毕业论文/项目/data/esa_worldcover
```

## 数据引用

- ESA WorldCover data access: <https://esa-worldcover.org/en/data-access>
- WorldCover 2021 v200 DOI: <https://doi.org/10.5281/zenodo.7254221>
