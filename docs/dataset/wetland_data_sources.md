# 湿地数据源说明

## 数据目标

自建湿地弱监督变化检测数据集以 Sentinel-2 双时相影像、GLC_FCS30D 年度土地覆盖产品和 ESA WorldCover 2021 独立土地覆盖产品为核心数据源。GLC_FCS30D 用于推导初始弱监督变化标签，ESA WorldCover 用于多源一致性验证和置信度筛选。

需要在论文中明确：当前湿地标签不是人工精标，而是土地覆盖产品差分推导标签，属于弱监督标签。

## 研究区

| Area ID | 中文名称 | 湿地类型 | 实验定位 |
| --- | --- | --- | --- |
| `hangzhou_xixi` | 杭州西溪湿地 | 城市湿地 / 河网湿地 | 小尺度湿地斑块与城市背景干扰 |
| `qiantang_estuary` | 钱塘江口 | 河口湿地 / 潮滩湿地 | 水体、潮滩与湿地之间的伪变化分析 |
| `poyang_lake` | 鄱阳湖 | 湖泊湿地 | 跨区域测试与季节水位波动分析 |
| `dongting_lake` | 洞庭湖 | 湖泊湿地 | 大型湖泊湿地训练样本补充 |
| `yellow_river_delta` | 黄河三角洲 | 滨海湿地 / 三角洲湿地 | 滨海湿地与土地开发变化分析 |
| `chongming_dongtan` | 崇明东滩 | 滨海湿地 / 滩涂湿地 | 河口滩涂与滨海湿地样本补充 |

## 影像与土地覆盖产品

每个研究区当前包含 2018 与 2022 两个时相：

| Area ID | Sentinel-2 T1 | Sentinel-2 T2 | GLC_FCS30D T1 | GLC_FCS30D T2 | ESA WorldCover |
| --- | --- | --- | --- | --- | --- |
| `hangzhou_xixi` | `sentinel2_2018.tif` | `sentinel2_2022.tif` | `glc_2018.tif` | `glc_2022.tif` | `esa_worldcover_2021_semantic.tif` |
| `qiantang_estuary` | `sentinel2_2018.tif` | `sentinel2_2022.tif` | `glc_2018.tif` | `glc_2022.tif` | `esa_worldcover_2021_semantic.tif` |
| `poyang_lake` | `sentinel2_2018.tif` | `sentinel2_2022.tif` | `glc_2018.tif` | `glc_2022.tif` | `esa_worldcover_2021_semantic.tif` |
| `dongting_lake` | `sentinel2_2018.tif` | `sentinel2_2022.tif` | `glc_2018.tif` | `glc_2022.tif` | `esa_worldcover_2021_semantic.tif` |
| `yellow_river_delta` | `sentinel2_2018.tif` | `sentinel2_2022.tif` | `glc_2018.tif` | `glc_2022.tif` | `esa_worldcover_2021_semantic.tif` |
| `chongming_dongtan` | `sentinel2_2018.tif` | `sentinel2_2022.tif` | `glc_2018.tif` | `glc_2022.tif` | `esa_worldcover_2021_semantic.tif` |

## 本地目录结构

```text
D:/桌面/毕业论文/项目/data/
  raw_glc_fcs30d/
    GLC_FCS30D_20002022_*.tif
  raw_esa_worldcover/
    tiles/
      ESA_WorldCover_10m_2021_v200_*.tif
  raw/
    <area>/
      sentinel2_2018.tif
      sentinel2_2022.tif
      glc_2018.tif
      glc_2022.tif
  esa_worldcover/
    <area>/
      esa_worldcover_2021_raw_aligned.tif
      esa_worldcover_2021_semantic.tif
      esa_glc_t1_consistency.tif
      esa_glc_t2_consistency.tif
      esa_glc_any_consistency.tif
      esa_consistency_score.tif
  change_labels/
    <area>/
      binary_change_raw.tif
      binary_change_final.tif
      pseudo_change_mask.tif
      semantic_change.tif
  weak_labels/
    initial_change/
    confidence/
  processed/
    train/
    val/
    test/
    dataset_manifest.csv
```

## 处理后切片统计

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

## 数据层结论

当前数据层已经具备三个层次的监督信息：

- GLC_FCS30D 差分生成的初始弱监督变化标签。
- Sentinel-2 双时相光谱变化证据，用于筛除光谱不支持的疑似变化。
- ESA WorldCover 2021 多源一致性证据，用于进一步标记高置信样本和低置信区域。

这为后续开展噪声鲁棒训练、样本加权、伪变化抑制和跨区域泛化实验提供了数据基础。若后续继续增强时序一致性，建议补充 Dynamic World 或更多年份土地覆盖/遥感影像产品。
