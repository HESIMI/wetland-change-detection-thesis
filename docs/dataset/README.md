# 数据集与预处理说明

本目录记录硕士论文湿地弱监督变化检测实验的数据层建设情况。当前数据层重点服务三个问题：

- 土地覆盖产品差分推导标签不精确，需明确弱监督属性。
- 湿地季节性、水位波动、潮滩边界造成伪变化，需进行置信样本筛选。
- 复杂边界和破碎斑块识别困难，需保留语义变化和跨区域验证设置。

## 主要文档

| File | Description |
| --- | --- |
| `wetland_data_sources.md` | 自建湿地数据源、研究区和目录结构 |
| `initial_weak_labels.md` | GLC_FCS30D 差分生成初始弱标签的说明 |
| `esa_worldcover.md` | ESA WorldCover 2021 下载、对齐、重编码与一致性统计 |
| `weak_label_confidence.md` | 融合 Sentinel-2 与 ESA 证据后的置信样本筛选 |
| `data_quality_acceptance.md` | 按论文要求进行的数据质量验收与适配性评估 |
| `data_layer_status.md` | 当前数据层产出状态和统计表 |
| `label_construction_flow.md` | 标签构建流程图 |
| `public_datasets.md` | SECOND、HRSCD 等公开数据集说明 |

## 当前数据规模

| Split | Patch Count | Area Coverage |
| --- | ---: | --- |
| train | 2762 | `hangzhou_xixi`, `dongting_lake`, `yellow_river_delta` |
| val | 1036 | `qiantang_estuary`, `chongming_dongtan` |
| test | 427 | `poyang_lake` |

## 当前辅助数据

| Product | Role |
| --- | --- |
| Sentinel-2 2018/2022 | 双时相遥感影像与光谱变化证据 |
| GLC_FCS30D 2018/2022 | 初始弱监督变化标签来源 |
| ESA WorldCover 2021 | 多源一致性验证与置信度筛选辅助证据 |

后续若需要更扎实的时序一致性分析，可继续加入 Dynamic World 或更多年份影像/土地覆盖产品。
