# Initial Weak Label Construction

本阶段构建湿地弱监督变化标签的第一层结果，仅基于两期土地覆盖产品进行逐像素差分，不引入伪变化抑制、置信度筛选或边界修正。

## Inputs

输入数据来自 GLC_FCS30D 裁剪结果：

```text
D:/桌面/毕业论文/项目/data/glc_subsets/<area>/
  glc_2018.tif
  glc_2022.tif
```

其中：

- `glc_2018.tif`: T1 土地覆盖图
- `glc_2022.tif`: T2 土地覆盖图

## Outputs

输出目录：

```text
D:/桌面/毕业论文/项目/data/weak_labels/initial_change/<area>/
  lc_t1_2018.tif
  lc_t2_2022.tif
  initial_change.tif
```

输出说明：

- `lc_t1_2018.tif`: 规范化保存的第一时相土地覆盖图。
- `lc_t2_2022.tif`: 规范化保存的第二时相土地覆盖图。
- `initial_change.tif`: 初始二值变化图，由两期土地覆盖逐像素差分得到。

初始变化图定义为：

```text
initial_change = 1, if lc_t1 != lc_t2 and both pixels are valid
initial_change = 0, otherwise
```

其中土地覆盖编码 `0` 被视为无效或背景像素，不参与变化像素统计。

## Current Summary

| Area ID | Changed Pixels | Valid Change Ratio |
| --- | ---: | ---: |
| `chongming_dongtan` | 569271 | 0.0811 |
| `dongting_lake` | 1281122 | 0.0912 |
| `hangzhou_xixi` | 38450 | 0.1109 |
| `poyang_lake` | 433749 | 0.1750 |
| `qiantang_estuary` | 172990 | 0.1197 |
| `yellow_river_delta` | 1098496 | 0.1043 |

## Relation to Later Labels

`initial_change.tif` 是弱监督标签构建的初始结果。后续步骤将在此基础上继续进行：

- 伪变化识别：例如水体、湿地和潮滩之间的自然波动。
- 高低置信样本划分：结合多源土地覆盖产品或时序稳定性。
- 边界不确定区域处理：降低土地覆盖产品边界误差对训练的影响。

旧目录 `data/change_labels` 中的 `binary_change_raw.tif` 与本阶段结果接近，但该旧结果未单独形成弱监督标签层。当前 `data/weak_labels/initial_change` 作为论文中“初始弱监督标签”的标准输出。

