# 数据集与预处理说明

## 1. 数据目标

本课题的数据准备不是简单的“下载与裁剪”，而是面向湿地变化检测模型设计的数据构建，核心包括：

- 双时相遥感影像配对
- 湿地变化标签构建
- 语义变化编码生成
- 面向模型训练的 patch 切片

## 2. 数据来源

### 2.1 土地覆盖标签

- 数据集：`GLC_FCS30D`
- 类型：年度土地覆盖像素级分类产品
- 用途：作为 `2018` 与 `2022` 两个时相的土地覆盖参考图，用于自动构建变化标签

### 2.2 遥感影像

- 数据集：`Sentinel-2 L2A`
- 来源：Microsoft Planetary Computer 平台公开数据
- 时间范围：生长季 `4-10 月`
- 处理方式：筛选低云量影像后生成双时相合成图

## 3. 研究区域

当前已构建 6 个典型湿地区域：

1. `hangzhou_xixi`
2. `qiantang_estuary`
3. `poyang_lake`
4. `dongting_lake`
5. `yellow_river_delta`
6. `chongming_dongtan`

这些区域覆盖：

- 内陆湖泊湿地
- 河口湿地
- 滨海湿地
- 城市近郊湿地

有助于后续验证跨区域泛化能力。

## 4. 标签构建流程

### 4.1 原始年度标签提取

从 `GLC_FCS30D` 原始年度瓦片中提取：

- `2018`
- `2022`

并裁剪到各研究区范围。

### 4.2 二值变化标签

通过逐像素比较 `2018` 与 `2022` 的类别差异，生成：

- 原始变化图 `binary_change_raw`
- 最终变化图 `binary_change_final`

### 4.3 伪变化掩码

考虑到湿地场景中存在明显的季节性和水位波动，本项目额外构建：

- `pseudo_change_mask`

用于标记 `水体 ↔ 湿地`、`湿地 ↔ 潮滩` 等更可能由自然波动引起的变化候选区域。

### 4.4 语义变化标签

除二值变化外，还保留语义变化编码：

- `semantic_change = source_class * 1000 + target_class`

便于后续引入：

- 文本提示词
- 语义对齐损失
- 语义变化检测扩展实验

## 5. 数据切片与样本规模

为了适配深度学习训练，将大幅面图像切片为固定大小 patch，形成训练样本集。

当前已生成样本总数：

- 总样本数：`4225`
- 训练集：`2762`
- 验证集：`1036`
- 测试集：`427`

按区域统计如下：

| 区域 | patch 数 |
| --- | ---: |
| `dongting_lake` | 2492 |
| `chongming_dongtan` | 856 |
| `poyang_lake` | 427 |
| `yellow_river_delta` | 189 |
| `qiantang_estuary` | 180 |
| `hangzhou_xixi` | 81 |

## 6. 变化标签统计特征

当前多区域变化统计显示：

- `poyang_lake`、`qiantang_estuary`、`chongming_dongtan` 等区域中伪变化比例较高
- 这为“语义引导抑制误检”提供了明确实验动机

示例：

- `hangzhou_xixi`
  - raw changed pixels: `38479`
  - final changed pixels: `36766`
  - pseudo change pixels: `1713`

- `qiantang_estuary`
  - raw changed pixels: `172990`
  - final changed pixels: `94634`
  - pseudo change pixels: `78356`

- `poyang_lake`
  - raw changed pixels: `433749`
  - final changed pixels: `189493`
  - pseudo change pixels: `244256`

## 7. 当前数据阶段结论

当前数据集已经满足以下要求：

- 可支撑 baseline 训练
- 可支撑跨区域泛化实验
- 可支撑语义引导类模型设计
- 可支撑后续主模型筛选实验

因此，当前论文已具备从“数据准备阶段”进入“模型筛选与主实验阶段”的基础。
