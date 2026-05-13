# 湿地变化检测相关模型调研（2021-2025）

更新时间：`2026-05-13`

## 1. 调研目的

本目录汇总近几年与湿地变化检测相关的遥感变化检测模型，目标在于为后续主模型选择与结构设计提供依据。调研重点围绕以下四个方面展开：

- 高分辨率遥感影像下的计算效率
- 湿地场景中伪变化抑制能力
- 细碎斑块与复杂边界保持能力
- 跨区域泛化与语义先验利用能力

调研范围包括：

- 二值变化检测（BCD）
- 语义变化检测（SCD）
- Vision-Language 变化检测
- Foundation Model 适配路线

## 2. 主要技术路线

### 2.1 状态空间建模路线

代表模型：

- `ChangeMamba`
- `CDMamba`
- `CD-Lamba`
- `M-CD`

该路线的共同特点是利用状态空间模型获得更好的长程依赖建模效率，在大尺寸遥感影像和双时相输入场景中具有较好的计算优势。当前不足主要体现在语义先验较弱，对湿地中的自然波动型伪变化约束有限。

### 2.2 Transformer / ViT 路线

代表模型：

- `BIT`
- `ChangeFormer`
- `RCDT`
- `VcT`
- `ChangeViT`
- `STeInFormer`

该路线已经形成较成熟的对照体系，具备较强的全局上下文建模能力。其中，`ChangeViT` 与 `STeInFormer` 是当前最值得关注的非 Mamba 主干候选。

### 2.3 对象级解码与边界增强路线

代表模型：

- `MaskCD`
- `SARAS-Net`

该类方法更关注对象级变化、边界一致性与尺度敏感性，对湿地中常见的复杂边缘和破碎斑块具有较高参考价值。

### 2.4 语义与视觉语言路线

代表模型：

- `ChangeCLIP`
- `TextSCD`
- `Semantic-CD`
- `ClearSCD`
- `LSC-CD`
- `RemoteCLIP`
- `GeoRSCLIP`

该路线通过文本提示、开放词汇表达或遥感领域视觉语言预训练模型引入外部语义先验，在抑制伪变化、增强跨区域泛化方面具有明显潜力。

### 2.5 基础模型适配路线

代表模型：

- `BAN`
- `RoCD`
- `AdaptVFMs-RSCD`
- `FMT`
- `Changen2`

该路线关注基础视觉模型或多模态模型的迁移适配，更适用于泛化、零样本与预训练扩展问题，工程复杂度相对较高。

## 3. 综合比较

### 3.1 状态空间模型

优点：

- 计算复杂度较低
- 适合高分辨率遥感输入
- 长程依赖建模能力较强

局限：

- 局部细节保持能力依赖额外模块
- 对湿地伪变化的语义约束不足

### 3.2 Transformer / ViT

优点：

- 全局上下文建模成熟
- 对照体系完整，易于开展横向比较
- 社区复现资源相对丰富

局限：

- 计算与显存开销通常高于同规模状态空间模型
- 针对边界与小斑块问题常需额外设计

### 3.3 Vision-Language

优点：

- 可显式引入语义先验
- 适合误检抑制与开放类别泛化
- 对语义变化检测扩展友好

局限：

- 训练与推理成本更高
- prompt 设计和类别体系适配成本较高

### 3.4 Mask / 对象级解码

优点：

- 对边界和对象完整性更友好
- 更适合复杂纹理与碎片化区域

局限：

- 结构与训练过程较复杂
- 与现有二值变化检测框架整合成本较高

## 4. 与湿地任务的关联

湿地变化检测的难点主要表现为：

- 水体、潮滩与湿地之间的季节性或水位波动
- 破碎斑块与复杂边界
- 跨区域场景差异明显

因此，不同路线在本任务中的潜在作用可概括如下：

- `ChangeMamba / CD-Lamba / CDMamba`：适合作为高效主干
- `ChangeViT / STeInFormer`：适合作为强 Transformer 对照与主干候选
- `MaskCD`：适合作为边界与对象级解码参考
- `ChangeCLIP / TextSCD / Semantic-CD`：适合作为语义增强或语义扩展参考
- `BAN / RoCD`：适合作为基础模型适配路线的补充实验方向

## 5. 阶段性结论

综合模型能力、公开代码可得性与当前课题需求，主模型筛选应重点关注以下方向：

### 主干候选

- `ChangeMamba`
- `CDMamba`
- `CD-Lamba`
- `ChangeViT`
- `STeInFormer`

### 语义增强候选

- `ChangeCLIP`
- `LSC-CD`
- `TextSCD`
- `Semantic-CD`

### 解码头与边界增强候选

- `MaskCD`

### 基础模型路线候选

- `BAN`
- `RoCD`

当前阶段的结论并非预先固定某一模型名称，而是保留以下四类结构作为可比候选：

- `Mamba / SSM` 主干
- `Transformer / ViT` 主干
- `Mask / 对象级解码`
- `Vision-Language / Foundation Model` 语义增强

## 6. 本目录文件

- [models.csv](./models.csv)
  - 模型结构化信息表
- [open_source_shortlist.md](./open_source_shortlist.md)
  - 已核实公开代码仓库的模型列表
- [references.bib](./references.bib)
  - 参考文献 BibTeX
- [selection_notes.md](./selection_notes.md)
  - 主模型筛选说明
