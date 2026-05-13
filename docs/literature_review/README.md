# 湿地变化检测模型调研（2021-2025）

更新时间：`2026-05-13`

本目录用于整理近几年适合“湿地变化检测 / 遥感变化检测”方向的前沿模型，重点服务于后续的**主模型选择**与**模型融合设计**。当前目标不是预设 `Mamba-CLIP` 必然为最终主模型，而是先比较不同技术路线在以下问题上的适配性：

- 高分辨率遥感影像的计算效率
- 湿地场景中的伪变化抑制能力
- 细碎斑块与复杂边界的尺度感知能力
- 跨区域泛化与弱监督/语义先验利用能力

## 1. 调研结论先看

### 最值得优先实验的三条路线

1. **`ChangeMamba / CDMamba / CD-Lamba` 路线**
   - 优势：效率高、长程依赖建模强、适合高分辨率双时相输入。
   - 风险：纯视觉建模，对“湿地-水体-潮滩”伪变化的语义约束仍偏弱。
   - 判断：**最适合作为主干 backbone 候选。**

2. **`ChangeCLIP / TextSCD / Semantic-CD` 路线**
   - 优势：能显式引入语义和文本先验，天然适合“伪变化抑制”和开放类别泛化。
   - 风险：训练与推理开销更高，对 prompt 设计和类别体系更敏感。
   - 判断：**最适合作为语义增强分支，而不建议直接单独做唯一主模型。**

3. **`MaskCD` 路线**
   - 优势：对象级 mask 解码对边界和碎片区域更友好。
   - 风险：结构相对复杂，训练成本高于普通二值解码器。
   - 判断：**适合作为“解码头设计灵感”，尤其可补边界质量。**

### 当前最稳妥的主模型候选顺序

1. **`ChangeMamba` 系列主干 + 语义先验增强**
2. **`CDMamba / CD-Lamba` 这类“局部细节增强版 Mamba”**
3. **`MaskCD` 式解码头 + 高效 backbone**
4. **直接沿用 `ChangeCLIP` 作为主模型**

我的当前判断是：

> 如果论文目标是“在湿地变化检测上稳定拿到可解释的指标提升”，最稳的路线并不是直接把 `ChangeCLIP` 当主模型，而是**先选一个成熟的 Mamba 系主干，再叠加轻量语义引导或 prompt 模块**。

---

## 2. 快速对比表

> 说明：不同论文使用的数据集不同，指标不宜直接横向当成严格排行榜。这里的“结果亮点”仅用于判断其是否具备明确实验增益。

| 模型 | 年份 | 路线 | 核心思想 | 结果亮点（论文原文/摘要级） | 优点 | 局限 | 对湿地任务判断 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `SNUNet-CD` | 2021 | CNN Siamese | 基于 NestedUNet 的密集连接双分支变化检测 | 论文与代码均称相对多种 SOTA 指标提升明显 | 结构稳、易复现、边界保留较好 | 全局上下文建模弱，难抑制语义伪变化 | 适合做传统 CNN 基线 |
| `BIT` | 2021 | Transformer | 用少量 semantic tokens 建模双时相时空上下文 | 相比纯 CNN baseline，以更低计算成本取得更好精度 | 开启了遥感 CD 的 Transformer 路线，效率意识强 | 仍偏重全局建模，对局部细节需额外设计 | 适合做经典 Transformer 基线 |
| `ChangeFormer` | 2022 | Siamese Transformer | 分层 Transformer encoder + MLP decoder | 在两套数据集上优于以往方法 | 多尺度长程建模较强，社区使用广 | 参数和算力成本高于 CNN | 适合作为强基线，但不是最优效率路线 |
| `SARAS-Net` | 2023 | CNN/Attention | 关系感知 + 尺度感知 + cross-transformer | 在 LEVIR-CD / WHU-CD / DSIFN 上取得 SOTA 准确率 | 更关注尺度变化与双时相交互 | 仍以视觉关系建模为主，缺语义先验 | 对湿地碎片斑块有参考价值 |
| `MaskCD` | 2024 | Mask Classification | 用 mask classification 替代逐像素分类 | 作为首个该范式的 RSCD 模型，强调边界和对象完整性 | 对复杂边界、对象级变化更友好 | 训练复杂度更高，迁移到湿地需验证 | 很适合作为“解码头设计”参考 |
| `RS-Mamba` | 2024 | Visual Mamba | 多方向 selective scan，引入线性复杂度状态空间建模 | 在大幅面遥感 dense prediction 中达到 SOTA | 为遥感引入 Mamba，效率和大视野兼顾 | 不是专门为变化检测双时相交互设计 | 适合作为 Mamba 先验与扫描策略参考 |
| `ChangeMamba` | 2024 | Spatiotemporal Mamba | 面向 BCD/SCD/BDA 的时空状态空间模型 | 在 5 个 benchmark 上优于 CNN/Transformer 方法 | 当前最有代表性的 Mamba-CD 基线，效率与精度平衡好 | 语义先验不足，易受伪变化干扰 | **强烈建议作为主 baseline 之一** |
| `ChangeCLIP` | 2024 | Vision-Language | 将 CLIP 文本提示与双时相视觉特征联合建模 | 论文指出其为首个将 CLIP 引入 RSCD 的方法 | 最适合做语义引导与泛化增强 | 结构较重，prompt 体系对任务敏感 | **适合做语义分支，不一定适合作唯一主干** |
| `M-CD` | 2025 | Siamese Mamba | WACV 版本的 Mamba 双分支变化检测器 | 在 4 个常用数据集上显著优于现有 SOTA | Mamba 在通用 RSCD 上的强实现，适合复现实验 | 偏通用变化检测，未专门抑制湿地伪变化 | 值得纳入备选主模型 |
| `CDMamba` | 2025 | Local+Global Mamba | Mamba 全局建模 + 卷积局部细节增强 | 在 LEVIR-CD / CLCD 上 F1/IoU 分别提升 `2.10/3.00` 和 `2.44/2.91` | 直接补足 Mamba 的局部细节短板 | 仍是纯视觉路线 | **很适合湿地细碎边界场景** |
| `CD-Lamba` | 2025 | Locally Adaptive Mamba | 局部自适应扫描 + 跨时相扫描 + 窗口交互 | 在 4 个 benchmark 上实现更优的效率-精度平衡 | 对“变化区域连续性”建模更强 | 新方法，复现生态还不如 ChangeMamba 成熟 | **非常适合做你后续融合主干候选** |
| `TextSCD` | 2025 | Text-guided SCD | 用文本描述引导语义变化检测 | 在 SECOND 上带来显著 OA 改善 | 文本能直接约束“变化是什么” | 偏 SCD 任务，需从多类语义迁到湿地二值/语义变化 | 适合作为“文本引导模块”参考 |
| `Semantic-CD` | 2025 | Open-vocabulary SCD | 借助 CLIP 的开放词汇语义做二值+语义解耦 | 在 SECOND 上减少语义分类错误、掩码更准确 | 泛化能力强，适合开放类别场景 | 更偏语义变化检测，训练复杂度高 | 对后续开放词汇湿地变化很有价值 |

---

## 3. 推荐阅读顺序

### A. 主干骨架类

先读这些，决定最终 backbone：

1. `ChangeMamba`
2. `CDMamba`
3. `CD-Lamba`
4. `M-CD`
5. `RS-Mamba`

### B. 语义增强类

这些决定是否要保留 `CLIP / Prompt / Open-vocabulary` 路线：

1. `ChangeCLIP`
2. `TextSCD`
3. `Semantic-CD`
4. `RemoteCLIP`
5. `GeoRSCLIP`

### C. 边界与解码头类

这些决定是否要引入更强的 change decoder：

1. `MaskCD`
2. `SARAS-Net`
3. `ChangeFormer`
4. `BIT`

---

## 4. 对你这个湿地课题的具体启发

### 4.1 伪变化问题

湿地任务里的核心难点不是“看见差异”，而是区分：

- 季节性水位波动
- 潮滩/浅水/湿地之间的自然迁移
- 人工建设、围垦、永久转化

因此单纯视觉 backbone 往往不够。语义引导是必要的，但不必一步到位做重型多模态主模型。更合适的方式是：

- 用 `ChangeMamba / CD-Lamba` 做主干
- 用 `RemoteCLIP / ChangeCLIP / TextSCD` 提供语义先验
- 用轻量 prompt 或语义对齐 loss 进行约束

### 4.2 细小湿地斑块与复杂边界

这个问题最值得吸收的不是 `ChangeCLIP`，而是：

- `CDMamba` 的局部-全局结合
- `MaskCD` 的对象级 mask 解码
- `SARAS-Net` 的尺度与关系建模

### 4.3 跨区域泛化

泛化更可能来自以下几类机制：

- 开放词汇或文本先验：`Semantic-CD` / `TextSCD`
- 遥感领域专用 VLM：`RemoteCLIP` / `GeoRSCLIP`
- 更鲁棒的局部-全局扫描：`CD-Lamba`

---

## 5. 当前建议的模型选择策略

## 方案 A：最稳妥

**主干：`ChangeMamba` 或 `CD-Lamba`**

**增强：轻量语义引导**

- 文本提示只服务于“抑制伪变化”
- 不把整套 `ChangeCLIP` 原样搬进来
- 适合论文做出稳定指标提升

适用原因：

- 保留 Mamba 的效率优势
- 语义模块只做增量，不把训练复杂度拉太高
- 更适合你现在已经做好的湿地数据集和 baseline 流程

## 方案 B：更有创新性

**主干：`CDMamba / CD-Lamba`**

**解码：借鉴 `MaskCD` 或对象级边界增强模块**

**语义：接 `RemoteCLIP / GeoRSCLIP` 的文本向量**

适用原因：

- 兼顾局部细节、全局上下文和语义先验
- 很适合写成“结构化融合创新”
- 风险是工程复杂度更高

## 方案 C：不建议直接作为唯一主线

**直接以 `ChangeCLIP` 为唯一主模型**

原因：

- 多模态训练和 prompt 设计成本较高
- 对湿地标签体系的适配需要额外工作
- 你当前数据链路和 baseline 已经偏向标准变化检测框架，直接切过去代价偏大

---

## 6. 推荐最终纳入正式对比实验的模型

### 必须纳入

- `SNUNet-CD`
- `BIT`
- `ChangeFormer`
- `ChangeMamba`
- `ChangeCLIP`

### 如果算力和时间允许，强烈建议增加

- `CDMamba`
- `CD-Lamba`
- `MaskCD`

### 若后续要做语义变化扩展实验

- `TextSCD`
- `Semantic-CD`

---

## 7. 非 Mamba 前沿路线补充

为了避免主模型选择过早收敛到 `Mamba-CLIP`，这里补充当前最值得关注的非 Mamba 前沿方向。

### 7.1 强 Transformer 主干

这些模型可以直接和 Mamba 系竞争主模型地位：

- `STeInFormer`
  - 专门面向 RSCD 设计的时空交互 Transformer backbone。
  - 强项是双时相交互和频域 token mixing。
  - 判断：**非常适合当“非 Mamba 主模型候选”。**

- `ChangeViT`
  - 证明 plain ViT 在遥感变化检测上仍然有很强潜力。
  - 更擅长大尺度变化，但需额外模块补局部细节。
  - 判断：**非常适合当“强 Transformer 主干基线”。**

- `RCDT`
  - 以 relational cross-attention 为核心，结构较简洁。
  - 判断：**适合做计算成本较低的 Transformer 对照。**

- `VcT`
  - 较成熟的视觉 Transformer 变化检测模型。
  - 判断：**适合补一个非 Mamba 的公开视频代码基线。**

### 7.2 对象级/边界质量路线

- `MaskCD`
  - 对边界和碎片斑块最有帮助。
  - 如果你的湿地实验最后短板主要是复杂边缘、破碎块和对象完整性，那么它的重要性会非常高。
  - 判断：**最适合作为解码头设计灵感。**

### 7.3 语义与开放词汇路线

这些模型最适合解决湿地中的伪变化问题：

- `ChangeCLIP`
- `LSC-CD`
- `TextSCD`
- `Semantic-CD`
- `ClearSCD`

其中：

- `ChangeCLIP` 最适合直接借鉴 prompt 和视觉-文本对齐设计。
- `LSC-CD` 更轻，更适合做低成本语义增强。
- `TextSCD` 和 `Semantic-CD` 更适合后期扩展到语义变化检测。
- `ClearSCD` 适合作为语义变化关系建模参考。

### 7.4 Foundation Model 路线

这些模型不一定适合第一时间做主模型，但很值得保留为后续泛化实验和扩展方向：

- `BAN`
- `RoCD`
- `AdaptVFMs-RSCD`
- `FMT`
- `Changen2`

判断：

- `BAN`：适合做较稳的 foundation model 适配 baseline。
- `RoCD`：适合验证冻结视觉基础模型是否能提升泛化。
- `AdaptVFMs-RSCD`：更像语义变化检测和视觉基础模型结合的高级路线。
- `FMT`：偏探索。
- `Changen2`：更适合作为预训练或数据生成参考，而不是当前直接主模型。

---

## 8. 当前推荐的“主模型候选池”

如果不限定 Mamba，当前最值得优先复现和比对的候选池建议是：

### 第一梯队

- `ChangeMamba`
- `CDMamba`
- `CD-Lamba`
- `STeInFormer`
- `ChangeViT`
- `MaskCD`

### 第二梯队

- `ChangeCLIP`
- `RCDT`
- `BAN`
- `RoCD`

### 第三梯队

- `TextSCD`
- `Semantic-CD`
- `AdaptVFMs-RSCD`
- `Changen2`

---

## 9. 当前结论更新

当前并不能直接下结论说最终主模型一定是 `Mamba-CLIP`。

更合理的说法是：

> 最终主模型应从 `Mamba 主干`、`强 Transformer 主干`、`对象级解码路线`、`语义增强路线` 四类候选中，通过实验结果共同筛出。

---

## 10. 本目录文件说明

- [README.md](./README.md)
  - GitHub 首页式综述与选型建议
- [models.csv](./models.csv)
  - 可直接导入 Excel / pandas 的结构化模型表
- [open_source_shortlist.md](./open_source_shortlist.md)
  - 仅保留已核到公开代码仓库的可复现模型清单
- [references.bib](./references.bib)
  - BibTeX 参考文献
- [selection_notes.md](./selection_notes.md)
  - 面向论文落地的主模型选择建议

---

## 11. 直接结论

如果目标是尽快做出**论文可落地、指标可提升、结构可解释**的主模型，当前最建议优先验证：

1. **`ChangeMamba` 作为主干基线**
2. **`CDMamba / CD-Lamba` 作为更强 Mamba 变体**
3. **`STeInFormer / ChangeViT` 作为非 Mamba 主干对照**
4. **`RemoteCLIP / ChangeCLIP / TextSCD / LSC-CD` 只保留其语义引导思想**
5. **必要时借鉴 `MaskCD` 的解码头改善复杂边界**

因此，`Mamba-CLIP` 仍然是候选方案，但**更像一种融合方向，而不是当前唯一确定的最终主模型**。
