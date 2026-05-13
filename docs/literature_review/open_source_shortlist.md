# 开源代码可得模型清单

更新时间：`2026-05-13`

说明：

- 本清单只保留**已核到公开代码仓库**的模型。
- 优先使用**官方仓库**；若无官方仓库但有稳定工具箱复现，则单独标注。
- 主模型优先从这里选，避免把时间投入到“论文有、代码没有/不完整”的路线。

## 1. 强烈推荐优先复现

### 主干候选

1. `ChangeMamba`
   - 路线：Mamba 主干
   - 代码状态：**官方开源**
   - 价值：当前最成熟的 Mamba 变化检测基线之一
   - 仓库：https://github.com/ChenHongruixuan/ChangeMamba
   - 论文：https://arxiv.org/abs/2404.03425

2. `ChangeViT`
   - 路线：Plain ViT 主干
   - 代码状态：**官方开源**
   - 价值：最值得和 Mamba 正面对比的非 Mamba 主干之一
   - 仓库：https://github.com/zhuduowang/ChangeViT
   - 论文：https://arxiv.org/abs/2406.12847

3. `MaskCD`
   - 路线：Mask Classification
   - 代码状态：**官方开源**
   - 价值：最适合检验边界和破碎斑块问题
   - 仓库：https://github.com/AI4RS/MaskCD

4. `ChangeCLIP`
   - 路线：Vision-Language
   - 代码状态：**官方开源**
   - 价值：最适合做语义引导基线
   - 仓库：https://github.com/dyzy41/ChangeCLIP
   - 论文：https://www.sciencedirect.com/science/article/pii/S0924271624000042

5. `BAN`
   - 路线：Foundation Model Adaptation
   - 代码状态：**官方开源**
   - 价值：如果后面要验证基础模型路线，这是最稳的起点
   - 仓库：https://github.com/likyoo/BAN
   - 论文：https://arxiv.org/abs/2312.01163

## 2. 推荐作为补充对照

1. `BIT`
   - 代码状态：**官方开源**
   - 仓库：https://github.com/justchenhao/BIT_CD
   - 作用：经典 Transformer 基线

2. `ChangeFormer`
   - 代码状态：**官方开源**
   - 仓库：https://github.com/wgcban/ChangeFormer
   - 作用：强 Transformer 对照

3. `SARAS-Net`
   - 代码状态：**官方开源**
   - 仓库：https://github.com/f64051041/SARAS-Net
   - 作用：边界与尺度建模参考

4. `SNUNet-CD`
   - 代码状态：**官方开源**
   - 仓库：https://github.com/likyoo/Siam-NestedUNet
   - 作用：传统 CNN 基线

5. `RCDT`
   - 代码状态：**公开实现可获取**
   - 仓库/论文入口：https://arxiv.org/abs/2212.04869
   - 作用：轻量 Transformer 对照

6. `VcT`
   - 代码状态：**官方开源**
   - 仓库：https://github.com/Event-AHU/VcT_Remote_Sensing_Change_Detection
   - 作用：补充 Transformer 对照

7. `ClearSCD`
   - 代码状态：**官方开源**
   - 仓库：https://github.com/tangkai-RS/ClearSCD
   - 作用：语义变化检测参考

8. `DDPM-CD`
   - 路线：Diffusion-based CD
   - 代码状态：**官方开源**
   - 仓库：https://github.com/wgcban/ddpm-cd
   - 作用：前沿生成式变化检测路线参考

9. `ELGC-Net`
   - 路线：Efficient local-global aggregation
   - 代码状态：**官方开源**
   - 仓库：https://github.com/techmn/elgcnet
   - 作用：轻量高效对照

10. `SaDL_CD`
   - 路线：Semantic-aware dense representation learning
   - 代码状态：**官方开源**
   - 仓库：https://github.com/justchenhao/SaDL_CD
   - 作用：轻量语义感知路线参考

## 3. 工具箱可复现，但不是独立官方仓库优先

1. `STeInFormer`
   - 代码状态：**可通过 rschange 工具箱复现**
   - 工具箱：https://github.com/xwmaxwma/rschange
   - 论文：https://arxiv.org/abs/2412.17247
   - 说明：如果工具箱中配置和权重可用，仍然值得优先尝试。

2. `CD-Lamba`
   - 代码状态：**可通过 rschange 工具箱路线继续核验**
   - 工具箱：https://github.com/xwmaxwma/rschange
   - 论文：https://arxiv.org/abs/2501.15455

3. `CDMamba`
   - 代码状态：**官方开源已公开**
   - 仓库：https://github.com/zmoka-zht/CDMamba
   - 说明：虽然是 Mamba，但仍在这里保留，因为它非常值得做主干候选。

## 4. 暂不建议优先投入的“代码不确定/工程风险高”路线

这些论文可以看思想，但当前不建议优先作为第一轮主模型复现目标：

- `Semantic-CD`
- `TextSCD`
- `RoCD`
- `AdaptVFMs-RSCD`
- `FMT`
- `Changen2`
- `LSC-CD`
- `FFPNet`
- `HGINet`

原因：

- 有些论文当前未核到稳定公开代码仓库
- 有些只有论文或链接不稳定
- 有些更偏语义变化或基础模型扩展，工程代价高于你当前主线收益

## 5. 只从“开源可复现”角度给出的最终建议

如果现在就要进入第一轮复现，建议顺序如下：

1. `ChangeMamba`
2. `ChangeViT`
3. `MaskCD`
4. `ChangeCLIP`
5. `BAN`

如果你想把工作量控制住，最精简的论文候选池可以只保留：

- `ChangeMamba`
- `ChangeViT`
- `MaskCD`
- `ChangeCLIP`

这四个分别代表：

- 高效状态空间路线
- 强 Transformer 路线
- 对象级/边界路线
- 语义引导路线

这样后面做模型融合时，逻辑也最清楚。
