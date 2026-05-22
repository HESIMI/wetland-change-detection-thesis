# CDMamba-MaskCD 融合模型原型

## 目标定位

当前原型用于探索“面向弱监督标签与伪变化抑制的湿地遥感变化检测”中的主模型方向。核心思想是把变化检测从传统逐像素二分类，推进到由双时相全局-局部特征驱动的目标级变化推理。

```text
T1 image, T2 image
        ↓
CDMamba-style Siamese Encoder
        ↓
Multi-scale global-local bitemporal features
        ↓
Change Feature Adapter
        ↓
Change-aware Mask Query Generator
        ↓
Bitemporal Mask Interaction Decoder
        ↓
Mask Classification Head
        ↓
Binary Change Map
```

## 模块设计

### CDMamba-style Encoder

当前版本根据 LEVIR-CD mini baseline 判断进行了调整：MaskCD 在小样本上明显优于 CDMamba，因此融合模型不再让 CDMamba 主导最终输出，而是将其作为全局-局部变化特征增强器。第一版仍复用仓库内的 `ChangeMambaLite` 编码器组件，形成可直接训练的自包含原型。它承担两个职责：

- 提取双时相影像的多尺度特征；
- 通过 Mamba-style 全局扫描与局部卷积混合，生成变化敏感差异特征。

后续服务器环境稳定后，可将该部分替换为官方 CDMamba 中的 SRCM 与 AGLGF 模块。

### Change-aware Mask Query Generator

普通 MaskCD 主要依赖可学习 query。本原型改为从高层差异特征中生成 query，使 mask proposal 直接受变化区域驱动。当前默认采用 top-k change response selection，而不是固定池化，以减少背景 token 对 mask proposal 的干扰。

对应代码：

```text
src/wetland_cd/training/ours/change_aware_query.py
```

### MaskCD-dominant Output Fusion

mini 结果显示 MaskCD 的目标级 mask reasoning 更适合 LEVIR-CD，因此当前输出融合采用：

```text
final_logits = 0.7 * object_mask_logits + 0.3 * auxiliary_pixel_logits
```

其中：

- `object_mask_logits` 来自 mask proposal 与 changed / unchanged 分类；
- `auxiliary_pixel_logits` 来自 CDMamba-style diff feature 的像素级辅助分支；
- auxiliary pixel head 的作用是稳定早期训练，避免 mask proposal 召回不足。

### Bitemporal Mask Interaction Decoder

decoder 不只看差异特征，还分别与 T1、T2、diff feature 交互：

```text
query ↔ T1 feature
query ↔ T2 feature
query ↔ diff feature
```

这样可以为“伪变化抑制”提供结构基础：如果某区域在差异图中显著，但在双时相语义目标结构中不稳定，可在后续加入一致性约束或低置信抑制。

对应代码：

```text
src/wetland_cd/training/ours/bitemporal_mask_decoder.py
```

### Mask-level Consistency Loss

已加入独立 loss 模块，用于后续减少目标内部空洞和碎片：

```text
src/wetland_cd/training/ours/mask_consistency_loss.py
```

当前统一训练入口仍使用 BCE + Dice 训练最终二值变化图，确保第一版能稳定与现有 baseline 对比。后续可把 mask proposal 输出纳入 loss 计算。

## 当前可运行配置

可直接使用统一训练入口运行：

```bash
python -m src.wetland_cd.training.train --config configs/training/levir_cdmamba_maskcd.json
python -m src.wetland_cd.training.train --config configs/training/whu_cdmamba_maskcd.json
python -m src.wetland_cd.training.train --config configs/training/sysu_cdmamba_maskcd.json
```

说明性 YAML 配置位于：

```text
configs/ours/cdmamba_maskcd_levir.yaml
configs/ours/cdmamba_maskcd_whu.yaml
configs/ours/cdmamba_maskcd_sysu.yaml
```

## 已完成验证

本机已完成：

- 纯张量模型 smoke test；
- LEVIR-CD dataloader 单 batch 训练、验证、测试 smoke test；
- loss、metrics、checkpoint、history 保存链路验证。

smoke run：

```text
runs/smoke_levir_cdmamba_maskcd
```

## 后续升级路线

1. 用官方 CDMamba encoder 替换当前 `ChangeMambaLite` 编码器；
2. 将 MaskCD 官方 Mask2Former decoder 的候选 mask 机制接入当前 query 生成逻辑；
3. 增加 mask-level consistency loss，与 BCE + Dice 共同训练；
4. 在 LEVIR-CD、WHU-CD、SYSU-CD 上与 Siamese U-Net、ChangeFormer、CDMamba、MaskCD 对比；
5. 迁移到湿地弱监督数据，验证边界破碎、伪变化、水位波动场景下的收益。
