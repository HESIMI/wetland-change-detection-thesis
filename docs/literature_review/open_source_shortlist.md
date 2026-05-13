# 开源代码可复现模型清单

更新时间：`2026-05-13`

## 1. 说明

本清单仅保留已核实具有公开代码仓库的模型，用于后续复现与实验对比。模型按照“主干候选”“补充对照”“工具箱复现”三个层次组织。

## 2. 主干候选

### 2.1 `ChangeMamba`

- 路线：Mamba 主干
- 代码状态：官方开源
- 仓库：https://github.com/ChenHongruixuan/ChangeMamba
- 论文：https://arxiv.org/abs/2404.03425

### 2.2 `ChangeViT`

- 路线：Plain ViT 主干
- 代码状态：官方开源
- 仓库：https://github.com/zhuduowang/ChangeViT
- 论文：https://arxiv.org/abs/2406.12847

### 2.3 `MaskCD`

- 路线：Mask Classification
- 代码状态：官方开源
- 仓库：https://github.com/AI4RS/MaskCD

### 2.4 `ChangeCLIP`

- 路线：Vision-Language
- 代码状态：官方开源
- 仓库：https://github.com/dyzy41/ChangeCLIP
- 论文：https://www.sciencedirect.com/science/article/pii/S0924271624000042

### 2.5 `BAN`

- 路线：Foundation Model Adaptation
- 代码状态：官方开源
- 仓库：https://github.com/likyoo/BAN
- 论文：https://arxiv.org/abs/2312.01163

## 3. 补充对照

### Transformer / CNN 基线

- `BIT`
  - 仓库：https://github.com/justchenhao/BIT_CD

- `ChangeFormer`
  - 仓库：https://github.com/wgcban/ChangeFormer

- `SARAS-Net`
  - 仓库：https://github.com/f64051041/SARAS-Net

- `SNUNet-CD`
  - 仓库：https://github.com/likyoo/Siam-NestedUNet

- `VcT`
  - 仓库：https://github.com/Event-AHU/VcT_Remote_Sensing_Change_Detection

### 语义变化检测与扩展路线

- `ClearSCD`
  - 仓库：https://github.com/tangkai-RS/ClearSCD

- `DDPM-CD`
  - 仓库：https://github.com/wgcban/ddpm-cd

- `ELGC-Net`
  - 仓库：https://github.com/techmn/elgcnet

- `SaDL_CD`
  - 仓库：https://github.com/justchenhao/SaDL_CD

## 4. 工具箱可复现模型

### `STeInFormer`

- 代码状态：通过 `rschange` 工具箱提供实现
- 工具箱：https://github.com/xwmaxwma/rschange
- 论文：https://arxiv.org/abs/2412.17247

### `CD-Lamba`

- 代码状态：通过 `rschange` 工具箱相关实现继续核验
- 工具箱：https://github.com/xwmaxwma/rschange
- 论文：https://arxiv.org/abs/2501.15455

### `CDMamba`

- 代码状态：官方开源
- 仓库：https://github.com/zmoka-zht/CDMamba

## 5. 当前阶段适用性

结合当前课题的数据基础与实验条件，首轮复现模型覆盖以下四类路线：

- 状态空间主干：`ChangeMamba`
- Transformer 主干：`ChangeViT`
- 对象级解码：`MaskCD`
- 语义引导：`ChangeCLIP`

`BAN` 保留为基础模型适配路线，用于补充泛化能力相关实验。
