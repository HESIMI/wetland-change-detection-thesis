# 基于遥感影像的湿地变化检测模型研究

本仓库用于归档硕士毕业论文研究过程中的阶段性材料，主要包括文献调研、数据集构建说明、论文框架和实验进展。

## 研究问题

本课题围绕湿地变化检测中的以下问题展开：

- 高分辨率双时相遥感影像下的计算效率问题
- 水位波动与季节变化引起的伪变化干扰
- 复杂边界与破碎斑块区域的变化表征能力
- 跨区域场景下的模型泛化能力

当前纳入比较范围的技术路线包括：

- `Mamba / SSM`
- `Transformer / ViT`
- `Mask-based decoding`
- `Vision-Language modeling`
- `Foundation model adaptation`

## 仓库内容

- [docs/literature_review/README.md](./docs/literature_review/README.md)
  - 相关模型调研与分类整理
- [docs/literature_review/open_source_shortlist.md](./docs/literature_review/open_source_shortlist.md)
  - 公开代码可复现模型清单
- [docs/dataset/README.md](./docs/dataset/README.md)
  - 数据来源、标签构建与样本统计
- [docs/thesis_framework/README.md](./docs/thesis_framework/README.md)
  - 论文结构与技术路线
- [docs/progress/README.md](./docs/progress/README.md)
  - 阶段进展与实验情况

## 当前进展

截至当前阶段，已完成以下工作：

1. 基于 `GLC_FCS30D` 与 `Sentinel-2` 构建湿地变化检测数据集
2. 完成 6 个典型湿地区域的双时相样本整理与标签生成
3. 完成训练样本切片及 `train / val / test` 划分
4. 跑通 `Siamese UNet` baseline，建立训练与评估流程
5. 完成相关模型调研，并形成开源可复现候选清单

当前工作内容已由数据准备转入候选主模型复现与比较实验。

## 候选模型

现阶段重点关注的开源模型包括：

- `ChangeMamba`
- `ChangeViT`
- `MaskCD`
- `ChangeCLIP`
- `BAN`

上述模型分别对应状态空间建模、Transformer 主干、对象级解码、语义引导和基础模型适配等技术方向。

## 数据与代码说明

本仓库以研究文档为主，不直接托管大体量原始影像、切片样本和训练权重。完整实验数据、训练代码与服务器环境配置保留在本地项目目录与计算环境中。

## 研究区域示意

杭州西溪湿地变化示意：

![hangzhou_xixi_change](./assets/previews/hangzhou_xixi_change_final.png)

鄱阳湖变化示意：

![poyang_lake_change](./assets/previews/poyang_lake_change_final.png)
