# 湿地变化检测毕业论文项目

本仓库用于汇总硕士毕业论文阶段性工作，面向研究汇报与过程管理，集中展示当前的文献调研结果、数据集构建情况、论文框架与项目进展。

论文研究主题：

**基于遥感影像的湿地变化检测模型研究**

## 研究背景

湿地变化检测在生态保护、资源监管与区域规划中具有重要应用价值。相较于通用遥感变化检测任务，湿地场景具有以下特点：

- 水体、潮滩、湿地植被之间存在显著的季节性与水位波动
- 变化区域边界复杂，斑块形态破碎
- 高分辨率双时相影像对模型计算效率提出更高要求

因此，本课题的模型设计不预设固定技术路线，而以实验结果为依据，在以下候选方向中开展筛选与比较：

- `Mamba / SSM` 主干
- `Transformer / ViT` 主干
- `Mask / 对象级解码` 路线
- `CLIP / 文本语义引导` 路线
- `Foundation Model Adaptation` 路线

## 仓库结构

- [docs/literature_review/README.md](./docs/literature_review/README.md)
  - 遥感变化检测前沿模型调研
- [docs/literature_review/open_source_shortlist.md](./docs/literature_review/open_source_shortlist.md)
  - 已核实公开代码仓库的候选模型清单
- [docs/dataset/README.md](./docs/dataset/README.md)
  - 数据来源、标签构建、样本规模与数据处理流程
- [docs/thesis_framework/README.md](./docs/thesis_framework/README.md)
  - 论文结构与技术路线
- [docs/progress/README.md](./docs/progress/README.md)
  - 阶段性成果、baseline 实验与后续任务安排

## 当前工作基础

截至当前阶段，项目已完成以下基础工作：

1. 完成湿地变化检测数据准备与标签构建
2. 扩展形成 6 个典型湿地区域的多区域实验数据集
3. 完成 patch 切片及 `train / val / test` 划分
4. 跑通 `Siamese UNet` baseline，验证训练与评估流程
5. 完成近几年相关模型的系统性调研，并按“开源可复现”原则形成候选池

当前工作重点已经由数据准备转入主模型筛选与复现阶段。

## 已筛选的开源候选模型

当前纳入优先复现范围的代表性模型包括：

1. `ChangeMamba`
2. `ChangeViT`
3. `MaskCD`
4. `ChangeCLIP`
5. `BAN`

上述模型分别对应：

- 高效状态空间建模
- 强 Transformer 主干
- 对象级与边界增强
- 语义引导
- 基础模型适配

## 数据与代码说明

本仓库以**研究文档展示**为主，不直接托管大体量原始遥感数据、切片样本与训练权重。主要原因如下：

- 原始数据体量较大，不适合 GitHub 持续管理
- 训练样本与权重文件会随实验迭代频繁更新
- 当前仓库定位为论文过程归档与阶段汇报

完整实验环境、训练代码与服务器部署信息保留在本地项目目录与计算环境中，后续可根据需要进一步补充整理。

## 研究区域示意

杭州西溪湿地变化示意：

![hangzhou_xixi_change](./assets/previews/hangzhou_xixi_change_final.png)

鄱阳湖变化示意：

![poyang_lake_change](./assets/previews/poyang_lake_change_final.png)
