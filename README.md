# 面向弱监督标签与伪变化抑制的湿地遥感变化检测方法研究

本仓库是硕士毕业论文的核心代码与实验管理仓库，统一管理湿地弱监督变化检测的数据构建、公开数据集读取、baseline 训练、模型复现、结果统计与论文支撑文档。

当前论文方向已经从早期的单一 `Mamba-CLIP` 融合设想，调整为更贴合数据条件和论文问题链条的：

> 面向弱监督标签与伪变化抑制的湿地遥感变化检测方法研究

## 研究问题

本课题围绕湿地变化检测中的四个问题展开：

- 土地覆盖产品差分推导标签不精确，需显式建模弱监督标签噪声。
- 湿地季节性、水位波动、潮滩边界变化容易造成伪变化。
- 湿地边界复杂、斑块破碎，常规变化检测模型容易漏检或边界粗糙。
- 不同湖泊、河口、滨海湿地区域差异明显，需要验证跨区域泛化能力。

因此，本仓库不再以某一个预设主干为唯一目标，而是先建立统一数据和训练框架，再通过公开数据集、HRSCD 迁移和自建湿地弱监督数据逐步筛选主干模型与改进模块。

## 当前数据基础

| 数据 | 角色 | 当前状态 |
| --- | --- | --- |
| 自建湿地数据 | 论文主实验数据 | 6 个研究区，Sentinel-2 2018/2022 + GLC_FCS30D + ESA WorldCover |
| 弱监督标签 | 主线标签来源 | GLC_FCS30D 差分生成初始弱标签，已完成高低置信分层 |
| ESA WorldCover | 多源一致性验证 | 已用于辅助置信度分析和数据验收 |
| SECOND | 公开标准 benchmark | 已接入统一 dataloader，服务器正在跑 Siamese U-Net 正式 baseline |
| HRSCD balanced sample | 本机迁移调试数据 | 已重抽样为 train 220、val 40、test 40，三组均含 50% 变化样本 |

## 已完成

截至当前阶段，已完成：

1. 明确论文方向为弱监督标签、伪变化抑制、边界增强和跨区域泛化。
2. 完成 6 个湿地研究区的数据源设计、切片和统计。
3. 基于 GLC_FCS30D 构建初始弱监督变化标签。
4. 引入 ESA WorldCover 进行多源一致性验证。
5. 完成高置信变化、高置信未变化、低置信样本分层。
6. 完成数据质量验收表，确认数据可支撑弱监督标签构建、伪变化分析、湿地实验和泛化验证。
7. 建立统一训练框架，统一数据读取、训练轮数、输入尺寸、学习率策略、指标计算和结果保存格式。
8. 完成本机环境检查：数据读取、single-batch smoke test、可视化脚本、小 epoch 试训和结果表流程。
9. 扩大并重分层 HRSCD sample，解决原 val/test 空变化标签问题。
10. 在服务器上启动 SECOND + Siamese U-Net 正式 baseline 训练。

## 代码结构

- [scripts/data_preparation](./scripts/data_preparation)：数据下载、标签构建、样本切片、HRSCD 抽样与重分层脚本。
- [src/wetland_cd/training](./src/wetland_cd/training)：统一训练框架、dataloader、baseline 模型、损失函数、指标和训练入口。
- [configs/training](./configs/training)：SECOND、HRSCD、湿地数据的统一训练配置。
- [docs/dataset](./docs/dataset)：数据源、弱标签、置信样本、数据验收和公开数据集说明。
- [docs/experiments](./docs/experiments)：本机检查、结果表和实验准备状态。
- [docs/literature_review](./docs/literature_review)：公开模型调研与候选主干筛选。
- [docs/thesis_framework](./docs/thesis_framework)：论文框架与技术路线。
- [docs/progress](./docs/progress)：阶段进展与下一步工作。

## 关键文档

- [数据质量验收](./docs/dataset/data_quality_acceptance.md)
- [数据层状态](./docs/dataset/data_layer_status.md)
- [弱监督标签置信度筛选](./docs/dataset/weak_label_confidence.md)
- [HRSCD balanced sample](./docs/dataset/hrscd_balanced_sample.md)
- [本机实验准备状态](./docs/experiments/local_workstation_check.md)
- [统一训练框架说明](./src/wetland_cd/training/README.md)
- [模型调研清单](./docs/literature_review/README.md)
- [研发路线图](./docs/ROADMAP.md)

## 当前实验状态

本机已经完成调试职责，适合继续承担数据检查、快速 smoke test、可视化和脚本开发。

正式 baseline 与后续模型复现建议放到学校服务器完成。当前服务器实验：

```text
SECOND + Siamese U-Net formal baseline
run_dir: /tmp/hesimin/wetland-change-detection-thesis/runs/second_siamese_unet_formal
status: running, 47/50 epochs at last check
best val F1: 0.6850 at epoch 15
```

服务器结果文件包括：

```text
metrics.json
history.csv
history.jsonl
checkpoints/best.pt
checkpoints/last.pt
train.log
```

## 下一步

1. 等待 SECOND + Siamese U-Net 正式 baseline 跑完，整理正式指标表和训练曲线。
2. 在 SECOND 上继续复现 2-3 个 baseline，例如 ChangeViT、ChangeMamba/CDMamba、MaskCD 或 BAN。
3. 将表现稳定的模型迁移到 HRSCD balanced / larger HRSCD，检查复杂场景适应性。
4. 将候选模型迁移到自建湿地弱监督数据，重点分析高低置信样本、伪变化区域、边界和破碎斑块。
5. 在主干确定后设计弱标签噪声鲁棒训练、伪变化抑制和边界增强模块，并开展消融实验。

## 数据与版本管理说明

仓库不直接托管原始遥感影像、大规模 patch、完整公开数据集和模型权重。仓库主要保留：

- 数据处理与抽样脚本
- 统一训练代码
- 实验配置
- 指标统计与可视化脚本
- 论文支撑文档

大体量数据和训练结果保存在本机或服务器对应数据目录中。
