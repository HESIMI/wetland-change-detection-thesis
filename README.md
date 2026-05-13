# 湿地变化检测毕业论文项目

本仓库用于汇总硕士毕业论文阶段性工作，方便向导师展示当前的研究基础、文献调研、数据集构建、论文框架和项目进展。

论文暂定方向：

**基于遥感影像的湿地变化检测模型研究**

当前不预设最终主模型必须是 `Mamba-CLIP`，而是通过文献调研与实验对比，在以下候选路线中筛选最适合本课题的方案：

- `Mamba / SSM` 主干
- `Transformer / ViT` 主干
- `Mask / 对象级解码` 路线
- `CLIP / 文本语义引导` 路线
- `Foundation Model Adaptation` 路线

## 仓库内容

- [docs/literature_review/README.md](./docs/literature_review/README.md)
  - 近几年遥感变化检测前沿模型调研
- [docs/literature_review/open_source_shortlist.md](./docs/literature_review/open_source_shortlist.md)
  - 已核实具备开源代码的候选模型清单
- [docs/dataset/README.md](./docs/dataset/README.md)
  - 数据集来源、处理流程、样本规模与标签构建方法
- [docs/thesis_framework/README.md](./docs/thesis_framework/README.md)
  - 论文整体框架、章节安排与预期创新点
- [docs/progress/README.md](./docs/progress/README.md)
  - 当前阶段成果、baseline 结果和下一步计划

## 当前阶段结论

目前论文已经完成了以下关键基础工作：

1. 完成湿地变化检测任务的数据准备与标签构建
2. 扩展到 6 个典型湿地区域，形成多区域实验数据集
3. 切片生成可训练样本，并完成 `train / val / test` 划分
4. 跑通 `Siamese UNet` baseline，验证训练链路与评估流程
5. 完成近几年前沿模型调研，并按“是否开源可复现”筛出候选池

当前重点已从“数据准备”转向“主模型筛选与复现”。

## 建议优先复现的开源模型

当前优先级最高的候选为：

1. `ChangeMamba`
2. `ChangeViT`
3. `MaskCD`
4. `ChangeCLIP`
5. `BAN`

其中最适合当前课题比较的 4 条主路线是：

- 高效状态空间路线：`ChangeMamba`
- 强 Transformer 路线：`ChangeViT`
- 对象级边界路线：`MaskCD`
- 语义引导路线：`ChangeCLIP`

## 数据与代码说明

本仓库以**文档展示**为主，不直接上传大体量原始遥感数据、切片样本和训练权重。原因：

- 原始数据体量较大，不适合 GitHub 托管
- 后续实验数据与模型权重会持续变化
- 当前仓库用途以汇报和阶段性总结为主

如需完整实验代码与训练数据，可在导师沟通后通过本地项目目录或服务器环境进一步补充。

## 研究区域示意

杭州西溪湿地变化示意：

![hangzhou_xixi_change](./assets/previews/hangzhou_xixi_change_final.png)

鄱阳湖变化示意：

![poyang_lake_change](./assets/previews/poyang_lake_change_final.png)
