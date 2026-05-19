# 项目进展

## 当前方向

当前论文方向为：

> 面向弱监督标签与伪变化抑制的湿地遥感变化检测方法研究

研究重点已经从单一模型融合方案，调整为围绕数据噪声、伪变化、复杂边界和跨区域泛化的完整实验链条。

## 已完成工作

### 数据层

- 完成 6 个湿地研究区设计：
  - `hangzhou_xixi`
  - `qiantang_estuary`
  - `poyang_lake`
  - `dongting_lake`
  - `yellow_river_delta`
  - `chongming_dongtan`
- 完成 Sentinel-2 2018/2022 双时相数据整理。
- 完成 GLC_FCS30D 2018/2022 土地覆盖数据读取。
- 完成初始弱监督变化标签构建。
- 完成 ESA WorldCover 2021 对齐与多源一致性辅助分析。
- 完成高置信变化、高置信未变化、低置信样本分层。
- 完成湿地 train/val/test 切片与统计。
- 完成数据质量验收表，结论为数据可支撑弱监督标签构建、伪变化分析、湿地场景实验和跨区域泛化验证。

### 公开数据

- SECOND 已完成统一读取接口验证：
  - train: 2553
  - val: 415
  - test: 1694
- HRSCD small sample 已替换为 HRSCD balanced sample：
  - train: 220，其中变化 patch 110
  - val: 40，其中变化 patch 20
  - test: 40，其中变化 patch 20
- HRSCD balanced sample 已通过统一 reader 和 single-batch smoke test。

### 训练框架

- 已建立统一训练框架：
  - 数据读取
  - 输入尺寸
  - 训练轮数
  - 学习率策略
  - 指标计算
  - 结果保存格式
- 已实现 Siamese U-Net baseline。
- 已完成本机工作流检查：
  - 数据读取检查
  - single-batch smoke test
  - 代码跑通验证
  - 可视化脚本调试
  - 小 epoch 试训
  - 文档、表格、结果汇总

## 当前实验状态

### 本机

本机定位为调试环境，当前已经完善：

- 湿地、SECOND、HRSCD balanced sample 数据读取。
- 湿地、SECOND、HRSCD single-batch smoke test。
- 可视化脚本和结果表脚本。
- 本机不建议承担正式长时间训练。

### 服务器

服务器已启动 SECOND + Siamese U-Net 正式 baseline：

```text
server: GPU3090NODE3
repo: /tmp/hesimin/wetland-change-detection-thesis
data: /tmp/hesimin/datasets/SECOND
run: /tmp/hesimin/wetland-change-detection-thesis/runs/second_siamese_unet_formal
```

最近一次检查状态：

```text
progress: 47 / 50 epochs
best epoch: 15
best val F1: 0.6850
best val IoU: 0.5209
```

该结果尚未作为最终指标写入论文表格，需等待 50 epoch 完成并读取 `metrics.json`。

## 下一步动作

1. 等待 SECOND + Siamese U-Net formal baseline 完成。
2. 汇总正式 baseline 指标、训练曲线和预测可视化。
3. 在 SECOND 上继续复现至少两个 baseline，优先考虑：
   - `ChangeViT`
   - `ChangeMamba` 或 `CDMamba`
   - `MaskCD`
   - `BAN`
4. 将稳定 baseline 迁移到 HRSCD balanced / larger HRSCD。
5. 将候选模型迁移到湿地弱监督数据。
6. 根据湿地实验结果设计：
   - 置信度加权或噪声鲁棒训练
   - 伪变化抑制模块
   - 边界与破碎斑块增强模块

## 阶段性结论

当前仓库已经完成从“数据准备”到“统一训练框架”和“服务器正式 baseline 启动”的过渡。下一阶段的核心任务是把 SECOND baseline 结果收口，并开始多模型复现与湿地迁移实验。
