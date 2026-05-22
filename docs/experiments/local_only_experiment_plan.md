# 本机实验方案

## 资源判断

当前本机配置：

- GPU：NVIDIA GeForce RTX 2060
- 显存：6GB
- CUDA：可用
- D 盘剩余空间：约 189GB
- 已整理公开数据集：LEVIR-CD、WHU-CD、SYSU-CD，合计约 8.77GB

结论：本机可以承担论文主要实验，但需要采用 6GB 显存友好的设置。

## 实验定位

本机作为主实验平台：

- 数据读取检查；
- reduced baseline；
- 融合模型结构调试；
- 可视化；
- 湿地弱监督迁移实验；
- 论文中的趋势判断与模型选择依据。

服务器不再作为进度前置条件，仅作为可选的 official full run 补充。

## 推荐设置

公开数据集 reduced 实验：

```text
epoch: 20
batch_size: 1
train batches: 200
val batches: 50
test batches: 50
image_size: 256
num_workers: 0
```

如果显存稳定，可以把 `batch_size` 提升到 2；如果出现 OOM，保持 1。

## 一键运行

```powershell
powershell -ExecutionPolicy Bypass -File scripts/experiments/run_local_public_cd_reduced.ps1
```

可调整参数：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/experiments/run_local_public_cd_reduced.ps1 `
  -Epochs 20 `
  -BatchSize 1 `
  -TrainBatches 200 `
  -ValBatches 50 `
  -TestBatches 50
```

## 模型范围

当前本机 reduced 实验覆盖：

- Siamese U-Net；
- ChangeFormer；
- ChangeMambaLite；
- revised CDMamba-MaskCD。

官方 CDMamba 与官方 MaskCD 仍可保留为补充实验。考虑到 Windows + 6GB 显存限制，官方 CDMamba 的 Mamba/CUDA kernel 复现优先级降低，MaskCD official 更适合在本机做小 batch reduced run。

## 论文使用方式

本机 reduced 实验可用于：

- 模型筛选；
- 主模型方向判断；
- 消融实验设计；
- 可视化分析；
- 湿地弱监督数据迁移前的预实验。

正式论文表格可以分为：

- 本机统一框架结果；
- official implementation 补充结果；
- 湿地弱监督主实验结果。
