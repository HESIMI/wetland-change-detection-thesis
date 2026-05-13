# Training Module

本目录为当前训练代码入口，主要用于 baseline 模型训练与数据读取流程验证。

## Files

- `dataset.py`
  - 读取 `dataset_manifest.csv`，返回双时相影像与变化标签
- `models.py`
  - 当前 baseline 模型定义
- `train_baseline.py`
  - baseline 训练脚本
- `inspect_dataset.py`
  - 数据读取检查脚本

## Usage

检查数据读取：

```bash
python src/wetland_cd/training/inspect_dataset.py
```

运行 baseline：

```bash
python src/wetland_cd/training/train_baseline.py --epochs 5 --batch-size 4
```

实验输出建议保存到本地 `results/` 目录。

## Scope

当前版本仅包含 baseline 训练代码。后续主模型复现与改进模型实现将在此基础上继续扩展。
