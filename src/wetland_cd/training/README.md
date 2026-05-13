# Training

这部分是面向论文实验的第一版训练脚手架，目标是先把 baseline 跑通。

## 文件

- `dataset.py`
  - 读取 `dataset_manifest.csv`，返回双时相影像和变化标签。
- `models.py`
  - 一个轻量的 `Siamese UNet` baseline。
- `train_baseline.py`
  - baseline 训练入口。
- `inspect_dataset.py`
  - 快速检查数据加载是否正常。

## 先检查数据

```bash
python D:/桌面/文献/项目/training/inspect_dataset.py
```

## 训练 baseline

```bash
python D:/桌面/文献/项目/training/train_baseline.py --epochs 5 --batch-size 4
```

模型和指标会保存在：

`D:/桌面/文献/项目/runs/siamese_unet`

## 当前定位

这不是最终的 `Mamba-CLIP` 模型，而是论文实验的起点：

1. 先验证数据集和训练流程可用。
2. 先得到一个可比较的 baseline。
3. 后续再把 `ChangeMamba / ChangeCLIP / Mamba-CLIP` 逐步接进来。
