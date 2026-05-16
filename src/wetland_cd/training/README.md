# Training Module

本目录包含论文实验阶段的训练与数据读取入口，当前覆盖自建湿地变化检测数据集、SECOND 与 HRSCD-Clean。

## Files

- `dataset.py`: 自建湿地弱监督变化检测数据集读取器。
- `public_datasets.py`: SECOND 与 HRSCD-Clean 的统一读取器。
- `inspect_dataset.py`: 自建湿地数据集读取检查脚本。
- `inspect_public_datasets.py`: 公开数据集读取检查脚本。
- `models.py`: 当前 baseline 模型定义。
- `train_baseline.py`: Siamese U-Net baseline 训练入口。

## Public Dataset Output Format

`PublicSemanticChangeDataset` 统一返回以下字段：

- `t1`: 第一时相影像，形状为 `C x H x W`。
- `t2`: 第二时相影像，形状为 `C x H x W`。
- `image`: 双时相拼接影像，形状为 `2C x H x W`。
- `binary_mask`: 二值变化标签，形状为 `1 x H x W`。
- `semantic_t1`: 第一时相语义标签，形状为 `H x W`。
- `semantic_t2`: 第二时相语义标签，形状为 `H x W`。

## Usage

检查 SECOND：

```bash
python src/wetland_cd/training/inspect_public_datasets.py --dataset second --root D:/桌面/文献/论文/公开数据集/SECOND --split train
```

检查 HRSCD-Clean：

```bash
python src/wetland_cd/training/inspect_public_datasets.py --dataset hrscd --root D:/桌面/文献/论文/公开数据集/HRSCD_clean --split train
```

运行自建湿地 baseline：

```bash
python src/wetland_cd/training/train_baseline.py --epochs 5 --batch-size 4
```

