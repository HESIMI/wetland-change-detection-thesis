# Public Dataset Preparation

本文实验框架引入 SECOND 与 HRSCD-Clean 两个公开语义变化检测数据集，用于验证模型在标准精标数据与非完美标签场景下的适应性。

## SECOND

来源：Captain WHU Semantic Change Detection project page  
下载页：<https://captain-whu.github.io/SCD/>

当前本地状态：

- 原始压缩包：`D:/桌面/文献/论文/公开数据集/SECOND/second_dataset.zip`
- 解压目录：`D:/桌面/文献/论文/公开数据集/SECOND/extracted_7z`
- 训练集：`im1 / im2 / label1 / label2`
- 测试集：`test/im1 / test/im2 / test/label1 / test/label2`

数据格式：

- `im1`: 第一时相 RGB 影像。
- `im2`: 第二时相 RGB 影像。
- `label1`: 第一时相语义标签图。
- `label2`: 第二时相语义标签图。
- 二值变化标签由 `label1 != label2` 推导。

当前统计：

- train source: 2968 image pairs
- deterministic train split: 2553 image pairs
- deterministic validation split: 415 image pairs
- test: 1694 image pairs
- image size: 512 x 512

仓库读取器会从训练源中确定性划分 `train/val`，测试集直接使用官方 `test` 目录。

## HRSCD-Clean

来源：EPFL-ECEO/HRSCD_clean  
下载页：<https://huggingface.co/datasets/EPFL-ECEO/HRSCD_clean>

数据说明：

- 数据集包含 291 对双时相航空影像。
- 单景尺寸为 10000 x 10000，空间分辨率为 0.5 m。
- 每对影像配有二值变化掩码和两期语义分割图。
- 官方压缩包大小约 60.3 GB。

推荐下载命令：

```bash
python scripts/data_preparation/download_public_datasets.py --dataset hrscd --root D:/datasets_public
```

下载完成后，可将数据保留在英文路径下，避免部分 Python/Hugging Face 工具在 Windows 中文路径下出现缓存路径错误。训练时通过 `--root` 指向实际目录即可。

## Unified Reader

公开数据集统一使用：

```python
from wetland_cd.training.public_datasets import PublicSemanticChangeDataset

dataset = PublicSemanticChangeDataset(
    dataset="second",
    root="D:/桌面/文献/论文/公开数据集/SECOND",
    split="train",
)
```

统一输出字段：

- `t1`
- `t2`
- `image`
- `binary_mask`
- `semantic_t1`
- `semantic_t2`
- `sample_id`
- `dataset`
