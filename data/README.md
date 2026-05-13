# Data Directory

本目录用于存放本地实验所需的数据文件，但默认不纳入 Git 版本管理。

建议的本地组织方式如下：

```text
data/
  raw/
  raw_glc_fcs30d/
  glc_subsets/
  change_labels/
  processed/
```

说明：

- `raw/`
  - 双时相 `Sentinel-2` 合成影像与配套裁剪标签
- `raw_glc_fcs30d/`
  - 原始 `GLC_FCS30D` 年度瓦片
- `glc_subsets/`
  - 研究区年度标签子图
- `change_labels/`
  - 二值变化图、伪变化掩码与语义变化图
- `processed/`
  - patch 切片后的训练、验证与测试样本

当前数据体量较大，不直接上传至 GitHub。数据准备流程见：

- [../docs/dataset/README.md](../docs/dataset/README.md)
- [../scripts/data_preparation](../scripts/data_preparation)
