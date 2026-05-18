# 本机实验准备状态检查

## 结论

本机已经适合承担论文实验中的调试角色：数据读取检查、单 batch smoke test、代码跑通验证、可视化脚本调试、小 epoch 试训、文档和结果表汇总均已完成。

本机不建议承担正式长时间训练。正式 baseline、多模型复现、迁移实验和消融实验仍建议放到学校服务器运行。

## 检查清单

| Item | Status | Evidence |
| --- | --- | --- |
| 数据读取检查 | Passed | 湿地 train/val/test 可读取；SECOND train 2553；HRSCD balanced sample train 220 |
| 单 batch smoke test | Passed | `runs/local_checks/wetland_single_batch_smoke`、`runs/local_checks/second_single_batch_smoke` |
| 代码是否能跑通 | Passed | 训练核心模块与实验脚本 `py_compile` 通过 |
| 可视化脚本调试 | Passed | 已生成训练曲线和预测可视化 PNG |
| 小 epoch 试训 | Passed | `runs/local_checks/hrscd_sample_2epoch_trial` 完成 2 epoch |
| HRSCD balanced smoke test | Passed | `runs/local_checks/hrscd_balanced_single_batch_smoke` 完成 1 个 train/val/test batch |
| 文档、表格、结果汇总 | Passed | `docs/experiments/local_check_results.csv` 与本文件 |

## 已运行命令

数据读取检查：

```powershell
python src/wetland_cd/training/inspect_dataset.py
python src/wetland_cd/training/inspect_public_datasets.py --dataset second --root D:/桌面/文献/论文/公开数据集/SECOND --split train --batch-size 1
python src/wetland_cd/training/inspect_public_datasets.py --dataset hrscd --root D:/桌面/文献/论文/公开数据集/HRSCD_sample_balanced --split train --batch-size 1
```

单 batch smoke test：

```powershell
python src/wetland_cd/training/train.py --config configs/training/wetland_siamese_unet.json --epochs 1 --batch-size 1 --limit-train-batches 1 --limit-val-batches 1 --limit-test-batches 1 --outdir runs/local_checks/wetland_single_batch_smoke
python src/wetland_cd/training/train.py --config configs/training/second_siamese_unet.json --epochs 1 --batch-size 8 --limit-train-batches 1 --limit-val-batches 1 --limit-test-batches 1 --outdir runs/local_checks/second_single_batch_smoke
```

小 epoch 试训：

```powershell
python src/wetland_cd/training/train.py --config configs/training/hrscd_sample_siamese_unet.json --epochs 2 --batch-size 4 --outdir runs/local_checks/hrscd_sample_2epoch_trial
python src/wetland_cd/training/train.py --config configs/training/hrscd_sample_siamese_unet.json --epochs 1 --batch-size 4 --limit-train-batches 1 --limit-val-batches 1 --limit-test-batches 1 --outdir runs/local_checks/hrscd_balanced_single_batch_smoke
```

训练曲线与可视化：

```powershell
python scripts/experiments/plot_training_curves.py --run-dir runs/local_checks/hrscd_sample_2epoch_trial --output results/local_checks/hrscd_sample_2epoch_training_curves.png
python scripts/experiments/plot_training_curves.py --run-dir runs/local_checks/second_single_batch_smoke --output results/local_checks/second_smoke_training_curves.png
python scripts/experiments/plot_training_curves.py --run-dir runs/local_checks/wetland_single_batch_smoke --output results/local_checks/wetland_smoke_training_curves.png
python scripts/experiments/visualize_predictions.py --run-dir runs/local_checks/second_single_batch_smoke --split test --num-samples 4 --output-dir results/local_checks/second_smoke_visualizations
python scripts/experiments/visualize_predictions.py --run-dir runs/local_checks/wetland_single_batch_smoke --split test --num-samples 4 --output-dir results/local_checks/wetland_smoke_visualizations
python scripts/experiments/summarize_runs.py --runs-root runs/local_checks --output-csv results/local_checks/baseline_results.csv --output-md results/local_checks/baseline_results.md
```

## 本机结果表

| Run | Dataset | Model | Epochs | Best Val F1 | Test F1 | Test IoU | 用途 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `wetland_single_batch_smoke` | wetland | Siamese U-Net | 1 | 0.0019 | 0.1267 | 0.0677 | 本机湿地数据 smoke test |
| `second_single_batch_smoke` | SECOND | Siamese U-Net | 1 | 0.0000 | 0.0000 | 0.0000 | SECOND 单 batch 流程测试 |
| `hrscd_sample_2epoch_trial` | HRSCD sample | Siamese U-Net | 2 | 0.0000 | 0.0000 | 0.0000 | 小 epoch 试训与曲线脚本调试 |
| `hrscd_balanced_single_batch_smoke` | HRSCD balanced sample | Siamese U-Net | 1 | 0.0000 | 0.0000 | 0.0000 | 新均衡样本单 batch smoke test |

这些数值不是论文正式指标，只表示本机流程可运行。SECOND 和 HRSCD sample 的本机短训指标不具备模型比较意义。

## 重要观察

- HRSCD balanced sample 已替换旧的小样本，当前 train/val/test 均含约 50% 变化 patch；适合本机迁移调试，但仍不等同于完整 HRSCD 正式 benchmark。
- SECOND 数据读取和单 batch 训练已经跑通，适合作为服务器正式 baseline 的第一站。
- 湿地数据读取、单 batch 训练和预测可视化已经跑通，后续可在服务器完成迁移训练。
- 当前可视化脚本可以输出 T1、T2、GT、Prediction 和 Overlay，对后续 qualitative figures 有用。

## 主干模型选择依据

当前本机阶段只验证了 `Siamese U-Net` 作为轻量 baseline，不能直接据此选择最终主干。后续服务器正式实验建议按以下依据筛选主干：

| Criterion | Why It Matters |
| --- | --- |
| SECOND 上的标准 benchmark 表现 | 检查模型基本变化检测能力 |
| HRSCD 迁移稳定性 | 检查复杂场景和标签体系适应性 |
| 湿地数据上的伪变化抑制能力 | 对应论文核心问题 |
| 边界和破碎斑块表现 | 对应湿地复杂边界识别 |
| 训练显存与速度 | 决定是否适合服务器批量实验 |
| 是否易接入统一框架 | 避免多仓库、多训练逻辑导致结果不可比 |

建议服务器正式复现顺序：

1. SECOND 上跑完整 `Siamese U-Net` baseline。
2. SECOND 上加入两个强 baseline，例如 `ChangeViT` 和 `ChangeMamba`。
3. 将表现稳定的模型迁移到 HRSCD balanced sample；服务器阶段再扩大到官方 split 或完整 HRSCD。
4. 将候选主干迁移到湿地弱监督数据，重点分析高低置信样本和伪变化区域。
