# LEVIR-CD 本机 Reduced Baseline 结果

## 实验设置

- 数据集：LEVIR-CD 统一格式数据。
- 运行平台：本机 RTX 2060 6GB。
- 输入尺寸：256 x 256。
- batch size：1。
- 训练轮数：20 epoch。
- 采样规模：train 200 batch，val 50 batch，test 50 batch。
- 结果定位：用于本机可行性验证、模型排序初筛和后续主模型设计依据，不等同于完整论文最终指标。

## 测试结果

| Dataset | Model | Epochs | Best Epoch | Best Val F1 | Test P | Test R | Test F1 | Test IoU | Test OA |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LEVIR-CD | Siamese U-Net | 20 | 14 | 0.3704 | 0.5899 | 0.6247 | 0.6068 | 0.4356 | 0.9306 |
| LEVIR-CD | ChangeFormer | 20 | 20 | 0.1344 | 0.1881 | 0.8648 | 0.3090 | 0.1827 | 0.6687 |
| LEVIR-CD | ChangeMambaLite | 20 | 10 | 0.4567 | 0.6066 | 0.7952 | 0.6882 | 0.5246 | 0.9383 |
| LEVIR-CD | CDMamba-MaskCD prototype | 20 | 12 | 0.3379 | 0.3995 | 0.6702 | 0.5006 | 0.3339 | 0.8854 |

## 初步判断

在本机 reduced 设置下，ChangeMambaLite 的 F1 和 IoU 暂时最高，说明 Mamba 风格的变化特征建模在当前统一框架中有继续扩展价值。Siamese U-Net 表现稳定，可作为轻量 CNN baseline。ChangeFormer 在该设置下召回率高但误检明显，表现为 Precision 低、False Positive 多，后续需要调整学习率、类别不平衡处理或阈值策略后再判断。

CDMamba-MaskCD prototype 相比最早 mini 版本已有提升，但仍低于 ChangeMambaLite 和 Siamese U-Net。这个结果说明“MaskCD 框架 + CDMamba 特征增强”的方向可以继续作为创新结构探索，但当前简化版 mask decoder 和 query 机制还不够强，不能直接作为最终主模型定型。下一步应优先把 MaskCD 的目标级 mask 解码能力保留下来，再用 CDMamba/SRCM/AGLGF 作为特征增强模块，而不是只做简单特征拼接。

## 本地产物

- 指标表：`results/levir_local_reduced_results.csv`
- Markdown 表：`results/levir_local_reduced_results.md`
- 训练曲线：`runs/local_reduced_levir-cd_*/training_curves.png`
- 预测可视化：`runs/local_reduced_levir-cd_*/visualizations/test/*.png`

这些目录属于本地实验产物，默认不进入 Git。
