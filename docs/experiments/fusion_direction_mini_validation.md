# CDMamba-MaskCD 融合方向 mini 验证

## 验证目的

本次验证不作为正式论文指标，只用于判断融合模型方向是否值得继续推进：

- 是否能正常学习；
- 是否出现目标级 mask 推理的迹象；
- 是否相比纯像素解码更适合后续处理边界破碎和伪变化问题。

## 设置

数据集：LEVIR-CD  
输入尺寸：256 x 256  
训练规模：每轮 20 个 train batch，5 个 val batch，5 个 test batch  
训练轮数：5 epoch  
batch size：1  
学习率：1e-4  
指标：Precision、Recall、F1、IoU、Accuracy  

## 结果

| 模型 | train loss 变化 | train F1 变化 | best val F1 | test F1 | test IoU | 现象 |
|---|---:|---:|---:|---:|---:|---|
| ChangeMambaLite | 1.597 -> 1.391 | 0.048 -> 0.385 | 0.060 | 0.100 | 0.053 | 学习更快，但可视化中容易出现大片误检 |
| CDMamba-MaskCD | 1.561 -> 1.180 | 0.036 -> 0.318 | 0.050 | 0.016 | 0.008 | 预测更像局部目标级区域，但召回不足、验证波动较大 |

## 判断

融合方向具备继续推进价值，但第一版还不是稳定可用模型。

积极信号：

- 训练 loss 明显下降，说明结构可以反传并学习；
- train F1 从 0.036 提升到 0.318，说明不是无效结构；
- 可视化中预测更偏向局部 mask proposal，而不是满图噪声；
- 这与“从 pixel-level prediction 转向 object-level change reasoning”的论文设想一致。

主要问题：

- 验证集波动大；
- test F1 暂时低于 ChangeMambaLite；
- mask proposal 过早受阈值影响，容易出现召回不足；
- 当前 loss 仍只监督最终二值图，没有直接约束 mask proposal 的目标完整性。

## 下一步修改建议

1. 加入 mask sparsity / area regularization，抑制无效 proposal；
2. 将 `MaskConsistencyLoss` 接入训练入口，减少内部空洞和碎片；
3. 增加 auxiliary pixel head，让 early training 更稳定；
4. 将 query 生成从固定池化升级为 top-k change attention selection；
5. 正式训练时先用 LEVIR-CD 跑 50 到 100 epoch，再和 ChangeMambaLite、MaskCD official 对比。

## 当前结论

不建议放弃 CDMamba-MaskCD 融合方向。  
更合理的判断是：方向成立，但第一版 mask decoder 需要更强监督与正则化，才能把“目标级变化推理”的优势转化为稳定指标提升。
