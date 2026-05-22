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
| CDMamba-MaskCD, MaskCD-dominant | 1.673 -> 1.163 | 0.164 -> 0.276 | 0.055 | 0.181 | 0.099 | 相比第一版明显提升，但仍低于 MaskCD official mini |

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

## 已据此完成的代码调整

根据 CDMamba 与 MaskCD 在 LEVIR-CD mini-result 上的对比，融合模型已调整为 MaskCD-dominant：

- mask proposal 分支作为主输出；
- CDMamba-style encoder 作为变化特征增强器；
- query 生成由固定池化改为 top-k change response selection；
- 新增 auxiliary pixel head；
- 最终输出采用 `0.7 * object_mask_logits + 0.3 * auxiliary_pixel_logits`。

调整后同等 mini 设置下，test F1 从 0.016 提升到 0.181，说明改动方向有效。但与 MaskCD official mini 的 test F1 0.756 相比，当前简化 decoder 仍有明显差距。

## 下一步修改建议

1. 加入 mask sparsity / area regularization，抑制无效 proposal；
2. 将 `MaskConsistencyLoss` 接入训练入口，减少内部空洞和碎片；
3. 正式训练时先用 LEVIR-CD 跑 50 到 100 epoch，再和 ChangeMambaLite、MaskCD official 对比；
4. 若 MaskCD official 始终明显领先，则将融合模型定位为“MaskCD 的 CDMamba 特征增强版”，而不是完全独立范式替代。

## 当前结论

不建议放弃 CDMamba-MaskCD 融合方向。  
更合理的判断是：方向成立，但第一版 mask decoder 需要更强监督与正则化，才能把“目标级变化推理”的优势转化为稳定指标提升。
