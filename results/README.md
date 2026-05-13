# Results Directory

本目录用于保存实验输出结果，本地使用时可按照如下结构组织：

```text
results/
  baseline/
  changemamba/
  changevit/
  maskcd/
  changeclip/
```

建议保存内容包括：

- 训练日志
- 模型权重
- `metrics.json`
- 可视化结果图

训练权重和大体量中间结果默认不纳入 Git 版本管理。
