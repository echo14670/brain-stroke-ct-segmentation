# E3 Focal Tversky Loss 最终对照结果

仅改变损失函数，其余沿用baseline-v1：U-Net、红色掩码、无增强、256×256、seed42、冻结病例划分、阈值0.5。按验证集在两个预定候选中选择批次汇总版第11轮，最佳验证Dice 0.32277152。未根据测试或外部数据调整模型、阈值及参数。

| 指标 | baseline-v1 | E3 | 差值（E3−基线） |
|---|---:|---:|---:|
| 内部阳性Dice | 0.30155889 | 0.28681671 | -0.01474218 |
| 内部阳性IoU | 0.24027054 | 0.23283475 | -0.00743579 |
| 内部Precision | 0.52348424 | 0.62900301 | +0.10551877 |
| 内部Recall | 0.22386459 | 0.23325691 | +0.00939233 |
| Normal假阳性像素比例 | 0.00210891 | 0.00011249 | -0.00199642 |
| Bleeding Dice | 0.61249297 | 0.58326379 | -0.02922919 |
| Ischemia Dice | 0.00159895 | 0.00083248 | -0.00076647 |
| 外部图像级F1 | 0.54716981 | 0.60606061 | +0.05889079 |

## 结论与建议

本次未达到提高阳性分割Dice的预期：内部Dice下降0.01474，Bleeding与Ischemia分组Dice均下降。Recall仅提高0.00939，不能据此声称小病灶召回明显改善。Ischemia Dice仍接近0，是主要失败方向。

收益主要在减少误报：Normal假阳性像素比例约下降94.67%，Precision提高0.10552，外部图像级F1提高0.05889。按阳性Dice作为主指标时，暂不建议以该损失替换基线；可向组长报告减少误报的取舍，由团队与其他方向共同决定是否进入最终组合。两个候选、一个固定种子不支持统计显著性结论。

## 实验记录

- 第一组逐图计算：7轮，最佳第3轮，验证Dice 0.24845696；第4–7轮Dice为0，末轮全背景预测已通过验证集诊断确认。
- 第二组批次汇总：15轮，最佳第11轮，验证Dice 0.32277152；无Dice归零，第12轮降至0.03219503后恢复。单次固定种子仍有波动。
- 参数量7,765,409；第二组训练1146.3922秒。未额外测量纯推理延迟。
- 统一scripts/evaluate.py评估998个内部测试样本、200个外部样本；scripts/infer.py导出200张原尺寸外部预测掩码。
- 23项测试及CUDA AMP通过；权重SHA256匹配，checkpoint epoch=11；下载ZIP CRC通过。
- 基线已发布指标使用旧provisional名称；当前冻结切分只改元数据，病例成员关系沿用相同划分。

## 复现命令

```powershell
python scripts/train.py --config configs/experiments/E3_focal_tversky_batch.yaml
python scripts/evaluate.py --config configs/experiments/E3_focal_tversky_batch.yaml --checkpoint reports/experiments/E3_focal_tversky_batch/best.pt
python scripts/infer.py --config configs/experiments/E3_focal_tversky_batch.yaml --checkpoint reports/experiments/E3_focal_tversky_batch/best.pt
```

最终评估使用Kaggle Version 7。训练Version 6、原始Version 4与诊断Version 5均保留。权重和预测图片在本地结果包内，不加入Git。
