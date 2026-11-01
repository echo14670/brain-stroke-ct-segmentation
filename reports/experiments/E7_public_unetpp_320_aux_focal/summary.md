# E7：公开 U-Net++ 320×320 + 辅助 Focal Tversky

## 设置

该实验与 E6 使用相同数据、随机切分、U-Net++、EfficientNet-B0、数据增强、
AdamW、余弦退火、batch size 8、50 epochs 和阈值 0.5，仅修改损失函数：

`0.8 × (0.5 × Dice + 0.5 × BCE) + 0.2 × FocalTversky`

- Kaggle：https://www.kaggle.com/code/echo14aa/ct-e7-u-net-320-focal
- 测试集与 E6 完全一致，共 998 例
- E6/E7 的 `test_set.csv` SHA256 一致

## 最终结果

| 指标 | E6 | E7 | 差值 E7-E6 |
| --- | ---: | ---: | ---: |
| 最佳验证 Dice | 0.8636 | 0.8587 | -0.0049 |
| all-image Dice | 0.8466 | 0.8523 | +0.0057 |
| IoU | 0.8162 | 0.8197 | +0.0035 |
| Precision | 0.9144 | 0.8974 | -0.0170 |
| Recall | 0.8765 | 0.8860 | +0.0095 |
| 阳性病例 Dice | 0.8378 | 0.8449 | +0.0071 |
| 阳性病例 Precision | 0.9094 | 0.8925 | -0.0169 |
| 阳性病例 Recall | 0.8694 | 0.8795 | +0.0101 |

最佳验证 Dice 出现在第 40 轮。

像素混淆统计：TP 663917、FP 298086、FN 269567、TN 100963630。

## 结论

辅助 Focal Tversky 提高了召回和逐图像平均 Dice，但 Precision 下降，假阳性增加。
在全局像素 Dice 上，E7 没有超过 E6，因为 FP 增加幅度大于 FN 减少幅度。该结果说明
小权重 Focal 可以作为召回调节项，但当前设置没有带来稳定的分割质量突破。

## 产物

- 配置与指标：`config.yaml`、`metrics.json`
- 训练与测试记录：`train_history.json`、`run.log`、`test_set.csv`
- 可视化：`training_history_full.png`、`pixel_confusion_matrix.png`
- 预测样例：`test_predictions.png`、`viz_bleeding_grid.png`、`viz_ischemia_grid.png`
