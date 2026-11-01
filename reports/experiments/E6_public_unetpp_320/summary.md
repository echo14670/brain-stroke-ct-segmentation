# E6：公开 U-Net++ 的 320×320 复现

## 设置

该实验派生自 Kaggle 数据集页面关联的公开 U-Net++ Notebook。模型使用
EfficientNet-B0 ImageNet 预训练编码器、Dice+BCE `0.5/0.5`、数据增强、AdamW、
余弦退火、batch size 8、50 epochs 和阈值 0.5。相对公开代码仅把输入从 256×256
改为 320×320。

- Kaggle：https://www.kaggle.com/code/echo14aa/ct-e6-u-net-320
- 测试集：seed 42 随机图像级切分，共 998 例
- 类别数量：Bleeding 154、Ischemia 180、Normal 664

## 最终结果

| 指标 | 原始公开 U-Net++ | E6 |
| --- | ---: | ---: |
| all-image Dice | 0.8354 | 0.8466 |
| IoU | 0.8154 | 0.8162 |
| 阳性病例平均 Dice | 未报告 | 0.8378 |
| Precision | 未报告 | 0.9144 |
| Recall | 未报告 | 0.8765 |
| Pixel Accuracy | 未报告 | 0.9948 |

最佳验证 Dice 为 0.8636，出现在第 42 轮。

像素混淆统计：TP 644305、FP 237193、FN 289179、TN 101024523。

## 结论

只把公开 U-Net++ 的输入提高到 320×320，all-image Dice 提高约 0.0112，IoU 提高约
0.0008。提升幅度不大但方向明确，且验证曲线稳定，没有出现团队简单 U-Net 的验证集塌缩。

## 产物

- 配置与指标：`config.yaml`、`metrics.json`
- 训练与测试记录：`train_history.json`、`run.log`、`test_set.csv`
- 可视化：`training_history_full.png`、`pixel_confusion_matrix.png`
- 预测样例：`test_predictions.png`、`viz_bleeding_grid.png`、`viz_ischemia_grid.png`
