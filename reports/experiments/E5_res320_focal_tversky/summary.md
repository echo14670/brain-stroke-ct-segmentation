# E5：320×320 + Focal Tversky 组合实验

## 设置

基于 E4 的 320×320 简单 2D U-Net，将损失函数替换为 E3 的批次版 Focal Tversky。
数据、冻结切分、seed 42、batch size 8、优化器、15 轮上限、早停和阈值 0.5 均与 E4 一致。

## 最终结果

| 指标 | E4 | E5 | 差值 E5-E4 |
| --- | ---: | ---: | ---: |
| 内部阳性 Dice | 0.41495 | 0.32968 | -0.08526 |
| 内部阳性 IoU | 0.32054 | 0.26470 | -0.05584 |
| 内部 Precision | 0.49465 | 0.66818 | +0.17353 |
| 内部 Recall | 0.42828 | 0.28242 | -0.14586 |
| Normal 假阳性像素比例 | 0.003529 | 0.000148 | -0.003380 |
| Bleeding Dice | 0.58953 | 0.59313 | +0.00360 |
| Ischemia Dice | 0.24653 | 0.07553 | -0.17099 |
| External 分割 Dice | 0.44280 | 0.42068 | -0.02212 |
| External 图像级 F1 | 0.59292 | 0.66667 | +0.07375 |

最佳验证 Dice 为 0.34144，出现在第 11 轮，共完成 15 轮。

## 结论

直接替换 E4 的 Dice+BCE 后，模型明显提高 Precision 并压低假阳性，但牺牲了阳性 Dice、
Recall 和 Ischemia 分割，没有继承 E4 的主要收益。该结果不支持把 Focal Tversky 直接作为
最终分割模型的唯一损失，但支持后续尝试小权重辅助损失。

## 产物

- 训练与评估配置：`config.yaml`、`evaluation_config.yaml`
- 指标与训练记录：`metrics.json`、`train_summary.json`、`progress.json`、`run.log`
- 结果清单：`test_per_case.csv`、`external_test_per_case.csv`
- 曲线与验证记录：`curves.png`、`e5_tests.log`、`e5_cuda_smoke.log`
