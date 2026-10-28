# baseline-v1 正式基线结果

## 运行状态

- 基线标签：`baseline-v1`
- Kaggle kernel：`echo14aa/brain-stroke-baseline`
- Kaggle 版本：`11`
- 云端状态：`COMPLETE`
- 训练设备：`CUDA`
- 运行地址：https://www.kaggle.com/code/echo14aa/brain-stroke-baseline
- 训练用时：`1290.0` 秒
- 实际完成：`8/15` 个 epoch，触发早停
- 最佳检查点：第 `4` 个 epoch

## 控制组定义

- 简单 2D U-Net
- 256×256 单通道灰度输入
- 基础红色 OVERLAY 掩码
- Normal 样本使用全零掩码
- 无数据增强
- `Dice+BCE`
- 从零训练
- 固定随机种子 42
- 固定阈值 0.5
- 参数量：7,765,409

## 数据切分

- 冻结版本：`baseline-v1-seed42`
- 成员哈希：`db54033f24204e97faae21e80e15c8104ecb783f9feab5b3c894debf6f62f58c`
- Train：4654
- Validation：998
- Test：998
- External_Test：200

本次云端训练启动时，切分的原始元数据标签为 `baseline-provisional-v1-seed42`。训练完成后，团队确认并冻结了相同的病例成员关系，仅更新切分版本和冻结状态，未增删任何病例。`metrics.json` 保留原始运行记录，成员清单哈希记录在 `submission_manifest.json` 中。

## 核心指标

| 指标 | 结果 |
| --- | ---: |
| 最佳验证阳性病例 Dice | 0.3324 |
| 内部测试阳性病例 Dice | 0.3016 |
| 内部测试阳性病例 IoU | 0.2403 |
| 内部测试像素 Precision | 0.5235 |
| 内部测试像素 Recall | 0.2239 |
| Normal 假阳性像素比例 | 0.0021 |
| Bleeding Dice | 0.6125 |
| Ischemia Dice | 0.0016 |
| 内部测试图像级检测 F1 | 0.5536 |
| External_Test 阳性病例 Dice | 0.4032 |
| External_Test 图像级检测 F1 | 0.5472 |

## 结果判断

训练、验证、测试、外部推理和结果导出链路均已完成，最佳检查点保存正确。该成绩可以作为团队唯一的正式控制基线，但不能作为最终性能目标。

主要问题是模型在第 5 个 epoch 后发生塌缩：验证 Dice 从第 4 个 epoch 的 `0.3324` 降至 `0.0055`，随后连续 3 个 epoch 为 0，最终触发早停。最佳检查点仍保留第 4 个 epoch 的权重。

分组结果显示模型主要学习到了 Bleeding，Ischemia 几乎未分割成功。该问题由并行的 E1 数据质量修正分支处理，不回写或覆盖本基线。

## 提交文件

- `config.yaml`
- `evaluation_config.yaml`
- `metrics.json`
- `train_summary.json`
- `progress.json`
- `run.log`
- `curves.png`
- `test_per_case.csv`
- `external_test_per_case.csv`
- `inference_summary.json`
- `submission_manifest.json`

模型权重不进入 Git，通过 GitHub Release `baseline-v1` 交付。外部测试预测掩码保留在 Kaggle 输出中，不提交到 Git。

## 复现命令

```bash
python scripts/train.py --config configs/baseline.yaml
python scripts/evaluate.py --config configs/baseline.yaml
python scripts/infer.py --config configs/baseline.yaml
```
## 模型权重校验

- 文件名：`best.pt`
- 大小：31,126,149 字节
- SHA256：`6D6283CF77EF155A9B9B50752CBA5A318F6AE749064EC2A19FD262459CA07911`
- 可加载检查：通过
- 检查点 epoch：4
- 检查点验证 Dice：0.3324415020580968
