# E2：预训练 ResNet34 编码器

## 实验目的与变量

- 方向：把从零训练的简单 2D U-Net 编码器，替换为 ImageNet 预训练的 ResNet34 编码器。
- 唯一变量：**编码器**。数据、切分、损失（Dice+BCE）、输入尺寸（256）、评估脚本、预测阈值（0.5）均与正式基线 `baseline-v1` 一致。
- 实现：`segmentation-models-pytorch` 的 `Unet + resnet34`，`encoder_weights='imagenet'`，`in_channels=1`，`classes=1`；通过 `model.name: smp_unet` 选择。
- 参数量：24,430,097（基线简单 U-Net 为 7,765,409）。

## 配置摘要

- 配置：`configs/experiments/E2_resnet34.yaml`（主，seed=42）、`configs/experiments/E2_resnet34_seed2.yaml`（第二随机种子复跑，seed=2）
- 切分：`data/splits/main_split.json`（冻结版 `baseline-v1-seed42`，train 4654 / validation 998 / test 998，外部 200）
- 训练：epochs 15、batch 8、lr 3e-4、AMP、早停 patience 4

## 结果

### 两个随机种子（内部测试·阳性病例）

| 运行 | 最佳 epoch | Dice | IoU | Precision | Recall | Normal 假阳性像素比 | 外部图像级 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 基线 | 4 | 0.3016 | 0.2403 | 0.5235 | 0.2239 | 0.002109 | 0.5472 |
| E2 seed=42 | 2 | 0.2618 | 0.2146 | 0.9223 | 0.1252 | 0.0000225 | 0.6207 |
| E2 seed=2 | 1 | 0.1861 | 0.1364 | 0.7906 | 0.0499 | 0.000214 | 0.5315 |

### 分组 Dice（内部测试）

| 运行 | Bleeding | Ischemia |
| --- | ---: | ---: |
| 基线 | 0.6125 | 0.0016 |
| E2 seed=42 | 0.5332 | 0.0000 |
| E2 seed=2 | 0.3786 | 0.0005 |

### 图像级检测（E2 seed=42）

- 内部测试：tp=138 / fp=21 / fn=196 / tn=643，Precision 0.868，Recall 0.413，F1 0.560（基线：tp=284 / fp=408 / fn=50 / tn=256，F1 0.554）
- 外部测试：tp=36 / fp=10 / fn=34 / tn=120，Precision 0.783，Recall 0.514，F1 0.621

## 关键发现：验证集塌缩，且两个种子均复现

- seed=42：第 1 轮 val_dice 0.217 → 第 2 轮 0.268（最佳、保存）→ 第 3 轮起 **0.0**，第 6 轮早停。
- seed=2：最佳仅到**第 1 轮**，之后同样塌缩。
- 说明：验证 Dice 归零意味着模型塌缩为"几乎全部预测为背景"。**该现象与随机种子无关，可复现**，是 Dice+BCE 在极端类别不平衡（病灶像素极少）下的训练不稳定，而非编码器实现问题。基线（简单 U-Net）在早期轮次同样出现过该塌缩。
- 后果：用于评估的 `best.pt` 是塌缩前的早期半成品，因此 Dice/Recall 显著低于基线。

## 与基线的对照解读

- Dice、IoU、Recall 均低于基线；Precision 与 Normal 假阳性指标看似大幅改善（假阳性约降 94 倍），本质是模型"少预测"造成的**工作点变保守**，不能视为模型更强。
- 外部图像级 F1：seed=42 为 0.6207（高于基线），但 seed=2 为 0.5315（低于基线）→ **外部提升不稳定，不能作为结论**。
- 进入"进展"判定（分工文档 2.3）：仅命中第 3 条"Normal 假阳性明显下降"，但该项由召回骤降换得，属于弱进展。

## 局限与归属

1. **训练不稳定**：两个种子均塌缩，说明瓶颈在训练配方（损失/类别不平衡），预训练编码器无法单独解决；这正好为 E3（Focal Tversky Loss）和数据侧改进提供了依据。
2. **Ischemia 约等于 0**：基线使用 `mask_mode: red`，只提取红色前景，缺血病灶多为蓝色标记，红色掩码中缺失 → 缺血分割学不到。**该问题归属 E1（红蓝并集掩码）**，不是编码器的问题；E1 冻结后需用新掩码统一复跑。
3. **验证轮次过少**：最佳 checkpoint 落在第 1–2 轮，早停过早，不足以代表 ResNet34 的真实上限。
4. **切分口径**：基线 `metrics.json` 记录为 provisional 切分，本实验使用冻结切分 `baseline-v1-seed42`；两者分组样本数完全一致（Bleeding 164 / Ischemia 170 / Normal 664），实为同一份切分，但严格对照前建议基线用冻结切分复跑一次。

## 结论

- **不建议单独采用预训练 ResNet34 作为最终模型**：在该固定训练配方下，它导致验证集塌缩，表现为保守预测（高精度、极低假阳性、低召回、Ischemia 未检出），分割与召回均劣于基线。
- **编码器不是当前瓶颈**；主要瓶颈是类别不平衡下的训练不稳定（损失/数据），指向 E3 与 E1。
- 可作为**最终组合的潜在方向**（例如与 E3 或 E1 组合），但前提是先解决塌缩问题，并复跑验证。

## 归档说明

- 完整可复现归档：`reports/experiments/E2_resnet34_seed2/`（含 `run.log`、`curves.png`、逐病例 CSV）。
- 本目录（seed=42）：含 `config.yaml`、`metrics.json`、`train_summary.json`、`evaluation_config.yaml`；原始 `run.log`、`curves.png` 与逐病例 CSV 未归档（当时未下载）。
- 模型权重不进入 Git，应通过 GitHub Release 或共享盘交付。
