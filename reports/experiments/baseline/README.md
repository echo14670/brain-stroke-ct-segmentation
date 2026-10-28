# 基线实验目录

本目录保存 `baseline-v1` 正式控制组的配置、日志、指标、曲线和逐病例结果。

基线定义为基础红色掩码、无数据增强、固定切分、简单 2D U-Net、256×256、Dice+BCE、从零训练。模型权重通过 GitHub Release `baseline-v1` 交付，不提交到 Git。

E1 数据质量修正、E2 预训练 ResNet34、E3 Focal Tversky Loss 和 E4 320×320 输入均从 `baseline-v1` 并行开始，不能覆盖本目录中的基线结果。
