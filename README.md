# 医学图像处理期末项目：任务四

## 项目信息

- 选题：任务四 Brain Stroke CT Dataset
- 任务类型：CT 图像中的脑卒中病灶分割
- 分割目标：缺血性脑卒中、出血性脑卒中区域，共 1 个前景类别
- Kaggle 数据源：https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset
- 课程材料：[期末项目+第五章.pdf](docs/期末项目+第五章.pdf)

## 工作区结构

```text
.
├── data/
│   └── raw/
│       └── Brain_Stroke_CT_Dataset/
├── docs/
│   └── 期末项目+第五章.pdf
├── README.md
└── .gitignore
```

## 数据集概览

| 分组 | 图像数 | DICOM | PNG | OVERLAY | MASKS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bleeding | 1093 | 1093 | 1093 | 1093 | 0 |
| Ischemia | 1130 | 1130 | 1130 | 1130 | 0 |
| Normal | 4427 | 4427 | 4427 | 0 | 0 |
| External_Test | 200 | 200 | 200 | 200 | 200 |
| 合计 | 6850 | 6850 | 6850 | 2423 | 200 |

- 工作区数据集共 16,324 个文件，约 1.43 GiB。
- 主数据集共 6650 例，其中病灶组 2223 例，Normal 4427 例。
- 外部测试集共 200 例，标签分布在 `External_Test/labels.csv` 中：阴性 130 例，阳性 70 例。
- PNG 图像以 512x512 为主，但主数据集有 248 例、外部测试集有 8 例尺寸不是 512x512。
- 主数据集各分组内，PNG、DICOM 和 OVERLAY 的文件编号可以一一对应。
- 外部测试集的 PNG、DICOM、OVERLAY、MASKS 和 labels.csv 编号可以一一对应。
- 外部测试集与主数据集没有重复图像编号。

## 检查结论

1. `OVERLAY` 不是单通道二值掩码，而是叠加了红色或蓝色病灶标记的 RGB/RGBA 可视化图。若用它训练分割模型，需要先通过颜色或原图差分提取前景掩码。
2. 主数据集的 Normal 组没有 OVERLAY。若构建统一分割训练集，应将 Normal 样本配成全零掩码，或仅使用病灶组做病灶学习、用 Normal 做假阳性控制。
3. `External_Test/MASKS` 是 0/255 的二值标注，但 7 个 `Stroke=1` 样本的掩码全零，存在标签与掩码不一致的风险，训练和评估前应单独核查。
4. 部分图像不是 512x512，读取后需要统一 resize 或 pad/crop，并同步处理 mask，避免图像和标注错位。
5. 外部测试集没有与主数据集重复编号，可作为独立测试划分；仍需按团队数据清洗策略再次确认是否存在同一患者或近似切片泄漏。

## 期末交付提醒

课程材料要求提交汇报 PPT、每人独立项目报告、代码与模型，并在汇报中包含项目背景、方法、定量结果、结论和团队分工。
## 基线代码

- 配置：`configs/baseline.yaml`
- 训练：`python scripts/train.py --config configs/baseline.yaml`
- 评估：`python scripts/evaluate.py --config configs/baseline.yaml`
- 外部测试推理：`python scripts/infer.py --config configs/baseline.yaml`
- 基线实现：从零训练的简单 2D U-Net、256x256 单通道灰度、Dice+BCE、统一阈值 0.5。
- 数据读取：Normal 使用全零掩码；Bleeding 和 Ischemia 从 OVERLAY 提取红色前景；External_Test 使用官方 MASKS。
- 训练进度：控制台显示 tqdm 进度条，同时写入 `reports/experiments/baseline/run.log` 和 `reports/experiments/baseline/progress.json`。
- 运行结果：`metrics.json`、`summary.md`、`curves.png`、`test_per_case.csv`、`external_test_per_case.csv` 和外部测试预测掩码。

当前 `data/splits/main_split.json` 是基线负责人生成的 provisional 固定切分，训练集 4654 例、验证集 998 例、内部测试集 998 例、External_Test 200 例。正式实验前必须由数据负责人用冻结版 `main_split.json` 替换，并重新运行基线。

Kaggle 云端入口位于 `kaggle/`。Notebook 会把嵌入的源码包解压到 `/kaggle/working`，因此云端运行时不需要访问 GitHub；使用 Kaggle CLI 推送和查看状态：

```powershell
kaggle kernels push -p kaggle
kaggle kernels status echo14aa/brain-stroke-baseline
kaggle kernels output echo14aa/brain-stroke-baseline -p outputs/kaggle-baseline
```
## 当前正式基线

`baseline-v1` 已完成 Kaggle GPU 训练，定义为基础红色掩码、无数据增强、简单 2D U-Net、256×256、Dice+BCE、从零训练。结果和逐病例记录位于 `reports/experiments/baseline/`，模型权重通过 GitHub Release `baseline-v1` 交付。

E1 数据质量修正、E2 预训练 ResNet34、E3 Focal Tversky Loss 和 E4 320×320 输入均可从 `baseline-v1` 并行开始；任何实验都不得覆盖该控制组结果。
