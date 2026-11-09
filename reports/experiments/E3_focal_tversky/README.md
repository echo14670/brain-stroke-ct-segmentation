# E3 交付说明

本目录是两组验证试验及最终评估的汇总记录。最终候选为 **batch=True，第11轮best.pt**。第一组仅用于验证比较与退化诊断，没有运行其内部测试或外部最终评估。

## 文件对应关系

| 文件 | 用途 |
|---|---|
| `configs/experiments/E3_focal_tversky.yaml` | 第一组：逐图计算，默认batch=False |
| `configs/experiments/E3_focal_tversky_batch.yaml` | 最终选择：整批汇总TP/FP/FN，batch=True |
| 本目录`config.yaml`、`train_summary.json`、`run.log`、`curves.png` | 第二组15轮训练记录 |
| `metrics.json`、两个`*_per_case.csv` | 最终候选的统一评估结果 |
| `final_selection.json` | 验证集选择依据、epoch、权重SHA256 |
| `summary.md` | 基线对照、结论、失败方向和复现命令 |
| `validation_trials/per_image/` | 第一组7轮训练与验证集退化诊断证据 |
| `submission_manifest.json` | 数据、权重、运行版本和交付文件索引 |

汇总目录名按任务要求保留`E3_focal_tversky`；其中最终模型的experiment/output_dir字段保留真实运行名称`E3_focal_tversky_batch`，没有改写原始指标。复现运行会写入配置指定目录。第一组配置会写入本汇总目录，复跑前应使用独立工作副本，避免覆盖历史记录。

## 运行方法

从仓库根目录运行，环境沿用团队基线的PyTorch 2.10.0+cu128。数据挂载到配置路径，或设置`BSC_DATA_ROOT`。

```powershell
python -m unittest discover -s tests -v
python scripts/train.py --config configs/experiments/E3_focal_tversky_batch.yaml
python scripts/evaluate.py --config configs/experiments/E3_focal_tversky_batch.yaml --checkpoint reports/experiments/E3_focal_tversky_batch/best.pt
python scripts/infer.py --config configs/experiments/E3_focal_tversky_batch.yaml --checkpoint reports/experiments/E3_focal_tversky_batch/best.pt
```

只复核已完成模型时，跳过train.py，将交付包中的best.pt放到上面指定路径后运行evaluate.py和infer.py。测试集和外部数据只用于最终报告，不用于继续选择参数或阈值。

Kaggle训练使用`kaggle/E3_focal_tversky_batch_train.ipynb`。仅最终评估使用`kaggle/E3_final_evaluation.ipynb`：挂载原数据集、两个离线wheel数据集以及本账号Notebook的Version 6输出；运行时会严格核对所选权重SHA256和epoch=11。该Notebook没有训练单元格。`kaggle/kernel-metadata.json`属于团队原始基线，上传E3时应使用自己的Notebook，不用该文件覆盖基线。

## 模型和预测交付

E3代码与实验报告已通过主仓库PR #4合并。本次按模型所有者明确要求，补交最终选择的第11轮权重，直接保存到配置默认读取的位置：

```text
reports/experiments/E3_focal_tversky_batch/best.pt
reports/experiments/E3_focal_tversky_batch/model_manifest.json
```

权重为31,126,405字节，SHA256为`522b14b7f4e85fbc9f2907a8fa5b7ce0a6abd62244b6f221ff0392d252b3a0db`。校验信息、选择依据及配置路径保存在model_manifest.json中。该路径可直接用于上面的evaluate.py和infer.py命令，不需要重新训练。

原有[实验Release](https://github.com/tw1l1ghtcc/brain-stroke-ct-segmentation/releases/tag/e3-focal-tversky-v1)继续提供best.pt、结果报告ZIP及校验文件，200张外部预测掩码位于Release中的E3_evaluation_reports.zip内。本次没有重新训练、修改参数或更新实验指标。

本次权重直接进入Git是模型所有者针对该文件的交付要求；其他模型、数据集及训练缓存仍遵循团队既有交付约定。
