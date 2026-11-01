# E4 320×320 输入分辨率：Issue 与 PR 草稿

> 计划 §1.6 要求 PR 包含关联 Issue、修改文件、训练评估命令、切分与种子、阳性 Dice/IoU/Precision/Recall、
> Normal 假阳性、External_Test 结果和结论。以下内容可直接粘到 GitHub。文本按 `AGENTS.md` 使用简体中文。

## Issue

**标题**：E4 320×320 输入分辨率对照实验

**负责人**：冯玄（成员 7，输入分辨率实验负责人）

**目标与背景**
基线 `baseline-v1` 在 256×256 输入下内部阳性 Dice 仅 0.3016，Ischemia 分组 Dice 0.0016（几乎全零预测）。
本实验只把输入分辨率改为 320×320，验证提高分辨率能否改善病灶（尤其是缺血性卒中）的分割与检出。

**使用的数据与配置**
- 数据：`data/raw/Brain_Stroke_CT_Dataset`
- 切分：`data/splits/main_split.json`（`baseline-v1-seed42`，frozen，4654/998/998 + 200）
- 配置：`configs/experiments/E4_res320.yaml`
- 唯一变量：`data.image_size` 256 → 320；模型、损失、优化器、batch 8、15 轮、早停 4、阈值 0.5、评估脚本全部与基线一致

**需要修改的文件范围**
- `configs/experiments/E4_res320.yaml`（新增）
- `reports/experiments/E4_res320/`（新增）
- `kaggle/build_e4_notebook.py`、`kaggle/E4_res320/`（新增，云端执行包）
- 不改 `src/`、`scripts/`、`data/splits/`

**训练与评估命令**

```powershell
python -m unittest discover -s tests -v
python scripts/train.py    --config configs/experiments/E4_res320.yaml
python scripts/evaluate.py --config configs/experiments/E4_res320.yaml
python scripts/infer.py    --config configs/experiments/E4_res320.yaml
```

**交付文件**：`reports/experiments/E4_res320/` 全套（配置、指标、日志、曲线、逐病例 CSV、分层统计、环境、切分快照）
**验收标准**：仅输入分辨率一个变量与基线不同；指标由统一 `scripts/evaluate.py` 生成；External_Test 不参与调参

## Pull Request

**关联 Issue**：#（待填）

**修改了哪些文件**
- `configs/experiments/E4_res320.yaml`
- `reports/experiments/E4_res320/`（23 个文件，含 `summary.md`、`metrics.json`、`plan_mapping.md`、`PR_draft.md`，详见 `submission_manifest.json`）
- `kaggle/build_e4_notebook.py`、`kaggle/E4_res320/`（6 份 Notebook + `kernel-metadata.json` + `dataset-metadata.json` + `make_samples.py` + `README.md`）
- 未修改 `src/`、`scripts/`、`tests/`、`data/splits/main_split.json`（与远端 `main` 逐文件 SHA256 一致）

**训练与评估命令**：同上

**使用的数据切分和随机种子**
- 切分：`baseline-v1-seed42`，frozen，成员哈希 `db54033f24204e97faae21e80e15c8104ecb783f9feab5b3c894debf6f62f58c`
- 切分文件 SHA256：`4853e4fe55dac627ae4a5daba918aa00f46feb156932e909cbce5802742a7052`
- 随机种子：42；阈值：0.5；batch size：8（与基线相同，未因显存调整）

**指标（内部测试 998 例，阳性 334 例）**

| 指标 | baseline-v1 | E4 | 差值 |
| --- | ---: | ---: | ---: |
| 阳性 Dice | 0.30156 | 0.41495 | +0.11339 |
| 阳性 IoU | 0.24027 | 0.32054 | +0.08027 |
| Precision | 0.52348 | 0.49465 | −0.02883 |
| Recall | 0.22386 | 0.42828 | +0.20442 |
| Bleeding Dice | 0.61249 | 0.58953 | −0.02297 |
| Ischemia Dice | 0.00160 | 0.24653 | +0.24493 |

**Normal 假阳性**：像素比例 0.002109 → 0.003529（约为基线的 1.67 倍）

**External_Test 结果**
- 阳性 Dice：0.40315 → 0.44280（+0.03965）
- 像素 Recall：0.33032 → 0.48585（+0.15553），Precision：0.58003 → 0.45692（−0.12311）
- 图像级 F1：0.54717 → 0.59292（+0.04575），漏检 12 例 → 3 例

**推理时间与显存（计划 §2.2-4 / §2.3 要求）**

| 口径 | 320×320 | 256×256 | 比值 |
| --- | ---: | ---: | ---: |
| 纯前向 batch 1（ms/图） | 5.872 | 3.608 | 1.627 |
| 纯前向 batch 16（ms/图） | 4.883 | 3.141 | 1.554 |
| 端到端（解码+resize+前向）batch 1 | 242.672（30 例均值） | — | — |

- 分辨率代价与像素比 1.5625 基本一致，说明只按像素线性变慢
- 端到端 242.672 ms/图中 GPU 前向仅占 ~2.4%，其余为 512×512 PNG 解码 + resize 的 CPU 开销（单进程上界口径）
- 训练显存：batch 8 @320 allocated 1.20 GiB / reserved 1.62 GiB（显存 14.56 GiB）；评估 batch 16 为 1.07 GiB

**结论：达到改进目标**
阳性 Dice +0.1134（远超计划 §2.3 的 +0.02 门槛），且 External_Test 的分割 Dice 与图像级 F1 均未下降。
收益主要来自"Ischemia 整类从学不出来到可用"与"大病灶召回"；按病灶大小分层，最小分箱（GT 占比 ≤0.304%）
Dice 为 −0.0267，**因此不宣称改善小病灶**。代价是 Normal 假阳性与外部误报上升。
建议进入最终组合，优先级 E4+E1 或 E4+E2；若与 E3 组合可同时压低假阳性。

**评审注意**
- 验证集 Dice 在第 6–8 轮塌缩为 0（与基线同现象），最佳轮次按验证集选择、未看测试集
- 单一固定种子；进入最终组合前需第二种子复跑
- 推理延迟已补测（`latency_benchmark.json`，T4 + torch 2.10.0+cu128 + 权重 epoch 5）；测速格先落脚本再用子进程执行，
  以避免 kernel 内缓存的预装 torch 2.11.0 污染口径
- 权重未进 Git，需按 §1.8 通过 Release/共享盘交付（`best.pt` 31,126,213 字节）
