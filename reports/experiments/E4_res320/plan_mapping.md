# E4 与《医学图像处理期末项目协作与分工方案》条款对应表

本表把 `docs/医学图像处理期末项目协作与分工方案.md` 中与 E4 相关的条款逐条映射到仓库产物，
便于组长（王圣杰）与评审人（林子涵）核对交付完整性。

## 一、§2.2 成员 7（输入分辨率实验负责人）详细任务与交付

| 条款 | 要求 | 对应产物 | 状态 |
| --- | --- | --- | --- |
| 1 | 从 `baseline-v1` 创建 `exp/res320` 分支 | 分支 `exp/res320` | ⚠️ 基点为本提交时的 `main`（`9836731`）。远端尚不存在 `baseline-v1` 标签（§1.5-3 由组长补），模型/损失/输入尺寸全部由配置选择，图结构未变 |
| 2 | 输入尺寸 256×256 → 320×320，图像与掩码同步缩放 | `configs/experiments/E4_res320.yaml`（`data.image_size: 320`）；`src/data/dataset.py` 图像 BILINEAR、掩码 NEAREST 同一原图缩放 | ✅ 云端断言 `(1,320,320)` 通过 |
| 3 | 相同 batch size；显存不足用 AMP/梯度累积，不改有效 batch | `train.batch_size: 8`、`train.amp: true` 与基线一致；实测 batch 8 仅占 1.62/14.56 GiB，无需累积 | ✅ |
| 4 | 比较阳性 Dice、小病灶 Recall、推理时间、显存、Normal 假阳性 | `metrics.json`、`lesion_stratified.json`、`latency_benchmark.json`、`e4_cuda_smoke.log`、`summary.md` | ✅ |
| 5 | 提交配置、训练日志、指标和结论 | `reports/experiments/E4_res320/`（23 项） | ✅ |
| 交付 1 | `E4_res320.yaml` 和必要的预处理调整 | `configs/experiments/E4_res320.yaml`（无需改预处理） | ✅ |
| 交付 2 | `reports/experiments/E4_res320/` 完整记录 | 见目录 | ✅ |
| 交付 3 | PR 与结果结论 | `reports/experiments/E4_res320/PR_draft.md` | ⏳ PR 待创建 |
| 验收 | 仅输入分辨率一个变量与基线不同 | 参数量 7,765,409 与基线完全相同；云端单变量断言逐项比对通过 | ✅ |

## 二、§1.7 结果目录必备文件

| 要求 | 产物 | 状态 |
| --- | --- | --- |
| `config.yaml` 最终实验配置 | `reports/experiments/E4_res320/config.yaml` | ✅ |
| `metrics.json` 统一评估结果 | `metrics.json`（内部 998 / External 200 / combined） | ✅ |
| `summary.md` 实验目的、对照、结论、是否建议采用 | `summary.md`（含基线指标、本实验指标、指标变化、是否达预期、主要失败案例、是否建议组合） | ✅ |
| `run.log` 训练与评估日志 | `run.log`（9 轮）+ `evaluate_output.md`（`evaluate.py` 原始输出） | ✅ |
| `curves.png`（可选） | `curves.png` | ✅ |
| `samples.png`（可选） | `samples.png` + `samples_index.json`（红=GT、绿=预测） | ✅ |

## 三、§2.3 统一评估指标清单

| 指标 | 值 | 出处 |
| --- | --- | --- |
| 阳性病例平均 Dice | 0.41495（基线 0.30156） | `metrics.json` |
| 阳性病例平均 IoU | 0.32054（基线 0.24027） | `metrics.json` |
| Precision / Recall | 0.49465 / 0.42828 | `metrics.json` |
| Normal 假阳性像素比例 | 0.003529（基线 0.002109） | `metrics.json` |
| Bleeding / Ischemia 分组 | Dice 0.58953 / 0.24653 | `metrics.json` |
| External_Test 图像级 F1 | 0.59292（基线 0.54717） | `inference_summary.json`、`metrics.json` |
| 参数量 | 7,765,409（与基线相同） | `train_summary.json` |
| 训练时间 | 1401.5 s（稳定期 100 s/epoch，9 轮早停） | `train_summary.json` |
| 单张推理时间 | 纯前向 5.872 ms/图 @320 batch1；端到端 242.672 ms/图 | `latency_benchmark.json` |

## 四、§2.4 文件责任边界自检

| 文件 | 归属 | 本次改动 |
| --- | --- | --- |
| `data/splits/main_split.json` | 数据负责人 | ❌ 未改（SHA256 `4853e4fe…7052` 与冻结值一致） |
| `scripts/train.py`、`scripts/evaluate.py`、`scripts/infer.py` | 基线负责人 | ❌ 未改（与 `main` 逐文件 SHA256 一致） |
| `src/data/`、`src/models/`、`src/losses/`、`src/evaluation/` | 数据/基线/损失负责人 | ❌ 未改 |
| `tests/` | 基线负责人 | ❌ 未改 |
| `configs/baseline.yaml` | 基线负责人 | ❌ 未改 |
| `configs/experiments/E4_res320.yaml` | **成员 7（本人）** | ✅ 新增 |
| `kaggle/` | 计划未列入责任表，沿用基线负责人的 `kaggle/` 先例 | ✅ 新增 `build_e4_notebook.py` 与 `E4_res320/`，**请基线负责人确认** |
| 模型权重、预测输出、训练缓存 | 不入 Git（§1.8） | ✅ 仅交付 zip（`best.pt` 31,126,213 字节） |

## 五、§1.9 任务完成标准（五条）

| 标准 | 状态 |
| --- | --- |
| 代码或文档已经评审并合并 | ⏳ 待 PR 评审（需非本人批准） |
| 相关配置和实验记录已提交 | ✅ 已提交到 `exp/res320` 分支，待推送 |
| 最小运行命令通过 | ✅ `train.py` / `evaluate.py` / `infer.py` 均已在云端实际运行 |
| 指标由统一评估脚本生成 | ✅ `scripts/evaluate.py`，未做任何本地改写 |
| Issue 已更新并关闭 | ⏳ 待创建 Issue #5 |

## 六、尚未完成的流程项（责任划分）

| 项 | 条款 | 责任人 |
| --- | --- | --- |
| 创建 `baseline-v1` 标签与 Release | §1.5-3 | 组长 王圣杰 |
| 创建 Issue「E4 320×320 输入分辨率」 | §1.4 | 组长 王圣杰 |
| 推送 `exp/res320` 并创建 PR | §1.6 | 成员 7 冯玄 |
| PR 评审（非本人） | §1.6 | 基线负责人 林子涵 / 组长 |
| Squash Merge 并删除分支 | §1.6 | 组长 |
| 权重上传 Release/共享盘 | §1.8 | 成员 7 冯玄 |
| 第二种子复跑（仅入选最终组合时） | §2.3 | 成员 7 冯玄（配置已可复制改写 `seed`） |
| `kaggle/` 目录归属确认 | §1.2 / §1.3 未覆盖 | 基线负责人 林子涵 |
