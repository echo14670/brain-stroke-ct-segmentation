# E4：320×320 输入分辨率（Kaggle 上传目录）

本目录是 E4 实验的云端执行包，包含 `kernel-metadata.json` 与两份由生成脚本产出的 Notebook。
Notebook 内已内嵌 `configs/ scripts/ src/ data/splits/ tests/` 和基线对照结果，云端不访问 GitHub、不联网。

## 文件

| 文件 | 用途 |
| --- | --- |
| `E4_res320_kaggle.ipynb` | 离线版第 1 版：只跑单测、单变量断言、320×320 数据形状和显存压测，**不训练** |
| `E4_res320_train.ipynb` | 离线版第 2 版：验证通过后自动训练 15 轮并打包 |
| `E4_res320_full.ipynb` | **一步到底版**：训练 + 显存/延迟基准 + 统一评估 + 外部推理 + 打包 |
| `E4_res320_latency.ipynb` | **推理延迟补测版**：不训练不评估，只输出 `latency_benchmark.json`（需挂载 `best.pt`） |
| `E4_res320_online_kaggle.ipynb` | **联网版第 1 版**：只依赖 1 个数据数据集，torch 由 pip 安装并锁死 `2.10.0+cu128` |
| `E4_res320_online_train.ipynb` | **联网版第 2 版**：验证通过后自动训练 |
| `dataset-metadata.json` | 自建私有数据集用的元数据（公开数据集失效时走这条路） |
| `make_samples.py` | 本地生成 `samples.png` 样例图（只用 numpy + Pillow，不需要 torch/GPU） |
| `kernel-metadata.json` | kernel id `wtbcdj/brain-stroke-e4-res320`；T4、GPU 开启、Internet 关闭、挂载 3 个数据集 |

## 推理延迟补测（计划 §2.2-4 / §2.3 要求）

正式训练完成后如需补齐"单张推理时间"，用 `E4_res320_latency.ipynb`：

1. **必挂 3 个输入**：`ozguraslank/brain-stroke-ct-dataset` + `amitmalakarjoy/torch-2100-cu128` +
   `koumeimaki/vllm019-cp311-cu128-wheelhouse`（环境格强制，缺任一都会直接报错）；
2. GPU 选 T4，Internet 关闭；
3. 权重 **可选**：纯前向延迟只取决于网络结构与输入尺寸，与权重数值无关。若想记录 epoch，把结果包
   `brain-stroke-e4-*-e4-results.zip` 上传为私有数据集（或 *Add Input → Your Work* 选上一次 Notebook 输出），
   环境格会在 `/kaggle/input` 下递归找 `best.pt`；找不到就用同结构随机初始化权重，并在 JSON 里标
   `"weights": {"source": "random_init"}`；
4. Run All，约 2 分钟，输出 `latency_benchmark.json`（纯前向 320/256 × batch 1/16 + 端到端 30 例均值）。

测速格不在 kernel 进程内直接跑：它把脚本写到 `repo/e4_latency_bench.py`，再用 `subprocess` 以 `cwd=repo` 启动
新解释器执行。这样才能同时满足两个条件——`src` 可导入（kernel 的 cwd 是 `/kaggle/working`，
不在 `sys.path` 里），以及加载的是 2.10.0+cu128（环境格在 pip 替换之前已 `import torch`，
内核里缓存的是 Kaggle 预装的 2.11.0）。原始输出同时写入 `e4_latency_bench.log`。

## 样例图生成（本地，无需 GPU）

```powershell
python kaggle\E4_res320\make_samples.py --predictions <解压后的 predictions\external 目录>
```

输出 `reports/experiments/E4_res320/samples.png` 与 `samples_index.json`（红=GT、绿=预测、黄=重叠）。

## 数据来源的三条路

1. **公开数据集仍可用**：挂载 `ozguraslank/brain-stroke-ct-dataset` + 两个 wheel 数据集，用离线版 Notebook、Internet 关闭。
2. **公开数据集已被删除或转私有**（在 Add Input 里把筛选切成 `Datasets` 后搜不到）：用仓库里的
   `data/raw/Brain_Stroke_CT_Dataset`（1.43 GB，16,342 个文件）自建一个私有数据集，再配合**联网版** Notebook：

   ```powershell
   Copy-Item kaggle\E4_res320\dataset-metadata.json data\raw\
   kaggle datasets create -p data\raw --dir-mode zip
   Remove-Item data\raw\dataset-metadata.json
   ```

   也可以直接在网页上 *Datasets* → *New Dataset* 拖拽上传同一个目录的 zip。
   数据集创建完成后，把 Input 里挂载它的路径交给 Notebook——环境格会自动在 `/kaggle/input` 下递归查找
   含 `Bleeding / Ischemia / Normal / External_Test` 的 `Brain_Stroke_CT_Dataset` 目录。
3. **只有 wheel 数据集失效**：仍可用公开 CT 数据集，但改挂联网版 Notebook。


两份 Notebook 都由 `kaggle/build_e4_notebook.py` 生成，改动配置或代码后重新生成，不要手改 ipynb：

```powershell
cd kaggle
python build_e4_notebook.py            # 生成 E4_res320_kaggle.ipynb（验证版）
python build_e4_notebook.py --formal   # 生成 E4_res320_train.ipynb（正式训练版）
```

## 上传与执行

前提：本机已有 `%USERPROFILE%\.kaggle\kaggle.json`，且 Kaggle 账号能挂载上面三个数据集。

```powershell
# 第 1 版：验证（kernel-metadata.json 的 code_file 需指向 E4_res320_kaggle.ipynb）
kaggle kernels push -p kaggle/E4_res320
kaggle kernels status wtbcdj/brain-stroke-e4-res320
kaggle kernels output wtbcdj/brain-stroke-e4-res320 -p outputs/kaggle-e4-verify
```

确认 `e4_tests.log`、`e4_cuda_smoke.log` 里的单变量断言、数据形状和显存占用都通过后，再切第 2 版：

```powershell
# 把 code_file 改为 E4_res320_train.ipynb 后重新 push，形成 Version 2
kaggle kernels push -p kaggle/E4_res320
kaggle kernels output wtbcdj/brain-stroke-e4-res320 -p outputs/kaggle-e4-train
```

## 实验口径

- 唯一变量：`data.image_size` 256 → 320；模型、损失、优化器、`batch_size: 8`、15 轮上限、
  早停 4 轮、种子 42、阈值 0.5、冻结切分 `baseline-v1-seed42`、统一 `scripts/evaluate.py` 全部与基线一致。
- 图像 BILINEAR、掩码 NEAREST，从同一张原图同步缩放到 320×320。
- 显存不足时不允许改 batch size；`scripts/train.py` 目前没有梯度累积，需要时由基线负责人评估。
- 测试集与 External_Test 只用于最终报告，不参与调参、选轮次或改阈值。

## 取回后

把结果目录内容复制到仓库 `reports/experiments/E4_res320/`，并由 `summary.md` 给出与
`reports/experiments/baseline/metrics.json` 的逐项对照（Dice、IoU、Precision、Recall、
Normal 假阳性、Bleeding/Ischemia 分组、External 图像级 F1、参数量与耗时）。
