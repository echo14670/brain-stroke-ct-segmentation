# E5：320×320 + Focal Tversky

本目录用于上传 E4 与 E3 的组合实验。唯一新增的模型变量是损失函数：
在 E4 的 320×320 输入上使用 E3 的 batch 版 Focal Tversky Loss。

生成 Notebook：

```powershell
python kaggle\build_e5_notebook.py
```

上传并启动 T4 GPU：

```powershell
kaggle kernels push -p kaggle/E5_res320_focal_tversky
kaggle kernels status echo14aa/brain-stroke-e5-res320-focal-tversky
```

训练进度同时输出到 Notebook 标准输出、`run.log` 和 `progress.json`。
完成后会继续运行统一评估、外部推理并打包结果。
