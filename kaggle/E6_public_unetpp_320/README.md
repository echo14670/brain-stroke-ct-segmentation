# E6：公开 U-Net++ 320×320

本目录是公开 U-Net++ Notebook 的直接派生版本，仅将输入尺寸从
256×256 改为 320×320。模型、损失、增强、训练轮数和评估逻辑保持公开代码设置。

生成 Notebook：

```powershell
python kaggle\build_e6_public_320.py --source <公开Notebook路径>
```

上传并启动 T4 GPU：

```powershell
kaggle kernels push -p kaggle/E6_public_unetpp_320
kaggle kernels status echo14aa/brain-stroke-e6-public-unetpp-320
```
