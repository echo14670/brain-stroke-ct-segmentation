# E7：公开 U-Net++ 320×320 + 辅助 Focal Tversky

本实验沿用 E6 的公开 U-Net++、320×320、EfficientNet-B0、数据增强、
AdamW、余弦退火和 50 epochs，仅将损失函数改为：

`0.8 × (0.5 Dice + 0.5 BCE) + 0.2 × FocalTversky`

生成并上传：

```powershell
python kaggle\build_e7_public_aux_focal.py
kaggle kernels push -p kaggle/E7_public_unetpp_320_aux_focal
kaggle kernels status echo14aa/brain-stroke-e7-public-unetpp-320-aux-focal
```
