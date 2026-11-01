# E1 数据质量审计

本次只审计，不修改原始图像、掩码、标签或冻结切分。

| 项目 | 结果 |
| --- | --- |
| `split_version` | `baseline-v1-seed42` |
| `split_provisional` | `False` |
| `split_sha256` | `4853e4fe55dac627ae4a5daba918aa00f46feb156932e909cbce5802742a7052` |
| `split_counts` | `{'train': 4654, 'validation': 998, 'test': 998, 'external_test': 200}` |
| `discovered` | `6850` |
| `audited` | `6850` |
| `read_errors` | `0` |
| `main_non_512` | `248` |
| `external_non_512` | `8` |
| `external_non_square` | `8` |
| `mask_size_mismatch` | `0` |
| `main_positive_empty_red` | `0` |
| `main_positive_empty_union` | `0` |
| `blue_added_by_group` | `{'Ischemia': 1130}` |
| `external_positive_empty_mask` | `7` |
| `external_label_mask_conflict` | `7` |
| `exact_duplicate_groups` | `20` |
| `cross_split_duplicate_groups` | `10` |
| `split_unchanged` | `True` |

## 解释与后续处理

- cases.csv 是逐病例记录；anomalies.csv 是异常候选，不能直接据此删除病例。
- 红蓝并集复用基线的颜色提取实现；对照图仅使用训练集，需人工核验新增区域。
- 外部标签与掩码冲突保持原样并单独报告，不补画标签，不用外部结果调参。
- 原始图像与掩码尺寸或坐标不一致时需逐例核查，不能仅通过缩放宣称修复。
- 完全重复检测使用灰度像素和尺寸的 SHA256，不能排除近似切片或同患者泄漏。
- 如发现泄漏需要改变病例集合，须与基线负责人协作创建新切分并重跑基线，不能私改 baseline-v1。
- 本报告为数据审计，不是训练或性能评估结果。

## 训练集 OVERLAY 核验

检查 Ischemia/14127：红色填充外沿可见蓝色轮廓，红色掩码为 6171 像素，红蓝并集增加 717 像素。当前证据说明本实验主要纳入了颜色标记的轮廓像素，不能称为发现全新的病灶区域，亦不能据此预判性能提高。详见 mask_difference_detail.png；样例按训练集新增像素数量选取，不代表全部样本均已人工核验。
