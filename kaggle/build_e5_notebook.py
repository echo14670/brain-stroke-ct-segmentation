"""Build the E5 combined 320x320 and Focal Tversky Kaggle notebook."""
from __future__ import annotations

import json
from pathlib import Path

import build_e4_notebook as base

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"
CONFIG = "configs/experiments/E5_res320_focal_tversky.yaml"
OUTPUT_DIR = "reports/experiments/E5_res320_focal_tversky"
EXPERIMENT = "E5_res320_focal_tversky"


def _replace_experiment_names(notebook: dict) -> dict:
    for cell in notebook["cells"]:
        replacements = {
            "brain-stroke-e4-": "brain-stroke-e5-",
            "E4_res320": EXPERIMENT,
            "E4_VERIFIED": "E5_VERIFIED",
            "e4_tests.log": "e5_tests.log",
            "e4_cuda_smoke.log": "e5_cuda_smoke.log",
            "e4-latency": "e5-latency",
            "e4-overview": "e5-overview",
            "e4-runtime-setup": "e5-runtime-setup",
            "e4-verification": "e5-verification",
            "e4-train": "e5-train",
            "e4-final-evaluation": "e5-final-evaluation",
            "e4-export": "e5-export",
            "E4": "E5",
        }
        lines = []
        for line in cell["source"]:
            if line.startswith("payload = "):
                lines.append(line)
                continue
            for old, new in replacements.items():
                line = line.replace(old, new)
            lines.append(line)
        cell["source"] = lines
        cell["id"] = cell["id"].replace("e4-", "e5-")
    return notebook


def _patch_overview(notebook: dict) -> None:
    source = """# E5：320×320 + Focal Tversky 组合实验

本实验把 E4 的 320×320 输入分辨率与 E3 的 batch 版 Focal Tversky Loss
组合，用于检验“提高召回”与“减少假阳性”能否同时成立。

## 实验口径

- 数据、冻结切分、简单 2D U-Net、batch size 8、优化器、15 轮上限、
  早停 4 轮、seed 42 和阈值 0.5 均继承 E4。
- `data.image_size` 为 320，`loss` 与 E3 batch 配置逐项一致。
- 图像使用 BILINEAR、掩码使用 NEAREST，从同一原图同步缩放。
- Notebook 先执行单元测试和 CUDA 冒烟验证，再自动训练、统一评估和外部推理。
- 每个 epoch 结束后写出 `progress.json`；训练标准输出和 `run.log` 会显示
  epoch、batch、loss、验证 Dice/IoU 和耗时。
- 最佳权重仅按验证集 Dice 选择；测试集和 External_Test 不参与调参。
"""
    for cell in notebook["cells"]:
        if cell["id"] == "e5-overview":
            cell["source"] = source.splitlines(keepends=True)
            return
    raise RuntimeError("E5 overview cell was not generated")


def _patch_verification(notebook: dict) -> None:
    for cell in notebook["cells"]:
        if cell["id"] != "e5-verification":
            continue
        source = "".join(cell["source"])
        old_load = (
            "    with open('configs/experiments/E5_res320_focal_tversky.yaml', "
            "encoding='utf-8') as stream:\n"
            "        config = yaml.safe_load(stream)"
        )
        new_load = (
            "    with open('configs/experiments/E3_focal_tversky_batch.yaml', "
            "encoding='utf-8') as stream:\n"
            "        e3 = yaml.safe_load(stream)\n"
            "    with open('configs/experiments/E5_res320_focal_tversky.yaml', "
            "encoding='utf-8') as stream:\n"
            "        config = yaml.safe_load(stream)"
        )
        source = source.replace(old_load, new_load, 1)
        source = source.replace(
            "    # 单变量断言：除 image_size 及命名/输出目录外，其余必须与基线逐项相同",
            "    # 组合断言：继承 E4，并逐项复用 E3 的损失配置",
            1,
        )
        source = source.replace(
            "    for key in ('model', 'loss', 'train', 'evaluation'):\n"
            "        assert config[key] == baseline[key], key",
            "    for key in ('model', 'train', 'evaluation'):\n"
            "        assert config[key] == baseline[key], key\n"
            "    assert config['loss'] == e3['loss'], 'loss must match E3 batch config'",
            1,
        )
        source = source.replace(
            "    print('单变量校验通过：仅 image_size 256 -> 320')",
            "    assert config['loss']['name'] == 'focal_tversky'\n"
            "    assert config['loss']['params']['batch'] is True\n"
            "    print('组合配置校验通过：E4 320x320 + E3 batch Focal Tversky')",
            1,
        )
        cell["source"] = source.splitlines(keepends=True)
        return
    raise RuntimeError("E5 verification cell was not generated")


def build() -> dict:
    base.CONFIG = CONFIG
    notebook = base.build(
        formal=True,
        online=False,
        full=True,
        latency_only=False,
    )
    notebook = _replace_experiment_names(notebook)
    notebook["cells"] = [
        cell for cell in notebook["cells"] if cell["id"] != "e5-latency"
    ]
    _patch_overview(notebook)
    _patch_verification(notebook)
    return notebook


def main() -> None:
    destination = KAGGLE / EXPERIMENT / f"{EXPERIMENT}_full.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(build(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Generated: {destination.relative_to(ROOT).as_posix()}")
    print(f"Size: {destination.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
