"""Create the public U-Net++ notebook variant with 320x320 inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def patch_notebook(source: Path) -> dict:
    notebook = json.loads(source.read_text(encoding="utf-8"))
    replacements = (
        ("    IMG_H = 256\n    IMG_W = 256", "    IMG_H = 320\n    IMG_W = 320"),
        (
            "def crop_and_resize_pair(image, mask, target_h=256, target_w=256):",
            "def crop_and_resize_pair(image, mask, target_h=320, target_w=320):",
        ),
        (
            "# Resize lại về đúng chuẩn (256x256) như mô hình mong đợi",
            "# Resize lại về đúng chuẩn (320x320) như mô hình mong đợi",
        ),
        (
            "model_graph = draw_graph(model, input_size=(1, 3, 256, 256), expand_nested=True)",
            "model_graph = draw_graph(model, input_size=(1, 3, 320, 320), expand_nested=True)",
        ),
    )
    for cell in notebook["cells"]:
        text = "".join(cell.get("source", []))
        for old, new in replacements:
            text = text.replace(old, new)
        cell["source"] = text.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    title = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# E6：公开 U-Net++ 的 320×320 复现\n",
            "\n",
            "本 Notebook 派生自公开 U-Net++ 方案，只把输入尺寸从 256×256 改为 320×320。\n",
            "EfficientNet-B0、ImageNet 预训练、Dice+BCE、数据增强、随机切分、batch size 8、\n",
            "AdamW、余弦退火、50 epochs 和阈值 0.5 均保持公开代码的设置。\n",
        ],
    }
    notebook["cells"].insert(0, title)
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Pulled public .ipynb")
    parser.add_argument(
        "--output",
        default="kaggle/E6_public_unetpp_320/E6_public_unetpp_320.ipynb",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    notebook = patch_notebook(source)
    destination.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Generated: {destination}")
    print(f"Size: {destination.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
