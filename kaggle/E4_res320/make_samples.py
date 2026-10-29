"""生成 E4 的典型成功/失败样例图（只用 numpy + Pillow，不需要 torch）。

输入：External_Test 原图、官方 MASKS、evaluate.py 的逐病例指标、infer.py 导出的预测掩码。
输出：reports/experiments/E4_res320/samples.png 与 samples_index.json。

用法（在仓库根目录执行）：
    python kaggle/E4_res320/make_samples.py --predictions <解压后的 predictions/external 目录>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/Brain_Stroke_CT_Dataset/External_Test"
OUTPUT_DIR = ROOT / "reports/experiments/E4_res320"
CELL = 256          # 每张图缩放到 256×256
PAD = 8
TITLE_H = 64
FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
    "C:/Windows/Fonts/msyhl.ttc",
    "C:/Windows/Fonts/simhei.ttf",     # 黑体
    "C:/Windows/Fonts/simsun.ttc",     # 宋体
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _font(size: int = 13):
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:      # Pillow < 10.1 不支持 size 参数
        return ImageFont.load_default()


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = image.convert("L").resize(size, resample=Image.Resampling.NEAREST)
    return np.asarray(mask) > 0


def _panel(image_id: str, prediction_dir: Path, lines: tuple[str, str, str]) -> Image.Image:
    with Image.open(DATA / "PNG" / f"{image_id}.png") as source:
        grey = source.convert("L")
        size = grey.size
        base = np.stack([np.asarray(grey)] * 3, axis=-1).astype(np.float32)
    truth = _read_mask(DATA / "MASKS" / f"{image_id}.png", size)
    predicted = _read_mask(prediction_dir / f"{image_id}.png", size)

    # GT 红色、预测绿色，重叠区域自然混成黄色
    base[truth] = base[truth] * 0.45 + np.array([255, 40, 40]) * 0.55
    base[predicted] = base[predicted] * 0.55 + np.array([40, 230, 90]) * 0.45

    picture = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    picture = picture.resize((CELL, CELL), resample=Image.Resampling.BILINEAR)

    cell = Image.new("RGB", (CELL, CELL + TITLE_H), (22, 22, 22))
    cell.paste(picture, (0, TITLE_H))
    draw = ImageDraw.Draw(cell)
    draw.text((6, 2), f"{image_id}  {lines[0]}", font=_font(17), fill=(255, 255, 255))
    draw.text((6, 23), lines[1], font=_font(15), fill=(205, 205, 205))
    draw.text((6, 42), lines[2], font=_font(15), fill=(160, 160, 160))
    return cell


def select_cases(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    positive = [r for r in rows if int(r["ground_truth_positive"]) > 0]
    normal = [r for r in rows if int(r["ground_truth_positive"]) == 0]
    positive.sort(key=lambda r: -float(r["dice"]))
    normal.sort(key=lambda r: -int(r["predicted_positive"]))

    zero_dice = [r for r in positive if float(r["dice"]) == 0.0]
    partial = [r for r in positive if 0.0 < float(r["dice"]) < 0.35]
    picked = (
        [("成功", r) for r in positive[:2]]
        + [("失败-漏检", r) for r in zero_dice[:1]]
        + [("失败-低Dice", r) for r in partial[:2]]
        + [("假阳性", r) for r in normal[:2] if int(r["predicted_positive"]) > 0]
    )

    chosen: list[dict[str, object]] = []
    for tag, row in picked:
        case = {
            "tag": tag,
            "image_id": row["image_id"],
            "dice": round(float(row["dice"]), 4),
            "recall": round(float(row["recall"]), 4),
            "precision": round(float(row["precision"]), 4),
            "gt_pixels": int(row["ground_truth_positive"]),
            "predicted_pixels": int(row["predicted_positive"]),
            "pixels": int(row["pixels"]),
        }
        ratio = case["predicted_pixels"] / max(case["pixels"], 1) * 100
        if tag == "假阳性":
            case["lines"] = (
                f"{tag}",
                "GT 0 px",
                f"误报 {case['predicted_pixels']} px（{ratio:.1f}%）",
            )
        elif tag == "失败-漏检":
            case["lines"] = (
                f"{tag}",
                "Dice 0.000",
                f"病灶 {case['gt_pixels']} px 完全未检出",
            )
        else:
            case["lines"] = (
                f"{tag} Dice {case['dice']:.3f}",
                f"R {case['recall']:.3f}  P {case['precision']:.3f}",
                f"GT {case['gt_pixels']} px → 预测 {case['predicted_pixels']} px",
            )
        chosen.append(case)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="infer.py 导出的 predictions/external 目录")
    parser.add_argument("--tag", default="E4_res320（320×320）")
    args = parser.parse_args()

    prediction_dir = Path(args.predictions).expanduser().resolve()
    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"预测目录不存在: {prediction_dir}")

    rows = _load_rows(OUTPUT_DIR / "external_test_per_case.csv")
    chosen = select_cases(rows)

    panels = []
    for case in chosen:
        panels.append(
            _panel(str(case["image_id"]), prediction_dir, case["lines"])
        )

    columns = 4
    rows_count = (len(panels) + columns - 1) // columns
    width = columns * CELL + (columns + 1) * PAD
    height = rows_count * (CELL + TITLE_H) + (rows_count + 1) * PAD
    canvas = Image.new("RGB", (width, height), (32, 32, 32))
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        canvas.paste(
            panel,
            (PAD + column * (CELL + PAD), PAD + row * (CELL + TITLE_H + PAD)),
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT_DIR / "samples.png")
    (OUTPUT_DIR / "samples_index.json").write_text(
        json.dumps(
            {
                "experiment": "E4_res320",
                "tag": args.tag,
                "legend": "红=GT 掩码，绿=模型预测，黄=重叠；数据来源 External_Test（官方 MASKS）",
                "cells": chosen,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"samples.png: {OUTPUT_DIR / 'samples.png'} ({canvas.size[0]}x{canvas.size[1]})")
    for case in chosen:
        print(f"  {case['tag']} {case['image_id']} dice={case['dice']} recall={case['recall']}")


if __name__ == "__main__":
    main()
