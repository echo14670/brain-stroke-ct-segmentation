from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import dataset_for_split, ensure_split  # noqa: E402
from src.evaluation.metrics import detection_metrics  # noqa: E402
from src.models import UNet2D  # noqa: E402
from src.utils import (  # noqa: E402
    load_config,
    resolve_data_root,
    resolve_path,
    seed_everything,
    write_json,
)


def _load_labels(data_root: Path) -> dict[str, int]:
    with (data_root / "External_Test" / "labels.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        return {
            str(row["image_id"]): int(row["Stroke"])
            for row in csv.DictReader(stream)
        }


def infer(config_path: Path, checkpoint_path: Path | None) -> None:
    config = load_config(config_path)
    seed_everything(int(config["project"]["seed"]))
    data_root = resolve_data_root(config["data"]["root"])
    output_dir = resolve_path(config["project"]["output_dir"])
    checkpoint_path = checkpoint_path or (output_dir / "best.pt")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    split_payload = ensure_split(
        root=data_root,
        split_path=resolve_path(config["data"]["split_path"]),
        seed=int(config["project"]["seed"]),
        generate_if_missing=bool(
            config["data"]["generate_split_if_missing"]
        ),
        train_fraction=float(config["data"]["train_fraction"]),
        val_fraction=float(config["data"]["val_fraction"]),
        test_fraction=float(config["data"]["test_fraction"]),
    )
    dataset = dataset_for_split(
        root=data_root,
        split_payload=split_payload,
        split_name="external_test",
        image_size=int(config["data"]["image_size"]),
        mask_mode=str(config["data"]["mask_mode"]),
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D(
        in_channels=int(config["model"]["in_channels"]),
        out_channels=int(config["model"]["out_channels"]),
        base_channels=int(config["model"]["base_channels"]),
    ).to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    threshold = float(config["evaluation"]["threshold"])
    prediction_dir = output_dir / "predictions" / "external"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    labels = _load_labels(data_root)
    predicted_labels: list[int] = []
    actual_labels: list[int] = []

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            probabilities = torch.sigmoid(model(images))
            predictions = (probabilities >= threshold).cpu().numpy()
            for index, prediction in enumerate(predictions):
                image_id = str(batch["image_id"][index])
                source_path = (
                    data_root
                    / "External_Test"
                    / "PNG"
                    / f"{image_id}.png"
                )
                with Image.open(source_path) as source:
                    size = source.size
                mask = Image.fromarray(
                    prediction[0].astype(np.uint8) * 255, mode="L"
                ).resize(size, resample=Image.Resampling.NEAREST)
                mask.save(prediction_dir / f"{image_id}.png")
                predicted_labels.append(int(prediction.any()))
                actual_labels.append(int(labels[image_id]))

    detection = detection_metrics(actual_labels, predicted_labels)
    write_json(output_dir / "inference_summary.json", detection)
    print(
        f"External inference complete | images={len(predicted_labels)} "
        f"| detection_f1={detection['f1']:.4f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict External_Test masks")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    checkpoint = resolve_path(args.checkpoint) if args.checkpoint else None
    infer(resolve_path(args.config), checkpoint)


if __name__ == "__main__":
    main()
