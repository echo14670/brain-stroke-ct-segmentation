from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import dataset_for_split, ensure_split  # noqa: E402
from src.evaluation.engine import collect_case_metrics  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    aggregate_case_metrics,
    group_case_metrics,
    detection_metrics,
)
from src.models import build_model  # noqa: E402
from src.utils import (  # noqa: E402
    load_config,
    resolve_data_root,
    resolve_path,
    seed_everything,
    write_json,
)


def _load_labels(data_root: Path) -> dict[str, int]:
    path = data_root / "External_Test" / "labels.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {
            str(row["image_id"]): int(row["Stroke"])
            for row in csv.DictReader(stream)
        }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "image_id",
        "group",
        "dice",
        "iou",
        "precision",
        "recall",
        "tp",
        "fp",
        "fn",
        "tn",
        "predicted_positive",
        "ground_truth_positive",
        "pixels",
        "threshold",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(config_path: Path, checkpoint_path: Path | None) -> dict[str, object]:
    config = load_config(config_path)
    seed_everything(int(config["project"]["seed"]))
    data_root = resolve_data_root(config["data"]["root"])
    output_dir = resolve_path(config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
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
    checkpoint_path = checkpoint_path or (output_dir / "best.pt")
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = config["model"]
    model = build_model(model_config, pretrained=False).to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    threshold = float(config["evaluation"]["threshold"])

    def rows_for(split_name: str) -> list[dict[str, object]]:
        dataset = dataset_for_split(
            root=data_root,
            split_payload=split_payload,
            split_name=split_name,
            image_size=int(config["data"]["image_size"]),
            mask_mode=str(config["data"]["mask_mode"]),
            training=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(config["evaluation"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["evaluation"]["num_workers"]),
            pin_memory=device.type == "cuda",
        )
        return collect_case_metrics(
            model=model,
            loader=loader,
            device=device,
            threshold=threshold,
        )

    test_rows = rows_for("test")
    external_rows = rows_for("external_test")
    internal_labels = [
        0 if row["group"] == "Normal" else 1 for row in test_rows
    ]
    internal_predictions = [
        1 if int(row["predicted_positive"]) > 0 else 0 for row in test_rows
    ]
    external_labels_map = _load_labels(data_root)
    external_labels = [
        external_labels_map[str(row["image_id"])] for row in external_rows
    ]
    external_predictions = [
        1 if int(row["predicted_positive"]) > 0 else 0 for row in external_rows
    ]
    combined_rows = test_rows + external_rows
    metrics = {
        "experiment": config["project"]["name"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split_version": split_payload["version"],
        "split_provisional": bool(split_payload.get("provisional", False)),
        "threshold": threshold,
        "internal_test": {
            "overall": aggregate_case_metrics(test_rows),
            "groups": group_case_metrics(test_rows),
            "image_detection": detection_metrics(
                internal_labels, internal_predictions
            ),
        },
        "external_test": {
            "overall_segmentation": aggregate_case_metrics(external_rows),
            "image_detection": detection_metrics(
                external_labels, external_predictions
            ),
        },
        "combined_segmentation": aggregate_case_metrics(combined_rows),
    }
    write_json(output_dir / "metrics.json", metrics)
    _write_rows(output_dir / "test_per_case.csv", test_rows)
    _write_rows(output_dir / "external_test_per_case.csv", external_rows)
    shutil.copy2(config_path, output_dir / "evaluation_config.yaml")

    internal = metrics["internal_test"]["overall"]
    external_detection = metrics["external_test"]["image_detection"]
    split_status = "provisional" if metrics["split_provisional"] else "frozen"
    summary = (
        f"# Experiment Result: {config['project']['name']}\n\n"
        f"- Split: `{metrics['split_version']}` "
        f"({split_status})\n"
        f"- Checkpoint epoch: `{metrics['checkpoint_epoch']}`\n"
        f"- Threshold: `{threshold}`\n"
        f"- Internal positive-case Dice: `{internal['positive_mean_dice']:.4f}`\n"
        f"- Internal positive-case IoU: `{internal['positive_mean_iou']:.4f}`\n"
        f"- Internal Precision: `{internal['positive_pixel_precision']:.4f}`\n"
        f"- Internal Recall: `{internal['positive_pixel_recall']:.4f}`\n"
        "- Internal Normal false-positive pixel ratio: "
        f"`{internal['normal_false_positive_pixel_ratio']:.6f}`\n"
        f"- External image-level F1: `{external_detection['f1']:.4f}`\n"
    )
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the baseline")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    checkpoint = resolve_path(args.checkpoint) if args.checkpoint else None
    evaluate(resolve_path(args.config), checkpoint)


if __name__ == "__main__":
    main()
