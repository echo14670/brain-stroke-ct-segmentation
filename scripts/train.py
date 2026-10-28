from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import dataset_for_split, ensure_split  # noqa: E402
from src.evaluation.engine import collect_case_metrics  # noqa: E402
from src.evaluation.metrics import aggregate_case_metrics  # noqa: E402
from src.losses import DiceBCELoss  # noqa: E402
from src.models import UNet2D  # noqa: E402
from src.utils import (  # noqa: E402
    load_config,
    resolve_data_root,
    resolve_path,
    seed_everything,
    write_json,
)


class RunLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.stream = path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message, flush=True)
        self.stream.write(message + "\n")
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


def _make_loader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def _save_curves(history: list[dict[str, float]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        epochs = [row["epoch"] for row in history]
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, [row["train_loss"] for row in history])
        axes[0].plot(epochs, [row["validation_loss"] for row in history])
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend(["train", "validation"])
        axes[1].plot(epochs, [row["validation_dice"] for row in history])
        axes[1].set_title("Validation positive-case Dice")
        axes[1].set_xlabel("Epoch")
        figure.tight_layout()
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
    except Exception as error:
        print(f"WARNING: could not save curves: {error}", flush=True)


def train(
    config_path: Path,
    *,
    max_batches: int | None = None,
    max_epochs: int | None = None,
) -> dict[str, object]:
    config = load_config(config_path)
    project = config["project"]
    data_config = config["data"]
    model_config = config["model"]
    loss_config = config["loss"]
    train_config = config["train"]
    seed = int(project["seed"])
    seed_everything(seed)

    data_root = resolve_data_root(data_config["root"])
    output_dir = resolve_path(project["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir / "run.log")
    progress_path = output_dir / "progress.json"
    split_path = resolve_path(data_config["split_path"])
    split_payload = ensure_split(
        root=data_root,
        split_path=split_path,
        seed=seed,
        generate_if_missing=bool(data_config["generate_split_if_missing"]),
        train_fraction=float(data_config["train_fraction"]),
        val_fraction=float(data_config["val_fraction"]),
        test_fraction=float(data_config["test_fraction"]),
    )
    shutil.copy2(config_path, output_dir / "config.yaml")
    write_json(output_dir / "split_snapshot.json", split_payload)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.environ.get("REQUIRE_CUDA") == "1" and device.type != "cuda":
        raise RuntimeError("REQUIRE_CUDA=1 but CUDA is not available")
    image_size = int(data_config["image_size"])
    mask_mode = str(data_config["mask_mode"])
    train_dataset = dataset_for_split(
        root=data_root,
        split_payload=split_payload,
        split_name="train",
        image_size=image_size,
        mask_mode=mask_mode,
        training=True,
        horizontal_flip=False,
    )
    validation_dataset = dataset_for_split(
        root=data_root,
        split_payload=split_payload,
        split_name="validation",
        image_size=image_size,
        mask_mode=mask_mode,
        training=False,
    )
    pin_memory = bool(data_config["pin_memory"]) and device.type == "cuda"
    train_loader = _make_loader(
        train_dataset,
        batch_size=int(train_config["batch_size"]),
        num_workers=int(data_config["num_workers"]),
        pin_memory=pin_memory,
        shuffle=True,
    )
    validation_loader = _make_loader(
        validation_dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        num_workers=int(data_config["num_workers"]),
        pin_memory=pin_memory,
        shuffle=False,
    )

    model = UNet2D(
        in_channels=int(model_config["in_channels"]),
        out_channels=int(model_config["out_channels"]),
        base_channels=int(model_config["base_channels"]),
    ).to(device)
    criterion = DiceBCELoss(
        bce_weight=float(loss_config["bce_weight"]),
        dice_weight=float(loss_config["dice_weight"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(train_config["scheduler_factor"]),
        patience=int(train_config["scheduler_patience"]),
    )
    use_amp = bool(train_config["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.log(
        f"Baseline start | device={device} | image_size={image_size} | "
        f"parameters={parameter_count:,}"
    )
    logger.log(
        f"Cases | train={len(train_dataset)} | validation={len(validation_dataset)} "
        f"| split={split_payload['version']} | provisional={split_payload.get('provisional')}"
    )

    epochs = int(train_config["epochs"])
    if max_epochs is not None:
        epochs = min(epochs, max_epochs)
    patience = int(train_config["early_stopping_patience"])
    progress_every = int(train_config["progress_every"])
    best_dice = -1.0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    started_at = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen_batches = 0
        epoch_started = time.time()
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{epochs}",
            dynamic_ncols=True,
            leave=False,
        )
        for batch_index, batch in enumerate(progress, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, enabled=use_amp
            ):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())
            seen_batches += 1
            progress.set_postfix(loss=f"{running_loss / seen_batches:.4f}")
            if progress_every > 0 and batch_index % progress_every == 0:
                logger.log(
                    f"epoch={epoch} batch={batch_index}/{len(train_loader)} "
                    f"loss={running_loss / seen_batches:.4f}"
                )
            if max_batches is not None and batch_index >= max_batches:
                break

        train_loss = running_loss / max(seen_batches, 1)
        validation_rows = collect_case_metrics(
            model=model,
            loader=validation_loader,
            device=device,
            threshold=float(config["evaluation"]["threshold"]),
        )
        validation_metrics = aggregate_case_metrics(validation_rows)
        validation_losses: list[float] = []
        model.eval()
        with torch.inference_mode():
            for batch in validation_loader:
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type, enabled=use_amp
                ):
                    logits = model(images)
                    validation_losses.append(float(criterion(logits, masks)))
        validation_loss = float(np.mean(validation_losses))
        validation_dice = float(validation_metrics["positive_mean_dice"])
        scheduler.step(validation_dice)

        elapsed = time.time() - epoch_started
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_dice": validation_dice,
            "validation_iou": float(validation_metrics["positive_mean_iou"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": elapsed,
        }
        history.append(row)
        logger.log(
            f"epoch={epoch}/{epochs} train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} val_dice={validation_dice:.4f} "
            f"val_iou={row['validation_iou']:.4f} seconds={elapsed:.1f}"
        )
        write_json(
            progress_path,
            {
                "status": "running",
                "epoch": epoch,
                "epochs": epochs,
                "latest": row,
                "best_validation_dice": max(best_dice, validation_dice),
                "elapsed_seconds": time.time() - started_at,
            },
        )

        checkpoint = {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "best_validation_dice": max(best_dice, validation_dice),
            "config": config,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if validation_dice > best_dice:
            best_dice = validation_dice
            stale_epochs = 0
            torch.save(checkpoint, output_dir / "best.pt")
            logger.log(f"Saved new best checkpoint | val_dice={best_dice:.4f}")
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            logger.log(f"Early stopping after {stale_epochs} stale epochs")
            break

    _save_curves(history, output_dir / "curves.png")
    result = {
        "status": "completed",
        "best_validation_dice": best_dice,
        "epochs_completed": len(history),
        "parameter_count": parameter_count,
        "elapsed_seconds": time.time() - started_at,
        "history": history,
    }
    write_json(output_dir / "train_summary.json", result)
    write_json(progress_path, result)
    logger.log(
        f"Training complete | best_val_dice={best_dice:.4f} "
        f"| elapsed_seconds={result['elapsed_seconds']:.1f}"
    )
    logger.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 2D U-Net baseline")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional debug limit for a smoke run",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional debug epoch limit for a smoke run",
    )
    args = parser.parse_args()
    train(
        resolve_path(args.config),
        max_batches=args.max_batches,
        max_epochs=args.max_epochs,
    )


if __name__ == "__main__":
    main()
