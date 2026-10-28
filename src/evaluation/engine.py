from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from src.evaluation.metrics import binary_case_metrics


@torch.inference_mode()
def collect_case_metrics(
    *,
    model: torch.nn.Module,
    loader: Iterable[dict[str, object]],
    device: torch.device,
    threshold: float,
) -> list[dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= threshold).cpu().numpy().astype(np.uint8)
        targets = masks.cpu().numpy().astype(np.uint8)
        for index, (prediction, target) in enumerate(zip(predictions, targets)):
            metrics = binary_case_metrics(prediction[0], target[0])
            metrics.update(
                {
                    "image_id": str(batch["image_id"][index]),
                    "group": str(batch["group"][index]),
                    "threshold": threshold,
                }
            )
            rows.append(metrics)
    return rows
