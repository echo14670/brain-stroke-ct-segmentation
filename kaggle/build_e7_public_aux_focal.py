"""Build the public U-Net++ 320 variant with auxiliary Focal Tversky loss."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


LOSS_SOURCE = r'''
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits.float())
        targets = targets.float()
        dimensions = (0, 1, 2, 3)
        tp = (probabilities * targets).sum(dim=dimensions)
        fp = (probabilities * (1.0 - targets)).sum(dim=dimensions)
        fn = ((1.0 - probabilities) * targets).sum(dim=dimensions)
        index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return (1.0 - index).clamp(min=self.eps, max=1.0).pow(self.gamma).mean()


class HybridSegLoss(nn.Module):
    def __init__(
        self,
        w_dice=0.5,
        w_bce=0.5,
        focal_weight=0.2,
        alpha=0.3,
        beta=0.7,
        gamma=0.75,
    ):
        super().__init__()
        self.w_dice = w_dice
        self.w_bce = w_bce
        self.focal_weight = focal_weight
        self.dice = DiceLoss(mode="binary", smooth=1.0)
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)

    def forward(self, logits, targets):
        public_loss = (
            self.w_dice * self.dice(logits, targets)
            + self.w_bce * self.bce(logits, targets)
        )
        auxiliary_loss = self.focal(logits, targets)
        return (
            (1.0 - self.focal_weight) * public_loss
            + self.focal_weight * auxiliary_loss
        )


criterion = HybridSegLoss(
    w_dice=cfg.DICE_WEIGHT,
    w_bce=cfg.BCE_WEIGHT,
    focal_weight=cfg.FOCAL_WEIGHT,
)
'''.strip()


def patch_notebook(source: Path) -> dict:
    notebook = json.loads(source.read_text(encoding="utf-8"))
    loss_patched = False
    config_patched = False
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell.get("source", []))
        if "    BCE_WEIGHT  = 0.5" in text:
            text = text.replace(
                "    BCE_WEIGHT  = 0.5",
                "    BCE_WEIGHT  = 0.5\n    FOCAL_WEIGHT = 0.2",
                1,
            )
            config_patched = True
        if "class HybridSegLoss(nn.Module):" in text:
            text = LOSS_SOURCE + "\n"
            loss_patched = True
        cell["source"] = text.splitlines(keepends=True)
        cell["execution_count"] = None
        cell["outputs"] = []
    if not config_patched or not loss_patched:
        raise RuntimeError("Could not locate the public config or loss cell")

    title = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# E7：公开 U-Net++ 320×320 + 辅助 Focal Tversky\n",
            "\n",
            "本 Notebook 与 E6 的输入、模型、数据增强、优化器和训练轮数一致。\n",
            "损失函数采用：\n",
            "\n",
            "`0.8 × (0.5 × DiceLoss + 0.5 × BCEWithLogits) + 0.2 × FocalTversky`\n",
        ],
    }
    notebook["cells"].insert(0, title)
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="kaggle/E6_public_unetpp_320/E6_public_unetpp_320.ipynb",
    )
    parser.add_argument(
        "--output",
        default="kaggle/E7_public_unetpp_320_aux_focal/E7_public_unetpp_320_aux_focal.ipynb",
    )
    args = parser.parse_args()
    source = Path(args.source).resolve()
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(patch_notebook(source), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Generated: {destination}")
    print(f"Size: {destination.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
