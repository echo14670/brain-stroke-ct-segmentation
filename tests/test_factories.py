from __future__ import annotations

import unittest
from pathlib import Path

import torch
import yaml

from src.losses import DiceBCELoss, build_loss
from src.models import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ModelFactoryTest(unittest.TestCase):
    def test_custom_unet_accepts_320_input(self) -> None:
        model = build_model(
            {
                "name": "unet",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 4,
            }
        )
        output = model(torch.zeros(1, 1, 320, 320))
        self.assertEqual(tuple(output.shape), (1, 1, 320, 320))

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_model({"name": "unknown"})

    def test_custom_model_class_path(self) -> None:
        model = build_model(
            {
                "name": "custom_unet",
                "class_path": "src.models.unet:UNet2D",
                "params": {
                    "in_channels": 1,
                    "out_channels": 1,
                    "base_channels": 4,
                },
            }
        )
        output = model(torch.zeros(1, 1, 256, 256))
        self.assertEqual(tuple(output.shape), (1, 1, 256, 256))


class LossFactoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logits = torch.randn(2, 1, 16, 16, requires_grad=True)
        self.targets = (
            torch.rand(2, 1, 16, 16, device=self.logits.device) > 0.95
        ).float()

    def test_dice_bce(self) -> None:
        loss = build_loss({"name": "dice_bce"})
        self.assertIsInstance(loss, DiceBCELoss)
        self.assertTrue(torch.isfinite(loss(self.logits, self.targets)))

    def test_custom_loss_class_path(self) -> None:
        loss = build_loss(
            {
                "name": "custom_dice_bce",
                "class_path": "src.losses.dice_bce:DiceBCELoss",
                "params": {
                    "bce_weight": 1.0,
                    "dice_weight": 1.0,
                },
            }
        )
        self.assertIsInstance(loss, DiceBCELoss)
        value = loss(self.logits, self.targets)
        value.backward()
        self.assertTrue(torch.isfinite(value))

    def test_unknown_loss_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_loss({"name": "unknown"})


class BaselineConfigTest(unittest.TestCase):
    def _load(self, relative_path: str) -> dict[str, object]:
        with (PROJECT_ROOT / relative_path).open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    def test_baseline_uses_builtin_modules(self) -> None:
        config = self._load("configs/baseline.yaml")
        self.assertEqual(config["data"]["mask_mode"], "red")
        self.assertEqual(config["model"]["name"], "unet")
        self.assertEqual(config["loss"]["name"], "dice_bce")


if __name__ == "__main__":
    unittest.main()
