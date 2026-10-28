from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch
import yaml

from src.losses import DiceBCELoss, FocalTverskyLoss, build_loss
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

    @unittest.skipUnless(
        importlib.util.find_spec("segmentation_models_pytorch"),
        "segmentation-models-pytorch is not installed",
    )
    def test_smp_resnet34(self) -> None:
        model = build_model(
            {
                "name": "smp_unet",
                "encoder_name": "resnet34",
                "encoder_weights": None,
                "in_channels": 1,
                "out_channels": 1,
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

    def test_focal_tversky(self) -> None:
        loss = build_loss(
            {
                "name": "focal_tversky",
                "alpha": 0.7,
                "beta": 0.3,
                "gamma": 0.75,
            }
        )
        self.assertIsInstance(loss, FocalTverskyLoss)
        value = loss(self.logits, self.targets)
        value.backward()
        self.assertTrue(torch.isfinite(value))

    def test_unknown_loss_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_loss({"name": "unknown"})


class ExperimentConfigTest(unittest.TestCase):
    def _load(self, relative_path: str) -> dict[str, object]:
        with (PROJECT_ROOT / relative_path).open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    def test_e1_changes_mask_only(self) -> None:
        config = self._load("configs/experiments/E1_data_quality.yaml")
        self.assertEqual(config["data"]["mask_mode"], "red_blue")
        self.assertEqual(config["model"]["name"], "unet")
        self.assertEqual(config["loss"]["name"], "dice_bce")

    def test_e2_changes_encoder_only(self) -> None:
        config = self._load("configs/experiments/E2_resnet34.yaml")
        self.assertEqual(config["data"]["mask_mode"], "red")
        self.assertEqual(config["model"]["name"], "smp_unet")
        self.assertEqual(config["model"]["encoder_name"], "resnet34")
        self.assertEqual(config["loss"]["name"], "dice_bce")

    def test_e3_changes_loss_only(self) -> None:
        config = self._load("configs/experiments/E3_focal_tversky.yaml")
        self.assertEqual(config["data"]["mask_mode"], "red")
        self.assertEqual(config["model"]["name"], "unet")
        self.assertEqual(config["loss"]["name"], "focal_tversky")

    def test_e4_changes_resolution_only(self) -> None:
        config = self._load("configs/experiments/E4_res320.yaml")
        self.assertEqual(config["data"]["image_size"], 320)
        self.assertEqual(config["model"]["name"], "unet")
        self.assertEqual(config["loss"]["name"], "dice_bce")


if __name__ == "__main__":
    unittest.main()
