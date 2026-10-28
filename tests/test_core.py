from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data import build_provisional_split, extract_overlay_mask
from src.data.dataset import CaseRecord
from src.evaluation.metrics import binary_case_metrics, detection_metrics


class MetricsTest(unittest.TestCase):
    def test_binary_metrics(self) -> None:
        target = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        prediction = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        metrics = binary_case_metrics(prediction, target)
        self.assertAlmostEqual(metrics["dice"], 0.5)
        self.assertAlmostEqual(metrics["iou"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)

    def test_detection_metrics(self) -> None:
        metrics = detection_metrics([0, 0, 1, 1], [0, 1, 1, 0])
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertAlmostEqual(metrics["f1"], 0.5)


class MaskTest(unittest.TestCase):
    def test_overlay_colors(self) -> None:
        pixels = np.zeros((3, 3, 3), dtype=np.uint8)
        pixels[0, 0] = [220, 80, 80]
        pixels[0, 1] = [80, 80, 220]
        pixels[0, 2] = [0, 255, 0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlay.png"
            Image.fromarray(pixels, mode="RGB").save(path)
            red = extract_overlay_mask(path, "red")
            red_blue = extract_overlay_mask(path, "red_blue")
        self.assertEqual(int(red.sum()), 1)
        self.assertEqual(int(red_blue.sum()), 2)


class SplitTest(unittest.TestCase):
    def test_provisional_split_coverage(self) -> None:
        cases = [
            CaseRecord(f"{index:04d}", group)
            for group, count in (
                ("Bleeding", 12),
                ("Ischemia", 12),
                ("Normal", 12),
                ("External_Test", 3),
            )
            for index in range(count)
        ]
        payload = build_provisional_split(
            cases,
            seed=42,
            train_fraction=0.70,
            val_fraction=0.15,
            test_fraction=0.15,
        )
        keys = []
        for name in ("train", "validation", "test"):
            keys.extend(
                f"{item['group']}/{item['image_id']}"
                for item in payload["splits"][name]
            )
        external = {
            f"{item['group']}/{item['image_id']}"
            for item in payload["external_test"]
        }
        self.assertEqual(len(keys), 36)
        self.assertEqual(len(set(keys)), 36)
        self.assertFalse(set(keys) & external)


if __name__ == "__main__":
    unittest.main()
