from __future__ import annotations

import unittest

import torch

from src.losses import FocalTverskyLoss, build_loss


class FocalTverskyTest(unittest.TestCase):
    def test_new_factory_interface(self):
        criterion = build_loss({
            "name": "focal_tversky",
            "class_path": "src.losses.focal_tversky:FocalTverskyLoss",
            "params": {"alpha": 0.3, "beta": 0.7, "gamma": 0.75, "smooth": 1.0},
        })
        self.assertIsInstance(criterion, FocalTverskyLoss)
        self.assertEqual(criterion.beta, 0.7)

    def test_hand_calculated_value(self):
        # p=(0.5,0.5), y=(1,0): TP=FP=FN=0.5, TI=1.5/2=0.75.
        logits = torch.zeros(1, 1, 1, 2)
        target = torch.tensor([[[[1.0, 0.0]]]])
        self.assertAlmostEqual(FocalTverskyLoss()(logits, target).item(),
                               0.25 ** 0.75, places=6)

    def test_false_negatives_have_greater_penalty(self):
        criterion = FocalTverskyLoss()
        target = torch.tensor([[[[1.0, 0.0]]]])
        missed = torch.tensor([[[[-2.0, -2.0]]]])
        false_alarm = torch.tensor([[[[2.0, 2.0]]]])
        self.assertGreater(criterion(missed, target).item(),
                           criterion(false_alarm, target).item())

    def test_normal_case_gradient_suppresses_false_positives(self):
        logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
        loss = FocalTverskyLoss()(logits, torch.zeros_like(logits))
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertTrue((logits.grad > 0).all().item())

    def test_saturated_predictions_have_finite_gradients(self):
        for foreground in (False, True):
            for prediction in (-100.0, 100.0):
                with self.subTest(foreground=foreground, prediction=prediction):
                    logits = torch.full((1, 1, 8, 8), prediction, requires_grad=True)
                    target = torch.full_like(logits, float(foreground))
                    loss = FocalTverskyLoss()(logits, target)
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss).item())
                    self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_mixed_batch_half_precision_at_training_resolution(self):
        # Summing 256*256 probabilities in float16 can overflow.
        logits = torch.zeros(2, 1, 256, 256, dtype=torch.float16, requires_grad=True)
        target = torch.zeros_like(logits)
        target[1, :, 100:110, 100:110] = 1.0
        loss = FocalTverskyLoss()(logits, target)
        loss.backward()
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_numerical_gradient(self):
        logits = torch.tensor([[[[-0.5, 0.5]]]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(FocalTverskyLoss(), (logits, target)))

    def test_per_image_reduction(self):
        criterion = FocalTverskyLoss()
        logits = torch.tensor([[[[-1.0, 1.0]]], [[[0.0, 0.0]]]])
        target = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 0.0]]]])
        separate = (criterion(logits[:1], target[:1]) + criterion(logits[1:], target[1:])) / 2
        torch.testing.assert_close(criterion(logits, target), separate)

    def test_invalid_shape_and_parameters(self):
        with self.assertRaises(ValueError):
            FocalTverskyLoss()(torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2))
        for kwargs in ({"alpha": -1}, {"gamma": 0}, {"smooth": 0}, {"eps": 1}, {"beta": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FocalTverskyLoss(**kwargs)

    def test_batch_hand_calculation_and_factory(self):
        # Two images: one positive pixel, three negative pixels; p=0.5.
        # TP=.5, FP=1.5, FN=.5 => TI=1.5/2.3, error=8/23.
        criterion = build_loss({
            "class_path": "src.losses.focal_tversky:FocalTverskyLoss",
            "params": {"batch": True},
        })
        logits = torch.zeros(2, 1, 1, 2)
        targets = torch.tensor([[[[1., 0.]]], [[[0., 0.]]]])
        self.assertTrue(criterion.batch)
        self.assertAlmostEqual(criterion(logits, targets).item(), (8/23)**.75, places=6)
        self.assertNotAlmostEqual(criterion(logits, targets).item(),
                                  FocalTverskyLoss()(logits, targets).item(), places=5)

    def test_batch_gradient_with_empty_and_small_lesion_images(self):
        logits = torch.full((2, 1, 256, 256), -4., requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[1, :, 100:110, 100:110] = 1.
        loss = FocalTverskyLoss(batch=True)(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertTrue((logits.grad[targets == 1] < 0).all().item())
        self.assertTrue((logits.grad[targets == 0] > 0).all().item())

    def test_batch_numerical_gradient_and_single_image(self):
        logits = torch.tensor([[[[-.5, .5]]], [[[.2, -.3]]]],
                              dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([[[[1., 0.]]], [[[0., 0.]]]], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(FocalTverskyLoss(batch=True), (logits, targets)))
        torch.testing.assert_close(FocalTverskyLoss(batch=True)(logits[:1], targets[:1]),
                                   FocalTverskyLoss()(logits[:1], targets[:1]))
        with self.assertRaises(TypeError):
            FocalTverskyLoss(batch="true")


if __name__ == "__main__":
    unittest.main()
