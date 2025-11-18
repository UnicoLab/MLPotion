import unittest
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mlpotion.core.exceptions import EvaluationError
from mlpotion.core.results import EvaluationResult
from mlpotion.frameworks.pytorch.evaluation.evaluators import ModelEvaluator
from tests.core import TestBase  # optional; gives temp_dir etc., but not strictly needed


class DummyEvalConfig:
    """Minimal stand-in for PyTorchEvaluationConfig used in tests.

    We only provide attributes actually read by the evaluator:
    - device
    - loss_fn
    - max_batches (optional)
    """

    def __init__(
        self,
        device: str = "cpu",
        loss_fn: Any | None = None,
        max_batches: int | None = None,
    ) -> None:
        self.device = device
        self.loss_fn = loss_fn
        self.max_batches = max_batches


class SmallNet(nn.Module):
    def __init__(self, in_features: int = 4, out_features: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestModelEvaluator(TestBase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(123)
        self.evaluator = ModelEvaluator()

    # ------------------------------------------------------------------ #
    # End-to-end: supervised, (inputs, targets) batches
    # ------------------------------------------------------------------ #
    def test_evaluate_supervised_with_default_mse(self) -> None:
        """(inputs, targets) batches with default MSE should give near-zero loss
        when targets are model outputs."""
        model = SmallNet(in_features=4, out_features=2)

        # Build a dataset where targets are exactly model(inputs),
        # so MSE should be ~ 0
        x = torch.randn(20, 4)
        with torch.no_grad():
            y = model(x)

        ds = TensorDataset(x, y)
        dl = DataLoader(ds, batch_size=5)

        config = DummyEvalConfig(device="cpu", loss_fn="mse")
        result = self.evaluator.evaluate(model, dl, config)

        self.assertIsInstance(result, EvaluationResult)
        self.assertIn("loss", result.metrics)
        self.assertLess(result.metrics["loss"], 1e-6)

    # ------------------------------------------------------------------ #
    # End-to-end: unsupervised, inputs-only batches
    # ------------------------------------------------------------------ #
    def test_evaluate_unsupervised_inputs_only(self) -> None:
        """When only inputs are provided, evaluator uses loss_fn(outputs, inputs)."""
        model = nn.Identity()

        x = torch.randn(10, 3)
        ds = TensorDataset(x)  # only inputs
        dl = DataLoader(ds, batch_size=4)

        config = DummyEvalConfig(device="cpu", loss_fn="mse")
        result = self.evaluator.evaluate(model, dl, config)

        self.assertIsInstance(result, EvaluationResult)
        self.assertIn("loss", result.metrics)
        self.assertLess(result.metrics["loss"], 1e-6)  # Identity vs inputs

    # ------------------------------------------------------------------ #
    # End-to-end: callable loss_fn
    # ------------------------------------------------------------------ #
    def test_evaluate_with_callable_loss_fn(self) -> None:
        """Custom callable loss_fn should be used as-is."""
        model = SmallNet(in_features=4, out_features=1)

        x = torch.randn(8, 4)
        y = torch.randn(8, 1)
        ds = TensorDataset(x, y)
        dl = DataLoader(ds, batch_size=4)

        # Custom loss always returns 42
        def custom_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return torch.tensor(42.0, device=outputs.device)

        config = DummyEvalConfig(device="cpu", loss_fn=custom_loss)
        result = self.evaluator.evaluate(model, dl, config)

        self.assertAlmostEqual(result.metrics["loss"], 42.0, places=5)

    # ------------------------------------------------------------------ #
    # _create_loss_fn behaviour
    # ------------------------------------------------------------------ #
    def test_create_loss_fn_with_unknown_name_falls_back_to_mse(self) -> None:
        """Unknown loss name should fall back to MSELoss."""
        config = DummyEvalConfig(device="cpu", loss_fn="something_weird")
        loss_fn = self.evaluator._create_loss_fn(config)

        self.assertIsInstance(loss_fn, nn.MSELoss)

    def test_create_loss_fn_with_module_instance(self) -> None:
        """If loss_fn is an nn.Module, it should be returned unchanged."""
        module_loss = nn.L1Loss()
        config = DummyEvalConfig(device="cpu", loss_fn=module_loss)

        loss_fn = self.evaluator._create_loss_fn(config)
        self.assertIs(loss_fn, module_loss)

    # ------------------------------------------------------------------ #
    # _prepare_batch behaviour
    # ------------------------------------------------------------------ #
    def test_prepare_batch_with_tuple(self) -> None:
        """_prepare_batch should handle (inputs, targets)."""
        device = torch.device("cpu")
        x = torch.randn(3, 2)
        y = torch.randn(3, 2)

        inputs, targets = self.evaluator._prepare_batch((x, y), device=device)
        self.assertIsInstance(inputs, torch.Tensor)
        self.assertIsInstance(targets, torch.Tensor)
        self.assertEqual(inputs.shape, x.shape)
        self.assertEqual(targets.shape, y.shape)

    def test_prepare_batch_with_list(self) -> None:
        """_prepare_batch should handle [inputs, targets]."""
        device = torch.device("cpu")
        x = torch.randn(2, 2)
        y = torch.randn(2, 2)

        inputs, targets = self.evaluator._prepare_batch([x, y], device=device)
        self.assertEqual(inputs.shape, x.shape)
        self.assertEqual(targets.shape, y.shape)

    def test_prepare_batch_with_inputs_only(self) -> None:
        """_prepare_batch should handle inputs-only batch."""
        device = torch.device("cpu")
        x = torch.randn(4, 3)

        inputs, targets = self.evaluator._prepare_batch(x, device=device)
        self.assertEqual(inputs.shape, x.shape)
        self.assertIsNone(targets)

    def test_prepare_batch_raises_if_inputs_not_tensor_after_move(self) -> None:
        """Non-tensor inputs after _move_to_device should raise EvaluationError."""
        device = torch.device("cpu")
        bad_batch = ("not_a_tensor", "also_not_tensor")

        with self.assertRaises(EvaluationError):
            _ = self.evaluator._prepare_batch(bad_batch, device=device)

    # ------------------------------------------------------------------ #
    # Empty dataloader behaviour
    # ------------------------------------------------------------------ #
    def test_evaluate_raises_on_empty_dataloader(self) -> None:
        """Evaluator should raise if the dataloader yields no batches."""
        model = SmallNet(in_features=4, out_features=2)

        # Empty dataset: 0 samples
        x = torch.empty(0, 4)
        y = torch.empty(0, 2)
        ds = TensorDataset(x, y)
        dl = DataLoader(ds, batch_size=4)

        config = DummyEvalConfig(device="cpu", loss_fn="mse")

        with self.assertRaises(EvaluationError):
            _ = self.evaluator.evaluate(model, dl, config)

    # ------------------------------------------------------------------ #
    # max_batches support
    # ------------------------------------------------------------------ #
    def test_evaluate_respects_max_batches(self) -> None:
        """If config.max_batches is set, evaluation should stop after that many batches."""
        model = SmallNet(in_features=4, out_features=1)

        x = torch.randn(50, 4)
        y = torch.randn(50, 1)
        ds = TensorDataset(x, y)
        dl = DataLoader(ds, batch_size=5)

        # Counter via closure to see how many times loss_fn is called
        call_counter = {"count": 0}

        def counting_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            call_counter["count"] += 1
            return torch.mean((outputs - targets) ** 2)

        config = DummyEvalConfig(
            device="cpu",
            loss_fn=counting_loss,
            max_batches=3,
        )

        result = self.evaluator.evaluate(model, dl, config)

        self.assertIsInstance(result, EvaluationResult)
        self.assertLessEqual(call_counter["count"], 3)
        self.assertGreater(call_counter["count"], 0)


if __name__ == "__main__":
    unittest.main()
