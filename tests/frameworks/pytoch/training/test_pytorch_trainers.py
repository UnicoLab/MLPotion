import unittest

from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mlpotion.core.exceptions import TrainingError
from mlpotion.core.results import TrainingResult
from mlpotion.frameworks.pytorch.config import PyTorchTrainingConfig
from mlpotion.frameworks.pytorch.training import PyTorchModelTrainer
from tests.core import TestBase  # provides common setup (e.g. temp dirs, seeds)


class TestPyTorchModelTrainer(TestBase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(123)

        # Simple synthetic regression data: y = 3x + 1 + noise
        x = torch.linspace(-1.0, 1.0, steps=100).unsqueeze(1)
        noise = 0.1 * torch.randn_like(x)
        y = 3.0 * x + 1.0 + noise

        self.train_dataset = TensorDataset(x, y)
        self.train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)

        # Unsupervised-style dataset: inputs only
        self.unsup_dataset = TensorDataset(x)
        self.unsup_loader = DataLoader(self.unsup_dataset, batch_size=16, shuffle=True)

        # Simple linear model
        self.model = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    # ------------------------------------------------------------------ #
    # Basic supervised training
    # ------------------------------------------------------------------ #
    def test_train_supervised_returns_training_result(self) -> None:
        """Supervised training should produce a valid TrainingResult."""
        trainer = PyTorchModelTrainer()
        config = PyTorchTrainingConfig(
            epochs=3,
            learning_rate=1e-2,
            optimizer="adam",
            loss_fn="mse",
            device="cpu",
            verbose=False,
        )

        result = trainer.train(
            model=self.model,
            dataloader=self.train_loader,
            config=config,
            validation_dataloader=None,
        )

        self.assertIsInstance(result, TrainingResult)
        self.assertIsInstance(result.model, nn.Module)
        self.assertIn("loss", result.history)
        self.assertEqual(len(result.history["loss"]), config.epochs)
        self.assertIn("loss", result.metrics)
        self.assertIsInstance(result.metrics["loss"], float)
        self.assertGreater(result.training_time, 0.0)

    # ------------------------------------------------------------------ #
    # Unsupervised / inputs-only training
    # ------------------------------------------------------------------ #
    def test_train_unsupervised_inputs_only(self) -> None:
        """Training should work when batches contain only inputs (autoencoder-style)."""
        trainer = PyTorchModelTrainer()
        config = PyTorchTrainingConfig(
            epochs=2,
            learning_rate=1e-2,
            optimizer="adam",
            loss_fn="mse",
            device="cpu",
            verbose=False,
        )

        # Model maps R^1 -> R^1, we use inputs as targets implicitly
        autoencoder = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

        result = trainer.train(
            model=autoencoder,
            dataloader=self.unsup_loader,
            config=config,
            validation_dataloader=None,
        )

        self.assertIsInstance(result, TrainingResult)
        self.assertEqual(len(result.history["loss"]), config.epochs)

    # ------------------------------------------------------------------ #
    # max_batches_per_epoch / max_batches behaviour
    # ------------------------------------------------------------------ #
    def test_train_respects_max_batches_per_epoch(self) -> None:
        """Trainer should stop an epoch after max_batches_per_epoch batches."""
        # Dataloader with known number of batches: 100 samples / 10 batch size = 10 batches
        dataloader = DataLoader(self.train_dataset, batch_size=10, shuffle=False)

        class CountingModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(1, 1)
                self.call_count = 0

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.call_count += 1
                return self.linear(x)

        model = CountingModel()
        trainer = PyTorchModelTrainer()

        max_batches = 3
        config = PyTorchTrainingConfig(
            epochs=1,
            learning_rate=1e-2,
            optimizer="adam",
            loss_fn="mse",
            device="cpu",
            verbose=False,
            max_batches_per_epoch=max_batches,
        )

        _ = trainer.train(
            model=model,
            dataloader=dataloader,
            config=config,
            validation_dataloader=None,
        )

        # Model forward should have been called only for the limited number of batches
        self.assertLessEqual(
            model.call_count,
            max_batches,
            msg=f"Expected at most {max_batches} steps, got {model.call_count}",
        )

    def test_train_respects_max_batches_fallback_when_per_epoch_missing(self) -> None:
        """If max_batches_per_epoch is not set, trainer should fall back to max_batches."""
        dataloader = DataLoader(self.train_dataset, batch_size=10, shuffle=False)

        class CountingModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(1, 1)
                self.call_count = 0

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.call_count += 1
                return self.linear(x)

        model = CountingModel()
        trainer = PyTorchModelTrainer()

        max_batches = 4
        # Only max_batches provided (legacy / fallback name)
        config = PyTorchTrainingConfig(
            epochs=1,
            learning_rate=1e-2,
            optimizer="adam",
            loss_fn="mse",
            device="cpu",
            verbose=False,
            max_batches=max_batches,  # type: ignore[arg-type]
        )

        _ = trainer.train(
            model=model,
            dataloader=dataloader,
            config=config,
            validation_dataloader=None,
        )

        self.assertLessEqual(
            model.call_count,
            max_batches,
            msg=f"Expected at most {max_batches} steps, got {model.call_count}",
        )

    # ------------------------------------------------------------------ #
    # Validation support
    # ------------------------------------------------------------------ #
    def test_train_with_validation_tracks_val_loss_and_best_epoch(self) -> None:
        """Training with validation should record val_loss and best_epoch."""
        trainer = PyTorchModelTrainer()
        # Use the same loader as validation for simplicity
        config = PyTorchTrainingConfig(
            epochs=2,
            learning_rate=1e-2,
            optimizer="adam",
            loss_fn="mse",
            device="cpu",
            verbose=False,
        )

        result = trainer.train(
            model=self.model,
            dataloader=self.train_loader,
            config=config,
            validation_dataloader=self.train_loader,
        )

        self.assertIn("val_loss", result.history)
        self.assertEqual(len(result.history["val_loss"]), config.epochs)
        self.assertIn("val_loss", result.metrics)
        self.assertIsInstance(result.metrics["val_loss"], float)
        self.assertIsNotNone(result.best_epoch)
        self.assertGreaterEqual(result.best_epoch, 1)
        self.assertLessEqual(result.best_epoch, config.epochs)

    # ------------------------------------------------------------------ #
    # Error handling
    # ------------------------------------------------------------------ #
    def test_train_raises_training_error_on_empty_dataloader(self) -> None:
        """Trainer should raise TrainingError when dataloader yields no batches."""
        empty_dataset = TensorDataset(torch.empty(0, 1), torch.empty(0, 1))
        empty_loader = DataLoader(empty_dataset, batch_size=4)

        trainer = PyTorchModelTrainer()
        config = PyTorchTrainingConfig(
            epochs=1,
            learning_rate=1e-2,
            optimizer="adam",
            loss_fn="mse",
            device="cpu",
            verbose=False,
        )

        with self.assertRaises(TrainingError):
            _ = trainer.train(
                model=self.model,
                dataloader=empty_loader,
                config=config,
                validation_dataloader=None,
            )

    def test_train_with_unknown_optimizer_raises_training_error(self) -> None:
        """Unknown optimizer name should result in TrainingError."""
        trainer = PyTorchModelTrainer()
        config = PyTorchTrainingConfig(
            epochs=1,
            learning_rate=1e-3,
            optimizer="unknown_optimizer",
            loss_fn="mse",
            device="cpu",
            verbose=False,
        )

        with self.assertRaises(TrainingError):
            _ = trainer.train(
                model=self.model,
                dataloader=self.train_loader,
                config=config,
                validation_dataloader=None,
            )

    def test_prepare_batch_raises_on_non_tensor_inputs(self) -> None:
        """_prepare_batch should raise TrainingError when inputs are non-tensor after move."""
        trainer = PyTorchModelTrainer()
        device = torch.device("cpu")

        # Batch is a plain dict of non-tensors → should fail
        bad_batch: dict[str, Any] = {"x": "not-a-tensor"}

        with self.assertRaises(TrainingError):
            _ = trainer._prepare_batch(bad_batch, device=device)

    def test_prepare_batch_stacks_list_of_tensors(self) -> None:
        """_prepare_batch should stack list/tuple of tensors into (batch_size, ...) tensor."""
        trainer = PyTorchModelTrainer()
        device = torch.device("cpu")

        t1 = torch.tensor([1.0])
        t2 = torch.tensor([2.0])
        t3 = torch.tensor([3.0])
        batch = [t1, t2, t3]

        inputs, targets = trainer._prepare_batch(batch, device=device)
        self.assertIsNone(targets)
        self.assertIsInstance(inputs, torch.Tensor)
        self.assertEqual(inputs.shape, (3, 1))
        self.assertTrue(torch.allclose(inputs.squeeze(), torch.tensor([1.0, 2.0, 3.0])))


if __name__ == "__main__":
    unittest.main()
