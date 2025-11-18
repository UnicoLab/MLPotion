"""Unit tests for PyTorch ZenML steps."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from loguru import logger

# Check if pytorch is available
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None
    DataLoader = None

# Check if zenml is available
try:
    import zenml
    ZENML_AVAILABLE = True
except ImportError:
    ZENML_AVAILABLE = False
    zenml = None

# Only import steps if dependencies are available
if PYTORCH_AVAILABLE and ZENML_AVAILABLE:
    from mlpotion.integrations.zenml.pytorch.steps import (
        load_csv_data,
        load_streaming_csv_data,
        train_model,
        evaluate_model,
        export_model,
        save_model,
        load_model,
    )

from tests.core import TestBase


if PYTORCH_AVAILABLE:
    class SimpleModel(nn.Module):
        """Simple PyTorch model for testing."""

        def __init__(self, input_size: int = 5):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 16)
            self.fc2 = nn.Linear(16, 8)
            self.fc3 = nn.Linear(8, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
else:
    SimpleModel = None


@pytest.mark.skipif(not ZENML_AVAILABLE, reason="ZenML not installed")
@pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.integration
@pytest.mark.pytorch
class TestPyTorchZenMLSteps(TestBase):
    """Test PyTorch ZenML integration steps."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class-level fixtures."""
        super().setUpClass()
        logger.info("Setting up PyTorch ZenML tests")
        # Note: ZenML steps can be tested without initializing a ZenML client

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        logger.info(f"Setting up PyTorch ZenML tests in {self.temp_dir}")

        # Create sample CSV data
        self.n_samples = 100
        self.n_features = 5
        self.csv_file = self.test_subdir / "train_data.csv"

        rng = np.random.default_rng(42)
        data = {
            **{f"feature_{i}": rng.normal(size=self.n_samples) for i in range(self.n_features)},
            "target": rng.normal(size=self.n_samples),
        }
        df = pd.DataFrame(data)
        df.to_csv(self.csv_file, index=False)
        logger.info(f"Created CSV file: {self.csv_file}")

        # Create a simple model
        self.model = SimpleModel(input_size=self.n_features)
        logger.info("Created simple PyTorch model")

    def test_load_csv_data_step(self) -> None:
        """Test load_csv_data ZenML step."""
        logger.info("Testing load_csv_data step")

        # Call the step directly (not in pipeline)
        dataloader = load_csv_data(
            file_path=str(self.csv_file),
            batch_size=32,
            label_name="target",
            shuffle=True,
        )

        self.assertIsInstance(dataloader, DataLoader)

        # Verify dataloader structure
        for batch in dataloader:
            if isinstance(batch, tuple):
                features, labels = batch
                self.assertEqual(features.shape[1], self.n_features)
                self.assertIsNotNone(labels)
            break

        logger.info("✓ load_csv_data step working correctly")

    def test_load_streaming_csv_data_step(self) -> None:
        """Test load_streaming_csv_data ZenML step."""
        logger.info("Testing load_streaming_csv_data step")

        # Call the step directly (not in pipeline)
        dataloader = load_streaming_csv_data(
            file_path=str(self.csv_file),
            batch_size=32,
            label_name="target",
            chunksize=50,
        )

        self.assertIsInstance(dataloader, DataLoader)

        # Verify dataloader structure
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            if isinstance(batch, tuple):
                features, labels = batch
                self.assertEqual(features.shape[1], self.n_features)
                self.assertIsNotNone(labels)
            if batch_count >= 2:  # Just check a few batches
                break

        logger.info("✓ load_streaming_csv_data step working correctly")

    def test_train_model_step(self) -> None:
        """Test train_model ZenML step."""
        logger.info("Testing train_model step")

        # Load data
        dataloader = load_csv_data(
            file_path=str(self.csv_file),
            batch_size=32,
            label_name="target",
        )

        # Train model - returns tuple (model, metrics)
        trained_model, training_metrics = train_model(
            model=self.model,
            dataloader=dataloader,
            epochs=2,
            learning_rate=0.001,
            optimizer="adam",
            loss_fn="mse",
            device="cpu",
            verbose=0,
        )

        self.assertIsInstance(trained_model, nn.Module)
        self.assertIsInstance(training_metrics, dict)

        logger.info("✓ train_model step working correctly")

    def test_evaluate_model_step(self) -> None:
        """Test evaluate_model ZenML step."""
        logger.info("Testing evaluate_model step")

        # Load data
        dataloader = load_csv_data(
            file_path=str(self.csv_file),
            batch_size=32,
            label_name="target",
        )

        # Evaluate
        metrics = evaluate_model(
            model=self.model,
            dataloader=dataloader,
            loss_fn="mse",
            device="cpu",
            verbose=0,
        )

        self.assertIsInstance(metrics, dict)
        self.assertIn("loss", metrics)
        self.assertIn("evaluation_time", metrics)

        logger.info("✓ evaluate_model step working correctly")

    def test_save_and_load_model_steps(self) -> None:
        """Test save_model and load_model ZenML steps."""
        logger.info("Testing save_model and load_model steps")

        model_path = self.test_subdir / "saved_model.pth"

        # Save model
        saved_path = save_model(
            model=self.model,
            save_path=str(model_path),
            save_full_model=False,
        )

        self.assertEqual(saved_path, str(model_path))
        self.assertTrue(Path(model_path).exists())

        # Load model
        loaded_model = load_model(
            model_path=str(model_path),
            model_class=SimpleModel,
            map_location="cpu",
        )

        self.assertIsInstance(loaded_model, nn.Module)

        logger.info("✓ save_model and load_model steps working correctly")

    def test_export_model_step_state_dict(self) -> None:
        """Test export_model ZenML step with state_dict format."""
        logger.info("Testing export_model step (state_dict)")

        export_path = self.test_subdir / "exported_model.pth"

        # Export model
        exported_path = export_model(
            model=self.model,
            export_path=str(export_path),
            export_format="state_dict",
            device="cpu",
        )

        self.assertEqual(exported_path, str(export_path))
        self.assertTrue(Path(export_path).exists())

        logger.info("✓ export_model step (state_dict) working correctly")

    def test_export_model_step_torchscript(self) -> None:
        """Test export_model ZenML step with TorchScript format."""
        logger.info("Testing export_model step (TorchScript)")

        export_path = self.test_subdir / "exported_model.pt"

        # Create example input for tracing
        example_input = torch.randn(1, self.n_features)

        # Export model
        exported_path = export_model(
            model=self.model,
            export_path=str(export_path),
            export_format="torchscript",
            device="cpu",
            example_input=example_input,
            jit_mode="trace",
        )

        self.assertEqual(exported_path, str(export_path))
        self.assertTrue(Path(export_path).exists())

        logger.info("✓ export_model step (TorchScript) working correctly")

    def test_train_with_validation_step(self) -> None:
        """Test training with validation data."""
        logger.info("Testing train_model step with validation")

        # Create validation CSV
        val_csv_file = self.test_subdir / "val_data.csv"
        rng = np.random.default_rng(123)
        val_data = {
            **{f"feature_{i}": rng.normal(size=50) for i in range(self.n_features)},
            "target": rng.normal(size=50),
        }
        val_df = pd.DataFrame(val_data)
        val_df.to_csv(val_csv_file, index=False)

        # Load train and validation data
        train_dataloader = load_csv_data(
            file_path=str(self.csv_file),
            batch_size=32,
            label_name="target",
        )

        val_dataloader = load_csv_data(
            file_path=str(val_csv_file),
            batch_size=32,
            label_name="target",
        )

        # Train model with validation - returns tuple (model, metrics)
        trained_model, training_metrics = train_model(
            model=self.model,
            dataloader=train_dataloader,
            epochs=2,
            learning_rate=0.001,
            optimizer="adam",
            loss_fn="mse",
            device="cpu",
            validation_dataloader=val_dataloader,
            verbose=0,
        )

        self.assertIsInstance(trained_model, nn.Module)
        self.assertIsInstance(training_metrics, dict)

        logger.info("✓ train_model with validation working correctly")


if __name__ == "__main__":
    unittest.main()
