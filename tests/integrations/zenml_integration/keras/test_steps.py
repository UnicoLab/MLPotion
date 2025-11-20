"""Unit tests for Keras ZenML steps."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from loguru import logger

# Check if keras is available
try:
    import keras

    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    keras = None

# Check if zenml is available
try:
    import zenml

    ZENML_AVAILABLE = True
except ImportError:
    ZENML_AVAILABLE = False
    zenml = None

# Only import steps if dependencies are available
if KERAS_AVAILABLE and ZENML_AVAILABLE:
    from mlpotion.integrations.zenml.keras.steps import (
        load_data,
        transform_data,
        train_model,
        evaluate_model,
        export_model,
        save_model,
        load_model,
        inspect_model,
    )
    from mlpotion.frameworks.keras.data.loaders import CSVSequence

from tests.core import TestBase


@pytest.mark.skipif(not ZENML_AVAILABLE, reason="ZenML not installed")
@pytest.mark.skipif(not KERAS_AVAILABLE, reason="Keras not installed")
@pytest.mark.integration
@pytest.mark.keras
class TestKerasZenMLSteps(TestBase):
    """Test Keras ZenML integration steps."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class-level fixtures."""
        super().setUpClass()
        logger.info("Setting up Keras ZenML tests")
        # Note: ZenML steps can be tested without initializing a ZenML client

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        logger.info(f"Setting up Keras ZenML tests in {self.temp_dir}")

        # Create sample CSV data
        self.n_samples = 100
        self.n_features = 5
        self.csv_file = self.test_subdir / "train_data.csv"

        rng = np.random.default_rng(42)
        data = {
            **{
                f"feature_{i}": rng.normal(size=self.n_samples)
                for i in range(self.n_features)
            },
            "target": rng.normal(size=self.n_samples),
        }
        df = pd.DataFrame(data)
        df.to_csv(self.csv_file, index=False)
        logger.info(f"Created CSV file: {self.csv_file}")

        # Create a simple model
        self.model = keras.Sequential(
            [
                keras.layers.Dense(
                    16, activation="relu", input_shape=(self.n_features,)
                ),
                keras.layers.Dense(8, activation="relu"),
                keras.layers.Dense(1),
            ]
        )
        logger.info("Created simple Keras model")

    def test_load_data_step(self) -> None:
        """Test load_data ZenML step."""
        logger.info("Testing load_data step")

        # Call the step directly (not in pipeline)
        sequence = load_data(
            file_path=str(self.csv_file),
            batch_size=32,
            label_name="target",
            shuffle=True,
        )

        self.assertIsInstance(sequence, CSVSequence)
        self.assertGreater(len(sequence), 0)

        # Verify sequence structure
        batch_x, batch_y = sequence[0]
        self.assertEqual(batch_x.shape[1], self.n_features)
        self.assertIsNotNone(batch_y)

        logger.info("✓ load_data step working correctly")

    def test_train_model_step(self) -> None:
        """Test train_model ZenML step."""
        logger.info("Testing train_model step")

        # Load data
        sequence = load_data(
            file_path=str(self.csv_file),
            batch_size=32,
            label_name="target",
        )

        # Compile the model first
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        # Train model - returns tuple (model, metrics)
        trained_model, training_metrics = train_model(
            model=self.model,
            data=sequence,
            epochs=2,
            learning_rate=0.001,
            verbose=0,
        )

        self.assertIsInstance(trained_model, keras.Model)
        self.assertTrue(trained_model.built)
        self.assertIsInstance(training_metrics, dict)

        logger.info("✓ train_model step working correctly")

    def test_evaluate_model_step(self) -> None:
        """Test evaluate_model ZenML step."""
        logger.info("Testing evaluate_model step")

        # Load data
        sequence = load_data(
            file_path=str(self.csv_file),
            batch_size=32,
            label_name="target",
        )

        # Compile the model first
        self.model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"],
        )

        # Evaluate
        metrics = evaluate_model(
            model=self.model,
            data=sequence,
            verbose=0,
        )

        self.assertIsInstance(metrics, dict)
        self.assertIn("loss", metrics)

        logger.info("✓ evaluate_model step working correctly")

    def test_save_and_load_model_steps(self) -> None:
        """Test save_model and load_model ZenML steps."""
        logger.info("Testing save_model and load_model steps")

        model_path = self.test_subdir / "saved_model.keras"

        # Save model
        saved_path = save_model(
            model=self.model,
            save_path=str(model_path),
        )

        self.assertEqual(saved_path, str(model_path))
        self.assertTrue(Path(model_path).exists())

        # Load model
        loaded_model = load_model(
            model_path=str(model_path),
            inspect=True,
        )

        self.assertIsInstance(loaded_model, keras.Model)
        self.assertEqual(len(loaded_model.layers), len(self.model.layers))

        logger.info("✓ save_model and load_model steps working correctly")

    def test_export_model_step(self) -> None:
        """Test export_model ZenML step."""
        logger.info("Testing export_model step")

        export_path = self.test_subdir / "exported_model.keras"

        # Export model
        exported_path = export_model(
            model=self.model,
            export_path=str(export_path),
            export_format="keras",
        )

        self.assertEqual(exported_path, str(export_path))
        self.assertTrue(Path(export_path).exists())

        logger.info("✓ export_model step working correctly")

    def test_inspect_model_step(self) -> None:
        """Test inspect_model ZenML step."""
        logger.info("Testing inspect_model step")

        # Build the model first
        self.model.build(input_shape=(None, self.n_features))

        # Inspect model
        inspection = inspect_model(
            model=self.model,
            include_layers=True,
            include_signatures=True,
        )

        self.assertIsInstance(inspection, dict)
        self.assertIn("name", inspection)
        self.assertIn("parameters", inspection)
        self.assertIn("inputs", inspection)
        self.assertIn("outputs", inspection)

        logger.info("✓ inspect_model step working correctly")

    @pytest.mark.skip(
        reason="KerasCSVToCSVTransformer config needs refactoring for ZenML step usage"
    )
    def test_transform_data_step(self) -> None:
        """Test transform_data ZenML step."""
        logger.info("Testing transform_data step")

        # Load data
        sequence = load_data(
            file_path=str(self.csv_file),
            batch_size=32,
            label_name="target",
        )

        # Compile the model
        self.model.compile(optimizer="adam", loss="mse")

        output_path = self.test_subdir / "predictions.csv"

        # Transform data
        result_path = transform_data(
            dataset=sequence,
            model=self.model,
            data_output_path=str(output_path),
            data_output_per_batch=False,
        )

        self.assertEqual(result_path, str(output_path))

        logger.info("✓ transform_data step working correctly")


if __name__ == "__main__":
    unittest.main()
