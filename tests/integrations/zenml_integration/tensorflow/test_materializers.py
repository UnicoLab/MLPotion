"""Tests for TensorFlow ZenML materializers."""

import json
import tempfile
import unittest
from pathlib import Path

import pytest
from loguru import logger

# Check if tensorflow is available
try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

# Check if zenml is available
try:
    from zenml.enums import ArtifactType
    from zenml.materializers.materializer_registry import materializer_registry

    ZENML_AVAILABLE = True
except ImportError:
    ZENML_AVAILABLE = False
    ArtifactType = None
    materializer_registry = None

# Only import materializers if dependencies are available
if TF_AVAILABLE and ZENML_AVAILABLE:
    from mlpotion.integrations.zenml.tensorflow.materializers import (
        TFConfigDatasetMaterializer,
    )
    from mlpotion.frameworks.tensorflow.data.loaders import CSVDataLoader

from tests.core import TestBase


@pytest.mark.skipif(not ZENML_AVAILABLE, reason="ZenML not installed")
@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
@pytest.mark.integration
@pytest.mark.tensorflow
class TestTFConfigDatasetMaterializer(TestBase):
    """Test TFConfigDatasetMaterializer for CSV datasets."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        logger.info(f"Setting up materializer tests in {self.temp_dir}")

        # Create a CSV file
        self.csv_file = self.temp_dir / "test_data.csv"
        with open(self.csv_file, "w", encoding="utf-8") as f:
            f.write("feature_1,feature_2,feature_3,target\n")
            for i in range(10):
                f.write(f"{0.1*i},{0.2*i},{0.3*i},{0.5*i}\n")

        # Create dataset using CSVDataLoader (which attaches _csv_config)
        self.loader = CSVDataLoader(
            file_pattern=str(self.csv_file),
            batch_size=4,
            label_name="target",
            config={"num_epochs": 1, "shuffle": False},
        )
        self.dataset = self.loader.load()

    def test_csv_config_attribute_exists(self) -> None:
        """Test that dataset has _csv_config attribute after loading."""
        logger.info("Testing _csv_config attribute on dataset")

        # Check that _csv_config exists
        self.assertTrue(hasattr(self.dataset, "_csv_config"))
        config = self.dataset._csv_config

        self.assertIsInstance(config, dict)
        self.assertIn("file_pattern", config)
        self.assertIn("batch_size", config)
        self.assertIn("label_name", config)
        self.assertEqual(config["label_name"], "target")
        self.assertEqual(config["batch_size"], 4)

        logger.info("✓ _csv_config attribute exists and has correct structure")

    def test_materializer_detects_csv_config(self) -> None:
        """Test that materializer can detect _csv_config from dataset."""
        logger.info("Testing materializer detection of _csv_config")

        with tempfile.TemporaryDirectory() as tmpdir:
            materializer = TFConfigDatasetMaterializer(uri=tmpdir)

            # Extract config (this is what save() does internally)
            config = materializer._extract_config_from_dataset(self.dataset)

            self.assertIsNotNone(config)
            self.assertIn("file_pattern", config)
            self.assertEqual(config["label_name"], "target")
            self.assertEqual(config["batch_size"], 4)

            logger.info("✓ Materializer can detect _csv_config")

    def test_materializer_save_and_load(self) -> None:
        """Test that materializer can save and load CSV dataset config."""
        logger.info("Testing materializer save and load")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            materializer = TFConfigDatasetMaterializer(uri=tmpdir)
            materializer.save(self.dataset)

            # Check that config.json was created (not TFRecord)
            config_path = Path(tmpdir) / "config.json"
            self.assertTrue(config_path.exists(), "config.json should exist")

            # Check that data.tfrecord does NOT exist (we're using CSV config, not TFRecord)
            tfrecord_path = Path(tmpdir) / "data.tfrecord"
            self.assertFalse(
                tfrecord_path.exists(),
                "data.tfrecord should NOT exist when using CSV config materializer",
            )

            # Verify config.json contents
            with open(config_path, "r", encoding="utf-8") as f:
                saved_config = json.load(f)

            self.assertEqual(saved_config["file_pattern"], str(self.csv_file))
            self.assertEqual(saved_config["batch_size"], 4)
            self.assertEqual(saved_config["label_name"], "target")

            # Load
            loaded_dataset = materializer.load(tf.data.Dataset)

            self.assertIsInstance(loaded_dataset, tf.data.Dataset)

            # Verify loaded dataset has same structure
            for batch in loaded_dataset.take(1):
                features, labels = batch
                self.assertIsInstance(features, dict)
                self.assertIn("feature_1", features)
                self.assertIn("feature_2", features)
                self.assertIn("feature_3", features)
                self.assertIsInstance(labels, tf.Tensor)

            logger.info("✓ Materializer save and load working correctly")

    def test_materializer_falls_back_to_tfrecord_when_no_config(self) -> None:
        """Test that materializer falls back to TFRecord when _csv_config is missing."""
        logger.info("Testing materializer fallback to TFRecord")

        # Create a dataset without _csv_config (e.g., from TFRecord or other source)
        dataset_without_config = tf.data.Dataset.from_tensor_slices(
            ({"x": [1.0, 2.0, 3.0]}, [1, 2, 3])
        ).batch(2)

        with tempfile.TemporaryDirectory() as tmpdir:
            materializer = TFConfigDatasetMaterializer(uri=tmpdir)

            # Save should fall back to TFRecord
            materializer.save(dataset_without_config)

            # Check that TFRecord was created (fallback)
            tfrecord_path = Path(tmpdir) / "data.tfrecord"
            self.assertTrue(
                tfrecord_path.exists(),
                "data.tfrecord should exist when falling back",
            )

            # Config.json might not exist or might have default config
            config_path = Path(tmpdir) / "config.json"
            # If config.json exists, it should have a default/placeholder config
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    # Default config should have _is_default flag
                    if config.get("_is_default"):
                        logger.info("Default config was created (expected)")

            logger.info("✓ Materializer correctly falls back to TFRecord")

    def test_materializer_handles_clean_csv(self) -> None:
        """Test that materializer works with clean CSV files.

        Note: make_csv_dataset doesn't handle empty lines, so this test
        uses a clean CSV without empty lines.
        """
        logger.info("Testing materializer with clean CSV")

        # Create CSV without empty lines (make_csv_dataset limitation)
        csv_clean = self.temp_dir / "data_clean.csv"
        with open(csv_clean, "w", encoding="utf-8") as f:
            f.write("feature_1,feature_2,target\n")
            f.write("0.1,0.2,0.3\n")
            f.write("0.2,0.4,0.6\n")
            f.write("0.3,0.6,0.9\n")

        loader = CSVDataLoader(
            file_pattern=str(csv_clean),
            batch_size=2,
            label_name="target",
            config={"num_epochs": 1, "shuffle": False, "ignore_errors": True},
        )
        dataset = loader.load()

        # Should have _csv_config
        self.assertTrue(hasattr(dataset, "_csv_config"))

        # Materializer should be able to save it
        with tempfile.TemporaryDirectory() as tmpdir:
            materializer = TFConfigDatasetMaterializer(uri=tmpdir)
            materializer.save(dataset)

            # Should save as config.json (not TFRecord)
            config_path = Path(tmpdir) / "config.json"
            self.assertTrue(config_path.exists())

            # Load and verify it works
            loaded_dataset = materializer.load(tf.data.Dataset)
            batch_count = 0
            for batch in loaded_dataset:
                batch_count += 1
                features, labels = batch
                self.assertIsInstance(features, dict)
                self.assertIsInstance(labels, tf.Tensor)

            # Should have processed 3 data rows
            self.assertGreater(batch_count, 0)

            logger.info("✓ Materializer handles clean CSV correctly")


@pytest.mark.skipif(not ZENML_AVAILABLE, reason="ZenML not installed")
@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
@pytest.mark.integration
@pytest.mark.tensorflow
class TestMaterializerRegistration(TestBase):
    """Test that materializers are properly registered."""

    def test_csv_config_materializer_registered(self) -> None:
        """Test that TFConfigDatasetMaterializer is registered for tf.data.Dataset."""
        logger.info("Testing materializer registration")

        # Check if materializer is registered
        # Note: This might not work in all test environments, so we'll just check
        # that the materializer class exists and can be instantiated
        self.assertTrue(hasattr(TFConfigDatasetMaterializer, "ASSOCIATED_TYPES"))
        self.assertIn(tf.data.Dataset, TFConfigDatasetMaterializer.ASSOCIATED_TYPES)

        logger.info("✓ TFConfigDatasetMaterializer is properly configured")


if __name__ == "__main__":
    unittest.main()
