"""Tests for TensorFlow configuration classes."""

import unittest
from loguru import logger

import keras

from mlpotion.frameworks.tensorflow.config import ModelTrainingConfig
from tests.core import TestBase


class TestModelTrainingConfig(TestBase):
    """Test ModelTrainingConfig class."""

    def test_optimizer_attribute_exists(self) -> None:
        """Test that ModelTrainingConfig has 'optimizer' attribute (not 'optimizer_type')."""
        logger.info("Testing ModelTrainingConfig.optimizer attribute")

        config = ModelTrainingConfig()

        # Should have 'optimizer' attribute
        self.assertTrue(hasattr(config, "optimizer"))
        self.assertEqual(config.optimizer, "adam")  # Default value

        # Should NOT have 'optimizer_type' attribute
        self.assertFalse(hasattr(config, "optimizer_type"))

        logger.info("✓ ModelTrainingConfig has 'optimizer' attribute")

    def test_optimizer_can_be_set(self) -> None:
        """Test that optimizer can be set to different values."""
        logger.info("Testing ModelTrainingConfig.optimizer can be set")

        config = ModelTrainingConfig(optimizer="sgd")
        self.assertEqual(config.optimizer, "sgd")

        config = ModelTrainingConfig(optimizer="rmsprop")
        self.assertEqual(config.optimizer, "rmsprop")

        # Can also be an optimizer instance
        adam_opt = keras.optimizers.Adam(learning_rate=0.01)
        config = ModelTrainingConfig(optimizer=adam_opt)
        self.assertEqual(config.optimizer, adam_opt)

        logger.info("✓ ModelTrainingConfig.optimizer can be set correctly")

    def test_optimizer_with_learning_rate(self) -> None:
        """Test that optimizer works with learning_rate."""
        logger.info("Testing ModelTrainingConfig with optimizer and learning_rate")

        config = ModelTrainingConfig(
            optimizer="adam",
            learning_rate=0.001,
        )

        self.assertEqual(config.optimizer, "adam")
        self.assertEqual(config.learning_rate, 0.001)

        logger.info("✓ ModelTrainingConfig works with optimizer and learning_rate")

    def test_config_serialization(self) -> None:
        """Test that config can be serialized/deserialized (for ZenML)."""
        logger.info("Testing ModelTrainingConfig serialization")

        config = ModelTrainingConfig(
            epochs=5,
            batch_size=64,
            optimizer="adam",
            learning_rate=0.001,
            loss="mse",
        )

        # Convert to dict (as ZenML might do)
        config_dict = config.dict()
        self.assertIn("optimizer", config_dict)
        self.assertEqual(config_dict["optimizer"], "adam")
        self.assertNotIn("optimizer_type", config_dict)

        logger.info("✓ ModelTrainingConfig serializes correctly")


if __name__ == "__main__":
    unittest.main()
