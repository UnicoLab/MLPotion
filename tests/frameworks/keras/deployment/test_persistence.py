import unittest
import keras
from loguru import logger

from mlpotion.core.exceptions import ModelPersistenceError
from mlpotion.frameworks.keras.deployment.persistence import KerasModelPersistence
from tests.core import TestBase  # provides temp_dir, setup/teardown


class TestKerasModelPersistence(TestBase):
    def setUp(self) -> None:
        super().setUp()
        logger.info(f"Creating temp directory for {self.__class__.__name__}")
        self.model_path = self.temp_dir / "test_model.keras"
        logger.info(f"Model will be saved to: {self.model_path}")

        # Simple Keras model for testing save/load
        logger.info("Building test Keras model")
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=(4,)),
                keras.layers.Dense(8, activation="relu"),
                keras.layers.Dense(1),
            ]
        )
        logger.info(f"Test model summary:\n{self.model.summary(print_fn=str)}")

    def test_save_and_load_with_inspection(self) -> None:
        """Save a model and then load it back with inspection enabled."""
        logger.info("Testing save and load with inspection=True")

        persistence = KerasModelPersistence(
            path=self.model_path,
            model=self.model,
        )

        logger.info("Saving model...")
        persistence.save()

        logger.info("Checking that model file exists on disk")
        self.assertTrue(self.model_path.exists())
        self.assertTrue(self.model_path.is_file())

        logger.info("Loading model using a new persistence instance")
        loader = KerasModelPersistence(path=self.model_path)
        loaded_model, inspection = loader.load()

        logger.info("Asserting loaded model and inspection output")
        self.assertIsInstance(loaded_model, keras.Model)
        self.assertIsNotNone(inspection)
        self.assertIsInstance(inspection, dict)
        # The loader instance should keep a reference to the loaded model
        self.assertIs(loader.model, loaded_model)

    def test_save_without_model_raises_error(self) -> None:
        """Saving without an attached model should raise ModelPersistenceError."""
        logger.info("Testing save() when no model is attached")

        persistence = KerasModelPersistence(path=self.model_path)

        with self.assertRaises(ModelPersistenceError):
            persistence.save()

    def test_load_missing_path_raises_error(self) -> None:
        """Loading from a non-existent path should raise ModelPersistenceError."""
        missing_path = self.temp_dir / "non_existent_model.keras"
        logger.info(f"Testing load() from missing path: {missing_path}")

        persistence = KerasModelPersistence(path=missing_path)

        with self.assertRaises(ModelPersistenceError):
            persistence.load()

    def test_save_with_overwrite_false_raises_if_file_exists(self) -> None:
        """Saving with overwrite=False should fail if the target already exists."""
        logger.info("Testing save() with overwrite=False when file already exists")

        # First save to create the file
        first_persistence = KerasModelPersistence(
            path=self.model_path,
            model=self.model,
        )
        logger.info("Initial save to create model file")
        first_persistence.save()
        self.assertTrue(self.model_path.exists())

        # Second save with overwrite disabled should raise
        second_persistence = KerasModelPersistence(
            path=self.model_path,
            model=self.model,
        )

        logger.info("Attempting to save again with overwrite=False (should fail)")
        with self.assertRaises(ModelPersistenceError):
            second_persistence.save(overwrite=False)

    def test_load_without_inspection(self) -> None:
        """Load a model with inspect=False should return None for inspection."""
        logger.info("Testing load() with inspect=False")

        persistence = KerasModelPersistence(
            path=self.model_path,
            model=self.model,
        )
        logger.info("Saving model before load")
        persistence.save()

        loader = KerasModelPersistence(path=self.model_path)
        loaded_model, inspection = loader.load(inspect=False)

        logger.info("Asserting loaded model and absence of inspection info")
        self.assertIsInstance(loaded_model, keras.Model)
        self.assertIsNone(inspection)


if __name__ == "__main__":
    unittest.main()
