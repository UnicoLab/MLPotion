import unittest
from unittest.mock import MagicMock, patch

import keras
import numpy as np
from loguru import logger

from mlpotion.frameworks.keras.training.trainers import ModelTrainer
from tests.core import TestBase  # provides temp_dir, setUp/tearDown


class TestModelTrainer(TestBase):
    def setUp(self) -> None:
        super().setUp()
        logger.info(f"Setting up test data for {self.__class__.__name__}")

        self.num_samples = 32
        self.num_features = 4

        rng = np.random.default_rng(42)
        self.x = rng.normal(size=(self.num_samples, self.num_features)).astype(
            "float32"
        )
        self.y = rng.integers(0, 2, size=(self.num_samples, 1)).astype("float32")

        logger.info("Building test Keras model")
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.num_features,)),
                keras.layers.Dense(8, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        self.trainer = ModelTrainer()

    # ------------------------------------------------------------------ #
    # Happy path: compile + train in one shot with (x, y)
    # ------------------------------------------------------------------ #
    def test_train_compiles_and_trains_tuple_data(self) -> None:
        """Uncompiled model + compile_params should compile and train."""
        logger.info("Testing train() with compile_params and (x, y) data")

        compile_params = {
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "metrics": ["accuracy"],
        }
        fit_params = {"epochs": 2, "batch_size": 8, "verbose": 0}

        # Force the trainer to consider the model "uncompiled"
        with patch.object(
            self.trainer,
            "_is_compiled",
            return_value=False,
        ) as mock_is_compiled, patch.object(
            self.model,
            "compile",
            wraps=self.model.compile,
        ) as mock_compile:
            history = self.trainer.train(
                model=self.model,
                data=(self.x, self.y),
                compile_params=compile_params,
                fit_params=fit_params,
            )

        logger.info(f"Training history: {history}")

        mock_is_compiled.assert_called_once_with(self.model)
        mock_compile.assert_called_once()
        _, compile_kwargs = mock_compile.call_args
        for k, v in compile_params.items():
            self.assertIn(k, compile_kwargs)

        # History dict should have at least loss curve
        self.assertIsInstance(history, dict)
        self.assertIn("loss", history)
        self.assertIsInstance(history["loss"], list)
        self.assertEqual(len(history["loss"]), fit_params["epochs"])

    # ------------------------------------------------------------------ #
    # Error when uncompiled and no compile_params
    # ------------------------------------------------------------------ #
    def test_train_raises_if_uncompiled_and_no_compile_params(self) -> None:
        """Uncompiled model without compile_params should raise RuntimeError."""
        logger.info("Testing train() with uncompiled model and no compile_params")

        with patch.object(self.trainer, "_is_compiled", return_value=False):
            with self.assertRaises(RuntimeError):
                self.trainer.train(model=self.model, data=(self.x, self.y))

    # ------------------------------------------------------------------ #
    # Precompiled model: compile_params should be ignored (no recompile)
    # ------------------------------------------------------------------ #
    def test_train_uses_precompiled_model_and_ignores_compile_params(self) -> None:
        """If model is already compiled, compile_params should not recompile."""
        logger.info("Testing train() with precompiled model")

        # Compile once up-front for real
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        fit_params = {"epochs": 1, "batch_size": 8, "verbose": 0}

        # Force _is_compiled to True so _ensure_compiled won't recompile
        with patch.object(
            self.trainer,
            "_is_compiled",
            return_value=True,
        ) as mock_is_compiled, patch.object(
            self.model,
            "compile",
            wraps=self.model.compile,
        ) as mock_compile:
            history = self.trainer.train(
                model=self.model,
                data=(self.x, self.y),
                compile_params={"optimizer": "sgd", "loss": "mse"},
                fit_params=fit_params,
            )

        logger.info(f"Training history (precompiled): {history}")
        mock_is_compiled.assert_called_once_with(self.model)
        mock_compile.assert_not_called()

        self.assertIsInstance(history, dict)
        self.assertIn("loss", history)

    # ------------------------------------------------------------------ #
    # _call_fit dispatch behavior
    # ------------------------------------------------------------------ #
    def test_call_fit_with_tuple_data(self) -> None:
        """_call_fit should unpack (x, y) tuples into positional args."""
        logger.info("Testing _call_fit with (x, y) tuple")

        class DummyModel(keras.Model):
            def __init__(self) -> None:
                super().__init__()
                self.compiled_loss = object()
                self.optimizer = object()
                self.fit = MagicMock(return_value=keras.callbacks.History())

        dummy = DummyModel()
        fit_kwargs = {"epochs": 3, "batch_size": 4, "verbose": 0}

        _ = self.trainer._call_fit(
            model=dummy,
            data=(self.x, self.y),
            fit_kwargs=fit_kwargs,
        )

        dummy.fit.assert_called_once()
        args, kwargs = dummy.fit.call_args
        self.assertIs(args[0], self.x)
        self.assertIs(args[1], self.y)
        self.assertEqual(kwargs["epochs"], 3)
        self.assertEqual(kwargs["batch_size"], 4)
        self.assertEqual(kwargs["verbose"], 0)

    def test_call_fit_with_dict_data(self) -> None:
        """_call_fit should treat dict data as keyword args (x=..., y=...)."""
        logger.info("Testing _call_fit with dict data")

        class DummyModel(keras.Model):
            def __init__(self) -> None:
                super().__init__()
                self.compiled_loss = object()
                self.optimizer = object()
                self.fit = MagicMock(return_value=keras.callbacks.History())

        dummy = DummyModel()
        data = {"x": self.x, "y": self.y}
        fit_kwargs = {"epochs": 2}

        _ = self.trainer._call_fit(
            model=dummy,
            data=data,
            fit_kwargs=fit_kwargs,
        )

        dummy.fit.assert_called_once()
        _, kwargs = dummy.fit.call_args

        self.assertIs(kwargs["x"], self.x)
        self.assertIs(kwargs["y"], self.y)
        self.assertEqual(kwargs["epochs"], 2)

    def test_call_fit_with_other_data(self) -> None:
        """_call_fit should pass non-tuple/non-dict data as a single arg."""
        logger.info("Testing _call_fit with 'other' data type")

        class DummyModel(keras.Model):
            def __init__(self) -> None:
                super().__init__()
                self.compiled_loss = object()
                self.optimizer = object()
                self.fit = MagicMock(return_value=keras.callbacks.History())

        dummy = DummyModel()
        fake_dataset = object()
        fit_kwargs = {"epochs": 1}

        _ = self.trainer._call_fit(
            model=dummy,
            data=fake_dataset,
            fit_kwargs=fit_kwargs,
        )

        dummy.fit.assert_called_once()
        args, kwargs = dummy.fit.call_args

        self.assertIs(args[0], fake_dataset)
        self.assertEqual(kwargs["epochs"], 1)

    # ------------------------------------------------------------------ #
    # History conversion
    # ------------------------------------------------------------------ #
    def test_history_to_dict_with_valid_history(self) -> None:
        """_history_to_dict should convert real History to dict of lists."""
        logger.info("Testing _history_to_dict with real Keras History")

        # Small real training to get a History object
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        history_obj = self.model.fit(
            self.x,
            self.y,
            epochs=2,
            batch_size=8,
            verbose=0,
        )

        history_dict = self.trainer._history_to_dict(history_obj)
        logger.info(f"Converted history dict: {history_dict}")

        self.assertIsInstance(history_dict, dict)
        self.assertIn("loss", history_dict)
        self.assertEqual(len(history_dict["loss"]), 2)
        for key, values in history_dict.items():
            self.assertIsInstance(values, list)
            for v in values:
                self.assertIsInstance(v, float)

    def test_history_to_dict_with_non_history_returns_empty(self) -> None:
        """Non-History input should return an empty dict."""
        logger.info("Testing _history_to_dict with non-History input")

        history_dict = self.trainer._history_to_dict(history=object())
        self.assertEqual(history_dict, {})

    # ------------------------------------------------------------------ #
    # Validation / compiled state helpers
    # ------------------------------------------------------------------ #
    def test_validate_model_raises_for_non_keras_model(self) -> None:
        """_validate_model should reject non-keras.Model instances."""
        logger.info("Testing _validate_model with invalid model")

        with self.assertRaises(TypeError):
            self.trainer._validate_model(model=object())  # type: ignore[arg-type]

    def test_is_compiled_heuristic_with_dummy_objects(self) -> None:
        """_is_compiled should behave according to its own heuristic."""
        logger.info("Testing _is_compiled heuristic on dummy objects")

        class DummyUncompiled:
            compiled_loss = None
            optimizer = None

        class DummyCompiledLoss:
            compiled_loss = object()
            optimizer = None

        class DummyCompiledOptimizer:
            compiled_loss = None
            optimizer = object()

        self.assertFalse(self.trainer._is_compiled(DummyUncompiled()))  # both None
        self.assertTrue(self.trainer._is_compiled(DummyCompiledLoss()))  # compiled_loss set
        self.assertTrue(self.trainer._is_compiled(DummyCompiledOptimizer()))  # optimizer set


if __name__ == "__main__":
    unittest.main()
