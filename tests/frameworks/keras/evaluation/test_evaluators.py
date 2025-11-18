import unittest
from unittest.mock import MagicMock, patch

import keras
import numpy as np
from loguru import logger

from mlpotion.frameworks.keras.evaluation.evaluators import ModelEvaluator
from tests.core import TestBase  # provides temp_dir, setUp/tearDown


class TestModelEvaluator(TestBase):
    def setUp(self) -> None:
        super().setUp()
        logger.info(f"Setting up test data for {self.__class__.__name__}")

        # Small synthetic binary classification dataset
        self.num_samples = 32
        self.num_features = 8

        rng = np.random.default_rng(42)
        self.x = rng.normal(size=(self.num_samples, self.num_features)).astype(
            "float32"
        )
        self.y = rng.integers(0, 2, size=(self.num_samples, 1)).astype("float32")

        logger.info("Building test Keras model")
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.num_features,)),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        self.evaluator = ModelEvaluator()

    # ------------------------------------------------------------------ #
    # Happy path: compile + evaluate in one shot with (x, y) tuple
    # ------------------------------------------------------------------ #
    def test_evaluate_compiles_and_evaluates_tuple_data(self) -> None:
        """Uncompiled model + compile_params should compile and evaluate."""
        logger.info("Testing evaluate() with compile_params and (x, y) data")

        compile_params = {
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "metrics": ["accuracy"],
        }

        eval_params = {"batch_size": 8, "verbose": 0}

        # Force the evaluator to consider the model "uncompiled" so that
        # _ensure_compiled will call model.compile(**compile_params)
        with patch.object(
            self.evaluator,
            "_is_compiled",
            return_value=False,
        ) as mock_is_compiled, patch.object(
            self.model,
            "compile",
            wraps=self.model.compile,
        ) as mock_compile:
            metrics = self.evaluator.evaluate(
                model=self.model,
                data=(self.x, self.y),
                compile_params=compile_params,
                eval_params=eval_params,
            )

        logger.info(f"Evaluation metrics: {metrics}")

        mock_is_compiled.assert_called_once_with(self.model)
        mock_compile.assert_called_once()
        # ensure compile was called with our params
        called_args, called_kwargs = mock_compile.call_args
        self.assertEqual(called_args, ())
        for k, v in compile_params.items():
            self.assertIn(k, called_kwargs)

        # Result should be a dict[str, float]
        self.assertIsInstance(metrics, dict)
        self.assertIn("loss", metrics)
        for key, value in metrics.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, float)

    # ------------------------------------------------------------------ #
    # Error when uncompiled and no compile_params
    # ------------------------------------------------------------------ #
    def test_evaluate_raises_if_uncompiled_and_no_compile_params(self) -> None:
        """Uncompiled model without compile_params should raise RuntimeError."""
        logger.info(
            "Testing evaluate() with uncompiled model and no compile_params"
        )

        # Fresh uncompiled model
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.num_features,)),
                keras.layers.Dense(1),
            ]
        )

        # Force the evaluator to consider the model "uncompiled"
        with patch.object(self.evaluator, "_is_compiled", return_value=False):
            with self.assertRaises(RuntimeError):
                self.evaluator.evaluate(model=model, data=(self.x, self.y))

    # ------------------------------------------------------------------ #
    # Precompiled model: compile_params should be ignored (no recompile)
    # ------------------------------------------------------------------ #
    def test_evaluate_uses_precompiled_model_and_ignores_compile_params(
        self,
    ) -> None:
        """If model is already compiled, compile_params should not recompile."""
        logger.info("Testing evaluate() with precompiled model")

        # Compile once up-front for real
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        # Force _is_compiled to say True so _ensure_compiled won't touch compile()
        with patch.object(
            self.evaluator,
            "_is_compiled",
            return_value=True,
        ) as mock_is_compiled, patch.object(
            self.model,
            "compile",
            wraps=self.model.compile,
        ) as mock_compile:
            metrics = self.evaluator.evaluate(
                model=self.model,
                data=(self.x, self.y),
                compile_params={
                    "optimizer": "sgd",
                    "loss": "mse",
                },
                eval_params={"verbose": 0},
            )

        logger.info(f"Evaluation metrics: {metrics}")
        mock_is_compiled.assert_called_once_with(self.model)
        mock_compile.assert_not_called()

    # ------------------------------------------------------------------ #
    # _call_evaluate dispatch behavior
    # ------------------------------------------------------------------ #
    def test_call_evaluate_with_tuple_data(self) -> None:
        """_call_evaluate should unpack (x, y) tuples into positional args."""
        logger.info("Testing _call_evaluate with (x, y) tuple")

        class DummyModel(keras.Model):
            def __init__(self) -> None:
                super().__init__()
                self.compiled_loss = object()
                self.optimizer = object()
                self.evaluate = MagicMock(return_value={"loss": 1.0})

        dummy = DummyModel()
        eval_kwargs = {"batch_size": 4, "return_dict": True}

        self.evaluator._call_evaluate(
            model=dummy,
            data=(self.x, self.y),
            eval_kwargs=eval_kwargs,
        )

        dummy.evaluate.assert_called_once()
        args, kwargs = dummy.evaluate.call_args
        self.assertIs(args[0], self.x)
        self.assertIs(args[1], self.y)
        self.assertEqual(kwargs["batch_size"], 4)
        self.assertTrue(kwargs["return_dict"])

    def test_call_evaluate_with_dict_data(self) -> None:
        """_call_evaluate should treat dict data as keyword args (x=..., y=...)."""
        logger.info("Testing _call_evaluate with dict data")

        class DummyModel(keras.Model):
            def __init__(self) -> None:
                super().__init__()
                self.compiled_loss = object()
                self.optimizer = object()
                self.evaluate = MagicMock(return_value={"loss": 1.0})

        dummy = DummyModel()
        data = {"x": self.x, "y": self.y}
        eval_kwargs = {"return_dict": True, "batch_size": 4}

        self.evaluator._call_evaluate(
            model=dummy,
            data=data,
            eval_kwargs=eval_kwargs,
        )

        dummy.evaluate.assert_called_once()
        _, kwargs = dummy.evaluate.call_args

        self.assertIs(kwargs["x"], self.x)
        self.assertIs(kwargs["y"], self.y)
        self.assertEqual(kwargs["batch_size"], 4)
        self.assertTrue(kwargs["return_dict"])

    def test_call_evaluate_with_other_data(self) -> None:
        """_call_evaluate should pass non-tuple/non-dict data as a single arg."""
        logger.info("Testing _call_evaluate with 'other' data type")

        class DummyModel(keras.Model):
            def __init__(self) -> None:
                super().__init__()
                self.compiled_loss = object()
                self.optimizer = object()
                self.evaluate = MagicMock(return_value={"loss": 1.0})

        dummy = DummyModel()
        fake_dataset = object()
        eval_kwargs = {"return_dict": True}

        self.evaluator._call_evaluate(
            model=dummy,
            data=fake_dataset,
            eval_kwargs=eval_kwargs,
        )

        dummy.evaluate.assert_called_once()
        args, kwargs = dummy.evaluate.call_args

        self.assertIs(args[0], fake_dataset)
        self.assertTrue(kwargs["return_dict"])

    # ------------------------------------------------------------------ #
    # Non-dict result wrapping
    # ------------------------------------------------------------------ #
    def test_evaluate_wraps_non_dict_result(self) -> None:
        """If model.evaluate returns a scalar, it should be wrapped into a dict."""
        logger.info(
            "Testing evaluate() wrapping non-dict result into {'metric_0': ...}"
        )

        class DummyModel(keras.Model):
            def __init__(self) -> None:
                super().__init__()
                # Pretend already compiled so _ensure_compiled does nothing
                self.compiled_loss = object()
                self.optimizer = object()
                self.evaluate = MagicMock(return_value=0.5)

        dummy = DummyModel()

        # Force _is_compiled to True so we don't call compile
        with patch.object(self.evaluator, "_is_compiled", return_value=True):
            metrics = self.evaluator.evaluate(
                model=dummy,
                data=(self.x, self.y),
            )

        logger.info(f"Wrapped evaluation metrics: {metrics}")

        self.assertIsInstance(metrics, dict)
        self.assertIn("metric_0", metrics)
        self.assertIsInstance(metrics["metric_0"], float)
        self.assertAlmostEqual(metrics["metric_0"], 0.5, places=6)

    # ------------------------------------------------------------------ #
    # Validation / compiled state helpers
    # ------------------------------------------------------------------ #
    def test_validate_model_raises_for_non_keras_model(self) -> None:
        """_validate_model should reject non-keras.Model instances."""
        logger.info("Testing _validate_model with invalid model")

        with self.assertRaises(TypeError):
            self.evaluator._validate_model(model=object())  # type: ignore[arg-type]

    def test_is_compiled_detects_compiled_and_uncompiled(self) -> None:
        """_is_compiled should behave according to its own heuristic."""
        logger.info("Testing _is_compiled on dummy models")

        class DummyUncompiled:
            compiled_loss = None
            optimizer = None

        class DummyCompiledLoss:
            compiled_loss = object()
            optimizer = None

        class DummyCompiledOptimizer:
            compiled_loss = None
            optimizer = object()

        self.assertFalse(self.evaluator._is_compiled(DummyUncompiled()))  # both None
        self.assertTrue(self.evaluator._is_compiled(DummyCompiledLoss()))  # compiled_loss set
        self.assertTrue(
            self.evaluator._is_compiled(DummyCompiledOptimizer())
        )  # optimizer set


if __name__ == "__main__":
    unittest.main()
