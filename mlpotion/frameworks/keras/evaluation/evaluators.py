from dataclasses import dataclass
from typing import Any, Mapping

import keras
from keras import Model
from keras.utils import Sequence
from loguru import logger

from mlpotion.core.protocols import ModelEvaluator as ModelEvaluatorProtocol
from mlpotion.core.results import EvaluationResult
from mlpotion.frameworks.keras.config import ModelEvaluationConfig
from mlpotion.utils import trycatch
from mlpotion.core.exceptions import ModelEvaluatorError


@dataclass(slots=True)
class ModelEvaluator(ModelEvaluatorProtocol[Model, Sequence]):
    """Generic evaluator for Keras 3 models.

    This class implements the `ModelEvaluatorProtocol` for Keras models. It wraps the
    `model.evaluate()` method to provide a consistent evaluation interface.

    It ensures that the evaluation result is always returned as a dictionary of metric names
    to values, regardless of how the model was compiled or what arguments were passed.

    Example:
        ```python
        import keras
        import numpy as np
        from mlpotion.frameworks.keras import ModelEvaluator

        # Prepare data
        X_test = np.random.rand(20, 10)
        y_test = np.random.randint(0, 2, 20)

        # Define model
        model = keras.Sequential([
            keras.layers.Dense(1, activation='sigmoid')
        ])

        # Initialize evaluator
        evaluator = ModelEvaluator()

        # Evaluate
        metrics = evaluator.evaluate(
            model=model,
            data=(X_test, y_test),
            compile_params={
                "optimizer": "adam",
                "loss": "binary_crossentropy",
                "metrics": ["accuracy"]
            },
            eval_params={"batch_size": 32}
        )

        print(metrics)  # {'loss': 0.693..., 'accuracy': 0.5...}
        ```
    """

    # ------------------------------------------------------------------ #
    # Public API (protocol implementation)
    # ------------------------------------------------------------------ #
    @trycatch(
        error=ModelEvaluatorError,
        success_msg="✅ Successfully evaluated Keras model",
    )
    def evaluate(
        self,
        model: Model,
        dataset: Any,
        config: ModelEvaluationConfig,
    ) -> EvaluationResult:
        """Evaluate a Keras model on the given data.

        Args:
            model: The Keras model to evaluate.
            dataset: The evaluation data. Can be a tuple `(x, y)`, a dictionary, or a `Sequence`.
            config: Configuration object containing evaluation parameters.

        Returns:
            EvaluationResult: An object containing the evaluation metrics.
        """
        self._validate_model(model)

        # Prepare eval parameters
        eval_kwargs = {
            "batch_size": config.batch_size,
            "verbose": config.verbose,
            "return_dict": True,
        }

        # Add any framework-specific options
        eval_kwargs.update(config.framework_options)

        # We assume the model is already compiled. If not, Keras will raise an error
        # unless we provide compile params, but EvaluationConfig doesn't typically carry them.
        # The user should ensure the model is compiled (e.g. after loading or training).
        if not self._is_compiled(model):
            logger.warning(
                "Model is not compiled. Evaluation might fail if loss/metrics are not defined."
            )

        logger.info("Evaluating Keras model...")
        logger.debug(f"Evaluation data type: {type(dataset)!r}")
        logger.debug(f"Evaluation parameters: {eval_kwargs}")

        import time

        start_time = time.time()

        result = self._call_evaluate(model=model, data=dataset, eval_kwargs=eval_kwargs)

        evaluation_time = time.time() - start_time

        # At this point, result should be a dict[str, float]
        if not isinstance(result, dict):
            # Defensive fallback if user or Keras changed behavior
            logger.warning(
                f"`model.evaluate` did not return a dict (got {type(result)!r}). "
                "Wrapping into a dict under key 'metric_0'."
            )
            result = {"metric_0": float(result)}

        metrics = {str(k): float(v) for k, v in result.items()}
        logger.info(f"Evaluation result: {metrics}")

        return EvaluationResult(
            metrics=metrics,
            config=config,
            evaluation_time=evaluation_time,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _validate_model(self, model: Model) -> None:
        if not isinstance(model, keras.Model):
            raise TypeError(
                f"ModelEvaluator expects a keras.Model, got {type(model)!r}"
            )

    def _is_compiled(self, model: Model) -> bool:
        """Best-effort check whether the model appears compiled."""
        # Keras 3 models have `compiled_loss` and `optimizer` after compile
        has_compiled_loss = getattr(model, "compiled_loss", None) is not None
        has_optimizer = getattr(model, "optimizer", None) is not None
        return bool(has_compiled_loss or has_optimizer)

    def _ensure_compiled(
        self,
        model: Model,
        compile_params: Mapping[str, Any] | None,
    ) -> None:
        """Compile the model if needed and possible."""
        if self._is_compiled(model):
            if compile_params:
                logger.warning(
                    "Model already appears compiled. `compile_params` will be ignored."
                )
            return

        if not compile_params:
            raise RuntimeError(
                "Model does not appear to be compiled and no `compile_params` "
                "were provided. Please compile the model manually or pass "
                "compile_params to ModelEvaluator.evaluate."
            )

        logger.info("Compiling model with provided compile_params.")
        logger.debug(f"compile_params: {compile_params}")
        model.compile(**compile_params)

    def _call_evaluate(
        self,
        model: Model,
        data: Any,
        eval_kwargs: dict[str, Any],
    ) -> Any:
        """Call `model.evaluate` with flexible handling of data formats."""
        # (x, y) tuple
        if isinstance(data, tuple) and len(data) in {2, 3}:
            logger.debug("Data detected as (x, y) or (x, y, sample_weight) tuple.")
            return model.evaluate(*data, **eval_kwargs)

        # dict-like: {"x": x, "y": y, ...}
        if isinstance(data, dict) and "x" in data:
            logger.debug("Data detected as dict with 'x' key.")
            return model.evaluate(**data, **eval_kwargs)

        # Anything else (dataset, generator, x-only)
        logger.debug("Data passed directly to model.evaluate.")
        return model.evaluate(data, **eval_kwargs)
