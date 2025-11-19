from dataclasses import dataclass
from typing import Any, Mapping

import keras
from keras import Model
from keras.utils import Sequence
from loguru import logger

from mlpotion.core.protocols import ModelEvaluator as ModelEvaluatorProtocol
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
    def evaluate(self, model: Model, data: Any, **kwargs: Any) -> dict[str, float]:
        """Evaluate a Keras model on the given data.

        Args:
            model: The Keras model to evaluate.
            data: The evaluation data. Can be a tuple `(x, y)`, a dictionary, or a `Sequence`.
            **kwargs: Additional arguments:
                - `compile_params` (dict): Arguments for `model.compile()` if not already compiled.
                - `eval_params` (dict): Arguments for `model.evaluate()` (batch_size, verbose, etc.).

        Returns:
            dict[str, float]: A dictionary mapping metric names to their values.
        """
        self._validate_model(model)

        compile_params: Mapping[str, Any] | None = kwargs.pop("compile_params", None)
        eval_params: Mapping[str, Any] | None = kwargs.pop("eval_params", None)

        if kwargs:
            logger.warning(
                "Unused evaluation kwargs in ModelEvaluator: "
                f"{list(kwargs.keys())}"
            )

        self._ensure_compiled(model=model, compile_params=compile_params)

        eval_kwargs = dict(eval_params or {})
        # We always want a dict back to satisfy the protocol contract
        eval_kwargs["return_dict"] = True

        logger.info("Evaluating Keras model...")
        logger.debug(f"Evaluation data type: {type(data)!r}")
        logger.debug(f"Evaluation parameters: {eval_kwargs}")

        result = self._call_evaluate(model=model, data=data, eval_kwargs=eval_kwargs)

        # At this point, result should be a dict[str, float]
        if not isinstance(result, dict):
            # Defensive fallback if user or Keras changed behavior
            logger.warning(
                f"`model.evaluate` did not return a dict (got {type(result)!r}). "
                "Wrapping into a dict under key 'metric_0'."
            )
            result = {"metric_0": float(result)}

        logger.info(f"Evaluation result: {result}")
        return {str(k): float(v) for k, v in result.items()}

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
