from dataclasses import dataclass
from typing import Any, Mapping

import keras
from keras import Model
from loguru import logger

from mlpotion.core.protocols import ModelEvaluatorProtocol
from mlpotion.utils import trycatch
from mlpotion.core.exceptions import ModelEvaluatorError


@dataclass(slots=True)
class KerasModelEvaluator(ModelEvaluatorProtocol[Model]):
    """Generic evaluator for Keras 3 models (pure `keras`, no `tensorflow` import).

    This class implements `ModelEvaluatorProtocol[keras.Model]` and wraps
    `model.evaluate(...)` in a consistent way:

    - Accepts various `data` forms:
        * `(x, y)` tuple (NumPy, tensors, etc.)
        * dict like `{"x": x, "y": y}`
        * dataset-like or generator that `model.evaluate` understands
    - Ensures `return_dict=True` so the result is always `dict[str, float]`.
    - Optionally compiles the model if it's not compiled yet.

    Recognized kwargs:
        - compile_params: Optional[Mapping[str, Any]]
            * Passed to `model.compile(**compile_params)` if the model appears
              uncompiled.

        - eval_params: Optional[Mapping[str, Any]]
            * Extra params forwarded to `model.evaluate(...)` (e.g.
              `batch_size`, `verbose`, `steps`).
              `return_dict` is always forced to True.

    Example:
        ```python
        import numpy as np
        import keras

        x_test = np.random.rand(100, 32).astype("float32")
        y_test = np.random.randint(0, 2, size=(100,))

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(32,)),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        evaluator = KerasModelEvaluator()

        # Compile + evaluate in one shot
        metrics = evaluator.evaluate(
            model,
            data=(x_test, y_test),
            compile_params={
                "optimizer": "adam",
                "loss": "binary_crossentropy",
                "metrics": ["accuracy"],
            },
            eval_params={"batch_size": 32, "verbose": 0},
        )

        print(metrics)  # {'loss': ..., 'accuracy': ...}
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
            model: Compiled or uncompiled Keras model.
            data: Evaluation data; can be:
                - (x, y)
                - {"x": x, "y": y, ...}
                - dataset-like object understood by `model.evaluate`.
            **kwargs:
                compile_params: Optional[Mapping[str, Any]]
                eval_params: Optional[Mapping[str, Any]]

        Returns:
            Dictionary mapping metric names to values.
        """
        self._validate_model(model)

        compile_params: Mapping[str, Any] | None = kwargs.pop("compile_params", None)
        eval_params: Mapping[str, Any] | None = kwargs.pop("eval_params", None)

        if kwargs:
            logger.warning(
                "Unused evaluation kwargs in KerasModelEvaluator: "
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
                f"KerasModelEvaluator expects a keras.Model, got {type(model)!r}"
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
                "compile_params to KerasModelEvaluator.evaluate."
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
