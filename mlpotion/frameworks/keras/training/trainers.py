from dataclasses import dataclass
from typing import Any, Mapping

import keras
from keras import Model
from keras.utils import Sequence
from loguru import logger

from mlpotion.core.protocols import ModelTrainer as ModelTrainerProtocol
from mlpotion.utils import trycatch
from mlpotion.core.exceptions import ModelTrainerError


@dataclass(slots=True)
class ModelTrainer(ModelTrainerProtocol[Model, Sequence]):
    """Generic trainer for Keras 3 models.

    This class implements the `ModelTrainerProtocol` for Keras models, providing a standardized
    interface for training. It wraps the standard `model.fit()` method but adds flexibility
    and consistency checks.

    It supports:
    - Automatic model compilation if `compile_params` are provided.
    - Handling of various data formats (tuples, dicts, generators).
    - Standardized return format (dictionary of history metrics).

    Example:
        ```python
        import keras
        import numpy as np
        from mlpotion.frameworks.keras import ModelTrainer

        # Prepare data
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        
        # Define model
        model = keras.Sequential([
            keras.layers.Dense(1, activation='sigmoid')
        ])

        # Initialize trainer
        trainer = ModelTrainer()

        # Train
        history = trainer.train(
            model=model,
            data=(X_train, y_train),
            compile_params={
                "optimizer": "adam",
                "loss": "binary_crossentropy",
                "metrics": ["accuracy"]
            },
            fit_params={
                "epochs": 5,
                "batch_size": 32,
                "verbose": 1
            }
        )
        
        print(history['loss'])
        ```
    """

    # ------------------------------------------------------------------ #
    # Public API (protocol implementation)
    # ------------------------------------------------------------------ #
    @trycatch(
        error=ModelTrainerError,
        success_msg="✅ Successfully trained Keras model",
    )
    def train(self, model: Model, data: Any, **kwargs: Any) -> dict[str, list[float]]:
        """Train a Keras model on the provided data.

        Args:
            model: The Keras model to train.
            data: The training data. Can be a tuple `(x, y)`, a dictionary, a `Sequence`, or a generator.
            **kwargs: Additional arguments:
                - `compile_params` (dict): Arguments for `model.compile()` (optimizer, loss, metrics).
                - `fit_params` (dict): Arguments for `model.fit()` (epochs, batch_size, callbacks, etc.).

        Returns:
            dict[str, list[float]]: A dictionary containing the training history (metrics per epoch).
        """
        self._validate_model(model)

        compile_params: Mapping[str, Any] | None = kwargs.pop("compile_params", None)
        fit_params: Mapping[str, Any] | None = kwargs.pop("fit_params", None)

        if kwargs:
            logger.warning(
                "Unused training kwargs in ModelTrainer: "
                f"{list(kwargs.keys())}"
            )

        # Compile if needed
        self._ensure_compiled(model=model, compile_params=compile_params)

        fit_kwargs = dict(fit_params or {})

        logger.info("Starting Keras model training...")
        logger.debug(f"Training data type: {type(data)!r}")
        logger.debug(f"Fit parameters: {fit_kwargs}")

        history_obj = self._call_fit(model=model, data=data, fit_kwargs=fit_kwargs)

        # Convert History object to dict[str, list[float]]
        history_dict = self._history_to_dict(history_obj)

        logger.info("Training completed.")
        logger.debug(f"Training history: {history_dict}")
        return history_dict

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _validate_model(self, model: Model) -> None:
        if not isinstance(model, keras.Model):
            raise TypeError(
                f"ModelTrainer expects a keras.Model, got {type(model)!r}"
            )

    def _is_compiled(self, model: Model) -> bool:
        """Best-effort check whether the model appears compiled."""
        has_compiled_loss = getattr(model, "compiled_loss", None) is not None
        has_optimizer = getattr(model, "optimizer", None) is not None
        return bool(has_compiled_loss or has_optimizer)

    def _ensure_compiled(
        self,
        model: Model,
        compile_params: Mapping[str, Any] | None,
    ) -> None:
        """Compile the model if it is not already compiled."""
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
                "`compile_params` to ModelTrainer.train."
            )

        logger.info("Compiling model with provided compile_params.")
        logger.debug(f"compile_params: {compile_params}")
        model.compile(**compile_params)

    def _call_fit(
        self,
        model: Model,
        data: Any,
        fit_kwargs: dict[str, Any],
    ) -> keras.callbacks.History:
        """Call `model.fit` with flexible handling of data formats.

        Returns:
            Keras History object.
        """
        # (x, y) or (x, y, sample_weight)
        if isinstance(data, tuple) and len(data) in {2, 3}:
            logger.debug("Training data detected as (x, y) or (x, y, sample_weight).")
            history = model.fit(*data, **fit_kwargs)
        # dict-like with 'x' (and maybe 'y')
        elif isinstance(data, dict) and "x" in data:
            logger.debug("Training data detected as dict with 'x' key.")
            history = model.fit(**data, **fit_kwargs)
        else:
            # dataset-like, generator, or x-only
            logger.debug("Training data passed directly to model.fit.")
            history = model.fit(data, **fit_kwargs)

        if not isinstance(history, keras.callbacks.History):
            # This would be very unusual, but just in case...
            logger.warning(
                "model.fit did not return a keras.callbacks.History; "
                "training history may be unavailable."
            )
        return history

    def _history_to_dict(
        self,
        history: keras.callbacks.History | Any,
    ) -> dict[str, list[float]]:
        """Convert a History object (or fallback) to a dict of metric curves."""
        if isinstance(history, keras.callbacks.History):
            hist = history.history or {}
            return {
                str(name): [float(v) for v in values]
                for name, values in hist.items()
            }

        # Fallback: no history available
        logger.warning(
            "No valid History object; returning empty history dict."
        )
        return {}
