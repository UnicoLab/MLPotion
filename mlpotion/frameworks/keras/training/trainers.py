from dataclasses import dataclass
from typing import Any, Mapping

import keras
from keras import Model
from loguru import logger

from mlpotion.core.protocols import ModelTrainerProtocol  # adjust import


@dataclass(slots=True)
class KerasModelTrainer(ModelTrainerProtocol[Model]):
    """Generic trainer for Keras 3 models (pure `keras`, no `tensorflow` import).

    This class implements `ModelTrainerProtocol[keras.Model]` and wraps
    `model.fit(...)` in a consistent, protocol-friendly way.

    It supports:

    - Compiling the model if it is not compiled yet (via `compile_params`).
    - Training on various data formats:
        * (x, y) or (x, y, sample_weight)
        * dict-like `{"x": x, "y": y, ...}`
        * dataset-like or generator understood by `model.fit`.
    - Passing arbitrary `fit` options via `fit_params`.
    - Returning a **history dict**: `metric_name -> list[float]`.

    Recognized kwargs:
        - compile_params: Optional[Mapping[str, Any]]
            Parameters passed to `model.compile(**compile_params)` if the model
            does not appear to be compiled.

        - fit_params: Optional[Mapping[str, Any]]
            Parameters forwarded to `model.fit(...)`, e.g.:
                * epochs
                * batch_size
                * validation_data
                * callbacks
                * verbose
                * steps_per_epoch
                * validation_steps

    Example:
        ```python
        import numpy as np
        import keras

        x_train = np.random.rand(1000, 32).astype("float32")
        y_train = np.random.randint(0, 2, size=(1000,))

        x_val = np.random.rand(200, 32).astype("float32")
        y_val = np.random.randint(0, 2, size=(200,))

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(32,)),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        trainer = KerasModelTrainer()

        history = trainer.train(
            model=model,
            data=(x_train, y_train),
            compile_params={
                "optimizer": "adam",
                "loss": "binary_crossentropy",
                "metrics": ["accuracy"],
            },
            fit_params={
                "epochs": 10,
                "batch_size": 32,
                "validation_data": (x_val, y_val),
                "verbose": 1,
            },
        )

        print(history.keys())      # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
        print(history["loss"])     # list of loss per epoch
        ```
    """

    # ------------------------------------------------------------------ #
    # Public API (protocol implementation)
    # ------------------------------------------------------------------ #
    def train(self, model: Model, data: Any, **kwargs: Any) -> dict[str, list[float]]:
        """Train a Keras model on the provided data.

        Args:
            model: Keras model to train.
            data: Training data; can be:
                - (x, y)
                - (x, y, sample_weight)
                - dict-like with 'x' and 'y' keys
                - dataset-like object
            **kwargs:
                compile_params: Optional[Mapping[str, Any]]
                fit_params: Optional[Mapping[str, Any]]

        Returns:
            History dictionary: metric name -> list of values per epoch.
        """
        self._validate_model(model)

        compile_params: Mapping[str, Any] | None = kwargs.pop("compile_params", None)
        fit_params: Mapping[str, Any] | None = kwargs.pop("fit_params", None)

        if kwargs:
            logger.warning(
                "Unused training kwargs in KerasModelTrainer: "
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
                f"KerasModelTrainer expects a keras.Model, got {type(model)!r}"
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
                "`compile_params` to KerasModelTrainer.train."
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
