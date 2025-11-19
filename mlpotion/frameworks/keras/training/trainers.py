from dataclasses import dataclass
from typing import Any, Mapping

import keras
from keras import Model
from keras.utils import Sequence
from loguru import logger

from mlpotion.core.protocols import ModelTrainer as ModelTrainerProtocol
from mlpotion.core.results import TrainingResult
from mlpotion.frameworks.keras.config import ModelTrainingConfig
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
    def train(
        self,
        model: Model,
        dataset: Any,
        config: ModelTrainingConfig,
        validation_dataset: Any | None = None,
    ) -> TrainingResult[Model]:
        """Train a Keras model using the provided dataset and configuration.

        Args:
            model: The Keras model to train.
            dataset: The training data. Can be a tuple `(x, y)`, a dictionary, a `Sequence`, or a generator.
            config: Configuration object containing training parameters.
            validation_dataset: Optional validation data.

        Returns:
            TrainingResult[Model]: An object containing the trained model, training history, and metrics.
        """
        self._validate_model(model)

        # Prepare compile parameters from config
        compile_params = {
            "optimizer": self._get_optimizer(config),
            "loss": config.loss,
            "metrics": config.metrics,
        }

        # Compile if needed or if forced by config (though we usually respect existing compilation)
        # Here we'll ensure it's compiled. If the user wants to use their own compilation,
        # they should probably compile it before passing it, but our config implies we control it.
        # However, to be safe and flexible:
        if not self._is_compiled(model):
            if not config.optimizer_type or not config.loss:
                raise RuntimeError(
                    "Model is not compiled and config does not provide optimizer_type and loss. "
                    "Either compile the model beforehand or provide optimizer_type and loss in config."
                )
            logger.info("Compiling model with config parameters.")
            model.compile(**compile_params)
        else:
            logger.info("Model already compiled. Using existing compilation settings.")

        # Prepare fit parameters
        fit_kwargs = {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "verbose": config.verbose,
            "shuffle": config.shuffle,
            "validation_split": config.validation_split,
        }
        
        if validation_dataset is not None:
            fit_kwargs["validation_data"] = validation_dataset

        # Add any framework-specific options
        fit_kwargs.update(config.framework_options)

        logger.info("Starting Keras model training...")
        logger.debug(f"Training data type: {type(dataset)!r}")
        logger.debug(f"Fit parameters: {fit_kwargs}")

        import time
        start_time = time.time()

        history_obj = self._call_fit(model=model, data=dataset, fit_kwargs=fit_kwargs)
        
        training_time = time.time() - start_time

        # Convert History object to dict[str, list[float]]
        history_dict = self._history_to_dict(history_obj)
        
        # Extract final metrics
        final_metrics = {}
        for k, v in history_dict.items():
            if v:
                final_metrics[k] = v[-1]

        logger.info("Training completed.")
        logger.debug(f"Training history: {history_dict}")
        
        return TrainingResult(
            model=model,
            history=history_dict,
            metrics=final_metrics,
            config=config,
            training_time=training_time,
            best_epoch=None, # Keras history doesn't explicitly track "best" unless using callbacks
        )

    def _get_optimizer(self, config: ModelTrainingConfig) -> Any:
        """Get optimizer from config."""
        # If learning_rate is specified, we might need to instantiate the optimizer
        # instead of just passing the string name, to apply the LR.
        if config.learning_rate:
            try:
                # Try to get the optimizer class from the string name
                opt_cls = getattr(keras.optimizers, config.optimizer_type, None)
                if opt_cls is None:
                    # Fallback for common names if case differs or alias
                    if config.optimizer_type.lower() == "adam":
                        opt_cls = keras.optimizers.Adam
                    elif config.optimizer_type.lower() == "sgd":
                        opt_cls = keras.optimizers.SGD
                    elif config.optimizer_type.lower() == "rmsprop":
                        opt_cls = keras.optimizers.RMSprop
                
                if opt_cls:
                    return opt_cls(learning_rate=config.learning_rate)
            except Exception as e:
                logger.warning(f"Failed to instantiate optimizer {config.optimizer_type} with LR {config.learning_rate}: {e}")
        
        return config.optimizer_type

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
