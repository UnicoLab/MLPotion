from pathlib import Path
from typing import Any

import keras
from keras import Model
from loguru import logger

from mlpotion.core.exceptions import ModelPersistenceError
from mlpotion.core.protocols import ModelPersistence as ModelPersistenceProtocol
from mlpotion.frameworks.keras.models.inspection import ModelInspector
from mlpotion.utils import trycatch


class ModelPersistence(ModelPersistenceProtocol[Model]):
    """Persistence helper for Keras models.

    This class manages saving and loading of Keras models. It supports standard Keras
    formats (`.keras`, `.h5`) and SavedModel directories. It also integrates with
    `ModelInspector` to provide model metadata upon loading.

    Attributes:
        path (Path): The file path for the model artifact.
        model (Model | None): The Keras model instance (optional).

    Example:
        ```python
        import keras
        from mlpotion.frameworks.keras import ModelPersistence

        # Define model
        model = keras.Sequential([keras.layers.Dense(1)])

        # Save
        saver = ModelPersistence(path="models/my_model.keras", model=model)
        saver.save()

        # Load
        loader = ModelPersistence(path="models/my_model.keras")
        loaded_model, metadata = loader.load(inspect=True)
        print(metadata['parameters'])
        ```
    """

    def __init__(self, path: str | Path, model: Model | None = None) -> None:
        self._path = Path(path)
        self._model = model

    # ------------------------------------------------------------------ #
    # Public attributes (properties for flexibility)
    # ------------------------------------------------------------------ #
    @property
    def path(self) -> Path:
        """Filesystem path where the model is saved/loaded."""
        return self._path

    @path.setter
    def path(self, value: str | Path) -> None:
        self._path = Path(value)

    @property
    def model(self) -> Model | None:
        """Currently attached Keras model (may be None before loading)."""
        return self._model

    @model.setter
    def model(self, value: Model | None) -> None:
        self._model = value

    # ------------------------------------------------------------------ #
    # Save / Load
    # ------------------------------------------------------------------ #
    @trycatch(
        error=ModelPersistenceError,
        success_msg="✅ Successfully saved Keras model",
    )
    def save(
        self,
        overwrite: bool = True,
        **kwargs: Any,
    ) -> None:
        """Save the attached model to disk.

        Args:
            overwrite: Whether to overwrite the file if it already exists.
            **kwargs: Additional arguments passed to `model.save()`.

        Raises:
            ModelPersistenceError: If no model is attached or if the file exists and `overwrite` is False.
        """
        model = self._ensure_model()
        target = self._path

        if target.exists() and not overwrite:
            raise ModelPersistenceError(
                f"Target path already exists and overwrite=False: {target!s}"
            )

        logger.info(f"Saving Keras model to: {target!s}")
        target.parent.mkdir(parents=True, exist_ok=True)

        # Keras 3 generally infers format from the path; `save_format` is
        # deprecated / discouraged in newer APIs, so we do NOT pass it.
        model.save(target.as_posix(), **kwargs)
        logger.info("Keras model saved successfully.")

    @trycatch(
        error=ModelPersistenceError,
        success_msg="✅ Successfully loaded Keras model",
    )
    def load(
        self,
        *,
        inspect: bool = True,
        **kwargs: Any,
    ) -> tuple[Model, dict[str, Any] | None]:
        """Load a Keras model from disk.

        Args:
            inspect: Whether to inspect the loaded model and return metadata.
            **kwargs: Additional arguments passed to `keras.models.load_model()`.

        Returns:
            tuple[Model, dict[str, Any] | None]: A tuple containing the loaded model and
            optional inspection metadata.

        Raises:
            ModelPersistenceError: If the model file cannot be found or loaded.
        """
        path = self._ensure_path_exists()

        logger.info(f"Loading Keras model from: {path!s}")
        model = keras.models.load_model(path.as_posix(), **kwargs)

        self._model = model  # keep instance in sync

        inspection_result: dict[str, Any] | None = None
        if inspect:
            logger.info("Inspecting loaded Keras model with ModelInspector.")
            inspector = ModelInspector()
            inspection_result = inspector.inspect(model)

        return model, inspection_result

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _ensure_model(self) -> Model:
        """Return the attached model or raise a persistence error."""
        if self._model is None:
            raise ModelPersistenceError(
                "No Keras model attached to ModelPersistence. "
                "Pass a model to the constructor or set `.model` first."
            )
        return self._model

    def _ensure_path_exists(self) -> Path:
        """Return the path if it exists, otherwise raise a persistence error."""
        if self._path is None:
            raise ModelPersistenceError("No path configured for persistence.")
        if not self._path.exists():
            raise ModelPersistenceError(
                f"Model file or directory does not exist: {self._path!s}"
            )
        return self._path
