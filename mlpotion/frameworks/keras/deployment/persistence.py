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
    """Simple persistence helper for Keras models.

    This class focuses on **whole-model** save/load using the native
    Keras APIs:

    - Saving:
        - `model.save("model.keras")`
        - `model.save("model.h5")`
        - `model.save("some/dir")` (SavedModel, depending on backend/version)

    - Loading:
        - `keras.models.load_model("model.keras")`
        - `keras.models.load_model("model.h5")`
        - `keras.models.load_model("some/dir")`

    Additionally, after loading, it can run a `ModelInspector` to
    collect useful introspection metadata (input/output signatures, etc).

    Args:
        path: Path to save/load the model artifact.
        model: Optional Keras model instance to be saved.

    Example:
        Basic save/load:

        ```python
        import keras
        from mlpotion.frameworks.keras import ModelPersistence

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(32,)),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(1),
            ]
        )

        # Create persistence helper bound to a path and model
        persistence = ModelPersistence(
            path="artifacts/my_model.keras",
            model=model,
        )

        # Save the model
        persistence.save()

        # Later: load the model from disk
        loader = ModelPersistence(path="artifacts/my_model.keras")
        loaded_model, inspection = loader.load()
        ```

    Example:
        Overwriting and controlling inspection:

        ```python
        persistence = ModelPersistence(
            path="artifacts/my_model.keras",
            model=model,
        )

        # Overwrite existing file and pass extra kwargs to model.save()
        persistence.save(
            overwrite=True,
            include_optimizer=False,
        )

        # Load without inspection metadata
        loaded_model, inspection = persistence.load(inspect=False)
        assert inspection is None
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

        This uses `model.save(path, **kwargs)`. The format is inferred
        from the path:

        - `*.keras` → Keras v3 native format
        - `*.h5` → HDF5
        - directory or other → backend-dependent SavedModel-like formats

        Args:
            overwrite: If False and the file already exists, raise
                `ModelPersistenceError` instead of overwriting.
            **kwargs: Extra keyword arguments forwarded to `model.save()`.

        Raises:
            ModelPersistenceError: If `model` is not set or if overwrite is
                disabled and the target already exists.

        Example:
            ```python
            persistence = ModelPersistence(
                path="models/my_model.keras",
                model=my_model,
            )
            persistence.save(include_optimizer=False)
            ```
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
        """Load a Keras model from disk and optionally inspect it.

        Args:
            inspect: If True, run `ModelInspector` on the loaded
                model and return its metadata as a dict. If False, skip
                inspection and return `None` for the metadata.
            **kwargs: Extra keyword arguments forwarded to
                `keras.models.load_model()`.

        Returns:
            A tuple of:
                - Loaded Keras model.
                - Inspection metadata dict, or None if `inspect=False`.

        Raises:
            ModelPersistenceError: If `path` does not exist.

        Example:
            ```python
            loader = ModelPersistence(path="models/my_model.keras")
            model, inspection = loader.load()

            # Optionally update the instance's model reference:
            loader.model = model
            ```
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
