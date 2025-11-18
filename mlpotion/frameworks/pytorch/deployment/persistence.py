"""PyTorch model persistence."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger

from mlpotion.core.exceptions import ModelPersistenceError
from mlpotion.core.protocols import ModelPersistence as ModelPersistenceProtocol
from mlpotion.utils import trycatch


@dataclass(slots=True)
class ModelPersistence(ModelPersistenceProtocol[nn.Module]):
    """Simple persistence helper for PyTorch models.

    This class focuses on saving and loading **PyTorch nn.Module** models
    in two main ways:

    - **state_dict** (recommended):
        * `torch.save(model.state_dict(), path)`
        * Loaded by instantiating a model class and calling `load_state_dict`.

    - **full model** (less portable):
        * `torch.save(model, path)`
        * Loaded by deserializing the whole object (pickled).

    Args:
        path: Filesystem path to save/load the model.
        model: Optional PyTorch model instance attached to this helper.

    Example:
        Save / load state_dict (recommended):

        ```python
        from mlpotion.frameworks.pytorch import ModelPersistence
        import torch.nn as nn

        class MyModel(nn.Module):
            ...

        model = MyModel()

        persistence = ModelPersistence(
            path="artifacts/my_model.pth",
            model=model,
        )

        # Save state_dict
        persistence.save(save_full_model=False)

        # Later, load into a fresh instance
        loader = ModelPersistence(path="artifacts/my_model.pth")
        loaded_model = loader.load(model_class=MyModel)
        ```

    Example:
        Save / load full model object (less portable):

        ```python
        from mlpotion.frameworks.pytorch import ModelPersistence

        persistence = ModelPersistence(
            path="artifacts/my_model_full.pt",
            model=model,
        )

        # Save full model
        persistence.save(save_full_model=True)

        # Load it back
        loader = ModelPersistence(path="artifacts/my_model_full.pt")
        loaded_model = loader.load()
        ```
    """

    path: str | Path
    model: nn.Module | None = None

    # ------------------------------------------------------------------ #
    # Public attributes as properties (for flexibility)
    # ------------------------------------------------------------------ #
    @property
    def path_obj(self) -> Path:
        """Return the model path as a `Path`."""
        return Path(self.path)

    @path_obj.setter
    def path_obj(self, value: str | Path) -> None:
        self.path = str(value)

    # ------------------------------------------------------------------ #
    # Save / Load
    # ------------------------------------------------------------------ #
    @trycatch(
        error=ModelPersistenceError,
        success_msg="✅ Successfully saved PyTorch model",
    )
    def save(
        self,
        *,
        save_full_model: bool = False,
        **torch_save_kwargs: Any,
    ) -> None:
        """Save the attached PyTorch model to disk.

        Args:
            save_full_model:
                - If False (default & recommended), save `model.state_dict()`.
                - If True, save the entire model object (pickled).
            **torch_save_kwargs:
                Extra keyword arguments forwarded to `torch.save()`.

        Raises:
            ModelPersistenceError: If no model is attached or saving fails.
        """
        model = self._ensure_model()
        path = self.path_obj

        logger.info(
            "Saving PyTorch model to {path} ({mode})",
            path=str(path),
            mode="full model" if save_full_model else "state_dict",
        )

        path.parent.mkdir(parents=True, exist_ok=True)

        if save_full_model:
            logger.warning(
                "Saving a full model object. This is less portable and may break "
                "if the code structure changes. Prefer saving a state_dict for "
                "long-term storage."
            )
            torch.save(model, path, **torch_save_kwargs)
        else:
            torch.save(model.state_dict(), path, **torch_save_kwargs)

        logger.info("PyTorch model saved successfully.")

    @trycatch(
        error=ModelPersistenceError,
        success_msg="✅ Successfully loaded PyTorch model",
    )
    def load(
        self,
        *,
        model_class: type[nn.Module] | None = None,
        map_location: str | torch.device | None = "cpu",
        strict: bool = True,
        model_kwargs: dict[str, Any] | None = None,
        **torch_load_kwargs: Any,
    ) -> nn.Module:
        """Load a PyTorch model from disk.

        Behavior depends on what was saved:

        - If the checkpoint is an `nn.Module`, it's returned directly.
        - If the checkpoint is a dict-like, it's treated as a state_dict, and
          a model instance is created (or reused) to load it.

        Args:
            model_class:
                Model class used when loading from a state_dict checkpoint.
                Required if:
                    * the checkpoint is dict-like AND
                    * no `self.model` is attached.
            map_location:
                Device mapping passed to `torch.load` (default: "cpu").
            strict:
                Whether to enforce that the keys in the state_dict exactly match
                the model's own `state_dict()` keys. Same semantics as PyTorch.
            model_kwargs:
                Keyword arguments passed to `model_class` when instantiating
                a new model for state_dict loading.
            **torch_load_kwargs:
                Extra kwargs forwarded to `torch.load()`.

        Returns:
            Loaded `nn.Module` instance (also stored in `self.model`).

        Raises:
            ModelPersistenceError: If loading fails or configuration is invalid.
        """
        path = self._ensure_path_exists()

        logger.info("Loading PyTorch model from {path}", path=str(path))

        checkpoint = torch.load(path, map_location=map_location, **torch_load_kwargs)

        # Case 1: full model was saved
        if isinstance(checkpoint, nn.Module):
            logger.info("Detected full-model checkpoint (nn.Module).")
            self.model = checkpoint
            logger.info("PyTorch model loaded successfully from full-model checkpoint.")
            return checkpoint

        # Case 2: dict-like checkpoint (state_dict or wrapped)
        if isinstance(checkpoint, dict):
            logger.info("Detected dict-like checkpoint; treating as state_dict.")
            state_dict = self._extract_state_dict(checkpoint)

            # If we already have a model attached, reuse it; otherwise, we need model_class
            if self.model is not None:
                model = self.model
                logger.info(
                    "Using attached model instance of type {cls} for state_dict loading.",
                    cls=type(model).__name__,
                )
            else:
                if model_class is None:
                    raise ModelPersistenceError(
                        "model_class is required when loading from a state_dict "
                        "checkpoint if no model is attached."
                    )
                model = self._instantiate_model(model_class, model_kwargs)
                self.model = model

            missing, unexpected = model.load_state_dict(state_dict, strict=strict)

            if strict:
                logger.debug("State_dict loaded with strict=True (no mismatch error raised).")
            else:
                if missing:
                    logger.warning(f"Missing keys in state_dict: {missing}")
                if unexpected:
                    logger.warning(f"Unexpected keys in state_dict: {unexpected}")

            logger.info("PyTorch model loaded successfully from state_dict checkpoint.")
            return model

        # Case 3: unsupported checkpoint structure
        raise ModelPersistenceError(
            f"Unsupported checkpoint type: {type(checkpoint)!r}. "
            "Expected nn.Module or dict-like object."
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _ensure_model(self) -> nn.Module:
        """Return attached model or raise a persistence error."""
        if self.model is None:
            raise ModelPersistenceError(
                "No PyTorch model attached to ModelPersistence. "
                "Pass a model to the constructor or set `.model` first."
            )
        return self.model

    def _ensure_path_exists(self) -> Path:
        """Return the path if it exists, otherwise raise a persistence error."""
        path = self.path_obj
        if not path.exists():
            raise ModelPersistenceError(f"Model file does not exist: {path!s}")
        return path

    def _extract_state_dict(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        """Extract a state_dict from a checkpoint dict.

        Supports:
            - direct state_dict (param/buffer keys)
            - wrapped state_dict under `'model_state_dict'`
        """
        if "model_state_dict" in checkpoint and isinstance(
            checkpoint["model_state_dict"], dict
        ):
            logger.debug("Using 'model_state_dict' key from checkpoint dict.")
            return checkpoint["model_state_dict"]

        logger.debug("Using checkpoint dict directly as state_dict.")
        return checkpoint

    def _instantiate_model(
        self,
        model_class: type[nn.Module],
        model_kwargs: dict[str, Any] | None,
    ) -> nn.Module:
        """Instantiate a model given its class and optional kwargs."""
        kwargs = dict(model_kwargs or {})
        logger.info(
            "Instantiating model class {cls} with kwargs={kwargs}",
            cls=model_class.__name__,
            kwargs=kwargs,
        )
        try:
            return model_class(**kwargs)
        except TypeError as exc:  # noqa: BLE001
            raise ModelPersistenceError(
                f"Failed to instantiate model_class {model_class.__name__}: {exc!s}"
            ) from exc
