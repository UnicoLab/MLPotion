from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import keras
from keras import Model, InputSpec
from loguru import logger

from mlpotion.core.protocols import ModelExporter as ModelExporterProtocol
from mlpotion.utils import trycatch
from mlpotion.core.exceptions import ModelExporterError


class ModelExporter(ModelExporterProtocol[Model]):
    """Generic exporter for Keras 3 models (no tensorflow imports).

    This class implements `ModelExporterProtocol[keras.Model]` and supports
    two main families of exports:

    1. **Whole-model serialization** (Keras-native):
        - `.keras` or `.h5` via `model.save(...)`

    2. **Inference exports** (for serving, other runtimes):
        - `model.export(path, format=...)` (e.g. `"tf_saved_model"`, `"onnx"`)
        - optional custom serving endpoint + input specs via
          `keras.export.ExportArchive`

    All configuration is passed via `**kwargs`, so the API stays generic.

    Recognized kwargs:
        - export_format: Optional[str]
            * `"keras"` or `"h5"` → use `model.save(...)`
            * anything else (e.g. `"tf_saved_model"`, `"onnx"`) → use `model.export(...)`
              unless overridden by a custom endpoint.

        - dataset: Optional[Iterable[Any]]
            * Optional iterable of batched inputs to "warm up" the model
              (run one mini-batch forward pass before export).

        - endpoint_name: Optional[str]
            * If provided, triggers ExportArchive-based export with a custom
              endpoint name.

        - input_specs: Optional[Sequence[keras.InputSpec]]
            * Input specs for the endpoint when using ExportArchive.

        - config: Optional[Mapping[str, Any]]
            * Extra keyword arguments forwarded to `model.export(...)` or
              `model.save(...)`.

    Example:
        ```python
        import keras
        from mlpotion.frameworks.keras import ModelExporter

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(32,)),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(1),
            ]
        )

        exporter = ModelExporter()

        # 1) Native Keras v3 format
        exporter.export(model, "my_model.keras")

        # 2) SavedModel for serving (TF backend) via model.export
        exporter.export(
            model,
            "saved_model_dir",
            export_format="tf_saved_model",
        )

        # 3) Custom endpoint + InputSpec via ExportArchive
        input_spec = keras.InputSpec(shape=(None, 32), dtype="float32")
        exporter.export(
            model,
            "custom_endpoint_model",
            endpoint_name="serve",
            input_specs=[input_spec],
            export_format="tf_saved_model",
        )
        ```
    """

    default_endpoint_name: str = "serve"

    # ------------------------------------------------------------------ #
    # Public API (protocol implementation)
    # ------------------------------------------------------------------ #
    @trycatch(
        error=ModelExporterError,
        success_msg="✅ Successfully Exported model",
    )
    def export(self, model: Model, path: str, **kwargs: Any) -> None:
        """Export a Keras model to disk.

        Args:
            model: Keras model to export.
            path: Path or directory where to export.
            **kwargs: See class docstring for supported keys.

        Example:
            ```python
            from mlpotion.frameworks.keras import ModelExporter

            exporter = ModelExporter()
            exporter.export(
                model=model,
                path="my_model.keras",
            )
            ```
        """
        export_path = Path(path)

        export_format: str | None = kwargs.pop("export_format", None)
        dataset: Iterable[Any] | None = kwargs.pop("dataset", None)
        endpoint_name: str | None = kwargs.pop("endpoint_name", None)
        input_specs: Sequence[InputSpec] | None = kwargs.pop("input_specs", None)
        config: Mapping[str, Any] | None = kwargs.pop("config", None)

        if kwargs:
            logger.warning(
                "Unused export kwargs passed to ModelExporter: "
                f"{list(kwargs.keys())}"
            )

        self._validate_model(model)
        self._validate_config(config)

        # Determine mode if export_format isn't explicitly set
        if export_format is None:
            export_format = self._infer_export_format_from_path(export_path)

        logger.info(
            f"Exporting Keras model '{model.name}' to {export_path!s} "
            f"with format '{export_format}'"
        )

        # Optional warm-up pass
        self._warmup_if_needed(model=model, dataset=dataset)

        # Choose strategy
        try:
            if self._is_native_keras_format(export_format):
                self._save_native_keras(model=model, path=export_path, config=config)
            elif endpoint_name is not None or input_specs is not None:
                self._export_with_export_archive(
                    model=model,
                    path=export_path,
                    endpoint_name=endpoint_name or self.default_endpoint_name,
                    input_specs=input_specs,
                    export_format=export_format,
                )
            else:
                self._export_with_model_export(
                    model=model,
                    path=export_path,
                    export_format=export_format,
                    config=config,
                )
        except ValueError as err:
            logger.warning(
                f"Export error: {err} "
                "(you may need to build the model by calling it on example data "
                "before exporting)"
            )

        logger.info(f"Model export completed: {export_path!s}")

    # ------------------------------------------------------------------ #
    # Validation helpers
    # ------------------------------------------------------------------ #
    def _validate_model(self, model: Model) -> None:
        if not isinstance(model, keras.Model):
            raise TypeError(
                f"ModelExporter expects a keras.Model, got {type(model)!r}"
            )

    def _validate_config(self, config: Mapping[str, Any] | None) -> None:
        if config is not None and not isinstance(config, Mapping):
            raise TypeError("`config` must be a mapping (dict-like) or None.")

    # ------------------------------------------------------------------ #
    # Format helpers
    # ------------------------------------------------------------------ #
    def _infer_export_format_from_path(self, path: Path) -> str:
        """Infer export format from the path extension."""
        suffix = path.suffix.lower()
        if suffix == ".keras":
            return "keras"
        if suffix == ".h5":
            return "h5"
        # Default to SavedModel-style export if no clear file extension
        return "tf_saved_model"

    @staticmethod
    def _is_native_keras_format(export_format: str) -> bool:
        return export_format in {"keras", "h5"}

    # ------------------------------------------------------------------ #
    # Warm-up
    # ------------------------------------------------------------------ #
    def _warmup_if_needed(
        self,
        model: Model,
        dataset: Iterable[Any] | None,
    ) -> None:
        """Run a small batch through the model if a dataset is provided."""
        if dataset is None:
            logger.debug("No dataset provided; skipping warm-up forward pass.")
            return

        logger.info("Running warm-up batch through the model before export.")
        try:
            for batch in dataset:
                model(batch)
                break  # one batch is enough
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Warm-up failed (ignored): {exc}")

    # ------------------------------------------------------------------ #
    # Export strategies
    # ------------------------------------------------------------------ #
    def _save_native_keras(
        self,
        model: Model,
        path: Path,
        config: Mapping[str, Any] | None,
    ) -> None:
        """Save to `.keras` or `.h5` using `model.save`."""
        logger.info(f"Saving model as native Keras file at {path!s}")
        path.parent.mkdir(parents=True, exist_ok=True)

        save_kwargs: dict[str, Any] = {}
        if config:
            save_kwargs.update(config)

        model.save(path.as_posix(), **save_kwargs)
        logger.info("Model saved in native Keras format.")

    def _export_with_model_export(
        self,
        model: Model,
        path: Path,
        export_format: str,
        config: Mapping[str, Any] | None,
    ) -> None:
        """Export using `model.export(...)` (Keras 3 inference export API)."""
        if not hasattr(model, "export"):
            raise RuntimeError(
                "This Keras model has no `.export` method. "
                "Make sure you're using Keras 3 with inference export support."
            )

        logger.info(
            f"Exporting model via `model.export(format='{export_format}')` "
            f"to {path!s}"
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        export_kwargs: dict[str, Any] = {"format": export_format}
        if config:
            export_kwargs.update(config)

        # Keras 3: model.export(path, format="tf_saved_model" | "onnx" | ...)
        model.export(path.as_posix(), **export_kwargs)

        logger.info("Model exported successfully with `model.export`.")
        logger.info(
            "Note: For `tf_saved_model` format you can later reload the artifact "
            "with `tf.saved_model.load(path)` in a TensorFlow environment."
        )

    def _export_with_export_archive(
        self,
        model: Model,
        path: Path,
        endpoint_name: str,
        input_specs: Sequence[InputSpec] | None,
        export_format: str,
    ) -> None:
        """Export using keras.export.ExportArchive with a custom endpoint."""
        try:
            from keras.export import ExportArchive
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "keras.export.ExportArchive is not available. "
                "Ensure you're using a Keras version with export support."
            ) from exc

        logger.info(
            f"Exporting model via ExportArchive with endpoint '{endpoint_name}' "
            f"and format '{export_format}' to {path!s}"
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        archive = ExportArchive()
        archive.track(model)

        if input_specs is None:
            logger.warning(
                "No `input_specs` provided for custom endpoint. "
                "You may want to pass a list[keras.InputSpec] to define "
                "a stable input signature."
            )

        archive.add_endpoint(
            name=endpoint_name,
            fn=model.call,
            input_signature=list(input_specs) if input_specs is not None else None,
        )

        # In Keras 3, ExportArchive writes a SavedModel-like artifact.
        archive.write_out(path.as_posix())

        logger.info("Model exported successfully with ExportArchive.")
        logger.info(
            "Note: For TF backends, the resulting artifact is a SavedModel "
            "that can be reloaded with `tf.saved_model.load(path)`."
        )
