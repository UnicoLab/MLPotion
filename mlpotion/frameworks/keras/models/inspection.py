from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from loguru import logger
import keras  # pure Keras 3 import

from mlpotion.core.protocols import ModelInspector as ModelInspectorProtocol
from mlpotion.utils import trycatch
from mlpotion.core.exceptions import ModelInspectorError

ModelLike = Any  # usually keras.Model, but duck-typed on attributes


@dataclass(slots=True)
class ModelInspector(ModelInspectorProtocol[ModelLike]):
    """Inspector for Keras models.

    This class analyzes Keras models to extract metadata such as input/output shapes,
    parameter counts, layer details, and signatures. It is useful for validating models
    before training or deployment, and for generating model reports.

    Attributes:
        include_layers (bool): Whether to include detailed information about each layer.
        include_signatures (bool): Whether to include model signatures (if available).

    Example:
        ```python
        import keras
        from mlpotion.frameworks.keras import ModelInspector

        model = keras.Sequential([keras.layers.Dense(1, input_shape=(10,))])
        inspector = ModelInspector()
        
        info = inspector.inspect(model)
        print(f"Total params: {info['parameters']['total']}")
        print(f"Inputs: {info['inputs']}")
        ```
    """

    include_layers: bool = True
    include_signatures: bool = True


    @trycatch(
        error=ModelInspectorError,
        success_msg="✅ Successfully inspected Keras model",
    )
    def inspect(self, model: ModelLike) -> dict[str, Any]:
        """Inspect a Keras model and return structured metadata.

        Args:
            model: The Keras model to inspect.

        Returns:
            dict[str, Any]: A dictionary containing model metadata:
                - `name`: Model name.
                - `backend`: Keras backend used.
                - `trainable`: Whether the model is trainable.
                - `inputs`: List of input specifications.
                - `outputs`: List of output specifications.
                - `parameters`: Dictionary of parameter counts.
                - `layers`: List of layer details (if `include_layers=True`).
                - `signatures`: Model signatures (if `include_signatures=True`).
        """
        if not isinstance(model, keras.Model):
            raise TypeError(
                f"ModelInspector expects a keras.Model, got {type(model)!r}"
            )

        logger.info("Inspecting Keras model...")

        backend_name = self._get_backend_name()

        info: dict[str, Any] = {
            "name": model.name,
            "backend": backend_name,
            "trainable": model.trainable,
        }

        info["inputs"] = self._get_inputs(model)
        info["input_names"] = [input["name"] for input in info["inputs"]]
        info["outputs"] = self._get_outputs(model)
        info["output_names"] = [output["name"] for output in info["outputs"]]
        info["parameters"] = self._get_param_counts(model)

        if self.include_signatures:
            info["signatures"] = self._get_signatures(model)

        if self.include_layers:
            info["layers"] = self._get_layers_summary(model)

        logger.debug(f"Keras model inspection result: {info}")
        return info

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #
    def _get_backend_name(self) -> str:
        """Return the current Keras backend name in a Keras 3–friendly way."""
        # Keras 3: keras.config.backend()
        try:
            return keras.config.backend()
        except AttributeError:
            # Fallback for tf.keras-style API, if ever run there
            try:
                return keras.backend.backend()
            except Exception:  # noqa: BLE001
                return "unknown"

    def _get_inputs(self, model: keras.Model) -> list[dict[str, Any]]:
        """Extract input specifications (index, name, shape, dtype)."""
        logger.debug("Extracting model input specs...")

        inputs = model.inputs
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        result: list[dict[str, Any]] = []
        for idx, inp in enumerate(inputs):
            result.append(
                {
                    "index": idx,
                    "name": getattr(inp, "name", f"input_{idx}"),
                    "shape": tuple(inp.shape),
                    "dtype": str(inp.dtype),
                }
            )

        return result

    def _get_outputs(self, model: keras.Model) -> list[dict[str, Any]]:
        """Extract output specifications (index, name, shape, dtype)."""
        logger.debug("Extracting model output specs...")

        outputs = model.outputs
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        result: list[dict[str, Any]] = []
        for idx, out in enumerate(outputs):
            result.append(
                {
                    "index": idx,
                    "name": getattr(out, "name", f"output_{idx}"),
                    "shape": tuple(out.shape),
                    "dtype": str(out.dtype),
                }
            )

        return result

    def _get_param_counts(self, model: keras.Model) -> dict[str, int]:
        """Compute parameter counts for a Keras 3 model."""
        logger.debug("Computing model parameter counts...")

        def num_params(weight: Any) -> int:
            """Return number of parameters in a weight tensor."""
            shape = getattr(weight, "shape", None)
            if shape is None:
                return 0
            try:
                return int(np.prod(shape))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed computing param count for {weight}: {exc}")
                return 0

        trainable_params = sum(num_params(w) for w in model.trainable_weights)
        non_trainable_params = sum(num_params(w) for w in model.non_trainable_weights)

        return {
            "trainable": int(trainable_params),
            "non_trainable": int(non_trainable_params),
            "total": int(trainable_params + non_trainable_params),
        }

    def _get_signatures(self, model: keras.Model) -> dict[str, Any]:
        """Extract signatures if the model exposes them.

        Note:
            For standard Keras models (`Sequential`, functional, subclassed)
            and models saved/loaded with `model.save()` / `keras.models.load_model`,
            this is usually `{}`.

            Some advanced workflows (e.g. Keras on TF with SavedModel under
            the hood) might populate `model.signatures`, in which case we
            attempt a best-effort introspection without importing TensorFlow.
        """
        logger.debug("Extracting signatures (if any)...")

        signatures = getattr(model, "signatures", None)
        if not signatures or not isinstance(signatures, Mapping):
            return {}

        result: dict[str, Any] = {}
        for name, fn in signatures.items():
            logger.debug(f"Inspecting signature '{name}'...")
            # Best-effort duck-typed inspection
            inputs: Any = {}
            outputs: Any = {}

            if hasattr(fn, "structured_input_signature"):
                try:
                    _, input_spec = fn.structured_input_signature
                    inputs = str(input_spec)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"Failed to read structured_input_signature for '{name}': {exc}"
                    )

            if hasattr(fn, "structured_outputs"):
                try:
                    outputs = str(fn.structured_outputs)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"Failed to read structured_outputs for '{name}': {exc}"
                    )

            result[name] = {
                "inputs": inputs,
                "outputs": outputs,
            }

        return result

    def _get_layers_summary(self, model: keras.Model) -> list[dict[str, Any]]:
        """Extract a per-layer summary."""
        logger.debug("Extracting per-layer summary...")

        layers_summary: list[dict[str, Any]] = []
        for layer in model.layers:
            layer_info: dict[str, Any] = {
                "name": layer.name,
                "class_name": layer.__class__.__name__,
                "trainable": layer.trainable,
            }

            input_shape = getattr(layer, "input_shape", None)
            output_shape = getattr(layer, "output_shape", None)

            if input_shape is not None:
                layer_info["input_shape"] = input_shape
            if output_shape is not None:
                layer_info["output_shape"] = output_shape

            layers_summary.append(layer_info)

        return layers_summary
