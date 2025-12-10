"""Pre-configured ZenML steps for TensorFlow.

These are convenience steps that create common configurations.
"""
import keras
import tensorflow as tf

from mlpotion.integrations.zenml.tensorflow.steps import (
    load_data,
    optimize_data,
    transform_data,
    train_model,
    evaluate_model,
    export_model,
    save_model,
    load_model,
    inspect_model,
)

# import and register materializers
from zenml.logger import get_logger
from zenml.materializers.materializer_registry import materializer_registry

# model materializer
from mlpotion.integrations.zenml.keras.materializers import KerasModelMaterializer
from mlpotion.integrations.zenml.tensorflow.materializers import (
    TFConfigDatasetMaterializer,
)

logger = get_logger(__name__)

# Import version for display
try:
    import mlpotion

    _MLPOTION_VERSION = mlpotion.__version__
except (ImportError, AttributeError):
    _MLPOTION_VERSION = "unknown"

materializer_registry.register_materializer_type(keras.Model, KerasModelMaterializer)
# dataset materializers
# Try CSV materializer first (lightweight, stores only config)
# Falls back to TFRecord materializer if CSV config not available

# Module-level flag to track if version was displayed
_VERSION_DISPLAYED = False

# Register CSV materializer with higher priority (overwrite any existing registration)
# This ensures it's used instead of TFRecord materializer
try:
    materializer_registry.register_and_overwrite_type(
        key=tf.data.Dataset,
        type_=TFConfigDatasetMaterializer,
    )
    # Display version info on first import (only once per session)
    if not _VERSION_DISPLAYED:
        print(
            f"✨ MLPotion v{_MLPOTION_VERSION} - TensorFlow + ZenML integration ready! 🚀"
        )
        _VERSION_DISPLAYED = True
except Exception as e:
    logger.warning(f"Failed to register TFConfigDatasetMaterializer: {e}")


__all__ = [
    "load_data",
    "optimize_data",
    "transform_data",
    "train_model",
    "evaluate_model",
    "export_model",
    "save_model",
    "load_model",
    "inspect_model",
]
