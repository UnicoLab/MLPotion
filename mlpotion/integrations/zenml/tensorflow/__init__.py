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
from zenml.enums import ArtifactType
from zenml.logger import get_logger
from zenml.materializers.materializer_registry import materializer_registry

# model materializer
from mlpotion.integrations.zenml.keras.materializers import KerasModelMaterializer
from mlpotion.integrations.zenml.tensorflow.materializers import (
    TFConfigDatasetMaterializer,
)

logger = get_logger(__name__)

materializer_registry.register_materializer_type(keras.Model, KerasModelMaterializer)
# dataset materializers
# Try CSV materializer first (lightweight, stores only config)
# Falls back to TFRecord materializer if CSV config not available


# Register CSV materializer with higher priority (overwrite any existing registration)
# This ensures it's used instead of TFRecord materializer
try:
    materializer_registry.register_and_overwrite_type(
        key=tf.data.Dataset,
        type_=TFConfigDatasetMaterializer,
        artifact_type=ArtifactType.DATA,
    )
    logger.info("Registered TFConfigDatasetMaterializer for tf.data.Dataset")
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
