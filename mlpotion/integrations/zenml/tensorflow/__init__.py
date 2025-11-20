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
from zenml.materializers.materializer_registry import materializer_registry

# model materializer
from mlpotion.integrations.zenml.keras.materializers import KerasModelMaterializer
from mlpotion.integrations.zenml.tensorflow.materializers import (
    TFConfigDatasetMaterializer,
)

materializer_registry.register_materializer_type(keras.Model, KerasModelMaterializer)
# dataset materializers
# Try CSV materializer first (lightweight, stores only config)
# Falls back to TFRecord materializer if CSV config not available


# Register CSV materializer with higher priority
materializer_registry.register_materializer_type(
    tf.data.Dataset, TFConfigDatasetMaterializer
)


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
