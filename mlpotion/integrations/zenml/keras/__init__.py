"""Pre-configured ZenML steps for Keras.

These are convenience steps that create common configurations.
"""

import keras

from mlpotion.integrations.zenml.keras.steps import (
    load_data,
    transform_data,
    train_model,
    evaluate_model,
    export_model,
    save_model,
    load_model,
    inspect_model,
)

# import and register materializers
from mlpotion.integrations.zenml.keras.materializers import KerasModelMaterializer

from zenml.materializers.materializer_registry import materializer_registry
materializer_registry.register_materializer_type(
    keras.Model,
    KerasModelMaterializer
)

__all__ = [
    "load_data",
    "transform_data",
    "train_model",
    "evaluate_model",
    "export_model",
    "save_model",
    "load_model",
    "inspect_model",
]
