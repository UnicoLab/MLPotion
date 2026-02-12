"""Pre-configured FlowyML steps and pipeline templates for Keras.

Each step wraps MLPotion Keras components and returns FlowyML assets
(Dataset, Model, Metrics) with automatic metadata extraction and lineage.

Pipeline templates provide ready-to-run DAGs for common workflows:
training, full lifecycle, evaluation, export, experiments, and scheduling.
"""

from mlpotion.integrations.flowyml.keras.steps import (
    evaluate_model,
    export_model,
    inspect_model,
    load_data,
    load_model,
    save_model,
    train_model,
    transform_data,
)
from mlpotion.integrations.flowyml.keras.pipelines import (
    create_keras_evaluation_pipeline,
    create_keras_experiment_pipeline,
    create_keras_export_pipeline,
    create_keras_full_pipeline,
    create_keras_scheduled_pipeline,
    create_keras_training_pipeline,
)

__all__ = [
    # Steps
    "load_data",
    "transform_data",
    "train_model",
    "evaluate_model",
    "export_model",
    "save_model",
    "load_model",
    "inspect_model",
    # Pipeline Templates
    "create_keras_training_pipeline",
    "create_keras_full_pipeline",
    "create_keras_evaluation_pipeline",
    "create_keras_export_pipeline",
    "create_keras_experiment_pipeline",
    "create_keras_scheduled_pipeline",
]
