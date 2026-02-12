"""FlowyML TensorFlow steps — Full-featured pipeline steps for TF/Keras workflows.

Each step leverages FlowyML's native capabilities:
- Artifact-centric design: returns Dataset, Model, Metrics with auto-extraction
- Supports caching, retry, GPU resources, tags, DAG wiring, and execution groups
- train_model integrates FlowymlKerasCallback for automatic tracking
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf
import keras
from loguru import logger

from flowyml.core.step import step

from flowyml.integrations.keras import FlowymlKerasCallback
from flowyml import Dataset, Model, Metrics

from mlpotion.frameworks.tensorflow.config import (
    DataLoadingConfig,
    DataOptimizationConfig,
    DataTransformationConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig,
)
from mlpotion.frameworks.tensorflow.data.loaders import CSVDataLoader
from mlpotion.frameworks.tensorflow.data.optimizers import DatasetOptimizer
from mlpotion.frameworks.tensorflow.data.transformers import DataToCSVTransformer
from mlpotion.frameworks.tensorflow.training.trainers import ModelTrainer
from mlpotion.frameworks.tensorflow.evaluation.evaluators import ModelEvaluator
from mlpotion.frameworks.tensorflow.deployment.exporters import ModelExporter
from mlpotion.frameworks.tensorflow.deployment.persistence import ModelPersistence
from mlpotion.frameworks.tensorflow.models.inspection import ModelInspector


# ---------------------------------------------------------------------------
# Data Steps
# ---------------------------------------------------------------------------


@step(
    name="tf_load_data",
    outputs=["dataset"],
    cache="code_hash",
    tags={"framework": "tensorflow", "component": "data_loader"},
)
def load_data(
    file_path: str,
    batch_size: int = 32,
    label_name: str = "target",
    column_names: list[str] | None = None,
) -> Dataset:
    """Load CSV data into a tf.data.Dataset, wrapped as a Dataset asset.

    Automatic metadata extraction captures batch size, source path,
    label name, and column configuration.

    Args:
        file_path: Glob pattern for CSV files.
        batch_size: Batch size.
        label_name: Target column name.
        column_names: Specific columns to load.

    Returns:
        Dataset asset wrapping the tf.data.Dataset.
    """
    config = DataLoadingConfig(
        file_pattern=file_path,
        batch_size=batch_size,
        label_name=label_name,
        column_names=column_names,
    )
    loader = CSVDataLoader(**config.dict())
    tf_dataset = loader.load()

    dataset_asset = Dataset.create(
        data=tf_dataset,
        name="tf_csv_dataset",
        properties={
            "source": file_path,
            "batch_size": batch_size,
            "label_name": label_name,
        },
        source=file_path,
        loader="CSVDataLoader",
        framework="tensorflow",
    )

    logger.info(f"📦 Loaded TF Dataset: batch_size={batch_size}, source={file_path}")
    return dataset_asset


@step(
    name="tf_optimize_data",
    inputs=["dataset"],
    outputs=["optimized_dataset"],
    cache="code_hash",
    tags={"framework": "tensorflow", "component": "data_optimizer"},
)
def optimize_data(
    dataset: tf.data.Dataset | Dataset,
    batch_size: int = 32,
    shuffle_buffer_size: int | None = None,
    prefetch: bool = True,
    cache: bool = False,
) -> Dataset:
    """Optimize a tf.data.Dataset with caching and prefetching, returned as Dataset asset.

    Returns a Dataset asset with lineage linked to the input dataset.

    Args:
        dataset: Input tf.data.Dataset or Dataset asset.
        batch_size: Batch size.
        shuffle_buffer_size: Shuffle buffer size.
        prefetch: Enable prefetching.
        cache: Enable dataset caching.

    Returns:
        Dataset asset wrapping the optimized tf.data.Dataset.
    """
    # Extract raw tf.data.Dataset from Dataset if wrapped
    raw_dataset = dataset.data if isinstance(dataset, Dataset) else dataset

    config = DataOptimizationConfig(
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        prefetch=prefetch,
        cache=cache,
    )
    optimizer = DatasetOptimizer(**config.dict())
    optimized = optimizer.optimize(raw_dataset)

    parent = dataset if isinstance(dataset, Dataset) else None
    dataset_asset = Dataset.create(
        data=optimized,
        name="tf_optimized_dataset",
        parent=parent,
        properties={
            "batch_size": batch_size,
            "prefetch": prefetch,
            "cache": cache,
            "shuffle_buffer_size": shuffle_buffer_size,
        },
        optimizer="DatasetOptimizer",
        framework="tensorflow",
    )

    logger.info(f"⚡ Optimized TF Dataset: cache={cache}, prefetch={prefetch}")
    return dataset_asset


@step(
    name="tf_transform_data",
    inputs=["dataset"],
    outputs=["transformed"],
    tags={"framework": "tensorflow", "component": "data_transformer"},
)
def transform_data(
    dataset: tf.data.Dataset | Dataset,
    model: keras.Model | Model,
    data_output_path: str,
    data_output_per_batch: bool = False,
) -> Dataset:
    """Transform data using a model and save predictions to CSV, returned as a Dataset asset.

    Returns a Dataset asset with lineage linked to the input dataset.

    Args:
        dataset: Input tf.data.Dataset or Dataset asset.
        model: Keras/TF model or Model asset for generating predictions.
        data_output_path: Output path for transformed data.
        data_output_per_batch: If True, output one file per batch.

    Returns:
        Dataset asset pointing to the output CSV with parent lineage.
    """
    # Extract raw objects from assets
    raw_dataset = dataset.data if isinstance(dataset, Dataset) else dataset
    raw_model = model.data if isinstance(model, Model) else model

    transformer = DataToCSVTransformer(
        dataset=raw_dataset,
        model=raw_model,
        data_output_path=data_output_path,
        data_output_per_batch=data_output_per_batch,
    )

    # Create minimal config for transform method
    config = DataTransformationConfig(
        file_pattern="",  # Not used since dataset is provided
        model_path="",  # Not used since model is provided
        model_input_signature={},  # Empty dict as model is provided directly
        data_output_path=data_output_path,
        data_output_per_batch=data_output_per_batch,
    )

    transformer.transform(dataset=None, model=None, config=config)

    parent = dataset if isinstance(dataset, Dataset) else None
    transformed = Dataset.create(
        data={"output_path": data_output_path},
        name="tf_transformed_data",
        parent=parent,
        properties={
            "output_path": data_output_path,
            "per_batch": data_output_per_batch,
        },
        source=data_output_path,
        transformer="DataToCSVTransformer",
    )

    logger.info(f"🔄 Transformed data saved to: {data_output_path}")
    return transformed


# ---------------------------------------------------------------------------
# Training Steps
# ---------------------------------------------------------------------------


@step(
    name="tf_train_model",
    inputs=["dataset"],
    outputs=["model", "training_metrics"],
    cache=False,
    retry=1,
    tags={"framework": "tensorflow", "component": "model_trainer"},
)
def train_model(
    model: keras.Model,
    data: tf.data.Dataset | Dataset,
    epochs: int = 10,
    learning_rate: float = 0.001,
    verbose: int = 1,
    validation_data: tf.data.Dataset | Dataset | None = None,
    callbacks: list[keras.callbacks.Callback] | None = None,
    experiment_name: str | None = None,
    project: str | None = None,
    log_model: bool = True,
) -> tuple[Model, Metrics]:
    """Train a TF/Keras model with FlowyML tracking integration.

    Automatically attaches a FlowymlKerasCallback for:
    - Dynamic capture of ALL training metrics
    - Live dashboard updates
    - Model artifact logging

    Returns a Model asset (via Model.from_keras with auto-extracted metadata)
    and a Metrics asset with training history.

    Args:
        model: Compiled Keras model.
        data: Training tf.data.Dataset or Dataset asset.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        verbose: Keras verbosity level.
        validation_data: Optional validation dataset or Dataset asset.
        callbacks: Additional Keras callbacks.
        experiment_name: Experiment name for FlowyML tracking.
        project: Project name for FlowyML dashboard.
        log_model: Whether to save model artifact after training.

    Returns:
        Tuple of (Model asset, Metrics asset).
    """
    # Extract raw data from Dataset if wrapped
    raw_data = data.data if isinstance(data, Dataset) else data
    raw_val = (
        validation_data.data
        if isinstance(validation_data, Dataset)
        else validation_data
    )

    all_callbacks = list(callbacks or [])

    # Auto-attach FlowyML callback
    flowyml_callback = None
    if experiment_name:
        flowyml_callback = FlowymlKerasCallback(
            experiment_name=experiment_name,
            project=project,
            log_model=log_model,
        )
        all_callbacks.append(flowyml_callback)

    config = ModelTrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        verbose=verbose,
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )

    trainer = ModelTrainer()
    result = trainer.train(
        model=model,
        dataset=raw_data,
        config=config,
        validation_dataset=raw_val,
    )

    # Collect raw metrics
    raw_metrics: dict[str, Any] = result.metrics if hasattr(result, "metrics") else {}
    raw_metrics["epochs_completed"] = epochs
    raw_metrics["learning_rate"] = learning_rate

    # Wrap as Model asset using from_keras for full auto-extraction
    model_asset = Model.from_keras(
        model,
        name="tf_trained_model",
        callback=flowyml_callback,
        epochs_requested=epochs,
    )

    # Wrap as Metrics asset
    metrics_asset = Metrics.create(
        metrics=raw_metrics,
        name="tf_training_metrics",
        tags={"stage": "training", "framework": "tensorflow"},
        properties={
            "epochs": epochs,
            "learning_rate": learning_rate,
            **{k: v for k, v in raw_metrics.items() if k != "history"},
        },
    )

    logger.info(
        f"🎯 TF training complete: {epochs} epochs, "
        f"metrics captured: {list(raw_metrics.keys())}"
    )
    return model_asset, metrics_asset


# ---------------------------------------------------------------------------
# Evaluation Steps
# ---------------------------------------------------------------------------


@step(
    name="tf_evaluate_model",
    inputs=["model", "dataset"],
    outputs=["metrics"],
    cache="input_hash",
    tags={"framework": "tensorflow", "component": "model_evaluator"},
)
def evaluate_model(
    model: keras.Model | Model,
    data: tf.data.Dataset | Dataset,
    verbose: int = 0,
) -> Metrics:
    """Evaluate a TF/Keras model and return a Metrics asset.

    Args:
        model: Trained Keras model or Model asset.
        data: Evaluation tf.data.Dataset or Dataset asset.
        verbose: Keras verbosity level.

    Returns:
        Metrics asset with evaluation results.
    """
    # Extract raw objects from assets
    raw_model = model.data if isinstance(model, Model) else model
    raw_data = data.data if isinstance(data, Dataset) else data

    config = ModelEvaluationConfig(verbose=verbose)
    evaluator = ModelEvaluator()
    result = evaluator.evaluate(model=raw_model, dataset=raw_data, config=config)

    raw_metrics = result.metrics if hasattr(result, "metrics") else {}

    metrics_asset = Metrics.create(
        metrics=raw_metrics,
        name="tf_evaluation_metrics",
        tags={"stage": "evaluation", "framework": "tensorflow"},
        properties=raw_metrics,
    )

    logger.info(f"📊 TF evaluation: {raw_metrics}")
    return metrics_asset


# ---------------------------------------------------------------------------
# Export / Save / Load Steps
# ---------------------------------------------------------------------------


@step(
    name="tf_export_model",
    inputs=["model"],
    outputs=["exported_model"],
    cache="code_hash",
    tags={"framework": "tensorflow", "component": "model_exporter"},
)
def export_model(
    model: keras.Model | Model,
    export_path: str,
    export_format: str = "keras",
) -> Model:
    """Export a TF/Keras model, returned as a Model asset.

    Args:
        model: Keras model or Model asset to export.
        export_path: Destination path.
        export_format: Format ('saved_model', 'tflite', 'keras').

    Returns:
        Model asset with export metadata.
    """
    raw_model = model.data if isinstance(model, Model) else model

    exporter = ModelExporter()
    exporter.export(
        model=raw_model,
        path=export_path,
        export_format=export_format,
    )

    model_asset = Model.from_keras(
        raw_model,
        name="tf_exported_model",
        export_path=export_path,
        export_format=export_format,
    )

    logger.info(f"📤 Exported TF model to: {export_path}")
    return model_asset


@step(
    name="tf_save_model",
    inputs=["model"],
    outputs=["saved_model"],
    tags={"framework": "tensorflow", "component": "model_persistence"},
)
def save_model(
    model: keras.Model | Model,
    save_path: str,
) -> Model:
    """Save a TF/Keras model to disk, returned as a Model asset.

    Args:
        model: Keras model or Model asset to save.
        save_path: Destination file path.

    Returns:
        Model asset with save location metadata.
    """
    raw_model = model.data if isinstance(model, Model) else model

    persistence = ModelPersistence(path=save_path, model=raw_model)
    persistence.save()

    model_asset = Model.from_keras(
        raw_model,
        name="tf_saved_model",
        save_path=save_path,
    )

    logger.info(f"💾 Saved TF model to: {save_path}")
    return model_asset


@step(
    name="tf_load_model",
    outputs=["model"],
    cache="code_hash",
    tags={"framework": "tensorflow", "component": "model_persistence"},
)
def load_model(
    model_path: str,
    inspect: bool = False,
) -> Model:
    """Load a TF/Keras model from disk, returned as a Model asset.

    Args:
        model_path: Path to the saved model.
        inspect: If True, log model inspection info.

    Returns:
        Model asset wrapping the loaded Keras model.
    """
    persistence = ModelPersistence(path=model_path)
    raw_model, inspection = persistence.load(inspect=inspect)

    model_asset = Model.from_keras(
        raw_model,
        name="tf_loaded_model",
        source_path=model_path,
    )

    if inspect and inspection:
        logger.info(f"🔍 Loaded TF model from: {model_path}, inspection: {inspection}")
    else:
        logger.info(f"🔍 Loaded TF model from: {model_path}")

    return model_asset


@step(
    name="tf_inspect_model",
    inputs=["model"],
    outputs=["inspection"],
    tags={"framework": "tensorflow", "component": "model_inspector"},
)
def inspect_model(
    model: keras.Model | Model,
    include_layers: bool = True,
    include_signatures: bool = True,
) -> Metrics:
    """Inspect a TF/Keras model and return detailed metadata as a Metrics asset.

    Args:
        model: Keras model or Model asset to inspect.
        include_layers: Include per-layer information.
        include_signatures: Include input/output signatures.

    Returns:
        Metrics asset with model inspection details.
    """
    raw_model = model.data if isinstance(model, Model) else model

    inspector = ModelInspector(
        include_layers=include_layers,
        include_signatures=include_signatures,
    )
    inspection = inspector.inspect(raw_model)

    metrics_asset = Metrics.create(
        metrics=inspection,
        name="tf_model_inspection",
        tags={"stage": "inspection", "framework": "tensorflow"},
        properties={
            "model_name": inspection.get("name", "unknown"),
            "total_params": inspection.get("parameters", {}).get("total"),
        },
    )

    logger.info(
        f"🔎 TF Model: {inspection.get('name', 'unknown')}, "
        f"params: {inspection.get('parameters', {}).get('total', '?')}"
    )
    return metrics_asset
