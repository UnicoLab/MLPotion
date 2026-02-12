"""FlowyML Keras steps — Full-featured pipeline steps for Keras workflows.

Each step leverages FlowyML's native capabilities:
- Artifact-centric design: returns Dataset, Model, Metrics with auto-extraction
- Supports caching, retry, GPU resources, tags, DAG wiring, and execution groups
- train_model integrates FlowymlKerasCallback for automatic tracking
"""

from __future__ import annotations
import uuid
from typing import Any

import keras
from loguru import logger

from flowyml.core.step import step

from flowyml.integrations.keras import FlowymlKerasCallback
from flowyml import Dataset, Model, Metrics

from mlpotion.frameworks.keras.config import (
    DataLoadingConfig,
    DataTransformationConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig,
)
from mlpotion.frameworks.keras.data.loaders import CSVDataLoader, CSVSequence
from mlpotion.frameworks.keras.data.transformers import CSVDataTransformer
from mlpotion.frameworks.keras.training.trainers import ModelTrainer
from mlpotion.frameworks.keras.evaluation.evaluators import ModelEvaluator
from mlpotion.frameworks.keras.deployment.exporters import ModelExporter
from mlpotion.frameworks.keras.deployment.persistence import ModelPersistence
from mlpotion.frameworks.keras.models.inspection import ModelInspector


# ---------------------------------------------------------------------------
# Data Steps
# ---------------------------------------------------------------------------


@step(
    name="keras_load_data",
    outputs=["dataset"],
    cache="code_hash",
    tags={"framework": "keras", "component": "data_loader"},
)
def load_data(
    file_path: str,
    batch_size: int = 32,
    label_name: str | None = None,
    column_names: list[str] | None = None,
    shuffle: bool = True,
    dtype: str = "float32",
) -> Dataset:
    """Load CSV data into a Keras-compatible CSVSequence, wrapped as a Dataset asset.

    Automatic metadata extraction captures batch count, batch size, source path,
    column names, and label information.

    Args:
        file_path: Glob pattern for CSV files (e.g., "data/*.csv").
        batch_size: Batch size for the sequence.
        label_name: Name of the label/target column.
        column_names: Specific columns to load (None = all).
        shuffle: Whether to shuffle the data.
        dtype: Data type for numeric conversion.

    Returns:
        Dataset asset wrapping the CSVSequence with auto-extracted metadata.
    """
    config = DataLoadingConfig(
        file_pattern=file_path,
        batch_size=batch_size,
        column_names=column_names,
        label_name=label_name,
        shuffle=shuffle,
        dtype=dtype,
    )
    loader = CSVDataLoader(**config.dict())
    sequence = loader.load()

    dataset = Dataset.create(
        data=sequence,
        name="keras_csv_dataset",
        properties={
            "source": file_path,
            "batch_size": batch_size,
            "batches": len(sequence),
            "label_name": label_name,
            "shuffle": shuffle,
            "dtype": dtype,
        },
        source=file_path,
        loader="CSVDataLoader",
        framework="keras",
    )

    logger.info(
        f"📦 Loaded dataset: {len(sequence)} batches, "
        f"batch_size={batch_size}, source={file_path}"
    )
    return dataset


@step(
    name="keras_transform_data",
    inputs=["dataset"],
    outputs=["transformed"],
    cache="code_hash",
    tags={"framework": "keras", "component": "data_transformer"},
)
def transform_data(
    dataset: Dataset,
    model: keras.Model,
    data_output_path: str,
    data_output_per_batch: bool = False,
    batch_size: int | None = None,
    feature_names: list[str] | None = None,
    input_columns: list[str] | None = None,
) -> Dataset:
    """Transform data using a Keras model and save predictions to CSV.

    Returns a Dataset asset with lineage linked to the input dataset.

    Args:
        dataset: Input Dataset asset wrapping a CSVSequence.
        model: Keras model for generating predictions.
        data_output_path: Output path for transformed data.
        data_output_per_batch: If True, output one file per batch.
        batch_size: Optional batch size override.
        feature_names: Optional feature names for output CSV.
        input_columns: Optional input columns to pass to model.

    Returns:
        Dataset asset pointing to the output CSV with parent lineage.
    """
    # Extract raw sequence from Dataset if wrapped
    raw_dataset = dataset.data if isinstance(dataset, Dataset) else dataset

    config = DataTransformationConfig(
        data_output_path=data_output_path,
        data_output_per_batch=data_output_per_batch,
        batch_size=batch_size,
        feature_names=feature_names,
        input_columns=input_columns,
    )
    transformer = CSVDataTransformer(
        dataset=raw_dataset,
        model=model,
        data_output_path=data_output_path,
        data_output_per_batch=data_output_per_batch,
        batch_size=batch_size,
        feature_names=feature_names,
        input_columns=input_columns,
    )
    transformer.transform(dataset=raw_dataset, model=model, config=config)

    # Wrap output as Dataset with parent lineage
    parent = dataset if isinstance(dataset, Dataset) else None
    transformed = Dataset.create(
        data={"output_path": data_output_path},
        name="keras_transformed_data",
        parent=parent,
        properties={
            "output_path": data_output_path,
            "per_batch": data_output_per_batch,
            "feature_names": feature_names,
        },
        source=data_output_path,
        transformer="CSVDataTransformer",
    )

    logger.info(f"🔄 Transformed data saved to: {data_output_path}")
    return transformed


# ---------------------------------------------------------------------------
# Training Steps
# ---------------------------------------------------------------------------


@step(
    name="keras_train_model",
    inputs=["dataset"],
    outputs=["model", "training_metrics"],
    cache=False,
    retry=1,
    tags={"framework": "keras", "component": "model_trainer"},
)
def train_model(
    model: keras.Model,
    data: CSVSequence | Dataset,
    epochs: int = 10,
    learning_rate: float = 0.001,
    verbose: int = 1,
    validation_data: CSVSequence | Dataset | None = None,
    callbacks: list[keras.callbacks.Callback] | None = None,
    experiment_name: str | None = None,
    project: str | None = None,
    log_model: bool = True,
) -> tuple[Model, Metrics]:
    """Train a Keras model with FlowyML tracking integration.

    Automatically attaches a FlowymlKerasCallback for:
    - Dynamic capture of ALL training metrics
    - Live dashboard updates
    - Model artifact logging

    Returns a Model asset (via Model.from_keras with auto-extracted metadata)
    and a Metrics asset with training history.

    Args:
        model: Compiled Keras model.
        data: Training data as CSVSequence or Dataset asset.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        verbose: Keras verbosity level.
        validation_data: Optional validation CSVSequence or Dataset.
        callbacks: Additional Keras callbacks (FlowyML callback auto-added).
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

    # Auto-attach FlowyML callback for tracking
    flowyml_callback = FlowymlKerasCallback(
        experiment_name=experiment_name or f"keras_train_{uuid.uuid4()}",
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
        framework_options={"callbacks": all_callbacks} if all_callbacks else {},
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
    if hasattr(result, "history") and result.history:
        raw_metrics["history"] = result.history
    raw_metrics["epochs_completed"] = epochs
    raw_metrics["learning_rate"] = learning_rate

    # Wrap as Model asset using from_keras for full auto-extraction
    model_asset = Model.from_keras(
        model,
        name="keras_trained_model",
        callback=flowyml_callback,
        epochs_requested=epochs,
        batch_size=getattr(raw_data, "batch_size", None),
    )

    # Wrap as Metrics asset
    metrics_asset = Metrics.create(
        metrics=raw_metrics,
        name="keras_training_metrics",
        tags={"stage": "training", "framework": "keras"},
        properties={
            "epochs": epochs,
            "learning_rate": learning_rate,
            **{k: v for k, v in raw_metrics.items() if k != "history"},
        },
    )

    logger.info(
        f"🎯 Training complete: {epochs} epochs, "
        f"metrics captured: {list(raw_metrics.keys())}"
    )
    return model_asset, metrics_asset


# ---------------------------------------------------------------------------
# Evaluation Steps
# ---------------------------------------------------------------------------


@step(
    name="keras_evaluate_model",
    inputs=["model", "dataset"],
    outputs=["metrics"],
    cache="input_hash",
    tags={"framework": "keras", "component": "model_evaluator"},
)
def evaluate_model(
    model: keras.Model | Model,
    data: CSVSequence | Dataset,
    verbose: int = 0,
) -> Metrics:
    """Evaluate a Keras model and return a Metrics asset.

    Args:
        model: Trained Keras model or Model asset.
        data: Evaluation data as CSVSequence or Dataset asset.
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
        name="keras_evaluation_metrics",
        tags={"stage": "evaluation", "framework": "keras"},
        properties=raw_metrics,
    )

    logger.info(f"📊 Evaluation: {raw_metrics}")
    return metrics_asset


# ---------------------------------------------------------------------------
# Export / Save / Load Steps
# ---------------------------------------------------------------------------


@step(
    name="keras_export_model",
    inputs=["model"],
    outputs=["exported_model"],
    cache="code_hash",
    tags={"framework": "keras", "component": "model_exporter"},
)
def export_model(
    model: keras.Model | Model,
    export_path: str,
    export_format: str | None = None,
) -> Model:
    """Export a Keras model to the specified format, returned as a Model asset.

    Args:
        model: Keras model or Model asset to export.
        export_path: Destination path.
        export_format: Format ('keras', 'saved_model', 'tflite').

    Returns:
        Model asset with export metadata.
    """
    raw_model = model.data if isinstance(model, Model) else model

    exporter = ModelExporter()
    config = {}
    if export_format:
        config["export_format"] = export_format
    exporter.export(model=raw_model, path=export_path, **config)

    model_asset = Model.from_keras(
        raw_model,
        name="keras_exported_model",
        export_path=export_path,
        export_format=export_format or "keras",
    )

    logger.info(f"📤 Exported model to: {export_path}")
    return model_asset


@step(
    name="keras_save_model",
    inputs=["model"],
    outputs=["saved_model"],
    tags={"framework": "keras", "component": "model_persistence"},
)
def save_model(
    model: keras.Model | Model,
    save_path: str,
) -> Model:
    """Save a Keras model to disk, returned as a Model asset.

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
        name="keras_saved_model",
        save_path=save_path,
    )

    logger.info(f"💾 Saved model to: {save_path}")
    return model_asset


@step(
    name="keras_load_model",
    outputs=["model"],
    cache="code_hash",
    tags={"framework": "keras", "component": "model_persistence"},
)
def load_model(
    model_path: str,
    inspect: bool = False,
) -> Model:
    """Load a Keras model from disk, returned as a Model asset.

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
        name="keras_loaded_model",
        source_path=model_path,
    )

    if inspect and inspection:
        logger.info(f"🔍 Loaded model from: {model_path}, inspection: {inspection}")
    else:
        logger.info(f"🔍 Loaded model from: {model_path}")

    return model_asset


@step(
    name="keras_inspect_model",
    inputs=["model"],
    outputs=["inspection"],
    tags={"framework": "keras", "component": "model_inspector"},
)
def inspect_model(
    model: keras.Model | Model,
    include_layers: bool = True,
    include_signatures: bool = True,
) -> Metrics:
    """Inspect a Keras model and return detailed metadata as a Metrics asset.

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
        name="keras_model_inspection",
        tags={"stage": "inspection", "framework": "keras"},
        properties={
            "model_name": inspection.get("name", "unknown"),
            "total_params": inspection.get("parameters", {}).get("total"),
        },
    )

    logger.info(
        f"🔎 Model: {inspection.get('name', 'unknown')}, "
        f"params: {inspection.get('parameters', {}).get('total', '?')}"
    )
    return metrics_asset
