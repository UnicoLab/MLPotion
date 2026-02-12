"""FlowyML PyTorch steps — Full-featured pipeline steps for PyTorch workflows.

Each step leverages FlowyML's native capabilities:
- Artifact-centric design: returns Dataset, Model, Metrics with auto-extraction
- Supports caching, retry, GPU resources, tags, DAG wiring, and execution groups
- Returns framework-native objects wrapped as FlowyML assets
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger

from flowyml.core.step import step

from flowyml import Dataset, Model, Metrics

from mlpotion.frameworks.pytorch.config import (
    DataLoadingConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig,
    ModelExportConfig,
)
from mlpotion.frameworks.pytorch.data.datasets import (
    CSVDataset,
    StreamingCSVDataset,
)
from mlpotion.frameworks.pytorch.data.loaders import CSVDataLoader
from mlpotion.frameworks.pytorch.training.trainers import ModelTrainer
from mlpotion.frameworks.pytorch.evaluation.evaluators import ModelEvaluator
from mlpotion.frameworks.pytorch.deployment.exporters import ModelExporter
from mlpotion.frameworks.pytorch.deployment.persistence import ModelPersistence


# ---------------------------------------------------------------------------
# Data Steps
# ---------------------------------------------------------------------------


@step(
    name="pytorch_load_csv_data",
    outputs=["dataset"],
    cache="code_hash",
    tags={"framework": "pytorch", "component": "data_loader"},
)
def load_csv_data(
    file_path: str,
    batch_size: int = 32,
    label_name: str | None = None,
    column_names: list[str] | None = None,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    dtype: str = "float32",
) -> Dataset:
    """Load CSV data into a PyTorch DataLoader, wrapped as a Dataset asset.

    Automatic metadata extraction captures batch size, source path,
    column names, and worker configuration.

    Args:
        file_path: Glob pattern for CSV files.
        batch_size: Batch size.
        label_name: Target column name.
        column_names: Specific columns to load.
        shuffle: Whether to shuffle.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for faster GPU transfer.
        drop_last: Drop the last incomplete batch.
        dtype: Data type for tensors.

    Returns:
        Dataset asset wrapping the PyTorch DataLoader.
    """
    # Convert dtype string to torch.dtype
    torch_dtype = getattr(torch, dtype)

    # Create dataset
    csv_dataset = CSVDataset(
        file_pattern=file_path,
        column_names=column_names,
        label_name=label_name,
        dtype=torch_dtype,
    )

    # Create DataLoader config
    config = DataLoadingConfig(
        file_pattern=file_path,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    # Create DataLoader using factory
    loader_factory = CSVDataLoader(**config.dict(exclude={"file_pattern", "config"}))
    dataloader = loader_factory.load(csv_dataset)

    dataset_asset = Dataset.create(
        data=dataloader,
        name="pytorch_csv_dataset",
        properties={
            "source": file_path,
            "batch_size": batch_size,
            "label_name": label_name,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "dtype": dtype,
        },
        source=file_path,
        loader="CSVDataLoader",
        framework="pytorch",
    )

    logger.info(
        f"📦 Loaded PyTorch DataLoader: batch_size={batch_size}, source={file_path}"
    )
    return dataset_asset


@step(
    name="pytorch_load_streaming_csv_data",
    outputs=["dataset"],
    cache=False,
    tags={"framework": "pytorch", "component": "data_loader", "mode": "streaming"},
)
def load_streaming_csv_data(
    file_path: str,
    batch_size: int = 32,
    label_name: str | None = None,
    column_names: list[str] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    chunksize: int = 10000,
    dtype: str = "float32",
) -> Dataset:
    """Load large CSV data via streaming into a PyTorch DataLoader, wrapped as a Dataset asset.

    Uses chunked reading for datasets that don't fit in memory.

    Args:
        file_path: Glob pattern for CSV files.
        batch_size: Batch size.
        label_name: Target column name.
        column_names: Specific columns.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for faster GPU transfer.
        chunksize: Number of rows per chunk.
        dtype: Data type for tensors.

    Returns:
        Dataset asset wrapping the streaming PyTorch DataLoader.
    """
    # Convert dtype string to torch.dtype
    torch_dtype = getattr(torch, dtype)

    # Create streaming dataset
    streaming_dataset = StreamingCSVDataset(
        file_pattern=file_path,
        column_names=column_names,
        label_name=label_name,
        chunksize=chunksize,
        dtype=torch_dtype,
    )

    # Create DataLoader config (no shuffle for streaming)
    config = DataLoadingConfig(
        file_pattern=file_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    loader_factory = CSVDataLoader(**config.dict(exclude={"file_pattern", "config"}))
    dataloader = loader_factory.load(streaming_dataset)

    dataset_asset = Dataset.create(
        data=dataloader,
        name="pytorch_streaming_dataset",
        properties={
            "source": file_path,
            "batch_size": batch_size,
            "chunksize": chunksize,
            "label_name": label_name,
            "num_workers": num_workers,
            "dtype": dtype,
            "mode": "streaming",
        },
        source=file_path,
        loader="StreamingCSVDataLoader",
        framework="pytorch",
    )

    logger.info(
        f"📦 Streaming DataLoader: batch_size={batch_size}, "
        f"chunksize={chunksize}, source={file_path}"
    )
    return dataset_asset


# ---------------------------------------------------------------------------
# Training Steps
# ---------------------------------------------------------------------------


@step(
    name="pytorch_train_model",
    inputs=["dataset"],
    outputs=["model", "training_metrics"],
    cache=False,
    retry=1,
    tags={"framework": "pytorch", "component": "model_trainer"},
)
def train_model(
    model: nn.Module,
    data: DataLoader | Dataset,
    epochs: int = 10,
    learning_rate: float = 0.001,
    optimizer: str = "adam",
    loss_fn: str = "mse",
    device: str = "cpu",
    validation_data: DataLoader | Dataset | None = None,
    verbose: bool = True,
    max_batches_per_epoch: int | None = None,
) -> tuple[Model, Metrics]:
    """Train a PyTorch model with full configuration.

    Returns a Model asset (via Model.from_pytorch with auto-extracted metadata)
    and a Metrics asset with training history.

    Args:
        model: PyTorch model (nn.Module).
        data: Training DataLoader or Dataset asset.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        optimizer: Optimizer type ('adam', 'sgd', 'adamw').
        loss_fn: Loss function name ('mse', 'cross_entropy').
        device: Device to train on ('cuda', 'cpu').
        validation_data: Optional validation DataLoader or Dataset.
        verbose: Whether to log per-epoch metrics.
        max_batches_per_epoch: Limit batches per epoch (for debugging).

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

    config = ModelTrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        verbose=verbose,
        max_batches_per_epoch=max_batches_per_epoch,
    )

    trainer = ModelTrainer()
    result = trainer.train(
        model=model,
        dataloader=raw_data,
        config=config,
        validation_dataloader=raw_val,
    )

    # Collect raw metrics
    raw_metrics: dict[str, Any] = result.metrics if hasattr(result, "metrics") else {}
    raw_metrics["epochs_completed"] = epochs
    raw_metrics["learning_rate"] = learning_rate
    raw_metrics["optimizer"] = optimizer
    raw_metrics["device"] = device

    # Wrap as Model asset using from_pytorch for full auto-extraction
    model_asset = Model.from_pytorch(
        result.model,
        name="pytorch_trained_model",
        training_history=raw_metrics,
        epochs_requested=epochs,
        optimizer_type=optimizer,
        loss_function=loss_fn,
    )

    # Wrap as Metrics asset
    metrics_asset = Metrics.create(
        metrics=raw_metrics,
        name="pytorch_training_metrics",
        tags={"stage": "training", "framework": "pytorch"},
        properties={
            "epochs": epochs,
            "learning_rate": learning_rate,
            "optimizer": optimizer,
            "device": device,
        },
    )

    logger.info(f"🎯 PyTorch training complete: {epochs} epochs, device={device}")
    return model_asset, metrics_asset


# ---------------------------------------------------------------------------
# Evaluation Steps
# ---------------------------------------------------------------------------


@step(
    name="pytorch_evaluate_model",
    inputs=["model", "dataset"],
    outputs=["metrics"],
    cache="input_hash",
    tags={"framework": "pytorch", "component": "model_evaluator"},
)
def evaluate_model(
    model: nn.Module | Model,
    data: DataLoader | Dataset,
    loss_fn: str = "mse",
    device: str = "cpu",
    verbose: bool = True,
    max_batches: int | None = None,
) -> Metrics:
    """Evaluate a PyTorch model and return a Metrics asset.

    Args:
        model: Trained PyTorch model or Model asset.
        data: Evaluation DataLoader or Dataset asset.
        loss_fn: Loss function name.
        device: Device for evaluation.
        verbose: Whether to log metrics.
        max_batches: Limit batches to evaluate.

    Returns:
        Metrics asset with evaluation results.
    """
    # Extract raw objects from assets
    raw_model = model.data if isinstance(model, Model) else model
    raw_data = data.data if isinstance(data, Dataset) else data

    config = ModelEvaluationConfig(
        batch_size=raw_data.batch_size or 32,
        verbose=verbose,
        device=device,
        framework_options={"loss_fn": loss_fn, "max_batches": max_batches},
    )

    evaluator = ModelEvaluator()
    result = evaluator.evaluate(
        model=raw_model,
        dataloader=raw_data,
        config=config,
    )

    raw_metrics = result.metrics if hasattr(result, "metrics") else {}

    metrics_asset = Metrics.create(
        metrics=raw_metrics,
        name="pytorch_evaluation_metrics",
        tags={"stage": "evaluation", "framework": "pytorch"},
        properties=raw_metrics,
    )

    logger.info(f"📊 PyTorch evaluation: {raw_metrics}")
    return metrics_asset


# ---------------------------------------------------------------------------
# Export / Save / Load Steps
# ---------------------------------------------------------------------------


@step(
    name="pytorch_export_model",
    inputs=["model"],
    outputs=["exported_model"],
    cache="code_hash",
    tags={"framework": "pytorch", "component": "model_exporter"},
)
def export_model(
    model: nn.Module | Model,
    export_path: str,
    export_format: str = "torchscript",
    sample_input: torch.Tensor | None = None,
) -> Model:
    """Export a PyTorch model to the specified format, returned as a Model asset.

    Args:
        model: PyTorch model or Model asset to export.
        export_path: Destination path.
        export_format: Format ('torchscript', 'onnx').
        sample_input: Sample input tensor (required for ONNX).

    Returns:
        Model asset with export metadata.
    """
    raw_model = model.data if isinstance(model, Model) else model

    config = ModelExportConfig(
        export_path=export_path,
        format=export_format,
    )
    exporter = ModelExporter()
    exporter.export(model=raw_model, config=config, sample_input=sample_input)

    model_asset = Model.from_pytorch(
        raw_model,
        name="pytorch_exported_model",
        export_path=export_path,
        export_format=export_format,
    )

    logger.info(f"📤 Exported PyTorch model to: {export_path}")
    return model_asset


@step(
    name="pytorch_save_model",
    inputs=["model"],
    outputs=["saved_model"],
    tags={"framework": "pytorch", "component": "model_persistence"},
)
def save_model(
    model: nn.Module | Model,
    save_path: str,
) -> Model:
    """Save a PyTorch model to disk, returned as a Model asset.

    Args:
        model: PyTorch model or Model asset to save.
        save_path: Destination file path.

    Returns:
        Model asset with save location metadata.
    """
    raw_model = model.data if isinstance(model, Model) else model

    persistence = ModelPersistence()
    persistence.save(model=raw_model, path=save_path)

    model_asset = Model.from_pytorch(
        raw_model,
        name="pytorch_saved_model",
        save_path=save_path,
    )

    logger.info(f"💾 Saved PyTorch model to: {save_path}")
    return model_asset


@step(
    name="pytorch_load_model",
    outputs=["model"],
    cache="code_hash",
    tags={"framework": "pytorch", "component": "model_persistence"},
)
def load_model(
    model_path: str,
    model_class: type | None = None,
    device: str | None = None,
) -> Model:
    """Load a PyTorch model from disk, returned as a Model asset.

    Args:
        model_path: Path to the saved model.
        model_class: Model class for state_dict loading.
        device: Device to load model onto.

    Returns:
        Model asset wrapping the loaded PyTorch model.
    """
    persistence = ModelPersistence()
    raw_model = persistence.load(
        path=model_path, model_class=model_class, device=device
    )

    model_asset = Model.from_pytorch(
        raw_model,
        name="pytorch_loaded_model",
        source_path=model_path,
    )

    logger.info(f"🔍 Loaded PyTorch model from: {model_path}")
    return model_asset
