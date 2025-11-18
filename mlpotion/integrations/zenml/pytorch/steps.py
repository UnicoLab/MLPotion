"""ZenML steps for PyTorch framework."""

import logging
from typing import Annotated, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from zenml import log_step_metadata, step
except ImportError:
    raise ImportError(
        "ZenML is required for this module. "
        "Install it with: poetry install --extras zenml"
    )

from mlpotion.frameworks.pytorch.config import (
    DataLoadingConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig,
    ModelExportConfig,
)
from mlpotion.frameworks.pytorch.data.datasets import (
    PyTorchCSVDataset,
    StreamingPyTorchCSVDataset,
)
from mlpotion.frameworks.pytorch.data.loaders import PyTorchDataLoaderFactory
from mlpotion.frameworks.pytorch.training.trainers import PyTorchModelTrainer
from mlpotion.frameworks.pytorch.evaluation.evaluators import PyTorchModelEvaluator
from mlpotion.frameworks.pytorch.deployment.exporters import PyTorchModelExporter
from mlpotion.frameworks.pytorch.deployment.persistence import PyTorchModelPersistence

logger = logging.getLogger(__name__)


@step
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
    metadata: dict[str, Any] | None = None,
) -> Annotated[DataLoader, "PyTorch DataLoader"]:
    """Load data from CSV files into PyTorch DataLoader."""
    logger.info(f"Loading data from: {file_path}")

    # Convert dtype string to torch.dtype
    torch_dtype = getattr(torch, dtype)

    # Create dataset
    dataset = PyTorchCSVDataset(
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

    # Create DataLoader using factory (exclude fields not accepted by PyTorchDataLoaderFactory)
    loader_factory = PyTorchDataLoaderFactory(**config.dict(exclude={"file_pattern", "config"}))
    dataloader = loader_factory.load(dataset)

    if metadata:
        log_step_metadata(metadata=metadata)

    return dataloader


@step
def load_streaming_csv_data(
    file_path: str,
    batch_size: int = 32,
    label_name: str | None = None,
    column_names: list[str] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    chunksize: int = 10000,
    dtype: str = "float32",
    metadata: dict[str, Any] | None = None,
) -> Annotated[DataLoader, "PyTorch DataLoader"]:
    """Load large CSV files as streaming PyTorch DataLoader."""
    logger.info(f"Loading streaming data from: {file_path}")

    # Convert dtype string to torch.dtype
    torch_dtype = getattr(torch, dtype)

    # Create streaming dataset
    dataset = StreamingPyTorchCSVDataset(
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
        shuffle=False,  # Streaming datasets don't support shuffle
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # Create DataLoader using factory (exclude fields not accepted by PyTorchDataLoaderFactory)
    loader_factory = PyTorchDataLoaderFactory(**config.dict(exclude={"file_pattern", "config"}))
    dataloader = loader_factory.load(dataset)

    if metadata:
        log_step_metadata(metadata=metadata)

    return dataloader


@step
def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 0.001,
    optimizer: str = "adam",
    loss_fn: str = "mse",
    device: str = "cpu",
    validation_dataloader: DataLoader | None = None,
    verbose: int = 1,
    max_batches_per_epoch: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Annotated[nn.Module, "Trained Model"]:
    """Train a PyTorch model."""
    logger.info(f"Training model for {epochs} epochs on {device}")

    config = ModelTrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        verbose=verbose,
        max_batches_per_epoch=max_batches_per_epoch,
    )

    trainer = PyTorchModelTrainer()
    result = trainer.train(
        model=model,
        dataloader=dataloader,
        config=config,
        validation_dataloader=validation_dataloader,
    )

    if metadata:
        log_step_metadata(
            metadata={
                **metadata,
                "history": result.history,
                "best_epoch": result.best_epoch,
                "final_metrics": result.metrics,
            }
        )

    return result.model


@step
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: str = "mse",
    device: str = "cpu",
    verbose: int = 1,
    max_batches: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Annotated[dict[str, float], "Evaluation Metrics"]:
    """Evaluate a PyTorch model."""
    logger.info(f"Evaluating model on {device}")

    config = ModelEvaluationConfig(
        batch_size=dataloader.batch_size or 32,
        verbose=verbose,
        device=device,
        framework_options={"loss_fn": loss_fn, "max_batches": max_batches},
    )

    evaluator = PyTorchModelEvaluator()
    result = evaluator.evaluate(
        model=model,
        dataloader=dataloader,
        config=config,
    )

    # Extract metrics and evaluation time from result
    metrics = {**result.metrics, "evaluation_time": result.evaluation_time}

    if metadata:
        log_step_metadata(metadata={**metadata, "metrics": metrics})

    return metrics


@step
def export_model(
    model: nn.Module,
    export_path: str,
    export_format: str = "state_dict",
    device: str = "cpu",
    example_input: torch.Tensor | None = None,
    jit_mode: str = "script",
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    opset_version: int = 14,
    metadata: dict[str, Any] | None = None,
) -> Annotated[str, "Export Path"]:
    """Export a PyTorch model to disk in various formats (TorchScript, ONNX, state_dict)."""
    logger.info(f"Exporting model to: {export_path} (format: {export_format})")

    config = ModelExportConfig(
        export_path=export_path,
        format=export_format,
        device=device,
        jit_mode=jit_mode,
        example_input=example_input,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
    )

    exporter = PyTorchModelExporter()
    result = exporter.export(model=model, config=config)

    if metadata:
        log_step_metadata(
            metadata={
                **metadata,
                "export_path": result.export_path,
                "format": result.format,
                "metadata": result.metadata,
            }
        )

    return str(result.export_path)


@step
def save_model(
    model: nn.Module,
    save_path: str,
    save_full_model: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Annotated[str, "Save Path"]:
    """Save a PyTorch model to disk (state_dict or full model)."""
    logger.info(f"Saving model to: {save_path}")

    persistence = PyTorchModelPersistence(path=save_path, model=model)
    persistence.save(save_full_model=save_full_model)

    if metadata:
        log_step_metadata(
            metadata={
                **metadata,
                "save_path": save_path,
                "save_full_model": save_full_model,
            }
        )

    return save_path


@step
def load_model(
    model_path: str,
    model_class: type[nn.Module] | None = None,
    map_location: str = "cpu",
    strict: bool = True,
    metadata: dict[str, Any] | None = None,
) -> Annotated[nn.Module, "Loaded Model"]:
    """Load a PyTorch model from disk."""
    logger.info(f"Loading model from: {model_path}")

    persistence = PyTorchModelPersistence(path=model_path)
    model = persistence.load(
        model_class=model_class,
        map_location=map_location,
        strict=strict,
    )

    if metadata:
        log_step_metadata(metadata={**metadata, "model_path": model_path})

    return model
