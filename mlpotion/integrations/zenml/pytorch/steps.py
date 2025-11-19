"""ZenML steps for PyTorch framework."""

import logging
from typing import Annotated, Any, Tuple
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
    CSVDataset,
    StreamingCSVDataset,
)
from mlpotion.frameworks.pytorch.data.loaders import CSVDataLoader
from mlpotion.frameworks.pytorch.training.trainers import ModelTrainer
from mlpotion.frameworks.pytorch.evaluation.evaluators import ModelEvaluator
from mlpotion.frameworks.pytorch.deployment.exporters import ModelExporter
from mlpotion.frameworks.pytorch.deployment.persistence import ModelPersistence

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
) -> Annotated[DataLoader, "PyTorchDataLoader"]:
    """Load data from CSV files into a PyTorch DataLoader.

    This step uses `CSVDataset` and `CSVDataLoader` to load data matching the specified file pattern.
    It returns a configured `DataLoader` ready for training or evaluation.

    Args:
        file_path: Glob pattern for CSV files (e.g., "data/*.csv").
        batch_size: Number of samples per batch.
        label_name: Name of the column to use as the label.
        column_names: List of specific columns to load.
        shuffle: Whether to shuffle the data.
        num_workers: Number of subprocesses to use for data loading.
        pin_memory: Whether to copy tensors into CUDA pinned memory.
        drop_last: Whether to drop the last incomplete batch.
        dtype: Data type for the features (e.g., "float32").
        metadata: Optional dictionary of metadata to log to ZenML.

    Returns:
        DataLoader: The configured PyTorch DataLoader.
    """
    logger.info(f"Loading data from: {file_path}")

    # Convert dtype string to torch.dtype
    torch_dtype = getattr(torch, dtype)

    # Create dataset
    dataset = CSVDataset(
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

    # Create DataLoader using factory (exclude fields not accepted by CSVDataLoader)
    loader_factory = CSVDataLoader(**config.dict(exclude={"file_pattern", "config"}))
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
) -> Annotated[DataLoader, "PyTorchDataLoader"]:
    """Load large CSV files as a streaming PyTorch DataLoader.

    This step uses `StreamingCSVDataset` to load data in chunks, making it suitable for
    datasets that do not fit in memory. It returns a `DataLoader` wrapping the iterable dataset.

    Args:
        file_path: Glob pattern for CSV files (e.g., "data/*.csv").
        batch_size: Number of samples per batch.
        label_name: Name of the column to use as the label.
        column_names: List of specific columns to load.
        num_workers: Number of subprocesses to use for data loading.
        pin_memory: Whether to copy tensors into CUDA pinned memory.
        chunksize: Number of rows to read into memory at a time per file.
        dtype: Data type for the features (e.g., "float32").
        metadata: Optional dictionary of metadata to log to ZenML.

    Returns:
        DataLoader: The configured streaming PyTorch DataLoader.
    """
    logger.info(f"Loading streaming data from: {file_path}")

    # Convert dtype string to torch.dtype
    torch_dtype = getattr(torch, dtype)

    # Create streaming dataset
    dataset = StreamingCSVDataset(
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

    # Create DataLoader using factory (exclude fields not accepted by CSVDataLoader)
    loader_factory = CSVDataLoader(**config.dict(exclude={"file_pattern", "config"}))
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
) -> Tuple[Annotated[nn.Module, "TrainedModel"], Annotated[dict[str, float], "TrainingMetrics"]]:
    """Train a PyTorch model using `ModelTrainer`.

    This step configures and runs a training session. It supports validation data,
    custom loss functions, and automatic device management.

    Args:
        model: The PyTorch model to train.
        dataloader: The training `DataLoader`.
        epochs: Number of epochs to train.
        learning_rate: Learning rate for the optimizer.
        optimizer: Name of the optimizer (e.g., "adam", "sgd").
        loss_fn: Name of the loss function (e.g., "mse", "cross_entropy").
        device: Device to train on ("cpu" or "cuda").
        validation_dataloader: Optional validation `DataLoader`.
        verbose: Verbosity mode (0 or 1).
        max_batches_per_epoch: Limit number of batches per epoch (useful for debugging).
        metadata: Optional dictionary of metadata to log to ZenML.

    Returns:
        Tuple[nn.Module, dict[str, float]]: The trained model and a dictionary of final metrics.
    """
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

    trainer = ModelTrainer()
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
    logger.info(f"{result=}")
    model = result.model
    metrics = result.metrics

    return model, metrics


@step
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: str = "mse",
    device: str = "cpu",
    verbose: int = 1,
    max_batches: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Annotated[dict[str, float], "EvaluationMetrics"]:
    """Evaluate a PyTorch model using `ModelEvaluator`.

    This step computes metrics on a given dataset using the provided model.

    Args:
        model: The PyTorch model to evaluate.
        dataloader: The evaluation `DataLoader`.
        loss_fn: Name of the loss function (e.g., "mse", "cross_entropy").
        device: Device to evaluate on ("cpu" or "cuda").
        verbose: Verbosity mode (0 or 1).
        max_batches: Limit number of batches to evaluate (useful for debugging).
        metadata: Optional dictionary of metadata to log to ZenML.

    Returns:
        dict[str, float]: A dictionary of computed metrics.
    """
    logger.info(f"Evaluating model on {device}")

    config = ModelEvaluationConfig(
        batch_size=dataloader.batch_size or 32,
        verbose=verbose,
        device=device,
        framework_options={"loss_fn": loss_fn, "max_batches": max_batches},
    )

    evaluator = ModelEvaluator()
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
) -> Annotated[str, "ExportPath"]:
    """Export a PyTorch model to disk using `ModelExporter`.

    This step exports the model to a specified format (TorchScript, ONNX, or state_dict).

    Args:
        model: The PyTorch model to export.
        export_path: The destination path for the exported model.
        export_format: The format to export to ("torchscript", "onnx", "state_dict").
        device: Device to use for export (important for tracing).
        example_input: Example input tensor (required for ONNX and TorchScript trace).
        jit_mode: TorchScript mode ("script" or "trace").
        input_names: List of input names for ONNX export.
        output_names: List of output names for ONNX export.
        dynamic_axes: Dictionary of dynamic axes for ONNX export.
        opset_version: ONNX opset version.
        metadata: Optional dictionary of metadata to log to ZenML.

    Returns:
        str: The path to the exported model artifact.
    """
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

    exporter = ModelExporter()
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
) -> Annotated[str, "SavePath"]:
    """Save a PyTorch model to disk using `ModelPersistence`.

    This step saves the model for later reloading. It supports saving just the state dict
    (recommended) or the full model object.

    Args:
        model: The PyTorch model to save.
        save_path: The destination path.
        save_full_model: Whether to save the full model object (pickle) instead of state dict.
        metadata: Optional dictionary of metadata to log to ZenML.

    Returns:
        str: The path to the saved model.
    """
    logger.info(f"Saving model to: {save_path}")

    persistence = ModelPersistence(path=save_path, model=model)
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
) -> Annotated[nn.Module, "LoadedModel"]:
    """Load a PyTorch model from disk using `ModelPersistence`.

    This step loads a previously saved model. If loading a state dict, `model_class`
    must be provided.

    Args:
        model_path: The path to the saved model.
        model_class: The class of the model (required for state dict loading).
        map_location: Device to load the model onto.
        strict: Whether to strictly enforce state dict keys match.
        metadata: Optional dictionary of metadata to log to ZenML.

    Returns:
        nn.Module: The loaded PyTorch model.
    """
    logger.info(f"Loading model from: {model_path}")

    persistence = ModelPersistence(path=model_path)
    model, _ = persistence.load(
        model_class=model_class,
        map_location=map_location,
        strict=strict,
    )

    if metadata:
        log_step_metadata(metadata={**metadata, "model_path": model_path})

    return model
