"""ZenML steps for Keras framework."""

import logging
from typing import Annotated, Any, Tuple
import keras

try:
    from zenml import log_step_metadata, step
except ImportError:
    raise ImportError(
        "ZenML is required for this module. "
        "Install it with: poetry install --extras zenml"
    )

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

logger = logging.getLogger(__name__)


@step
def load_data(
    file_path: str,
    batch_size: int = 32,
    label_name: str | None = None,
    column_names: list[str] | None = None,
    shuffle: bool = True,
    dtype: str = "float32",
    metadata: dict[str, Any] | None = None,
) -> Annotated[CSVSequence, "CSV Sequence"]:
    """Load data from CSV files into a Keras Sequence.

    This step uses `CSVDataLoader` to load data matching the specified file pattern.
    It returns a `CSVSequence` which can be used for training or evaluation.

    Args:
        file_path: Glob pattern for CSV files (e.g., "data/*.csv").
        batch_size: Number of samples per batch.
        label_name: Name of the column to use as the label.
        column_names: List of specific columns to load.
        shuffle: Whether to shuffle the data.
        dtype: Data type for the features (e.g., "float32").
        metadata: Optional dictionary of metadata to log to ZenML.

    Returns:
        CSVSequence: The loaded Keras Sequence.
    """
    logger.info(f"Loading data from: {file_path}")

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

    if metadata:
        log_step_metadata(metadata=metadata)

    return sequence


@step
def transform_data(
    dataset: CSVSequence,
    model: keras.Model,
    data_output_path: str,
    data_output_per_batch: bool = False,
    batch_size: int | None = None,
    feature_names: list[str] | None = None,
    input_columns: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Annotated[str, "Output Path"]:
    """Transform data using a Keras model and save predictions to CSV.

    This step uses `CSVDataTransformer` to run inference on a dataset using a provided model
    and saves the results to the specified output path.

    Args:
        dataset: The input dataset (`CSVSequence`).
        model: The Keras model to use for transformation.
        data_output_path: Path to save the transformed data (CSV).
        data_output_per_batch: Whether to save a separate file per batch.
        batch_size: Batch size for inference (overrides dataset batch size if provided).
        feature_names: Optional list of feature names for the output CSV.
        input_columns: Optional list of input columns to pass to the model.
        metadata: Optional dictionary of metadata to log to ZenML.

    Returns:
        str: The path to the saved output file(s).
    """
    logger.info(f"Transforming data and saving to: {data_output_path}")

    config = DataTransformationConfig(
        data_output_path=data_output_path,
        data_output_per_batch=data_output_per_batch,
        batch_size=batch_size,
        feature_names=feature_names,
        input_columns=input_columns,
    )

    transformer = CSVDataTransformer(
        dataset=dataset,
        model=model,
        data_output_path=data_output_path,
        data_output_per_batch=data_output_per_batch,
        batch_size=batch_size,
        feature_names=feature_names,
        input_columns=input_columns,
    )
    transformer.transform(dataset=dataset, model=model, config=config)

    if metadata:
        log_step_metadata(metadata=metadata)

    return data_output_path


@step
def train_model(
    model: keras.Model,
    data: CSVSequence,
    epochs: int = 10,
    validation_data: CSVSequence | None = None,
    learning_rate: float = 0.001,
    verbose: int = 1,
    callbacks: list[Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Tuple[Annotated[keras.Model, "Trained Model"], Annotated[dict[str, float], "Training Metrics"]]:
    """Train a Keras model using `ModelTrainer`.

    This step configures and runs a training session. It supports validation data,
    callbacks, and logging of training metrics.

    Args:
        model: The Keras model to train.
        data: The training dataset (`CSVSequence`).
        epochs: Number of epochs to train.
        validation_data: Optional validation dataset (`CSVSequence`).
        learning_rate: Learning rate for the Adam optimizer.
        verbose: Verbosity mode (0, 1, or 2).
        callbacks: List of Keras callbacks to apply during training.
        metadata: Optional dictionary of metadata to log to ZenML.

    Returns:
        Tuple[keras.Model, dict[str, float]]: The trained model and a dictionary of final metrics.
    """
    logger.info(f"Training model for {epochs} epochs")

    trainer = ModelTrainer()

    compile_params = {
        "optimizer": keras.optimizers.Adam(learning_rate=learning_rate),
        "loss": "mse",
        "metrics": ["mae"],
    }

    fit_params = {
        "epochs": epochs,
        "verbose": verbose,
        "validation_data": validation_data,
        "callbacks": callbacks,
    }

    history = trainer.train(
        model=model,
        data=data,
        compile_params=compile_params,
        fit_params=fit_params,
    )
    # Convert History → simple metrics dict (last epoch values)
    training_metrics: dict[str, float] = {}
    if hasattr(history, "history") and isinstance(history.history, dict):
        for key, values in history.history.items():
            if values:
                training_metrics[key] = float(values[-1])

    if metadata:
        log_step_metadata(metadata={**metadata, "history": history})

    return model, training_metrics


@step
def evaluate_model(
    model: keras.Model,
    data: CSVSequence,
    verbose: int = 1,
    metadata: dict[str, Any] | None = None,
) -> Annotated[dict[str, float], "Evaluation Metrics"]:
    """Evaluate a Keras model using `ModelEvaluator`.

    This step computes metrics on a given dataset using the provided model.

    Args:
        model: The Keras model to evaluate.
        data: The evaluation dataset (`CSVSequence`).
        verbose: Verbosity mode (0 or 1).
        metadata: Optional dictionary of metadata to log to ZenML.

    Returns:
        dict[str, float]: A dictionary of computed metrics.
    """
    logger.info("Evaluating model")

    evaluator = ModelEvaluator()

    eval_params = {
        "verbose": verbose,
    }

    metrics = evaluator.evaluate(
        model=model,
        data=data,
        eval_params=eval_params,
    )

    if metadata:
        log_step_metadata(metadata={**metadata, "metrics": metrics})

    return metrics


@step
def export_model(
    model: keras.Model,
    export_path: str,
    export_format: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Annotated[str, "Export Path"]:
    """Export a Keras model to disk using `ModelExporter`.

    This step exports the model to a specified format (e.g., SavedModel, H5, TFLite).

    Args:
        model: The Keras model to export.
        export_path: The destination path for the exported model.
        export_format: The format to export to (optional).
        metadata: Optional dictionary of metadata to log to ZenML.

    Returns:
        str: The path to the exported model artifact.
    """
    logger.info(f"Exporting model to: {export_path}")

    exporter = ModelExporter()

    config = {}
    if export_format:
        config["export_format"] = export_format

    exporter.export(
        model=model,
        path=export_path,
        **config,
    )

    if metadata:
        log_step_metadata(metadata={**metadata, "export_path": export_path})

    return export_path


@step
def save_model(
    model: keras.Model,
    save_path: str,
    metadata: dict[str, Any] | None = None,
) -> Annotated[str, "Save Path"]:
    """Save a Keras model to disk using `ModelPersistence`.

    This step saves the model for later reloading, typically preserving the optimizer state.

    Args:
        model: The Keras model to save.
        save_path: The destination path.
        metadata: Optional dictionary of metadata to log to ZenML.

    Returns:
        str: The path to the saved model.
    """
    logger.info(f"Saving model to: {save_path}")

    persistence = ModelPersistence(path=save_path, model=model)
    persistence.save()

    if metadata:
        log_step_metadata(metadata={**metadata, "save_path": save_path})

    return save_path


@step
def load_model(
    model_path: str,
    inspect: bool = True,
    metadata: dict[str, Any] | None = None,
) -> Annotated[keras.Model, "Loaded Model"]:
    """Load a Keras model from disk using `ModelPersistence`.

    This step loads a previously saved model. It can optionally inspect the loaded model
    to log metadata about its structure.

    Args:
        model_path: The path to the saved model.
        inspect: Whether to inspect the model after loading.
        metadata: Optional dictionary of metadata to log to ZenML.

    Returns:
        keras.Model: The loaded Keras model.
    """
    logger.info(f"Loading model from: {model_path}")

    persistence = ModelPersistence(path=model_path)
    model, inspection = persistence.load(inspect=inspect)

    if metadata:
        meta = {**metadata}
        if inspection:
            meta["inspection"] = inspection
        log_step_metadata(metadata=meta)

    return model


@step
def inspect_model(
    model: keras.Model,
    include_layers: bool = True,
    include_signatures: bool = True,
    metadata: dict[str, Any] | None = None,
) -> Annotated[dict[str, Any], "Model Inspection"]:
    """Inspect a Keras model using `ModelInspector`.

    This step extracts metadata about the model, such as layer configuration,
    input/output shapes, and parameter counts.

    Args:
        model: The Keras model to inspect.
        include_layers: Whether to include detailed layer information.
        include_signatures: Whether to include signature information.
        metadata: Optional dictionary of metadata to log to ZenML.

    Returns:
        dict[str, Any]: A dictionary containing the inspection results.
    """
    logger.info("Inspecting model")

    inspector = ModelInspector(
        include_layers=include_layers,
        include_signatures=include_signatures,
    )
    inspection = inspector.inspect(model)

    if metadata:
        log_step_metadata(metadata={**metadata, "inspection": inspection})

    return inspection
