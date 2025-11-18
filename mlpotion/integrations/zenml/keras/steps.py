"""ZenML steps for Keras framework."""

import logging
from typing import Annotated, Any
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
from mlpotion.frameworks.keras.training.trainers import KerasModelTrainer
from mlpotion.frameworks.keras.evaluation.evaluators import KerasModelEvaluator
from mlpotion.frameworks.keras.deployment.exporters import KerasModelExporter
from mlpotion.frameworks.keras.deployment.persistence import KerasModelPersistence
from mlpotion.frameworks.keras.models.inspection import KerasModelInspector

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
    """Load data from CSV files into Keras Sequence."""
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
    """Transform data using a Keras model and save predictions to CSV."""
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
) -> Annotated[keras.Model, "Trained Model"]:
    """Train a Keras model."""
    logger.info(f"Training model for {epochs} epochs")

    trainer = KerasModelTrainer()

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

    if metadata:
        log_step_metadata(metadata={**metadata, "history": history})

    return model


@step
def evaluate_model(
    model: keras.Model,
    data: CSVSequence,
    verbose: int = 1,
    metadata: dict[str, Any] | None = None,
) -> Annotated[dict[str, float], "Evaluation Metrics"]:
    """Evaluate a Keras model."""
    logger.info("Evaluating model")

    evaluator = KerasModelEvaluator()

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
    """Export a Keras model to disk."""
    logger.info(f"Exporting model to: {export_path}")

    exporter = KerasModelExporter()

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
    """Save a Keras model to disk."""
    logger.info(f"Saving model to: {save_path}")

    persistence = KerasModelPersistence(path=save_path, model=model)
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
    """Load a Keras model from disk."""
    logger.info(f"Loading model from: {model_path}")

    persistence = KerasModelPersistence(path=model_path)
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
    """Inspect a Keras model."""
    logger.info("Inspecting model")

    inspector = KerasModelInspector(
        include_layers=include_layers,
        include_signatures=include_signatures,
    )
    inspection = inspector.inspect(model)

    if metadata:
        log_step_metadata(metadata={**metadata, "inspection": inspection})

    return inspection
