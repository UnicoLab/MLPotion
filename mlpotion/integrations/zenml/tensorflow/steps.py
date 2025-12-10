import logging
from typing import Annotated, Any, Tuple
import tensorflow as tf
import keras

try:
    from zenml import log_step_metadata, step
except ImportError:
    raise ImportError(
        "ZenML is required for this module. "
        "Install it with: poetry install --extras zenml"
    )

# loading
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

logger = logging.getLogger(__name__)


@step
def load_data(
    file_path: str,
    batch_size: int = 32,
    label_name: str = "target",
    column_names: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Annotated[tf.data.Dataset, "TFDataset"]:
    """Load data from local CSV files using TensorFlow's efficient loading.

    This step uses `CSVDataLoader` to create a `tf.data.Dataset` from CSV files matching
    the specified pattern.

    Args:
        file_path: Glob pattern for CSV files (e.g., "data/*.csv").
        batch_size: Number of samples per batch.
        label_name: Name of the column to use as the label.
        column_names: List of specific columns to load.
        metadata: Optional dictionary of metadata to log to ZenML.

    Returns:
        tf.data.Dataset: The loaded TensorFlow dataset.
    """
    logger.info(f"Loading data from: {file_path}")

    # defining configuration
    config = DataLoadingConfig(
        file_pattern=file_path,
        batch_size=batch_size,
        label_name=label_name,
        column_names=column_names,
    )

    # initializing data loader
    loader = CSVDataLoader(**config.dict())
    # loading data
    dataset = loader.load()

    # adding metadata
    if metadata:
        log_step_metadata(metadata=metadata)

    return dataset


@step
def optimize_data(
    dataset: tf.data.Dataset,
    batch_size: int = 32,
    shuffle_buffer_size: int | None = None,
    prefetch: bool = True,
    cache: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Annotated[tf.data.Dataset, "TFDataset"]:
    """Optimize a TensorFlow dataset for training performance.

    This step applies optimizations like caching, shuffling, and prefetching to the dataset
    using `DatasetOptimizer`.

    Args:
        dataset: The input `tf.data.Dataset`.
        batch_size: Batch size (if re-batching is needed).
        shuffle_buffer_size: Size of the shuffle buffer.
        prefetch: Whether to prefetch data.
        cache: Whether to cache data in memory.
        metadata: Optional dictionary of metadata to log to ZenML.

    Returns:
        tf.data.Dataset: The optimized TensorFlow dataset.
    """
    logger.info("Optimizing dataset for training performance")

    config = DataOptimizationConfig(
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        prefetch=prefetch,
        cache=cache,
    )

    optimizer = DatasetOptimizer(**config.dict())
    dataset = optimizer.optimize(dataset)

    # adding metadata
    if metadata:
        log_step_metadata(metadata=metadata)

    return dataset


@step
def transform_data(
    dataset: tf.data.Dataset,
    model: keras.Model,
    data_output_path: str,
    data_output_per_batch: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Annotated[str, "OutputPath"]:
    """Transform data using a TensorFlow model and save predictions to CSV.

    This step uses `DataToCSVTransformer` to run inference on a dataset using a provided model
    and saves the results to the specified output path.

    Args:
        dataset: The input `tf.data.Dataset`.
        model: The Keras model to use for transformation.
        data_output_path: Path to save the transformed data (CSV).
        data_output_per_batch: Whether to save a separate file per batch.
        metadata: Optional dictionary of metadata to log to ZenML.

    Returns:
        str: The path to the saved output file(s).
    """
    logger.info(f"Transforming data and saving to: {data_output_path}")

    transformer = DataToCSVTransformer(
        dataset=dataset,
        model=model,
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

    if metadata:
        log_step_metadata(metadata=metadata)

    return data_output_path


@step
def train_model(
    model: keras.Model,
    dataset: tf.data.Dataset,
    epochs: int = 10,
    validation_dataset: tf.data.Dataset | None = None,
    learning_rate: float = 0.001,
    verbose: int = 1,
    metadata: dict[str, Any] | None = None,
) -> Tuple[
    Annotated[keras.Model, "TrainedModel"],
    Annotated[dict[str, list[float]], "TrainingHistory"],
]:
    """Train a TensorFlow/Keras model using `ModelTrainer`.

    This step configures and runs a training session. It supports validation data
    and logging of training metrics.

    Args:
        model: The Keras model to train.
        dataset: The training `tf.data.Dataset`.
        epochs: Number of epochs to train.
        validation_dataset: Optional validation `tf.data.Dataset`.
        learning_rate: Learning rate for the Adam optimizer.
        verbose: Verbosity mode (0, 1, or 2).
        metadata: Optional dictionary of metadata to log to ZenML.

    Returns:
        Tuple[keras.Model, dict[str, list[float]]]: The trained model and a dictionary of history metrics.
    """
    logger.info(f"Training model for {epochs} epochs")

    trainer = ModelTrainer()

    config = ModelTrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        verbose=verbose,
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )

    result = trainer.train(
        model=model,
        dataset=dataset,
        config=config,
        validation_dataset=validation_dataset,
    )

    # Result is TrainingResult object
    training_metrics = result.metrics

    if metadata:
        log_step_metadata(metadata={**metadata, "history": result.history})

    return model, training_metrics


@step
def evaluate_model(
    model: keras.Model,
    dataset: tf.data.Dataset,
    verbose: int = 1,
    metadata: dict[str, Any] | None = None,
) -> Annotated[dict[str, float], "EvaluationMetrics"]:
    """Evaluate a TensorFlow/Keras model using `ModelEvaluator`.

    This step computes metrics on a given dataset using the provided model.

    Args:
        model: The Keras model to evaluate.
        dataset: The evaluation `tf.data.Dataset`.
        verbose: Verbosity mode (0 or 1).
        metadata: Optional dictionary of metadata to log to ZenML.

    Returns:
        dict[str, float]: A dictionary of computed metrics.
    """
    logger.info("Evaluating model")

    evaluator = ModelEvaluator()

    config = ModelEvaluationConfig(
        verbose=verbose,
    )

    result = evaluator.evaluate(
        model=model,
        dataset=dataset,
        config=config,
    )

    metrics = result.metrics

    if metadata:
        log_step_metadata(metadata={**metadata, "metrics": metrics})

    return metrics


@step
def export_model(
    model: keras.Model,
    export_path: str,
    export_format: str = "keras",
    metadata: dict[str, Any] | None = None,
) -> Annotated[str, "ExportPath"]:
    """Export a TensorFlow/Keras model to disk using `ModelExporter`.

    This step exports the model to a specified format (e.g., Keras format, SavedModel).

    Args:
        model: The Keras model to export.
        export_path: The destination path for the exported model.
        export_format: The format to export to (default: "keras").
        metadata: Optional dictionary of metadata to log to ZenML.

    Returns:
        str: The path to the exported model artifact.
    """
    logger.info(f"Exporting model to: {export_path}")

    exporter = ModelExporter()

    exporter.export(
        model=model,
        path=export_path,
        export_format=export_format,
    )

    if metadata:
        log_step_metadata(metadata={**metadata, "export_path": export_path})

    return export_path


@step
def save_model(
    model: keras.Model,
    save_path: str,
    metadata: dict[str, Any] | None = None,
) -> Annotated[str, "SavePath"]:
    """Save a TensorFlow/Keras model to disk using `ModelPersistence`.

    This step saves the model for later reloading.

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
) -> Annotated[keras.Model, "LoadedModel"]:
    """Load a TensorFlow/Keras model from disk using `ModelPersistence`.

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
) -> Annotated[dict[str, Any], "ModelInspection"]:
    """Inspect a TensorFlow/Keras model using `ModelInspector`.

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
