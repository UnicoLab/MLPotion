import logging
from typing import Annotated, Any
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
)
from mlpotion.frameworks.tensorflow.data.loaders import TFCSVDataLoader
from mlpotion.frameworks.tensorflow.data.optimizers import TFDatasetOptimizer
from mlpotion.frameworks.tensorflow.data.transformers import TFDataToCSVTransformer
from mlpotion.frameworks.tensorflow.training.trainers import TFModelTrainer
from mlpotion.frameworks.tensorflow.evaluation.evaluators import TFModelEvaluator
from mlpotion.frameworks.tensorflow.deployment.exporters import TFModelExporter
from mlpotion.frameworks.tensorflow.deployment.persistence import TFModelPersistence
from mlpotion.frameworks.tensorflow.models.inspection import TFModelInspector

logger = logging.getLogger(__name__)


@step
def load_data(
    file_path: str,
    batch_size: int = 32,
    label_name: str = "target",
    column_names: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Annotated[tf.data.Dataset, "TF DataSet"]:
    """Load data from local CSV files using TensorFlow's efficient loading."""
    logger.info(f"Loading data from: {file_path}")

    # defining configuration
    config = DataLoadingConfig(
        file_pattern=file_path,
        batch_size=batch_size,
        label_name=label_name,
        column_names=column_names,
    )

    # initializing data loader
    loader = TFCSVDataLoader(**config.dict())
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
) -> Annotated[tf.data.Dataset, "TF DataSet"]:
    """Optimize TensorFlow dataset for training performance."""
    logger.info("Optimizing dataset for training performance")

    config = DataOptimizationConfig(
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        prefetch=prefetch,
        cache=cache,
    )

    optimizer = TFDatasetOptimizer(**config.dict())
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
) -> Annotated[str, "Output Path"]:
    """Transform data using a TensorFlow model and save predictions to CSV."""
    logger.info(f"Transforming data and saving to: {data_output_path}")

    transformer = TFDataToCSVTransformer(
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
) -> Annotated[keras.Model, "Trained Model"]:
    """Train a TensorFlow/Keras model."""
    logger.info(f"Training model for {epochs} epochs")

    trainer = TFModelTrainer()

    compile_params = {
        "optimizer": keras.optimizers.Adam(learning_rate=learning_rate),
        "loss": "mse",
        "metrics": ["mae"],
    }

    fit_params = {
        "epochs": epochs,
        "verbose": verbose,
        "validation_data": validation_dataset,
    }

    history = trainer.train(
        model=model,
        data=dataset,
        compile_params=compile_params,
        fit_params=fit_params,
    )

    if metadata:
        log_step_metadata(metadata={**metadata, "history": history})

    return model


@step
def evaluate_model(
    model: keras.Model,
    dataset: tf.data.Dataset,
    verbose: int = 1,
    metadata: dict[str, Any] | None = None,
) -> Annotated[dict[str, float], "Evaluation Metrics"]:
    """Evaluate a TensorFlow/Keras model."""
    logger.info("Evaluating model")

    evaluator = TFModelEvaluator()

    eval_params = {
        "verbose": verbose,
    }

    metrics = evaluator.evaluate(
        model=model,
        data=dataset,
        eval_params=eval_params,
    )

    if metadata:
        log_step_metadata(metadata={**metadata, "metrics": metrics})

    return metrics


@step
def export_model(
    model: keras.Model,
    export_path: str,
    export_format: str = "keras",
    metadata: dict[str, Any] | None = None,
) -> Annotated[str, "Export Path"]:
    """Export a TensorFlow/Keras model to disk."""
    logger.info(f"Exporting model to: {export_path}")

    exporter = TFModelExporter()

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
) -> Annotated[str, "Save Path"]:
    """Save a TensorFlow/Keras model to disk."""
    logger.info(f"Saving model to: {save_path}")

    persistence = TFModelPersistence(path=save_path, model=model)
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
    """Load a TensorFlow/Keras model from disk."""
    logger.info(f"Loading model from: {model_path}")

    persistence = TFModelPersistence(path=model_path)
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
    """Inspect a TensorFlow/Keras model."""
    logger.info("Inspecting model")

    inspector = TFModelInspector(
        include_layers=include_layers,
        include_signatures=include_signatures,
    )
    inspection = inspector.inspect(model)

    if metadata:
        log_step_metadata(metadata={**metadata, "inspection": inspection})

    return inspection