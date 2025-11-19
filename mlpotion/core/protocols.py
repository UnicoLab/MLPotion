"""Framework-agnostic protocols using Python 3.10+ type hints.

These protocols define interfaces that work across TensorFlow, PyTorch, and other frameworks.
"""
from typing import Any, Protocol, TypeVar, runtime_checkable

from mlpotion.core.config import EvaluationConfig, ExportConfig, TrainingConfig
from mlpotion.core.results import EvaluationResult, ExportResult, TrainingResult

# Type variables for generic types
ModelT = TypeVar("ModelT")
DatasetT = TypeVar("DatasetT")
DataT = TypeVar("DataT")


@runtime_checkable
class DataLoader(Protocol[DatasetT]):
    """Protocol for data loading components.

    This protocol defines the interface for loading data into a framework-specific format.
    Any class that implements a `load()` method returning a dataset satisfies this protocol.

    Type Parameters:
        DatasetT: The type of the dataset returned (e.g., `tf.data.Dataset`, `torch.utils.data.DataLoader`).

    Example:
        ```python
        class MyCSVLoader:
            def load(self) -> tf.data.Dataset:
                # Implementation details...
                return dataset

        loader: DataLoader = MyCSVLoader()
        ```
    """

    def load(self) -> DatasetT:
        """Load data and return a framework-specific dataset.

        Returns:
            DatasetT: The loaded dataset in a framework-specific format.
        """
        ...


@runtime_checkable
class DatasetOptimizer(Protocol[DatasetT]):
    """Protocol for dataset optimization components.

    This protocol defines the interface for optimizing datasets, such as applying batching,
    caching, prefetching, or shuffling.

    Type Parameters:
        DatasetT: The type of the dataset to be optimized.
    """

    def optimize(self, dataset: DatasetT) -> DatasetT:
        """Optimize a dataset for training or inference.

        Args:
            dataset: The input dataset to optimize.

        Returns:
            DatasetT: The optimized dataset, ready for consumption by a model.
        """
        ...


@runtime_checkable
class DataTransformer(Protocol[DatasetT, ModelT]):
    """Protocol for data transformation components.

    This protocol defines the interface for transforming datasets using a model,
    commonly used for tasks like generating embeddings or pre-processing data.

    Type Parameters:
        DatasetT: The type of the dataset.
        ModelT: The type of the model used for transformation.
    """

    def transform(
        self,
        dataset: DatasetT,
        model: ModelT,
        **kwargs: Any,
    ) -> DatasetT:
        """Transform a dataset using a model.

        Args:
            dataset: The input dataset to transform.
            model: The model to use for the transformation.
            **kwargs: Additional framework-specific arguments (e.g., batch_size, device).

        Returns:
            DatasetT: The transformed dataset.
        """
        ...


@runtime_checkable
class ModelTrainer(Protocol[ModelT, DatasetT]):
    """Protocol for model training components.

    This protocol defines the standard interface for training models across different frameworks.

    Type Parameters:
        ModelT: The type of the model to be trained.
        DatasetT: The type of the dataset used for training.
    """

    def train(
        self,
        model: ModelT,
        dataset: DatasetT,
        config: TrainingConfig,
        validation_dataset: DatasetT | None = None,
    ) -> TrainingResult[ModelT]:
        """Train a model using the provided dataset and configuration.

        Args:
            model: The model instance to train.
            dataset: The training dataset.
            config: Configuration object containing training parameters (epochs, learning rate, etc.).
            validation_dataset: Optional dataset for validation during training.

        Returns:
            TrainingResult[ModelT]: An object containing the trained model, training history, and metrics.
        """
        ...


@runtime_checkable
class ModelEvaluator(Protocol[ModelT, DatasetT]):
    """Protocol for model evaluation components.

    This protocol defines the interface for evaluating models.

    Type Parameters:
        ModelT: The type of the model to be evaluated.
        DatasetT: The type of the dataset used for evaluation.
    """

    def evaluate(
        self,
        model: ModelT,
        dataset: DatasetT,
        config: EvaluationConfig,
    ) -> EvaluationResult:
        """Evaluate a model on a given dataset.

        Args:
            model: The model to evaluate.
            dataset: The dataset to evaluate on.
            config: Configuration object containing evaluation parameters.

        Returns:
            EvaluationResult: An object containing the evaluation metrics.
        """
        ...


@runtime_checkable
class ModelExporter(Protocol[ModelT]):
    """Protocol for model export components.

    This protocol defines the interface for exporting models to various formats for deployment.

    Type Parameters:
        ModelT: The type of the model to be exported.
    """

    def export(
        self,
        model: ModelT,
        config: ExportConfig,
    ) -> ExportResult:
        """Export a model for serving or deployment.

        Args:
            model: The model to export.
            config: Configuration object containing export parameters (path, format, etc.).

        Returns:
            ExportResult: An object containing details about the exported model.
        """
        ...


@runtime_checkable
class ModelPersistence(Protocol[ModelT]):
    """Protocol for model persistence operations (save/load).

    This protocol defines the interface for saving and loading models to/from disk.

    Type Parameters:
        ModelT: The type of the model to be persisted.
    """

    def save(self, model: ModelT, path: str, **kwargs: Any) -> None:
        """Save a model to the specified path.

        Args:
            model: The model to save.
            path: The destination path.
            **kwargs: Additional framework-specific saving options.
        """
        ...

    def load(self, path: str, **kwargs: Any) -> tuple[ModelT, dict[str, Any]]:
        """Load a model from the specified path.

        Args:
            path: The source path to load from.
            **kwargs: Additional framework-specific loading options.

        Returns:
            tuple[ModelT, dict[str, Any]]: A tuple containing the loaded model and a dictionary of metadata (e.g., inspection results).
        """
        ...

@runtime_checkable
class ModelExporterProtocol(Protocol[ModelT]):
    """Protocol for model export."""

    def export(self, model: ModelT, path: str, **kwargs: Any) -> None:
        """Export model to disk.

        Args:
            model: Model to export
            path: Path to export to
            **kwargs: Framework-specific options
        """
        ...

@runtime_checkable
class ModelInspectorProtocol(Protocol[ModelT]):
    """Protocol for model inspection."""

    def inspect(self, model: ModelT) -> dict[str, Any]:
        """Inspect model and return input and output signatures."""
        ...

@runtime_checkable
class ModelEvaluatorProtocol(Protocol[ModelT]):
    """Protocol for model evaluation."""

    def evaluate(self, model: ModelT, data: Any, **kwargs: Any) -> dict[str, float]:
        """Evaluate model on data.

        Args:
            model: Model to evaluate.
            data: Evaluation data (arrays, dataset-like, etc.).
            **kwargs: Framework-specific options.

        Returns:
            Mapping of metric names to values.
        """
        ...

@runtime_checkable
class ModelTrainerProtocol(Protocol[ModelT]):
    """Protocol for model training."""

    def train(self, model: ModelT, data: Any, **kwargs: Any) -> dict[str, list[float]]:
        """Train model on data.

        Args:
            model: Model to train.
            data: Training data (arrays, datasets, etc.).
            **kwargs: Framework-specific options (epochs, batch_size, etc.).

        Returns:
            History of metrics as a mapping: metric name -> list of values per epoch.
        """
        ...

@runtime_checkable
class DataTransformer(Protocol[DatasetT, ModelT]):
    """Protocol for data transformation using models."""

    def transform(
        self,
        dataset: DatasetT,
        model: ModelT,
        batch_size: int = 32,
    ) -> DatasetT:
        """Transform dataset using a model.

        Args:
            dataset: Input dataset
            model: Model to use for transformation
            batch_size: Batch size for transformation

        Returns:
            Transformed dataset
        """
        ...