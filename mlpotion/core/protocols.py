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

    Any class implementing load() satisfies this protocol.

    Example:
        class CSVLoader:
            def load(self) -> tf.data.Dataset:
                return dataset

        loader: DataLoader = CSVLoader()  # Type checks!
    """

    def load(self) -> DatasetT:
        """Load data and return framework-specific dataset.

        Returns:
            Dataset in framework-specific format (tf.data.Dataset, DataLoader, etc.)
        """
        ...


@runtime_checkable
class DatasetOptimizer(Protocol[DatasetT]):
    """Protocol for dataset optimization components."""

    def optimize(self, dataset: DatasetT) -> DatasetT:
        """Optimize dataset for training/inference.

        Args:
            dataset: Input dataset to optimize

        Returns:
            Optimized dataset with batching, prefetching, etc.
        """
        ...


@runtime_checkable
class DataTransformer(Protocol[DatasetT, ModelT]):
    """Protocol for data transformation components."""

    def transform(
        self,
        dataset: DatasetT,
        model: ModelT,
        **kwargs: Any,
    ) -> DatasetT:
        """Transform dataset using a model.
        
        Args:
            dataset: Input dataset
            model: Model to use for transformation
            **kwargs: Extra arguments

        Returns:
            Transformed dataset
        """
        ...


@runtime_checkable
class ModelTrainer(Protocol[ModelT, DatasetT]):
    """Protocol for model training components.

    Type-safe across frameworks using generics.
    """

    def train(
        self,
        model: ModelT,
        dataset: DatasetT,
        config: TrainingConfig,
        validation_dataset: DatasetT | None = None,
    ) -> TrainingResult[ModelT]:
        """Train a model.

        Args:
            model: Model to train (framework-specific type)
            dataset: Training dataset
            config: Training configuration
            validation_dataset: Optional validation dataset

        Returns:
            Training result with trained model and metrics
        """
        ...


@runtime_checkable
class ModelEvaluator(Protocol[ModelT, DatasetT]):
    """Protocol for model evaluation components."""

    def evaluate(
        self,
        model: ModelT,
        dataset: DatasetT,
        config: EvaluationConfig,
    ) -> EvaluationResult:
        """Evaluate a model.

        Args:
            model: Model to evaluate
            dataset: Evaluation dataset
            config: Evaluation configuration

        Returns:
            Evaluation result with metrics
        """
        ...


@runtime_checkable
class ModelExporter(Protocol[ModelT]):
    """Protocol for model export components."""

    def export(
        self,
        model: ModelT,
        config: ExportConfig,
    ) -> ExportResult:
        """Export model for serving/deployment.

        Args:
            model: Model to export
            config: Export configuration

        Returns:
            Export result with path and metadata
        """
        ...


@runtime_checkable
class ModelPersistence(Protocol[ModelT]):
    """Protocol for model save/load operations."""

    def save(self, model: ModelT, path: str, **kwargs: Any) -> None:
        """Save model to disk.

        Args:
            model: Model to save
            path: Path to save to
            **kwargs: Framework-specific options
        """
        ...

    def load(self, path: str, **kwargs: Any) -> tuple[ModelT, dict[str, Any]]:
        """Load model from disk.

        Args:
            path: Path to load from
            **kwargs: Framework-specific options

        Returns:
            Loaded model and inspection result
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