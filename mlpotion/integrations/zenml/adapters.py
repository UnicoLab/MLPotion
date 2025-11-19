"""Generic ZenML adapters."""

from collections.abc import Callable
from typing import Any, TypeVar

from zenml import step

from mlpotion.core.protocols import (
    DataLoader,
    ModelEvaluator,
    ModelTrainer,
)

DatasetT = TypeVar("DatasetT")
ModelT = TypeVar("ModelT")


class ZenMLAdapter:
    """Adapter to convert MLPotion components to ZenML steps.

    This class provides static methods to automatically wrap MLPotion's core components
    (DataLoaders, ModelTrainers, ModelEvaluators) into ZenML steps. This allows for
    seamless integration of MLPotion's flexible components into ZenML pipelines without
    writing boilerplate step code.

    Example:
        ```python
        from zenml import pipeline
        from mlpotion.integrations.zenml.adapters import ZenMLAdapter
        from mlpotion.frameworks.tensorflow.data.loaders import CSVDataLoader
        from mlpotion.frameworks.tensorflow.training.trainers import ModelTrainer

        # 1. Create MLPotion components
        loader = CSVDataLoader(file_pattern="data/*.csv")
        trainer = ModelTrainer()

        # 2. Adapt to ZenML steps
        load_step = ZenMLAdapter.create_data_loader_step(loader)
        train_step = ZenMLAdapter.create_training_step(trainer)

        # 3. Define and run pipeline
        @pipeline
        def training_pipeline():
            dataset = load_step()
            train_step(model=my_model, dataset=dataset, config=train_config)
        ```
    """

    @staticmethod
    def create_data_loader_step(
        loader: DataLoader[DatasetT],
    ) -> Callable[[], DatasetT]:
        """Create a ZenML step from any `DataLoader`.

        This method wraps a `DataLoader` instance into a ZenML step function.
        The generated step takes no arguments and returns the dataset loaded by the loader.

        Args:
            loader: An instance implementing the `DataLoader` protocol.

        Returns:
            Callable[[], DatasetT]: A ZenML step function that returns the dataset.
        """

        @step
        def load_data() -> DatasetT:
            """Load data using the configured loader."""
            return loader.load()

        return load_data

    @staticmethod
    def create_training_step(
        trainer: ModelTrainer[ModelT, DatasetT],
    ) -> Callable[..., Any]:
        """Create a ZenML step from any `ModelTrainer`.

        This method wraps a `ModelTrainer` instance into a ZenML step function.
        The generated step accepts a model, a dataset, a configuration object, and optionally
        a validation dataset.

        Args:
            trainer: An instance implementing the `ModelTrainer` protocol.

        Returns:
            Callable[..., Any]: A ZenML step function that executes the training loop.
        """

        @step
        def train_model(
            model: ModelT,
            dataset: DatasetT,
            config: Any,
            validation_dataset: DatasetT | None = None,
        ) -> Any:
            """Train model using the configured trainer."""
            return trainer.train(model, dataset, config, validation_dataset)

        return train_model

    @staticmethod
    def create_evaluation_step(
        evaluator: ModelEvaluator[ModelT, DatasetT],
    ) -> Callable[..., Any]:
        """Create a ZenML step from any `ModelEvaluator`.

        This method wraps a `ModelEvaluator` instance into a ZenML step function.
        The generated step accepts a model, a dataset, and a configuration object.

        Args:
            evaluator: An instance implementing the `ModelEvaluator` protocol.

        Returns:
            Callable[..., Any]: A ZenML step function that executes the evaluation loop.
        """

        @step
        def evaluate_model(
            model: ModelT,
            dataset: DatasetT,
            config: Any,
        ) -> Any:
            """Evaluate model using the configured evaluator."""
            return evaluator.evaluate(model, dataset, config)

        return evaluate_model