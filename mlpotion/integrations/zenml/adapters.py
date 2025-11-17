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

    Example:
        # Create business logic
        loader = TFCSVDataLoader("data/*.csv")
        trainer = TFModelTrainer()

        # Adapt to ZenML
        load_step = ZenMLAdapter.create_data_loader_step(loader)
        train_step = ZenMLAdapter.create_training_step(trainer)

        # Use in pipeline
        @pipeline
        def ml_pipeline():
            dataset = load_step()
            result = train_step(model, dataset, config)
    """

    @staticmethod
    def create_data_loader_step(
        loader: DataLoader[DatasetT],
    ) -> Callable[[], DatasetT]:
        """Create ZenML step from any DataLoader.

        Args:
            loader: Any object implementing DataLoader protocol

        Returns:
            ZenML step function
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
        """Create ZenML step from any ModelTrainer.

        Args:
            trainer: Any object implementing ModelTrainer protocol

        Returns:
            ZenML step function
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
        """Create ZenML step from any ModelEvaluator.

        Args:
            evaluator: Any object implementing ModelEvaluator protocol

        Returns:
            ZenML step function
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