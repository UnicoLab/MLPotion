"""FlowyML Adapter — Bridge MLPotion protocols to FlowyML steps.

Provides factory methods that wrap MLPotion's protocol-compliant components
(DataLoader, ModelTrainer, ModelEvaluator) into fully-configured FlowyML
Step objects with asset outputs, caching, retry, resource specs, and tags.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from flowyml.assets.dataset import Dataset
from flowyml.assets.metrics import Metrics
from flowyml.assets.model import Model
from flowyml.core.step import Step, step

from mlpotion.core.protocols import DataLoader, ModelEvaluator, ModelTrainer

DatasetT = TypeVar("DatasetT")
ModelT = TypeVar("ModelT")


class FlowyMLAdapter:
    """Adapt MLPotion components into FlowyML pipeline steps.

    Unlike the ZenML adapter which returns raw objects, these steps return
    FlowyML Asset objects (Dataset, Model, Metrics) with automatic metadata
    extraction and lineage tracking.

    Example::

        from mlpotion.frameworks.keras.data.loaders import CSVDataLoader
        from mlpotion.integrations.flowyml import FlowyMLAdapter

        loader = CSVDataLoader(file_path="data.csv", batch_size=32)
        load_step = FlowyMLAdapter.create_data_loader_step(loader)

        # Use in a FlowyML pipeline
        pipeline = Pipeline("my_pipeline")
        pipeline.add_step(load_step)
    """

    @staticmethod
    def create_data_loader_step(
        loader: DataLoader[DatasetT],
        *,
        name: str | None = None,
        cache: bool | str | Callable = "code_hash",
        retry: int = 0,
        resources: Any | None = None,
        tags: dict[str, str] | None = None,
    ) -> Step:
        """Wrap a DataLoader as a FlowyML step returning a Dataset asset.

        Args:
            loader: Any MLPotion DataLoader protocol implementation.
            name: Step name (defaults to 'load_data').
            cache: Caching strategy ('code_hash', 'input_hash', False).
            retry: Number of retry attempts on failure.
            resources: ResourceRequirements for this step.
            tags: Metadata tags for observability.

        Returns:
            A FlowyML Step that loads data and returns a Dataset asset.
        """

        @step(
            name=name or "load_data",
            outputs=["dataset"],
            cache=cache,
            retry=retry,
            resources=resources,
            tags=tags or {"component": "data_loader"},
        )
        def load_data() -> Dataset:
            raw_data = loader.load()
            return Dataset.create(
                data=raw_data,
                name="loaded_dataset",
                tags={"source": type(loader).__name__},
            )

        return load_data

    @staticmethod
    def create_training_step(
        trainer: ModelTrainer[ModelT, DatasetT],
        *,
        name: str | None = None,
        cache: bool | str | Callable = False,
        retry: int = 0,
        resources: Any | None = None,
        tags: dict[str, str] | None = None,
    ) -> Step:
        """Wrap a ModelTrainer as a FlowyML step returning a Model asset.

        Args:
            trainer: Any MLPotion ModelTrainer protocol implementation.
            name: Step name (defaults to 'train_model').
            cache: Caching strategy (default: False for training).
            retry: Number of retry attempts on failure.
            resources: ResourceRequirements (e.g., GPU config).
            tags: Metadata tags for observability.

        Returns:
            A FlowyML Step that trains a model and returns a Model asset.
        """

        @step(
            name=name or "train_model",
            inputs=["model", "dataset", "config"],
            outputs=["trained_model"],
            cache=cache,
            retry=retry,
            resources=resources,
            tags=tags or {"component": "model_trainer"},
        )
        def train_model(
            model: ModelT,
            dataset: DatasetT,
            config: Any,
            validation_dataset: DatasetT | None = None,
        ) -> Model:
            result = trainer.train(model, dataset, config, validation_dataset)
            return Model.create(
                data=result.model,
                name="trained_model",
                training_history=getattr(result, "history", None),
                tags={"trainer": type(trainer).__name__},
            )

        return train_model

    @staticmethod
    def create_evaluation_step(
        evaluator: ModelEvaluator[ModelT, DatasetT],
        *,
        name: str | None = None,
        cache: bool | str | Callable = "input_hash",
        retry: int = 0,
        resources: Any | None = None,
        tags: dict[str, str] | None = None,
    ) -> Step:
        """Wrap a ModelEvaluator as a FlowyML step returning a Metrics asset.

        Args:
            evaluator: Any MLPotion ModelEvaluator protocol implementation.
            name: Step name (defaults to 'evaluate_model').
            cache: Caching strategy (default: 'input_hash').
            retry: Number of retry attempts on failure.
            resources: ResourceRequirements for this step.
            tags: Metadata tags for observability.

        Returns:
            A FlowyML Step that evaluates a model and returns a Metrics asset.
        """

        @step(
            name=name or "evaluate_model",
            inputs=["model", "dataset"],
            outputs=["metrics"],
            cache=cache,
            retry=retry,
            resources=resources,
            tags=tags or {"component": "model_evaluator"},
        )
        def evaluate_model(
            model: ModelT,
            dataset: DatasetT,
            config: Any | None = None,
        ) -> Metrics:
            result = evaluator.evaluate(model, dataset, config)
            return Metrics.create(
                name="evaluation_metrics",
                metrics=result.metrics if hasattr(result, "metrics") else result,
                tags={"evaluator": type(evaluator).__name__},
            )

        return evaluate_model
