"""Result types using dataclasses with Python 3.10+ features."""

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

ModelT = TypeVar("ModelT")


@dataclass
class TrainingResult(Generic[ModelT]):
    """Result from model training.

    Generic over model type for type safety.
    """

    model: ModelT
    history: dict[str, list[float]]
    metrics: dict[str, float]
    config: Any  # TrainingConfig (avoid circular import)
    training_time: float | None = None
    best_epoch: int | None = None

    def get_metric(self, name: str) -> float | None:
        """Get a specific metric value."""
        return self.metrics.get(name)

    def get_history(self, metric: str) -> list[float] | None:
        """Get history for a specific metric."""
        return self.history.get(metric)


@dataclass
class EvaluationResult:
    """Result from model evaluation."""

    metrics: dict[str, float]
    config: Any
    evaluation_time: float | None = None

    def get_metric(self, name: str) -> float | None:
        """Get a specific metric value."""
        return self.metrics.get(name)


@dataclass
class ExportResult:
    """Result from model export."""

    export_path: str
    format: str
    config: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Model exported to {self.export_path} (format: {self.format})"


@dataclass
class InspectionResult:
    """Result from model inspection."""

    name: str
    backend: str
    trainable: bool
    inputs: list[dict[str, Any]]
    input_names: list[str]
    outputs: list[dict[str, Any]]
    output_names: list[str]
    parameters: dict[str, int]
    signatures: dict[str, Any]  # only for tensorflow saved models
    layers: list[dict[str, Any]]



@dataclass
class LoadingResult(Generic[ModelT]):
    """Result from model loading."""

    model: ModelT
    path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformationResult:
    """Result from data transformation."""

    data_output_path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Data transformed and saved to {self.data_output_path}"