"""Result types using dataclasses with Python 3.10+ features."""

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

ModelT = TypeVar("ModelT")


@dataclass
class TrainingResult(Generic[ModelT]):
    """Result container for model training operations.

    This dataclass encapsulates all relevant information produced during a model training session.

    Attributes:
        model (ModelT): The trained model instance.
        history (dict[str, list[float]]): A dictionary mapping metric names to lists of values per epoch.
        metrics (dict[str, float]): A dictionary of the final metric values after training.
        config (Any): The configuration object used for this training session.
        training_time (float | None): The total time taken for training in seconds.
        best_epoch (int | None): The epoch number where the best performance was achieved (if applicable).
    """

    model: ModelT
    history: dict[str, list[float]]
    metrics: dict[str, float]
    config: Any  # TrainingConfig (avoid circular import)
    training_time: float | None = None
    best_epoch: int | None = None

    def get_metric(self, name: str) -> float | None:
        """Retrieve a specific final metric value.

        Args:
            name: The name of the metric to retrieve.

        Returns:
            float | None: The value of the metric, or None if not found.
        """
        return self.metrics.get(name)

    def get_history(self, metric: str) -> list[float] | None:
        """Retrieve the history of a specific metric over epochs.

        Args:
            metric: The name of the metric to retrieve history for.

        Returns:
            list[float] | None: The list of metric values per epoch, or None if not found.
        """
        return self.history.get(metric)


@dataclass
class EvaluationResult:
    """Result container for model evaluation operations.

    Attributes:
        metrics (dict[str, float]): A dictionary of evaluation metric values.
        config (Any): The configuration object used for this evaluation.
        evaluation_time (float | None): The total time taken for evaluation in seconds.
    """

    metrics: dict[str, float]
    config: Any
    evaluation_time: float | None = None

    def get_metric(self, name: str) -> float | None:
        """Retrieve a specific metric value.

        Args:
            name: The name of the metric to retrieve.

        Returns:
            float | None: The value of the metric, or None if not found.
        """
        return self.metrics.get(name)


@dataclass
class ExportResult:
    """Result container for model export operations.

    Attributes:
        export_path (str): The absolute path where the model was exported.
        format (str): The format of the exported model (e.g., 'saved_model', 'onnx').
        config (Any): The configuration object used for this export.
        metadata (dict[str, Any]): Additional metadata generated during export.
    """

    export_path: str
    format: str
    config: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Model exported to {self.export_path} (format: {self.format})"


@dataclass
class InspectionResult:
    """Result container for model inspection operations.

    Attributes:
        name (str): The name of the model.
        backend (str): The framework backend used (e.g., 'tensorflow', 'pytorch').
        trainable (bool): Whether the model is trainable.
        inputs (list[dict[str, Any]]): List of input specifications (shape, dtype, etc.).
        input_names (list[str]): List of input names.
        outputs (list[dict[str, Any]]): List of output specifications.
        output_names (list[str]): List of output names.
        parameters (dict[str, int]): Dictionary of parameter counts (total, trainable, non_trainable).
        signatures (dict[str, Any]): Model signatures (specific to TensorFlow SavedModel).
        layers (list[dict[str, Any]]): List of layer details (name, class, args).
    """

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
