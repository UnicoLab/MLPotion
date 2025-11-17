"""TensorFlow-specific configuration."""

from typing import Any, Literal

from pydantic import Field

from mlpotion.core.config import EvaluationConfig, ExportConfig, TrainingConfig


class TensorFlowTrainingConfig(TrainingConfig):
    """TensorFlow-specific training configuration."""

    optimizer: str = Field(default="adam", description="Optimizer name")
    loss: str | None = Field(default=None, description="Loss function")
    metrics: list[str] = Field(default_factory=list, description="Metrics to track")
    callbacks: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class TensorFlowEvaluationConfig(EvaluationConfig):
    """TensorFlow-specific evaluation configuration."""

    metrics: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class TensorFlowExportConfig(ExportConfig):
    """TensorFlow-specific export configuration."""

    format: Literal["saved_model", "h5", "keras"] = Field(default="saved_model")
    signatures: list[str] | None = Field(default=None)

    model_config = {"extra": "forbid"}