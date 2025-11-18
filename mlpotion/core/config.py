"""Framework-agnostic base configuration models using Pydantic 2.x.

This module contains only truly framework-agnostic configuration classes.
Framework-specific configurations should be defined in their respective
framework modules (keras, tensorflow, pytorch).
"""

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingConfig(BaseSettings):
    """Base training configuration - framework agnostic.

    Framework-specific configs should inherit from this.
    """

    epochs: int = Field(default=10, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=1, description="Batch size for training")
    learning_rate: float = Field(default=0.001, gt=0.0, description="Learning rate")
    validation_split: float = Field(default=0.0, ge=0.0, le=1.0)
    shuffle: bool = Field(default=True, description="Shuffle training data")
    verbose: int = Field(default=1, ge=0, le=2, description="Verbosity level")

    # Framework-specific options can go here
    framework_options: dict[str, Any] = Field(
        default_factory=dict, description="Framework-specific options"
    )

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix='train_',
    )


class EvaluationConfig(BaseSettings):
    """Base evaluation configuration."""

    batch_size: int = Field(default=32, ge=1)
    verbose: int = Field(default=1, ge=0, le=2)
    framework_options: dict[str, Any] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix='eval_',
    )


class ExportConfig(BaseSettings):
    """Base export configuration."""

    export_path: str = Field(..., description="Path to export model")
    format: str = Field(default="default", description="Export format")
    include_optimizer: bool = Field(default=False)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix='export_',
    )
