"""Framework-agnostic base configuration models using Pydantic 2.x.

This module contains only truly framework-agnostic configuration classes.
Framework-specific configurations should be defined in their respective
framework modules (keras, tensorflow, pytorch).
"""

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingConfig(BaseSettings):
    """Base configuration for model training.

    This class defines the standard configuration parameters for training models.
    Framework-specific configurations should inherit from this class.

    Attributes:
        epochs (int): Number of training epochs (must be >= 1).
        batch_size (int): Batch size for training (must be >= 1).
        learning_rate (float): Learning rate for the optimizer (must be > 0.0).
        validation_split (float): Fraction of data to use for validation (0.0 to 1.0).
        shuffle (bool): Whether to shuffle the training data.
        verbose (int): Verbosity level (0=silent, 1=progress bar, 2=one line per epoch).
        framework_options (dict[str, Any]): Dictionary for framework-specific options that don't fit standard fields.
    """

    epochs: int = Field(default=10, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=1, description="Batch size for training")
    learning_rate: float = Field(default=0.001, gt=0.0, description="Learning rate")
    validation_split: float = Field(default=0.0, ge=0.0, le=1.0, description="Validation split ratio")
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
    """Base configuration for model evaluation.

    Attributes:
        batch_size (int): Batch size for evaluation (must be >= 1).
        verbose (int): Verbosity level (0=silent, 1=progress bar, 2=one line per epoch).
        framework_options (dict[str, Any]): Dictionary for framework-specific options.
    """

    batch_size: int = Field(default=32, ge=1, description="Batch size for evaluation")
    verbose: int = Field(default=1, ge=0, le=2, description="Verbosity level")
    framework_options: dict[str, Any] = Field(default_factory=dict, description="Framework-specific options")

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix='eval_',
    )


class ExportConfig(BaseSettings):
    """Base configuration for model export.

    Attributes:
        export_path (str): Destination path for the exported model.
        format (str): Format identifier for the export (e.g., 'saved_model', 'onnx').
        include_optimizer (bool): Whether to include the optimizer state in the export.
        metadata (dict[str, Any]): Additional metadata to include with the export.
    """

    export_path: str = Field(..., description="Path to export model")
    format: str = Field(default="default", description="Export format")
    include_optimizer: bool = Field(default=False, description="Include optimizer state")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix='export_',
    )
