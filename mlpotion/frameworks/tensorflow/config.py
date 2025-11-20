"""TensorFlow-specific configuration."""

from typing import Any, Literal, Callable

import keras
import tensorflow as tf
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataLoadingConfig(BaseSettings):
    """Configuration for data loading."""

    file_pattern: str = Field(..., description="File pattern (glob) to load")
    batch_size: int = Field(default=32, ge=1)
    column_names: list[str] | None = Field(default=None, description="Columns to load")
    label_name: str | None = Field(default=None, description="Label column name")
    map_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = Field(
        default=None, description="Mapping function to apply to the dataset"
    )
    config: dict[str, Any] | None = Field(
        default=None, description="Extra configuration for the dataset loader"
    )

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix="data_",
    )


class DataOptimizationConfig(BaseSettings):
    """Configuration for dataset optimization."""

    batch_size: int = Field(default=32, ge=1)
    shuffle_buffer_size: int | None = Field(default=None, ge=1)
    prefetch: bool = Field(default=True)
    cache: bool = Field(default=False)

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix="opt_",
    )


class ModelLoadingConfig(BaseSettings):
    """Configuration for model loading."""

    model_path: str = Field(..., description="Path to model")
    model: keras.Model = Field(..., description="Model to use for transformation")
    model_input_signature: dict[str, tf.TensorSpec] | None = (None,)

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix="model_",
    )


class DataTransformationConfig(BaseSettings):
    """Configuration for data transformation."""

    # optional paths to load data and model from
    file_pattern: str = Field(..., description="File pattern (glob) to load")
    model_path: str = Field(..., description="Path to model")
    # alternatively, model can be passed as a keras.Model object and dataset can be passed as a tf.data.Dataset object
    model: keras.Model = Field(..., description="Model to use for transformation")
    dataset: tf.data.Dataset = Field(
        ..., description="Dataset to use for transformation"
    )
    model_input_signature: dict[str, tf.TensorSpec] | None = (None,)

    # batch size for transformation
    batch_size: int = Field(
        default=32, ge=1, description="Batch size for transformation"
    )
    data_output_path: str | None = Field(
        default=None, description="Path to save transformed data"
    )
    data_output_per_batch: bool = Field(
        default=False, description="Save data per batch"
    )
    config: dict[str, Any] | None = Field(
        default=None, description="Extra configuration for the data transformer"
    )

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix="transform_",
    )


class ModelPersistenceConfig(BaseSettings):
    """Configuration for model persistence."""

    path: str = Field(..., description="Path to model")
    model: keras.Model = Field(..., description="Model to use for persistence")
    save_format: str | None = Field(default=None, description="Save format")
    config: dict[str, Any] | None = Field(
        default=None, description="Extra configuration for the model persistence"
    )

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix="model_persist_",
    )


class ModelExportConfig(BaseSettings):
    """Base export configuration."""

    export_path: str = Field(..., description="Path to export model")
    format: str = Field(default="default", description="Export format")
    include_optimizer: bool = Field(default=False)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix="export_",
    )


class ModelEvaluationConfig(BaseSettings):
    """Base evaluation configuration."""

    batch_size: int = Field(default=32, ge=1)
    verbose: int = Field(default=1, ge=0, le=2)
    framework_options: dict[str, Any] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix="eval_",
    )


class ModelInspectionConfig(BaseSettings):
    """Configuration for model inspection."""

    format: Literal["saved_model", "h5", "keras"] = Field(default="saved_model")
    signatures: list[str] | None = Field(
        default=None, description="Signatures to inspect"
    )
    config: dict[str, Any] | None = Field(
        default=None, description="Extra configuration for the model inspection"
    )

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix="inspect_",
    )


class ModelTrainingConfig(BaseSettings):
    """Base training configuration - framework agnostic.

    Framework-specific configs should inherit from this.
    """

    epochs: int = Field(default=10, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=1, description="Batch size for training")
    learning_rate: float = Field(default=0.001, gt=0.0, description="Learning rate")
    validation_split: float = Field(default=0.0, ge=0.0, le=1.0)
    shuffle: bool = Field(default=True, description="Shuffle training data")
    verbose: int = Field(default=1, ge=0, le=2, description="Verbosity level")

    # Keras/TF specific fields
    optimizer_type: str = Field(default="adam", description="Optimizer name")
    loss: str | keras.losses.Loss = Field(default="mse", description="Loss function")
    metrics: list[str | keras.metrics.Metric] = Field(
        default_factory=lambda: ["mae"], description="List of metrics"
    )

    # Framework-specific options can go here
    framework_options: dict[str, Any] = Field(
        default_factory=dict, description="Framework-specific options"
    )

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix="train_",
        arbitrary_types_allowed=True,
    )
