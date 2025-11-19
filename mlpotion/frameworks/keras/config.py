from typing import Any, Literal

import keras
import numpy as np
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataLoadingConfig(BaseSettings):
    """Configuration for data loading."""

    file_pattern: str = Field(..., description="File pattern (glob) to load")
    batch_size: int = Field(default=32, ge=1)
    column_names: list[str] | None = Field(default=None, description="Columns to load")
    label_name: str | None = Field(default=None, description="Label column name")
    shuffle: bool = Field(default=True, description="Shuffle data")
    dtype: np.dtype | str = "float32"
    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix='data_',
    )

class DataTransformationConfig(BaseSettings):
    """Configuration for CSV→CSV transformations using a Keras model.

    This configuration matches the CSVDataTransformer class parameters,
    providing a clean interface for configuring data transformation operations.

    Fields:
        file_pattern: Optional glob pattern for CSV files. Can be used if you
            want the transformer (or a CSVDataLoader) to discover files.
            Not used directly by CSVDataTransformer, but reserved for future use.
        model_path: Optional path to a saved Keras model. Used when you want
            the transformer to resolve the model via ModelPersistence.
        data_output_path: Target path for transformed CSV data. If
            `data_output_per_batch=False`, this is a single CSV file. If True,
            this is treated as a directory (or its parent directory is used).
        data_output_per_batch: Whether to save one CSV per batch instead of
            a single concatenated CSV.
        batch_size: Optional override for batching when the input dataset is a
            DataFrame or NumPy array. If None, the transformer's own batch_size
            attribute is used.
        feature_names: Optional feature names for ndarray batches. If not set,
            uses `feature_0`, `feature_1`, etc.
        input_columns: Optional explicit list of columns to feed into the model.
            If None, they will be derived from model inspection (input names).
        config: Optional dict for future/extra settings. Currently unused by
            CSVDataTransformer but kept for parity and extensibility.
    """

    file_pattern: str | None = Field(
        default=None,
        description="Optional file pattern for CSV files (glob).",
    )
    model_path: str | None = Field(
        default=None,
        description="Optional path to a saved Keras model.",
    )
    data_output_path: str = Field(
        ...,
        description="Path to save transformed CSV data.",
    )
    data_output_per_batch: bool = Field(
        default=False,
        description="If True, save per-batch CSV files instead of a single file.",
    )
    batch_size: int | None = Field(
        default=None,
        ge=1,
        description="Optional batch size override for DataFrame/NumPy input.",
    )
    feature_names: list[str] | None = Field(
        default=None,
        description="Optional feature names for ndarray batches.",
    )
    input_columns: list[str] | None = Field(
        default=None,
        description="Optional explicit list of columns to feed into the model.",
    )
    config: dict[str, Any] | None = Field(
        default=None,
        description="Extra configuration for the data transformer.",
    )

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix="transform_",
    )


class ModelLoadingConfig(BaseSettings):
    """Configuration for model loading."""

    model_path: str = Field(..., description="Path to model")
    model: keras.Model = Field(..., description="Model to use for transformation")
    model_input_signature: dict[str, Any] | None = None,
    
    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix='model_',
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
        env_prefix='opt_',
    )



class ModelPersistenceConfig(BaseSettings):
    """Configuration for model persistence."""

    path: str = Field(..., description="Path to model")
    model: keras.Model = Field(..., description="Model to use for persistence")
    save_format: str | None = Field(default=None, description="Save format")
    config: dict[str, Any] | None = Field(default=None, description="Extra configuration for the model persistence")
    
    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix='model_persist_',
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
        env_prefix='export_',
    )


class ModelEvaluationConfig(BaseSettings):
    """Base evaluation configuration."""

    batch_size: int = Field(default=32, ge=1)
    verbose: int = Field(default=1, ge=0, le=2)
    framework_options: dict[str, Any] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix='eval_',
    )


class ModelInspectionConfig(BaseSettings):  
    """Configuration for model inspection."""

    format: Literal["saved_model", "h5", "keras"] = Field(default="saved_model")
    signatures: list[str] | None = Field(default=None, description="Signatures to inspect")
    config: dict[str, Any] | None = Field(default=None, description="Extra configuration for the model inspection")

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix='inspect_',
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

    # Keras specific fields
    optimizer_type: str = Field(default="adam", description="Optimizer name")
    loss: str | keras.losses.Loss = Field(default="mse", description="Loss function")
    metrics: list[str | keras.metrics.Metric] = Field(default_factory=lambda: ["mae"], description="List of metrics")

    # Framework-specific options can go here
    framework_options: dict[str, Any] = Field(
        default_factory=dict, description="Framework-specific options"
    )

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix='train_',
        arbitrary_types_allowed=True,
    )