from typing import Any, Callable
import keras
import tensorflow as tf

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict



class ModelLoadingConfig(BaseSettings):
    """Configuration for model loading."""

    model_path: str = Field(..., description="Path to model")
    model: keras.Model = Field(..., description="Model to use for transformation")
    model_input_signature: dict[str, tf.TensorSpec] | None = None,
    
    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix='model_',
    )



class KerasCSVTransformationConfig(BaseSettings):
    """Configuration for CSV→CSV transformations using a Keras model.

    This is intentionally simpler and Keras/pandas-only compared to the
    TensorFlow-oriented DataTransformationConfig.

    Fields:
        file_pattern: Optional glob pattern for CSV files. Can be used if you
            want the transformer (or a CSVDataLoader) to discover files.
            Not used directly by CSVDataTransformer, but reserved for future use.
        model_path: Optional path to a saved Keras model. Used when you want
            the transformer to resolve the model via KerasModelPersistence.
        data_output_path: Target path for transformed CSV data. If
            `data_output_per_batch=False`, this is a single CSV file. If True,
            this is treated as a directory (or its parent directory is used).
        data_output_per_batch: Whether to save one CSV per batch instead of
            a single concatenated CSV.
        batch_size: Optional override for batching when the input dataset is a
            DataFrame or NumPy array. If None, the transformer's own batch_size
            attribute is used.
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
    config: dict[str, Any] | None = Field(
        default=None,
        description="Extra configuration for the data transformer.",
    )

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix="keras_transform_",
    )
