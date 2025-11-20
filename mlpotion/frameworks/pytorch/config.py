"""PyTorch-specific configuration."""

from typing import Literal, Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataLoadingConfig(BaseSettings):
    """Configuration for PyTorch data loading."""

    file_pattern: str = Field(..., description="File pattern (glob) to load")
    batch_size: int = Field(default=32, ge=1)
    shuffle: bool = Field(default=True, description="Shuffle data")
    num_workers: int = Field(
        default=0, ge=0, description="Number of worker processes for data loading"
    )
    pin_memory: bool = Field(
        default=False, description="Pin memory for faster GPU transfer"
    )
    drop_last: bool = Field(default=False, description="Drop the last incomplete batch")
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
    num_workers: int = Field(default=0, ge=0, description="Number of worker processes")
    pin_memory: bool = Field(
        default=False, description="Pin memory for faster GPU transfer"
    )
    prefetch_factor: int | None = Field(
        default=None, ge=1, description="Number of batches to prefetch"
    )
    persistent_workers: bool = Field(
        default=False, description="Keep workers alive between epochs"
    )

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix="opt_",
    )


class ModelLoadingConfig(BaseSettings):
    """Configuration for model loading."""

    model_path: str = Field(..., description="Path to model")
    device: str = Field(default="cpu", description="Device to load model on")
    map_location: str | None = Field(
        default=None, description="Map location for loading model"
    )
    config: dict[str, Any] | None = Field(
        default=None, description="Extra configuration for model loading"
    )

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix="model_",
    )


class ModelPersistenceConfig(BaseSettings):
    """Configuration for model persistence."""

    path: str = Field(..., description="Path to save/load model")
    save_optimizer: bool = Field(default=False, description="Save optimizer state")
    save_scheduler: bool = Field(default=False, description="Save scheduler state")
    config: dict[str, Any] | None = Field(
        default=None, description="Extra configuration for model persistence"
    )

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix="model_persist_",
    )


class DataTransformationConfig(BaseSettings):
    """Configuration for data transformation with PyTorch models."""

    file_pattern: str | None = Field(
        default=None, description="File pattern (glob) to load"
    )
    model_path: str | None = Field(default=None, description="Path to model")
    batch_size: int = Field(
        default=32, ge=1, description="Batch size for transformation"
    )
    device: str = Field(default="cpu", description="Device for transformation")
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


class ModelTrainingConfig(BaseSettings):
    """Configuration for PyTorch model training.

    Designed to be flexible and match `PyTorchModelTrainer` expectations:

    - `optimizer` is a free string (validated in the trainer, not here).
    - `loss_fn` can be a string, nn.Module, or callable → typed as `Any`.
    - `max_batches_per_epoch` and `max_batches` are both supported; the
      trainer treats `max_batches_per_epoch` as primary and falls back to
      `max_batches` for backward compatibility.
    """

    # Core training settings
    epochs: int = Field(
        default=1,
        ge=1,
        description="Number of training epochs.",
    )
    learning_rate: float = Field(
        default=1e-3,
        gt=0,
        description="Learning rate for the optimizer.",
    )
    optimizer: str | None = Field(
        default="adam",
        description="Optimizer name (e.g. 'adam', 'sgd', 'adamw', 'rmsprop').",
    )
    loss_fn: Any = Field(
        default="mse",
        description=(
            "Loss specification. Can be a string key ('mse', 'cross_entropy', "
            "etc.), an nn.Module instance, or a callable(outputs, targets) -> loss."
        ),
    )
    device: str = Field(
        default="cpu",
        description="Device to train on, e.g. 'cpu', 'cuda', 'cuda:0'.",
    )
    verbose: bool = Field(
        default=True,
        description="Whether to log per-epoch metrics.",
    )

    # Batching limits
    max_batches_per_epoch: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum number of batches to process in each epoch. "
            "None means use all batches."
        ),
    )
    # Backwards-compatible alias used by tests / older configs
    max_batches: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Deprecated alias for max_batches_per_epoch. "
            "If max_batches_per_epoch is None, the trainer will fall back to this."
        ),
    )

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix="train_",
    )


class ModelEvaluationConfig(BaseSettings):
    """Configuration for PyTorch model evaluation."""

    batch_size: int = Field(default=32, ge=1)
    verbose: bool = Field(default=True, description="Whether to log evaluation metrics")
    device: Literal["cpu", "cuda", "mps"] = Field(default="cpu")
    framework_options: dict[str, Any] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix="eval_",
    )


class ModelExportConfig(BaseSettings):
    """Configuration for exporting PyTorch models.

    This config is intentionally minimal and export-specific so it can be
    used comfortably in tests and higher-level orchestration code.

    Required:
        - export_path: base path (with or without extension)
        - format: "torchscript", "onnx", or "state_dict"

    All other fields are optional and validated at *export* time,
    not at config construction time.
    """

    # ------------------------------------------------------------------ #
    # Required fields
    # ------------------------------------------------------------------ #
    export_path: str = Field(
        ...,
        description="Base path for exported artifact (with or without extension).",
    )

    format: str = Field(
        ...,
        description="Export format: 'torchscript', 'onnx', or 'state_dict'.",
    )

    # ------------------------------------------------------------------ #
    # Optional / format-specific fields
    # ------------------------------------------------------------------ #
    device: str = Field(
        default="cpu",
        description="Device string used for export (e.g. 'cpu', 'cuda', 'cuda:0').",
    )

    # TorchScript-specific
    jit_mode: str = Field(
        default="script",
        description="TorchScript mode: 'script' (default) or 'trace'.",
    )
    example_input: Any | None = Field(
        default=None,
        description=(
            "Example input tensor/structure. "
            "Required for ONNX and TorchScript trace mode; "
            "validated in the exporter."
        ),
    )

    # ONNX-specific
    input_names: list[str] | None = Field(
        default=None,
        description="ONNX input tensor names; defaults to ['input'] if not provided.",
    )
    output_names: list[str] | None = Field(
        default=None,
        description="ONNX output tensor names; defaults to ['output'] if not provided.",
    )
    dynamic_axes: dict[str, dict[int, str]] | None = Field(
        default=None,
        description="ONNX dynamic axes mapping (name -> axis_index -> axis_name).",
    )
    opset_version: int = Field(
        default=13,
        ge=1,
        description="ONNX opset version (default: 13).",
    )

    model_config = SettingsConfigDict(
        extra="forbid",  # reject unknown keys so bugs are caught early
        frozen=False,
        env_prefix="export_",
    )


class ModelInspectionConfig(BaseSettings):
    """Configuration for PyTorch model inspection."""

    format: Literal["state_dict", "torchscript", "onnx"] = Field(default="state_dict")
    show_parameters: bool = Field(default=True, description="Show model parameters")
    show_architecture: bool = Field(default=True, description="Show model architecture")
    config: dict[str, Any] | None = Field(
        default=None, description="Extra configuration for model inspection"
    )

    model_config = SettingsConfigDict(
        extra="forbid",
        frozen=False,
        env_prefix="inspect_",
    )
