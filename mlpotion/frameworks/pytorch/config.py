"""PyTorch-specific configuration."""

from typing import Literal, Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mlpotion.core.config import EvaluationConfig


class PyTorchTrainingConfig(BaseSettings):
    """Configuration for generic PyTorch training.

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
        env_prefix="pytorch_train_",
    )



class PyTorchEvaluationConfig(EvaluationConfig):
    """PyTorch-specific evaluation configuration."""

    device: Literal["cpu", "cuda", "mps"] = Field(default="cpu")

    model_config = {"extra": "forbid"}


class PyTorchExportConfig(BaseSettings):
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
    )