"""MLPotion: Type-safe ML components for TensorFlow and PyTorch.

This package works WITHOUT any frameworks installed (core only).
Install frameworks as needed:
    pip install mlpotion[tensorflow]  # TensorFlow support
    pip install mlpotion[pytorch]     # PyTorch support
    pip install mlpotion[all]         # Everything
"""

# Core exports (always available)
from mlpotion.core import (
    ConfigurationError,
    DataLoadingError,
    EvaluationConfig,
    EvaluationError,
    EvaluationResult,
    ExportConfig,
    ExportError,
    ExportResult,
    MLPotionError,
    TrainingConfig,
    TrainingError,
    TrainingResult,
)
from mlpotion.utils import get_available_frameworks, is_framework_available, trycatch

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Exceptions
    "MLPotionError",
    "DataLoadingError",
    "TrainingError",
    "EvaluationError",
    "ExportError",
    "ConfigurationError",
    # Config
    "TrainingConfig",
    "EvaluationConfig",
    "ExportConfig",
    # Results
    "TrainingResult",
    "EvaluationResult",
    "ExportResult",
    # Utils
    "is_framework_available",
    "get_available_frameworks",
    "trycatch",
]

# Framework-specific imports (only if installed)
_available = get_available_frameworks()

if "tensorflow" in _available:
    __all__.append("tensorflow")

if "torch" in _available:
    __all__.append("pytorch")
