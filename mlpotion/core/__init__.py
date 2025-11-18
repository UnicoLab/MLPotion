"""Core framework-agnostic components."""

from mlpotion.core.config import (
    TrainingConfig,
    EvaluationConfig,
    ExportConfig,
)
from mlpotion.core.exceptions import (
    MLPotionError,
    DataLoadingError,
    TrainingError,
    EvaluationError,
    ExportError,
    ConfigurationError,
    DataTransformationError,
    ModelPersistenceError,
)
from mlpotion.core.protocols import (
    DataLoader,
    DatasetOptimizer,
    DataTransformer,
    ModelTrainer,
    ModelEvaluator,
    ModelExporter,
    ModelPersistence,
    ModelInspectorProtocol,
    ModelExporterProtocol,
    ModelEvaluatorProtocol,
)
from mlpotion.core.results import (
    TrainingResult,
    EvaluationResult,
    ExportResult,
    TransformationResult,
)

__all__ = [
    # Config
    "TrainingConfig",
    "EvaluationConfig",
    "ExportConfig",
    # Exceptions
    "MLPotionError",
    "DataLoadingError",
    "TrainingError",
    "EvaluationError",
    "ExportError",
    "ConfigurationError",
    "DataTransformationError",
    "ModelPersistenceError",
    # Protocols
    "DataLoader",
    "DatasetOptimizer",
    "DataTransformer",
    "ModelTrainer",
    "ModelEvaluator",
    "ModelExporter",
    "ModelPersistence",
    "ModelInspectorProtocol",
    "ModelExporterProtocol",
    "ModelEvaluatorProtocol",
    # Results
    "TrainingResult",
    "EvaluationResult",
    "ExportResult",
    "TransformationResult",
]