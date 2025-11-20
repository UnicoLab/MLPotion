"""TensorFlow/Keras implementation.


This module is only available if TensorFlow is installed.
"""

from mlpotion.utils.framework import require_framework

# Ensure TensorFlow is installed
require_framework("tensorflow", "mlpotion[tensorflow, keras]")

# Safe to import TensorFlow components now
from mlpotion.frameworks.tensorflow.config import (  # noqa: E402
    ModelTrainingConfig,
    ModelEvaluationConfig,
    ModelExportConfig,
    DataLoadingConfig,
    DataOptimizationConfig,
    ModelLoadingConfig,
    ModelPersistenceConfig,
    DataTransformationConfig,
    ModelInspectionConfig,
)
from mlpotion.frameworks.tensorflow.data.loaders import CSVDataLoader, RecordDataLoader  # noqa: E402
from mlpotion.frameworks.tensorflow.data.optimizers import DatasetOptimizer  # noqa: E402
from mlpotion.frameworks.tensorflow.deployment.exporters import ModelExporter  # noqa: E402
from mlpotion.frameworks.tensorflow.deployment.persistence import ModelPersistence  # noqa: E402
from mlpotion.frameworks.tensorflow.evaluation.evaluators import ModelEvaluator  # noqa: E402
from mlpotion.frameworks.tensorflow.training.trainers import ModelTrainer  # noqa: E402

__all__ = [
    # Components
    "CSVDataLoader",
    "RecordDataLoader",
    "DatasetOptimizer",
    "ModelTrainer",
    "ModelEvaluator",
    "ModelExporter",
    "ModelPersistence",
    # Configs
    "DataLoadingConfig",
    "DataOptimizationConfig",
    "DataTransformationConfig",
    "ModelExportConfig",
    "ModelPersistenceConfig",
    "ModelEvaluationConfig",
    "ModelLoadingConfig",
    "ModelInspectionConfig",
    "ModelTrainingConfig",
]
