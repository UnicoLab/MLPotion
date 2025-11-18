"""TensorFlow/Keras implementation.

This module is only available if TensorFlow is installed.
"""

from mlpotion.utils.framework import require_framework

# Ensure TensorFlow is installed
require_framework("tensorflow", "mlpotion[tensorflow]")

# Safe to import TensorFlow components now
from mlpotion.frameworks.tensorflow.config import (
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
from mlpotion.frameworks.tensorflow.data.loaders import TFCSVDataLoader
from mlpotion.frameworks.tensorflow.data.optimizers import TFDatasetOptimizer
from mlpotion.frameworks.tensorflow.deployment.exporters import TFModelExporter
from mlpotion.frameworks.tensorflow.deployment.persistence import TFModelPersistence
from mlpotion.frameworks.tensorflow.evaluation.evaluators import TFModelEvaluator
from mlpotion.frameworks.tensorflow.training.trainers import TFModelTrainer

__all__ = [
    # Configs
    "ModelTrainingConfig",
    "ModelEvaluationConfig",
    "ModelExportConfig",
    "DataLoadingConfig",
    "DataOptimizationConfig",
    "ModelLoadingConfig",
    "ModelPersistenceConfig",
    "DataTransformationConfig",
    "ModelInspectionConfig",
    # Components
    "TFCSVDataLoader",
    "TFDatasetOptimizer",
    "TFModelTrainer",
    "TFModelEvaluator",
    "TFModelExporter",
    "TFModelPersistence",
]