"""TensorFlow/Keras implementation.

This module is only available if TensorFlow is installed.
"""

from mlpotion.utils.framework import require_framework

# Ensure TensorFlow is installed
require_framework("tensorflow", "mlpotion[tensorflow, keras]")

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
from mlpotion.frameworks.tensorflow.data.loaders import CSVDataLoader, RecordDataLoader
from mlpotion.frameworks.tensorflow.data.optimizers import DatasetOptimizer
from mlpotion.frameworks.tensorflow.deployment.exporters import ModelExporter
from mlpotion.frameworks.tensorflow.deployment.persistence import ModelPersistence
from mlpotion.frameworks.tensorflow.evaluation.evaluators import ModelEvaluator
from mlpotion.frameworks.tensorflow.training.trainers import ModelTrainer

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