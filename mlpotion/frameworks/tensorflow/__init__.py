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
from mlpotion.frameworks.tensorflow.data.loaders import TFCSVDataLoader as CSVDataLoader
from mlpotion.frameworks.tensorflow.data.optimizers import TFDatasetOptimizer as DatasetOptimizer
from mlpotion.frameworks.tensorflow.deployment.exporters import TFModelExporter as ModelExporter
from mlpotion.frameworks.tensorflow.deployment.persistence import TFModelPersistence as ModelPersistence
from mlpotion.frameworks.tensorflow.evaluation.evaluators import TFModelEvaluator as ModelEvaluator
from mlpotion.frameworks.tensorflow.training.trainers import TFModelTrainer as ModelTrainer

__all__ = [
    # Components
    "TFCSVDataLoader",
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