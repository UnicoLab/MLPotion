"""PyTorch implementation.

This module is only available if PyTorch is installed.
"""

from mlpotion.utils.framework import require_framework

# Ensure PyTorch is installed
require_framework("torch", "mlpotion[pytorch]")

# Safe to import PyTorch components now
from mlpotion.frameworks.pytorch.config import (
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
from mlpotion.frameworks.pytorch.data.datasets import PyTorchCSVDataset
from mlpotion.frameworks.pytorch.data.loaders import PyTorchDataLoaderFactory
from mlpotion.frameworks.pytorch.deployment.exporters import PyTorchModelExporter
from mlpotion.frameworks.pytorch.deployment.persistence import PyTorchModelPersistence
from mlpotion.frameworks.pytorch.evaluation.evaluators import PyTorchModelEvaluator
from mlpotion.frameworks.pytorch.training.trainers import PyTorchModelTrainer

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
    "PyTorchCSVDataset",
    "PyTorchDataLoaderFactory",
    "PyTorchModelTrainer",
    "PyTorchModelEvaluator",
    "PyTorchModelExporter",
    "PyTorchModelPersistence",
]