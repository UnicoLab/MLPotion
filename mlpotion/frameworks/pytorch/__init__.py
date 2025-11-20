"""PyTorch implementation.


This module is only available if PyTorch is installed.
"""

from mlpotion.utils.framework import require_framework

# Ensure PyTorch is installed
require_framework("torch", "mlpotion[pytorch]")

# Safe to import PyTorch components now
from mlpotion.frameworks.pytorch.config import (  # noqa: E402
    ModelTrainingConfig,
    ModelEvaluationConfig,
    ModelExportConfig,
    DataLoadingConfig,
    ModelLoadingConfig,
    ModelPersistenceConfig,
    ModelInspectionConfig,
    DataTransformationConfig,
)
from mlpotion.frameworks.pytorch.data.datasets import CSVDataset, StreamingCSVDataset  # noqa: E402
from mlpotion.frameworks.pytorch.data.loaders import CSVDataLoader  # noqa: E402
from mlpotion.frameworks.pytorch.data.transformers import DataToCSVTransformer  # noqa: E402
from mlpotion.frameworks.pytorch.data.transformers import PredictionFormatter  # noqa: E402
from mlpotion.frameworks.pytorch.deployment.exporters import ModelExporter  # noqa: E402
from mlpotion.frameworks.pytorch.deployment.persistence import ModelPersistence  # noqa: E402
from mlpotion.frameworks.pytorch.evaluation.evaluators import ModelEvaluator  # noqa: E402
from mlpotion.frameworks.pytorch.training.trainers import ModelTrainer  # noqa: E402

__all__ = [
    # Components
    "CSVDataset",
    "StreamingCSVDataset",
    "CSVDataLoader",
    "DataToCSVTransformer",
    "PredictionFormatter",
    "ModelExporter",
    "ModelPersistence",
    "ModelEvaluator",
    "ModelTrainer",
    # Configs
    "DataLoadingConfig",
    "DataTransformationConfig",
    "ModelExportConfig",
    "ModelPersistenceConfig",
    "ModelEvaluationConfig",
    "ModelTrainingConfig",
    "ModelLoadingConfig",
    "ModelInspectionConfig",
]
