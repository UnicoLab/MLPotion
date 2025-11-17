"""TensorFlow model trainers.

Provides TensorFlow-friendly API that wraps Keras implementation.
"""

from mlpotion.frameworks.keras.training.trainers import KerasModelTrainer

# Create TensorFlow-named alias
# Users see "TFModelTrainer", internally uses KerasModelTrainer
TFModelTrainer = KerasModelTrainer

__all__ = ["TFModelTrainer"]