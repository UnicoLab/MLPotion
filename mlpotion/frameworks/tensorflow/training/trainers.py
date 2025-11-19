"""TensorFlow model trainers.

This module re-exports the Keras `ModelTrainer` implementation, as TensorFlow 2.x
uses Keras as its high-level API.
"""

from mlpotion.frameworks.keras.training.trainers import ModelTrainer

__all__ = ["ModelTrainer"]