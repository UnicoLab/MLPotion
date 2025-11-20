"""TensorFlow model persistence.

This module re-exports the Keras `ModelPersistence` implementation, as TensorFlow 2.x
uses Keras as its high-level API.
"""

from mlpotion.frameworks.keras.deployment.persistence import ModelPersistence

__all__ = ["ModelPersistence"]
