"""TensorFlow model persistence."""

from mlpotion.frameworks.keras.deployment.persistence import KerasModelPersistence

# Create aliases
TFModelPersistence = KerasModelPersistence

__all__ = ["TFModelPersistence"]