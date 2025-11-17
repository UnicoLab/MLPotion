"""TensorFlow model inspection.

Re-exports Keras model inspection with TensorFlow-friendly names.
"""

from mlpotion.frameworks.keras.models.inspection import KerasModelInspector

# Create TensorFlow-named alias
# Users see "TFModelTrainer", internally uses KerasModelTrainer
TFModelInspector= KerasModelInspector

__all__ = ["TFModelInspector"]