"""TensorFlow model trainers.

Provides TensorFlow-friendly API that wraps Keras implementation.
"""

from mlpotion.frameworks.keras.utils.formatter import KerasPredictionFormatter

# Create TensorFlow-named alias
TFPredictionFormatter = KerasPredictionFormatter

__all__ = ["TFPredictionFormatter"]
