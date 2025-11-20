"""TensorFlow model inspection.

This module re-exports the Keras `ModelInspector` implementation, as TensorFlow 2.x
uses Keras as its high-level API.
"""

from mlpotion.frameworks.keras.models.inspection import ModelInspector

__all__ = ["ModelInspector"]
