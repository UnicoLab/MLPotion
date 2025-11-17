"""TensorFlow model evaluators."""

from mlpotion.frameworks.keras.evaluation.evaluators import KerasModelEvaluator

# Create alias
TFModelEvaluator = KerasModelEvaluator

__all__ = ["TFModelEvaluator"]