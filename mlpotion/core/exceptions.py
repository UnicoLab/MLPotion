"""Exception hierarchy for MLPotion."""


class MLPotionError(Exception):
    """Base exception for all MLPotion errors."""

    pass


class FrameworkNotInstalledError(MLPotionError):
    """Raised when a required framework is not installed."""

    pass


class DataLoadingError(MLPotionError):
    """Error during data loading."""

    pass


class TrainingError(MLPotionError):
    """Error during model training."""

    pass

class DataTransformationError(MLPotionError):
    """Error during data transformation."""

    pass


class EvaluationError(MLPotionError):
    """Error during model evaluation."""

    pass


class ExportError(MLPotionError):
    """Error during model export."""

    pass


class ConfigurationError(MLPotionError):
    """Error in configuration."""

    pass


class ValidationError(MLPotionError):
    """Error during validation."""

    pass

class ModelPersistenceError(MLPotionError):
    """Error during model persistence."""

    pass