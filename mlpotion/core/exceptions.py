"""Exception hierarchy for MLPotion."""


class MLPotionError(Exception):
    """Base exception for all MLPotion errors."""

    pass


class FrameworkNotInstalledError(MLPotionError):
    """Raised when a required framework (e.g., TensorFlow, PyTorch) is not installed or cannot be imported."""

    pass


class DataLoadingError(MLPotionError):
    """Raised when an error occurs during data loading or preprocessing."""

    pass


class TrainingError(MLPotionError):
    """Raised when an error occurs during the model training process."""

    pass

class DataTransformationError(MLPotionError):
    """Raised when an error occurs during data transformation operations."""

    pass


class EvaluationError(MLPotionError):
    """Raised when an error occurs during model evaluation."""

    pass


class ExportError(MLPotionError):
    """Raised when an error occurs during model export operations."""

    pass


class ConfigurationError(MLPotionError):
    """Raised when there is an issue with the provided configuration."""

    pass


class ValidationError(MLPotionError):
    """Raised when data or model validation fails."""

    pass

class ModelPersistenceError(MLPotionError):
    """Raised when an error occurs during model saving or loading."""

    pass


class ModelEvaluatorError(MLPotionError):
    """Raised when an error occurs specifically within a ModelEvaluator component."""

    pass

class ModelTrainerError(MLPotionError):
    """Raised when an error occurs specifically within a ModelTrainer component."""

    pass

class ModelInspectorError(MLPotionError):
    """Raised when an error occurs during model inspection."""

    pass

class ModelExporterError(MLPotionError):
    """Raised when an error occurs specifically within a ModelExporter component."""

    pass