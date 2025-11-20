import keras
from zenml.enums import ArtifactType
from zenml.logger import get_logger
from pathlib import Path
from typing import Any
from zenml.materializers.base_materializer import BaseMaterializer

logger = get_logger(__name__)


class KerasModelMaterializer(BaseMaterializer):
    """ZenML Materializer for Keras models.

    This materializer handles the serialization and deserialization of Keras models
    within ZenML pipelines. It uses the standard Keras `save` and `load_model` APIs.
    It stores the model as a `model.keras` file in the artifact store.
    """

    ASSOCIATED_TYPES = (keras.Model,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL

    def load(self, data_type: type[Any]) -> keras.Model:
        """Load a Keras model from the artifact store.

        Args:
            data_type: The type of the data to load (should be `keras.Model`).

        Returns:
            keras.Model: The loaded Keras model.
        """
        model_path = Path(self.uri) / "model.keras"
        logger.info(f"Loading model from: {model_path}")
        model = keras.models.load_model(str(model_path))
        logger.info("✅ Model loaded successfully")
        return model

    def save(self, model: keras.Model) -> None:
        """Save a Keras model to the artifact store.

        Args:
            model: The Keras model to save.
        """
        Path(self.uri).mkdir(parents=True, exist_ok=True)
        model_path = Path(self.uri) / "model.keras"
        logger.info(f"Saving model to: {model_path}")
        model.save(str(model_path))
        logger.info("✅ Model saved successfully")
