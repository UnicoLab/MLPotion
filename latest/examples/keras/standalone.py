"""Basic Keras usage WITHOUT ZenML.

This example demonstrates the core MLPotion Keras workflow:
1. Load data from CSV
2. Create a Keras model
3. Train the model
4. Evaluate the model
5. Save and export the model
"""

import tensorflow as tf
from loguru import logger

from mlpotion.frameworks.keras import (
    CSVDataLoader,
    ModelEvaluator,
    ModelPersistence,
    ModelTrainer,
    ModelTrainingConfig,
)


# =================== METHODS ===========================
def create_model(input_dim: int = 10) -> tf.keras.Model:
    """Create a simple feedforward neural network.

    Args:
        input_dim: Number of input features.

    Returns:
        Compiled Keras model.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae", "mse"],
    )
    return model


# =================== MAIN ===========================

# 1. Load data (♻️ REUSABLE)
logger.info("\n1. Loading data from CSV...")
loader = CSVDataLoader(
    file_pattern="docs/examples/data/sample.csv",
    label_name="target",
    batch_size=8,
    shuffle=True,
)
dataset = loader.load()
logger.info(f"Dataset created: {dataset}")

# 2. Create model (CUSTOM)
logger.info("\n2. Creating Keras model...")
model = create_model(input_dim=10)
logger.info(model.summary())

# 3. Train model (♻️ REUSABLE)
logger.info("\n3. Training model...")
trainer = ModelTrainer()
config = ModelTrainingConfig(
    epochs=10,
    batch_size=8,
    learning_rate=0.001,
    validation_split=0.2,
    verbose=1,
)

result = trainer.train(
    model=model,
    data=dataset,
    config=config,
)

logger.info("\nTraining completed!")
logger.info(f"Final training results: {result}")

# 4. Evaluate model (♻️ REUSABLE)
logger.info("\n4. Evaluating model...")
evaluator = ModelEvaluator()
eval_result = evaluator.evaluate(
    model=model,
    data=dataset,
    config=config,
)

logger.info("Evaluation completed!")
logger.info(f"Evaluation results: {eval_result}")

# 5. Save model (♻️ REUSABLE)
logger.info("\n5. Saving model...")
model_path = "/tmp/keras_model.keras"

persistence = ModelPersistence(
    path=model_path,
    model=model,
)
persistence.save(
    save_format="keras",
)
logger.info(f"Model saved to: {model_path}")

# 6. Load model (♻️ REUSABLE)
logger.info("\n6. Loading model...")
loaded_model, metadata = persistence.load()
logger.info(f"Model loaded successfully: {type(loaded_model)}")

logger.info("\n" + "=" * 60)
logger.info("Complete!")
logger.info("=" * 60)
