"""Basic Keras usage WITHOUT ZenML.

This example demonstrates the core MLPotion Keras workflow:
1. Load data from CSV
2. Create a Keras model
3. Train the model
4. Evaluate the model
5. Save and export the model
"""

import tensorflow as tf

from mlpotion.frameworks.keras import (
    CSVDataLoader as KerasCSVDataLoader,
    ModelEvaluator as KerasModelEvaluator,
    ModelPersistence as KerasModelPersistence,
    ModelTrainer as KerasModelTrainer,
    ModelTrainingConfig as KerasTrainingConfig,
)


def create_model(input_dim: int = 10) -> tf.keras.Model:
    """Create a simple feedforward neural network.

    Args:
        input_dim: Number of input features.

    Returns:
        Compiled Keras model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae", "mse"],
    )

    return model


def main() -> None:
    """Run basic Keras training pipeline."""
    print("=" * 60)
    print("MLPotion - Keras Basic Usage")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading data from CSV...")
    loader = KerasCSVDataLoader(
        file_pattern="examples/data/sample.csv",
        label_name="target",
        batch_size=8,
        shuffle=True,
    )
    dataset = loader.load()
    print(f"Dataset created: {dataset}")

    # 2. Create model
    print("\n2. Creating Keras model...")
    model = create_model(input_dim=10)
    print(model.summary())

    # 3. Train model
    print("\n3. Training model...")
    trainer = KerasModelTrainer()
    config = KerasTrainingConfig(
        epochs=10,
        batch_size=8,
        learning_rate=0.001,
        validation_split=0.2,
        verbose=1,
    ).dict()

    result = trainer.train(
        model=model,
        data=dataset,
        **config,
    )

    print(f"\nTraining completed!")
    print(f"Final training results: {result}")

    # 4. Evaluate model
    print("\n4. Evaluating model...")
    evaluator = KerasModelEvaluator()
    eval_result = evaluator.evaluate(
        model=model,
        data=dataset,
        **config,
    )

    print(f"Evaluation completed!")
    print(f"Evaluation results: {eval_result}")

    # 5. Save model
    print("\n5. Saving model...")
    model_path = "/tmp/keras_model.keras"

    persistence = KerasModelPersistence(
        path=model_path,
        model=model,
    )
    persistence.save(
        save_format="keras",
    )
    print(f"Model saved to: {model_path}")

    # 6. Load model
    print("\n6. Loading model...")
    loaded_model = persistence.load()
    print(f"Model loaded successfully: {type(loaded_model)}")

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
