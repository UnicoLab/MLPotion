"""Keras training pipeline WITH ZenML orchestration.

This example demonstrates how to use MLPotion's Keras components
within a ZenML pipeline for reproducible and tracked ML workflows.

Requirements:
    pip install zenml

Setup:
    zenml init  # Initialize ZenML repository
    export ZENML_RUN_SINGLE_STEPS_WITHOUT_STACK=true  # For testing without full stack
"""

import tensorflow as tf
from zenml import pipeline, step

from mlpotion.frameworks.keras import KerasTrainingConfig
from mlpotion.integrations.zenml.keras.steps import (
    evaluate_model,
    export_model,
    load_data,
    save_model,
    train_model,
)


@step
def create_model() -> tf.keras.Model:
    """Create and compile a Keras model.

    Returns:
        Compiled Keras model ready for training.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(10,)),
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


@step
def create_training_config() -> KerasTrainingConfig:
    """Create training configuration.

    Returns:
        Training configuration with hyperparameters.
    """
    return KerasTrainingConfig(
        epochs=10,
        batch_size=8,
        learning_rate=0.001,
        validation_split=0.2,
        verbose=1,
    )


@pipeline
def keras_training_pipeline(
    file_path: str = "examples/data/sample.csv",
    label_name: str = "target",
    model_save_path: str = "/tmp/keras_model.keras",
    export_path: str = "/tmp/keras_model_export",
):
    """Complete Keras training pipeline with ZenML.

    This pipeline orchestrates the entire ML workflow:
    1. Load data from CSV
    2. Create and configure model
    3. Train model
    4. Evaluate model
    5. Save model
    6. Export model for deployment

    Args:
        file_path: Path to CSV data file.
        label_name: Name of the target column.
        model_save_path: Path to save the trained model.
        export_path: Path to export the model for serving.
    """
    # Step 1: Load data
    dataset = load_data(
        file_path=file_path,
        label_name=label_name,
        batch_size=8,
        shuffle=True,
    )

    # Step 2: Create model and config
    model = create_model()
    config = create_training_config()

    # Step 3: Train model
    trained_model, training_metrics = train_model(
        model=model,
        dataset=dataset,
        config=config,
    )

    # Step 4: Evaluate model
    evaluation_metrics = evaluate_model(
        model=trained_model,
        dataset=dataset,
        config=config,
    )

    # Step 5: Save model
    save_model(
        model=trained_model,
        file_path=model_save_path,
        save_format="keras",
    )

    # Step 6: Export model for serving
    export_model(
        model=trained_model,
        export_path=export_path,
        export_format="saved_model",
    )

    return trained_model, training_metrics, evaluation_metrics


def main() -> None:
    """Run the Keras ZenML pipeline."""
    print("=" * 60)
    print("MLPotion - Keras ZenML Pipeline")
    print("=" * 60)

    # Run the pipeline
    print("\nRunning ZenML pipeline...")
    result = keras_training_pipeline()

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print("\nOutputs:")
    print(f"  - Trained model: {type(result[0])}")
    print(f"  - Training metrics: {result[1]}")
    print(f"  - Evaluation metrics: {result[2]}")


if __name__ == "__main__":
    main()
