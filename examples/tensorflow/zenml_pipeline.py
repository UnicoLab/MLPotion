"""TensorFlow training pipeline WITH ZenML orchestration.

This example demonstrates how to use MLPotion's TensorFlow components
within a ZenML pipeline for reproducible and tracked ML workflows.

Requirements:
    pip install zenml

Setup:
    zenml init  # Initialize ZenML repository
    export ZENML_RUN_SINGLE_STEPS_WITHOUT_STACK=true  # For testing without full stack
"""

import tensorflow as tf
from zenml import pipeline, step

from mlpotion.frameworks.tensorflow import ModelTrainingConfig
from mlpotion.integrations.zenml.tensorflow.steps import (
    evaluate_model,
    export_model,
    load_data,
    optimize_data,
    save_model,
    train_model,
)


@step(enable_cache=False)  # Disable caching to ensure fresh model
def create_model() -> tf.keras.Model:
    """Create and compile a TensorFlow model that accepts dict inputs.

    Returns:
        Compiled TensorFlow/Keras model ready for training.
    """
    # Create inputs for each feature (10 features: feature_0 to feature_9)
    # After batching, make_csv_dataset produces tensors with shape (batch_size,) for each scalar feature
    # The materializer now correctly preserves this shape as (None,) where None is the batch dimension
    inputs = {}
    feature_list = []

    for i in range(10):
        # Each input has shape (1,) per sample after batching and materializer roundtrip
        # The materializer preserves the concrete shape (batch_size, 1)
        inp = tf.keras.Input(shape=(1,), name=f"feature_{i}", dtype=tf.float32)
        inputs[f"feature_{i}"] = inp
        # Already shape (batch_size, 1), no need to reshape
        feature_list.append(inp)

    # Concatenate all features along the last axis
    # This will create shape (batch_size, 10)
    concatenated = tf.keras.layers.Concatenate(axis=-1)(feature_list)

    # Build the model architecture
    x = tf.keras.layers.Dense(64, activation="relu")(concatenated)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1)(x)

    # Create the functional model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae", "mse"],
    )

    return model


@step
def create_training_config() -> ModelTrainingConfig:
    """Create training configuration.

    Returns:
        Training configuration with hyperparameters.
    """
    return ModelTrainingConfig(
        epochs=10,
        batch_size=8,
        learning_rate=0.001,
        verbose=1,
    )


@pipeline(enable_cache=False)
def tensorflow_training_pipeline(
    file_path: str = "examples/data/sample.csv",
    label_name: str = "target",
    model_save_path: str = "/tmp/tensorflow_model.keras",
    export_path: str = "/tmp/tensorflow_model_export",
):
    """Complete TensorFlow training pipeline with ZenML.

    This pipeline orchestrates the entire ML workflow:
    1. Load data from CSV
    2. Optimize dataset for performance
    3. Create and configure model
    4. Train model
    5. Evaluate model
    6. Save model
    7. Export model for deployment

    Args:
        file_path: Path to CSV data file.
        label_name: Name of the target column.
        model_save_path: Path to save the trained model.
        export_path: Path to export the model for serving.
    """
    # Step 1: Load data
    dataset = load_data(
        file_path=file_path,
        batch_size=1,
        label_name=label_name,
    )

    # Step 2: Optimize dataset
    optimized_dataset = optimize_data(
        dataset=dataset,
        batch_size=8,
        shuffle_buffer_size=100,
    )

    # Step 3: Create model and config
    model = create_model()

    # Step 4: Train model
    _config_train = {
        "epochs": 10,
        "learning_rate": 0.001,
        "verbose": 1,
    }
    trained_model, training_metrics = train_model(
        model=model,
        dataset=optimized_dataset,
        **_config_train,
    )

    # Step 5: Evaluate model
    evaluation_metrics = evaluate_model(
        model=trained_model,
        dataset=optimized_dataset,
    )

    # Step 6: Save model
    save_model(
        model=trained_model,
        save_path=model_save_path,
    )

    # Step 7: Export model for serving
    export_model(
        model=trained_model,
        export_path=export_path,
        export_format="keras",
    )

    return trained_model, training_metrics, evaluation_metrics


if __name__ == "__main__":
    """Run the TensorFlow ZenML pipeline."""
    print("=" * 60)
    print("MLPotion - TensorFlow ZenML Pipeline")
    print("=" * 60)

    # Run the pipeline
    print("\nRunning ZenML pipeline...")
    result = tensorflow_training_pipeline()

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")

