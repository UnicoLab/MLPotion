"""Keras training pipeline WITH ZenML orchestration.

This example demonstrates how to use MLPotion's Keras components
within a ZenML pipeline for reproducible and tracked ML workflows.

Requirements:
    pip install zenml

Setup:
    zenml init  # Initialize ZenML repository
    export ZENML_RUN_SINGLE_STEPS_WITHOUT_STACK=true  # For testing without full stack
"""
import keras
from zenml import pipeline, step

from mlpotion.integrations.zenml.keras.steps import (
    evaluate_model,
    export_model,
    load_data,
    save_model,
    train_model,
)


@step
def create_model() -> keras.Model:
    """Create and compile a Keras model.

    Returns:
        Compiled Keras model ready for training.
    """
    model = keras.Sequential(
        [
            keras.layers.Dense(64, activation="relu", input_shape=(10,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae", "mse"],
    )

    return model


@pipeline(enable_cache=False)
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

    _config_train = {
        "epochs": 10,
        "learning_rate": 0.001,
        "verbose": 1,
    }
    # Step 3: Train model
    trained_model, training_metrics = train_model(
        model=model,
        data=dataset,
        **_config_train,
    )
    # Step 4: Evaluate model
    evaluation_metrics = evaluate_model(
        model=trained_model,
        data=dataset,
        verbose=1,
    )

    # # Step 5: Save model
    save_model(
        model=trained_model,
        save_path=model_save_path,
    )

    # # Step 6: Export model for serving
    export_model(
        model=trained_model,
        export_path=export_path,
        export_format="tf",
    )
    return trained_model, training_metrics, evaluation_metrics


if __name__ == "__main__":
    """Run the Keras ZenML pipeline."""
    print("=" * 60)
    print("MLPotion - Keras ZenML Pipeline")
    print("=" * 60)

    # Initialize ZenML (if not already initialized)
    try:
        from zenml.client import Client

        client = Client()
        print(f"✅ ZenML initialized. Active stack: {client.active_stack_model.name}")
    except Exception as e:
        print(f"⚠️  ZenML client error: {e}")
        print("Run 'zenml init' if you haven't already")

    # Run the pipeline
    print("\nRunning ZenML pipeline...")
    result = keras_training_pipeline()

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
