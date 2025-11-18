"""Basic TensorFlow usage WITHOUT ZenML.

This example demonstrates the core MLPotion TensorFlow workflow:
1. Load data from CSV
2. Optimize dataset for performance
3. Create a TensorFlow model
4. Train the model
5. Evaluate the model
6. Save and export the model
"""

import tensorflow as tf

from mlpotion.frameworks.tensorflow import (
    TFCSVDataLoader,
    TFDatasetOptimizer,
    TFModelEvaluator,
    TFModelPersistence,
    TFModelTrainer,
    TensorFlowTrainingConfig,
)


def main() -> None:
    """Run basic TensorFlow training pipeline."""
    print("=" * 60)
    print("MLPotion - TensorFlow Basic Usage")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading data...")
    loader = TFCSVDataLoader(
        file_pattern="examples/data/sample.csv",
        label_name="target",
    )
    dataset = loader.load()
    print(f"Dataset: {dataset}")

    # 2. Optimize dataset
    print("\n2. Optimizing dataset...")
    optimizer = TFDatasetOptimizer(batch_size=8, shuffle_buffer_size=100)
    dataset = optimizer.optimize(dataset)

    # 3. Create model
    print("\n3. Creating model...")
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
    print(model.summary())

    # 4. Train model
    print("\n4. Training model...")
    trainer = TFModelTrainer()
    config = TensorFlowTrainingConfig(
        epochs=10,
        batch_size=8,
        learning_rate=0.001,
        verbose=1,
    )
    result = trainer.train(model, dataset, config)

    print(f"\nTraining completed!")
    print(f"Training time: {result.training_time:.2f}s")
    print(f"Final loss: {result.metrics.get('loss', 'N/A'):.4f}")

    # 5. Evaluate model
    print("\n5. Evaluating model...")
    evaluator = TFModelEvaluator()
    eval_result = evaluator.evaluate(model, dataset, config)

    print(f"Evaluation completed in {eval_result.evaluation_time:.2f}s")
    print(f"Evaluation metrics:")
    for metric_name, metric_value in eval_result.metrics.items():
        print(f"  - {metric_name}: {metric_value:.4f}")

    # 6. Save model
    print("\n6. Saving model...")
    persistence = TFModelPersistence()
    model_path = "/tmp/tensorflow_model"
    persistence.save(model, model_path, save_format="tf")
    print(f"Model saved to: {model_path}")

    # 7. Load model
    print("\n7. Loading model...")
    loaded_model = persistence.load(model_path)
    print(f"Model loaded successfully: {type(loaded_model)}")

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()