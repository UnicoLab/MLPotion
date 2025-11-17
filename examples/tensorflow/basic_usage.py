"""Basic TensorFlow usage WITHOUT ZenML."""

import tensorflow as tf

from mlpotion.frameworks.tensorflow import (
    TFCSVDataLoader,
    TFDatasetOptimizer,
    TFModelEvaluator,
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
    optimizer = TFDatasetOptimizer(batch_size=32, shuffle_buffer_size=100)
    dataset = optimizer.optimize(dataset)

    # 3. Create model
    print("\n3. Creating model...")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # 4. Train model
    print("\n4. Training model...")
    trainer = TFModelTrainer()
    config = TensorFlowTrainingConfig(
        epochs=5,
        batch_size=32,
        learning_rate=0.001,
        verbose=1,
    )
    result = trainer.train(model, dataset, config)

    print(f"\nTraining completed!")
    print(f"Training time: {result.training_time:.2f}s")
    print(f"Final loss: {result.metrics['loss']:.4f}")

    # 5. Evaluate model
    print("\n5. Evaluating model...")
    evaluator = TFModelEvaluator()
    eval_result = evaluator.evaluate(model, dataset, config)

    print(f"Evaluation metrics: {eval_result.metrics}")

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()