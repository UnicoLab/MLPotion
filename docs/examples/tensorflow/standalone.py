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
    CSVDataLoader,
    DatasetOptimizer,
    ModelEvaluator,
    ModelPersistence,
    ModelTrainer,
    ModelTrainingConfig,
)


def main() -> None:
    """Run basic TensorFlow training pipeline."""
    print("=" * 60)
    print("MLPotion - TensorFlow Basic Usage")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading data...")
    loader = CSVDataLoader(
        file_pattern="examples/data/sample.csv",
        label_name="target",
        batch_size=1,  # Load unbatched, let DatasetOptimizer handle batching
    )
    dataset = loader.load()
    print(f"Dataset: {dataset}")

    # Unbatch the dataset first (since CSVDataLoader batches by default)
    dataset = dataset.unbatch()

    # Transform OrderedDict to single tensor
    def prepare_features(features, label):
        """Convert OrderedDict of features to single tensor."""
        feature_list = [features[key] for key in sorted(features.keys())]
        stacked_features = tf.stack(feature_list, axis=-1)
        return stacked_features, label

    dataset = dataset.map(prepare_features)

    # 2. Optimize dataset
    print("\n2. Optimizing dataset...")
    optimizer = DatasetOptimizer(batch_size=8, shuffle_buffer_size=100)
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
    trainer = ModelTrainer()
    config = ModelTrainingConfig(
        epochs=10,
        batch_size=8,
        learning_rate=0.001,
        verbose=1,
    )

    result = trainer.train(
        model=model,
        data=dataset,
        config=config,
    )

    print(f"\nTraining completed!")
    print(f"{result=}")

    # 5. Evaluate model
    print("\n5. Evaluating model...")
    evaluator = ModelEvaluator()
    from mlpotion.frameworks.tensorflow import ModelEvaluationConfig
    eval_config = ModelEvaluationConfig(batch_size=8, verbose=1)
    eval_result = evaluator.evaluate(
        model=model,
        data=dataset,
        config=eval_config,
    )
    print(f"{eval_result=}")

    # 6. Save model
    print("\n6. Saving model...")
    model_path = "/tmp/tensorflow_model.keras"
    persistence = ModelPersistence(
        path=model_path,
        model=model,
    )
    persistence.save(
        save_format=".keras",
    )
    print(f"Model saved to: {model_path}")

    # 7. Load model
    print("\n7. Loading model...")
    loaded_model, metadata = persistence.load()
    print(f"Model loaded successfully: {type(loaded_model)}")

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()