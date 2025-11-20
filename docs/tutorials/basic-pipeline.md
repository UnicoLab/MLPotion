# Your First Pipeline Tutorial 🎓

Let's build a complete end-to-end ML pipeline from scratch! This tutorial covers data loading, training, evaluation, and deployment.

## What We'll Build 🎯

A regression pipeline that:
1. Loads data from CSV
2. Trains a neural network
3. Evaluates performance
4. Saves the model
5. Exports for serving

**Time**: ~15 minutes
**Level**: Beginner

## Prerequisites 📋

```bash
poetry add mlpotion -E tensorflow
```

## Step 1: Prepare Your Data 📊

```python
# create_data.py
import pandas as pd
import numpy as np

# Generate synthetic house price data
np.random.seed(42)
n_samples = 10000

data = pd.DataFrame({
    'square_feet': np.random.randint(500, 5000, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'age_years': np.random.randint(0, 100, n_samples),
    'price': np.random.randint(100000, 1000000, n_samples)
})

# Add some correlation
data['price'] = (
    data['square_feet'] * 200 +
    data['bedrooms'] * 10000 +
    data['bathrooms'] * 15000 -
    data['age_years'] * 500 +
    np.random.randn(n_samples) * 50000
)

# Split into train/test
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)

print(f"✅ Created {len(train_data)} training samples")
print(f"✅ Created {len(test_data)} test samples")
```

## Step 2: Build the Pipeline 🏗️

```python
# pipeline.py
import tensorflow as tf
from mlpotion.frameworks.tensorflow import (
    CSVDataLoader,
    DatasetOptimizer,
    ModelTrainer,
    ModelEvaluator,
    ModelPersistence,
    ModelExporter,
    ModelTrainingConfig,
    ModelExportConfig,
)

def create_model(input_dim: int) -> tf.keras.Model:
    """Create a simple neural network."""
    return tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

def main():
    print("🚀 Starting ML Pipeline...")

    # 1. Load data
    print("\n📥 Loading data...")
    train_loader = CSVDataLoader(
        file_pattern="train.csv",
        label_name="price",
        batch_size=32,
    )
    test_loader = CSVDataLoader(
        file_pattern="test.csv",
        label_name="price",
        batch_size=32,
    )

    train_dataset = train_loader.load()
    test_dataset = test_loader.load()

    # 2. Optimize datasets
    print("⚡ Optimizing datasets...")
    optimizer = DatasetOptimizer(
        batch_size=32,
        cache=True,
        prefetch=True,
    )

    train_dataset = optimizer.optimize(train_dataset)
    test_dataset = optimizer.optimize(test_dataset)

    # 3. Create model
    print("\n🏗️ Creating model...")
    model = create_model(input_dim=4)  # 4 features

    # 4. Configure training
    print("⚙️ Configuring training...")
    config = ModelTrainingConfig(
        epochs=50,
        learning_rate=0.001,
        optimizer_type="adam",
        loss="mse",
        metrics=["mae", "mse"],
        early_stopping=True,
        early_stopping_patience=10,
        verbose=1,
    )

    # 5. Train model
    print("\n🎓 Training model...")
    trainer = ModelTrainer()
    result = trainer.train(
        model,
        train_dataset,
        config,
        validation_dataset=test_dataset,
    )

    print(f"\n✅ Training complete!")
    print(f"   Final loss: {result.metrics['loss']:.2f}")
    print(f"   Final MAE: {result.metrics['mae']:.2f}")
    print(f"   Best epoch: {result.best_epoch}")
    print(f"   Training time: {result.training_time:.2f}s")

    # 6. Evaluate
    print("\n📊 Evaluating model...")
    evaluator = ModelEvaluator()
    eval_result = evaluator.evaluate(result.model, test_dataset, config)

    print(f"✅ Evaluation complete!")
    print(f"   Test MAE: ${eval_result.metrics['mae']:,.2f}")
    print(f"   Test MSE: {eval_result.metrics['mse']:,.2f}")

    # 7. Save model
    print("\n💾 Saving model...")
    persistence = ModelPersistence(
        path="models/house_price_model",
        model=result.model,
    )
    persistence.save(save_format=".keras")

    print("✅ Model saved!")

    # 8. Export for serving
    print("\n📤 Exporting model...")
    exporter = ModelExporter()
    export_config = ModelExportConfig(format="saved_model")

    export_result = exporter.export(
        result.model,
        "exports/house_price_model",
        export_config,
    )

    print(f"✅ Model exported to: {export_result.export_path}")

    print("\n🎉 Pipeline complete!")

if __name__ == "__main__":
    main()
```

## Step 3: Run the Pipeline 🏃

```bash
# Create data
python create_data.py

# Run pipeline
python pipeline.py
```

You'll see output like:

```
🚀 Starting ML Pipeline...

📥 Loading data...
⚡ Optimizing datasets...

🏗️ Creating model...
⚙️ Configuring training...

🎓 Training model...
Epoch 1/50
250/250 [==============================] - 2s 8ms/step - loss: 52345.67 - mae: 178.32 - val_loss: 48234.56 - val_mae: 165.43
...
Epoch 25/50
250/250 [==============================] - 1s 4ms/step - loss: 15234.56 - mae: 98.21 - val_loss: 16543.21 - val_mae: 102.34

✅ Training complete!
   Final loss: 15234.56
   Final MAE: 98.21
   Best epoch: 25
   Training time: 45.23s

📊 Evaluating model...
✅ Evaluation complete!
   Test MAE: $102.34
   Test MSE: 16543.21

💾 Saving model...
✅ Model saved!

📤 Exporting model...
✅ Model exported to: exports/house_price_model

🎉 Pipeline complete!
```

## Step 4: Use the Model 🔮

```python
# predict.py
import tensorflow as tf
from mlpotion.frameworks.tensorflow import ModelPersistence

# Load model
persistence = ModelPersistence(
    path="models/house_price_model",
    model=None,  # Will be loaded
)
model, metadata = persistence.load()

# Make predictions
new_house = [[2500, 3, 2, 10]]  # sq_ft, beds, baths, age
prediction = model.predict(new_house)

print(f"Predicted price: ${prediction[0][0]:,.2f}")
```

## What You Learned 🎓

1. ✅ How to load data from CSV
2. ✅ How to optimize datasets for performance
3. ✅ How to configure and train models
4. ✅ How to evaluate model performance
5. ✅ How to save and export models

## Next Steps 🚀

- [ZenML Pipeline Tutorial →](zenml-integration.md) - Add MLOps superpowers
- [Advanced Training →](advanced-training.md) - Custom callbacks and more
- [Production Deployment →](deployment.md) - Ship to production

---

<p align="center">
  <strong>Congratulations! You've built your first complete ML pipeline!</strong> 🎉
</p>
