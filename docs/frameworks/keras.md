# Keras Guide 🎨

Complete guide to using MLPotion with Keras 3 - the user-friendly, backend-agnostic ML framework!

## Why Keras + MLPotion? 🤔

- **User-Friendly**: Simple, consistent API
- **Backend-Agnostic**: Switch between TensorFlow, PyTorch, and JAX
- **Fast Prototyping**: Build models quickly
- **Production-Ready**: Keras 3 is production-grade
- **MLPotion Benefits**: Type-safe, modular components

## Installation 📥

```bash
# Keras with TensorFlow backend (default)
poetry add mlpotion -E tensorflow

# Keras with PyTorch backend
poetry add mlpotion -E keras-pytorch

# Keras with JAX backend
poetry add mlpotion -E keras-jax
```

## Quick Example 🚀

```python
from mlpotion.frameworks.keras import (
    CSVDataLoader,
    ModelTrainer,
    ModelTrainingConfig,
)
import keras

# Load data
loader = CSVDataLoader("data.csv", label_name="target", batch_size=32)
dataset = loader.load()

# Create model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

# Train
trainer = ModelTrainer()
config = ModelTrainingConfig(
    epochs=10,
    learning_rate=0.001,
    optimizer="adam",
    loss="mse",
    metrics=["mae"],
)
result = trainer.train(model, dataset, config)

print(f"Final loss: {result.metrics['loss']:.4f}")
```

## Advanced Training 🎓

### Custom Optimizers, Loss, and Metrics

```python
import keras

# Custom optimizer instance
custom_optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    clipnorm=1.0,
)

# Custom loss instance
custom_loss = keras.losses.Huber(delta=1.0)

# Custom metrics
custom_metric = keras.metrics.MeanAbsoluteError(name="mae")

config = ModelTrainingConfig(
    epochs=50,
    batch_size=32,
    optimizer=custom_optimizer,  # Optimizer instance
    loss=custom_loss,            # Loss instance
    metrics=[custom_metric, "mse"],  # Mix of instances and strings
)

result = trainer.train(model, dataset, config)
```

### Callbacks and TensorBoard

```python
# Method 1: Pass callback instances
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

csv_logger = keras.callbacks.CSVLogger("training.log")

config = ModelTrainingConfig(
    epochs=100,
    learning_rate=0.001,
    callbacks=[early_stopping, csv_logger],
    use_tensorboard=True,  # Enabled by default
    tensorboard_log_dir="logs/keras_experiment",
)

# Method 2: Pass callback configs as dicts
config = ModelTrainingConfig(
    epochs=100,
    learning_rate=0.001,
    callbacks=[
        {
            "name": "EarlyStopping",
            "params": {
                "monitor": "val_loss",
                "patience": 5,
                "restore_best_weights": True,
            }
        },
        {
            "name": "ReduceLROnPlateau",
            "params": {
                "monitor": "val_loss",
                "factor": 0.5,
                "patience": 3,
            }
        },
    ],
)

result = trainer.train(model, dataset, config, validation_dataset=val_dataset)

# View TensorBoard
# tensorboard --logdir=logs/keras_experiment
```

## Complete Example 🚀

```python
from mlpotion.frameworks.keras import (
    CSVDataLoader,
    ModelTrainer,
    ModelTrainingConfig,
)
import keras

# Load data
loader = CSVDataLoader("data.csv", label_name="target", batch_size=32)
dataset = loader.load()

# Create model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

# Configure training with all features
config = ModelTrainingConfig(
    epochs=100,
    learning_rate=0.001,
    batch_size=32,

    # Custom optimizer
    optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),

    # Loss and metrics
    loss="mse",
    metrics=["mae", keras.metrics.RootMeanSquaredError()],

    # Callbacks
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        {"name": "ReduceLROnPlateau", "params": {"factor": 0.5, "patience": 5}},
    ],

    # TensorBoard
    use_tensorboard=True,
    tensorboard_log_dir="logs/my_experiment",
    tensorboard_params={
        "histogram_freq": 1,
        "write_graph": True,
    },
)

# Train
trainer = ModelTrainer()
result = trainer.train(model, dataset, config)

print(f"Training completed in {result.training_time:.2f}s")
print(f"Final loss: {result.metrics['loss']:.4f}")
```

For complete Keras documentation, see the [TensorFlow Guide](tensorflow.md) as Keras components use the same patterns!

---

<p align="center">
  <strong>Keras + MLPotion = Simplicity + Power!</strong> 🎨
</p>
