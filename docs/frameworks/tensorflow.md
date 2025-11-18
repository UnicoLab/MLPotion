# TensorFlow Guide 🔶

Complete guide to using MLPotion with TensorFlow - the production-ready ML framework.

## Why TensorFlow + MLPotion? 🤔

- **Production Ready**: Industry-standard deployment
- **Ecosystem**: Rich tooling (TensorBoard, TFX, TFLite)
- **Performance**: Optimized for TPUs and GPUs
- **Scalability**: Distributed training built-in
- **MLPotion Benefits**: Type-safe, modular components

## Installation 📥

```bash
pip install mlpotion[tensorflow]
```

This installs:
- `tensorflow>=2.15`
- `keras>=3.0`
- All TensorFlow-specific MLPotion components

## Quick Example 🚀

```python
from mlpotion.frameworks.tensorflow import (
    CSVDataLoader,
    ModelTrainer,
    ModelTrainingConfig,
)
import tensorflow as tf

# Load data
loader = CSVDataLoader("data.csv", label_name="target")
dataset = loader.load()

# Create model
model = tf.keras.Sequential([...])

# Train
trainer = ModelTrainer()
config = ModelTrainingConfig(epochs=10, learning_rate=0.001)
result = trainer.train(model, dataset, config)
```

## Data Loading 📊

### CSV Data Loader

```python
from mlpotion.frameworks.tensorflow import CSVDataLoader

loader = CSVDataLoader(
    file_pattern="data/*.csv",      # Supports glob patterns
    label_name="target",            # Column to use as label
    feature_columns=None,           # Auto-detect or specify
    batch_size=32,
    shuffle=True,
    shuffle_buffer_size=10000,
    num_parallel_reads=4,
    compression_type=None,          # or "GZIP", "ZLIB"
    header=True,
    field_delim=",",
    use_quote_delim=True,
    na_value="",
    select_columns=None,            # Subset of columns
)

dataset = loader.load()  # Returns tf.data.Dataset
```

**Features:**
- Automatic type inference
- Parallel reading for performance
- Memory-efficient streaming
- Glob pattern support for multiple files

### Dataset Optimization

```python
from mlpotion.frameworks.tensorflow import DatasetOptimizer

optimizer = DatasetOptimizer(
    batch_size=32,
    shuffle=True,
    shuffle_buffer_size=10000,
    cache=True,                     # Cache in memory
    cache_filename=None,            # Or cache to disk
    prefetch=True,                  # Prefetch for performance
    prefetch_buffer_size=tf.data.AUTOTUNE,
    repeat=True,                    # Repeat indefinitely
    num_parallel_calls=tf.data.AUTOTUNE,
)

optimized = optimizer.optimize(dataset)
```

**Performance Tips:**
- Use `cache()` for datasets that fit in memory
- Enable `prefetch()` to overlap data loading with training
- Set `num_parallel_calls=AUTOTUNE` for automatic optimization

## Model Training 🎓

### Basic Training

```python
from mlpotion.frameworks.tensorflow import ModelTrainer, ModelTrainingConfig

config = ModelTrainingConfig(
    epochs=10,
    learning_rate=0.001,
    batch_size=32,
    optimizer_type="adam",
    loss="mse",
    metrics=["mae"],
    verbose=1,
)

trainer = ModelTrainer()
result = trainer.train(model, train_dataset, config)

print(f"Final loss: {result.metrics['loss']:.4f}")
print(f"Training time: {result.training_time:.2f}s")
```

### Advanced Training Configuration

```python
config = ModelTrainingConfig(
    # Training parameters
    epochs=100,
    learning_rate=0.001,
    batch_size=32,

    # Optimizer configuration
    optimizer_type="adam",  # or "sgd", "adamw", "rmsprop"
    optimizer_kwargs={
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-7,
    },

    # Loss and metrics
    loss="mse",  # or custom loss function
    metrics=["mae", "mse"],

    # Validation
    validation_split=0.2,  # Use 20% for validation
    validation_freq=1,     # Validate every epoch

    # Early stopping
    early_stopping=True,
    early_stopping_patience=10,
    early_stopping_monitor="val_loss",
    early_stopping_min_delta=0.001,
    restore_best_weights=True,

    # Checkpointing
    checkpoint_filepath="checkpoints/model_{epoch:02d}",
    save_best_only=True,
    checkpoint_monitor="val_loss",
    checkpoint_mode="min",

    # Learning rate scheduling
    reduce_lr_on_plateau=True,
    reduce_lr_patience=5,
    reduce_lr_factor=0.5,
    reduce_lr_min_lr=1e-6,

    # Other
    verbose=1,
    class_weight=None,  # For imbalanced datasets
    sample_weight_mode=None,
)
```

### Custom Callbacks

```python
import tensorflow as tf

# Create custom callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="logs",
    histogram_freq=1,
)

csv_logger = tf.keras.callbacks.CSVLogger("training.log")

# Pass to trainer
config = ModelTrainingConfig(
    epochs=10,
    learning_rate=0.001,
    custom_callbacks=[tensorboard_callback, csv_logger],
)
```

### Distributed Training

```python
import tensorflow as tf
from mlpotion.frameworks.tensorflow import ModelTrainer, ModelTrainingConfig

# Create distribution strategy
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Create model within strategy scope
    model = tf.keras.Sequential([...])

    # Train as usual
    trainer = ModelTrainer()
    result = trainer.train(model, dataset, config)
```

## Model Evaluation 📊

```python
from mlpotion.frameworks.tensorflow import ModelEvaluator, ModelEvaluationConfig

config = ModelEvaluationConfig(
    batch_size=32,
    verbose=1,
    return_dict=True,
)

evaluator = ModelEvaluator()
result = evaluator.evaluate(model, test_dataset, config)

print(f"Test loss: {result.metrics['loss']:.4f}")
print(f"Test MAE: {result.metrics['mae']:.4f}")
```

## Model Persistence 💾

### Saving Models

```python
from mlpotion.frameworks.tensorflow import ModelPersistence

# Save in TensorFlow format (recommended)
persistence = ModelPersistence(
    path="models/my_model",
    model=model,
)
persistence.save(save_format=".keras")

# Save in H5 format
persistence_h5 = ModelPersistence(
    path="models/my_model.h5",
    model=model,
)
persistence_h5.save(save_format="h5")
```

### Loading Models

```python
# Load full model
persistence = ModelPersistence(
    path="models/my_model",
    model=None,  # Will be loaded
)
loaded_model = persistence.load()

# Load with custom objects
persistence_custom = ModelPersistence(
    path="models/my_model",
    model=None,
)
loaded_model = persistence_custom.load(
    custom_objects={"CustomLayer": CustomLayer}
)
```

## Model Export 📤

### SavedModel Format (Recommended for Serving)

```python
from mlpotion.frameworks.tensorflow import ModelExporter, ModelExportConfig

exporter = ModelExporter()

config = ModelExportConfig(
    format="saved_model",
    include_optimizer=False,  # Smaller size for inference
    signatures=None,  # Auto-detect
)

result = exporter.export(model, "exports/saved_model", config)
print(f"Exported to: {result.export_path}")
```

### TFLite (Mobile/Edge Devices)

```python
config = ModelExportConfig(
    format="tflite",
    optimization="default",  # or "float16", "int8"
    representative_dataset=None,  # For int8 quantization
)

result = exporter.export(model, "exports/model.tflite", config)
```

### TensorFlow.js (Web Deployment)

```python
config = ModelExportConfig(
    format="tfjs",
    quantization_dtype="float16",  # Smaller model size
)

result = exporter.export(model, "exports/tfjs_model", config)
```

## Model Inspection 🔍

```python
from mlpotion.frameworks.tensorflow import ModelInspector

inspector = ModelInspector()
info = inspector.inspect(model)

print(f"Model name: {info.name}")
print(f"Backend: {info.backend}")
print(f"Trainable: {info.trainable}")
print(f"Total params: {info.parameters['total']}")
print(f"Trainable params: {info.parameters['trainable']}")

# Inspect inputs
for inp in info.inputs:
    print(f"Input: {inp['name']}, shape: {inp['shape']}, dtype: {inp['dtype']}")

# Inspect outputs
for out in info.outputs:
    print(f"Output: {out['name']}, shape: {out['shape']}, dtype: {out['dtype']}")

# Inspect layers
for layer in info.layers:
    print(f"Layer: {layer['name']}, type: {layer['type']}, params: {layer['params']}")
```

## Common Patterns 🎯

### Pattern: Train-Val-Test Pipeline

```python
from mlpotion.frameworks.tensorflow import (
    CSVDataLoader,
    DatasetOptimizer,
    ModelTrainer,
    ModelEvaluator,
    ModelTrainingConfig,
)

# Load splits
train_loader = CSVDataLoader("train.csv", label_name="target")
val_loader = CSVDataLoader("val.csv", label_name="target")
test_loader = CSVDataLoader("test.csv", label_name="target")

train_data = train_loader.load()
val_data = val_loader.load()
test_data = test_loader.load()

# Optimize
optimizer = DatasetOptimizer(batch_size=32, cache=True, prefetch=True)
train_data = optimizer.optimize(train_data)
val_data = optimizer.optimize(val_data)
test_data = optimizer.optimize(test_data)

# Train
trainer = ModelTrainer()
config = ModelTrainingConfig(
    epochs=50,
    learning_rate=0.001,
    early_stopping=True,
    early_stopping_patience=10,
)

result = trainer.train(model, train_data, config, validation_dataset=val_data)

# Evaluate
evaluator = ModelEvaluator()
test_metrics = evaluator.evaluate(result.model, test_data, config)

print(f"Best epoch: {result.best_epoch}")
print(f"Test accuracy: {test_metrics.metrics['accuracy']:.2%}")
```

### Pattern: Hyperparameter Tuning

```python
from itertools import product

# Define hyperparameters to try
learning_rates = [0.0001, 0.001, 0.01]
batch_sizes = [16, 32, 64]
optimizers = ["adam", "rmsprop"]

best_val_loss = float('inf')
best_params = {}

for lr, bs, opt in product(learning_rates, batch_sizes, optimizers):
    print(f"\nTrying: lr={lr}, batch_size={bs}, optimizer={opt}")

    # Create fresh model
    model = create_model()

    # Configure training
    config = ModelTrainingConfig(
        epochs=30,
        learning_rate=lr,
        batch_size=bs,
        optimizer_type=opt,
        verbose=0,
        early_stopping=True,
        early_stopping_patience=5,
    )

    # Optimize dataset with new batch size
    optimizer = DatasetOptimizer(batch_size=bs)
    train_opt = optimizer.optimize(train_data)
    val_opt = optimizer.optimize(val_data)

    # Train
    result = trainer.train(model, train_opt, config, validation_dataset=val_opt)

    # Check if best
    val_loss = result.history['val_loss'][-1]
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = {'lr': lr, 'batch_size': bs, 'optimizer': opt}
        # Save best model
        persistence.save(result.model, "models/best_model")
        print(f"✨ New best! Val loss: {val_loss:.4f}")

print(f"\n🏆 Best params: {best_params}")
print(f"🏆 Best val loss: {best_val_loss:.4f}")
```

### Pattern: Mixed Precision Training

```python
import tensorflow as tf

# Enable mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Create model (automatically uses mixed precision)
model = create_model()

# Loss scaling for numerical stability
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

# Train as usual
config = ModelTrainingConfig(
    epochs=10,
    learning_rate=0.001,
    optimizer=optimizer,  # Pass configured optimizer
)

result = trainer.train(model, dataset, config)
```

### Pattern: Custom Training Loop

```python
from mlpotion.frameworks.tensorflow import ModelTrainer

class CustomTrainer(ModelTrainer):
    def train_step(self, model, batch):
        """Custom training step logic."""
        features, labels = batch

        with tf.GradientTape() as tape:
            predictions = model(features, training=True)
            loss = self.loss_fn(labels, predictions)

            # Add custom regularization
            regularization_loss = sum(model.losses)
            total_loss = loss + regularization_loss

        # Compute gradients
        gradients = tape.gradient(total_loss, model.trainable_variables)

        # Custom gradient clipping
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return {'loss': loss, 'total_loss': total_loss}

# Use custom trainer
custom_trainer = CustomTrainer()
result = custom_trainer.train(model, dataset, config)
```

## Integration with TensorFlow Ecosystem 🔧

### TensorBoard Integration

```python
import tensorflow as tf
from mlpotion.frameworks.tensorflow import ModelTrainer, ModelTrainingConfig

# Create TensorBoard callback
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir="logs",
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq="epoch",
    profile_batch="10,20",
)

config = ModelTrainingConfig(
    epochs=50,
    learning_rate=0.001,
    custom_callbacks=[tensorboard],
)

trainer = ModelTrainer()
result = trainer.train(model, dataset, config)

# View results
# tensorboard --logdir=logs
```

### TensorFlow Datasets Integration

```python
import tensorflow_datasets as tfds
from mlpotion.frameworks.tensorflow import DatasetOptimizer

# Load from TFDS
(train_ds, val_ds, test_ds), info = tfds.load(
    'mnist',
    split=['train[:80%]', 'train[80%:]', 'test'],
    with_info=True,
    as_supervised=True,
)

# Optimize with MLPotion
optimizer = DatasetOptimizer(batch_size=32, cache=True, prefetch=True)
train_ds = optimizer.optimize(train_ds)

# Train as usual
result = trainer.train(model, train_ds, config)
```

## Best Practices 💡

1. **Use Dataset Optimization**: Always use `DatasetOptimizer` for better performance
2. **Enable Prefetching**: Set `prefetch=True` to overlap data loading with training
3. **Cache Small Datasets**: Use `cache=True` if dataset fits in memory
4. **Early Stopping**: Enable to prevent overfitting
5. **SavedModel Format**: Use for production deployment
6. **Mixed Precision**: Enable for faster training on modern GPUs

## Troubleshooting 🔧

### Issue: Out of Memory

**Solution:** Reduce batch size or use gradient accumulation

```python
config = ModelTrainingConfig(
    batch_size=16,  # Smaller batches
    # Or use gradient accumulation
)
```

### Issue: Slow Data Loading

**Solution:** Enable optimization and parallel reading

```python
loader = CSVDataLoader(
    file_pattern="data.csv",
    num_parallel_reads=8,  # More parallel readers
)

optimizer = DatasetOptimizer(
    prefetch=True,
    num_parallel_calls=tf.data.AUTOTUNE,
)
```

## Next Steps 🚀

- [PyTorch Guide →](pytorch.md) - Compare with PyTorch
- [ZenML Integration →](../integrations/zenml.md) - Add MLOps
- [API Reference →](../api/frameworks/tensorflow.md) - Detailed API docs

---

<p align="center">
  <strong>TensorFlow + MLPotion = Production ML Made Easy!</strong> 🔶
</p>
