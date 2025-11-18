# ZenML Pipeline Tutorial 🔄

Transform your MLPotion pipeline into a production-ready MLOps workflow with ZenML tracking, versioning, and reproducibility!

**Note:** ZenML is just one integration example. MLPotion is designed to be **orchestrator-agnostic** and works with Prefect, Airflow, Kubeflow, and any other orchestration platform. See [ZenML Integration Guide](../integrations/zenml.md) for extending to other orchestrators.

## Prerequisites 📋

```bash
pip install mlpotion[tensorflow,zenml]
zenml init
```

## Converting Your Pipeline 🔄

### Before: Standalone Pipeline

```python
from mlpotion.frameworks.tensorflow import TFCSVDataLoader, TFModelTrainer
import keras

# Create model
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1),
])

# Load data
loader = TFCSVDataLoader(
    file_pattern="data.csv",
    label_name="target",
    batch_size=32,
)
dataset = loader.load()

# Train
trainer = TFModelTrainer()
compile_params = {
    "optimizer": keras.optimizers.Adam(learning_rate=0.001),
    "loss": "mse",
    "metrics": ["mae"],
}
fit_params = {"epochs": 10, "verbose": 1}
history = trainer.train(model, dataset, compile_params, fit_params)
```

### After: ZenML Pipeline

```python
from zenml import pipeline
from mlpotion.integrations.zenml.tensorflow.steps import (
    load_data,
    train_model,
    evaluate_model,
)
import keras

# Create model
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1),
])

@pipeline
def ml_pipeline(model):
    # Load data into tf.data.Dataset
    dataset = load_data(
        file_path="data.csv",
        batch_size=32,
        label_name="target",
    )

    # Train model - returns (model, history)
    trained_model, history = train_model(
        model=model,
        dataset=dataset,
        epochs=10,
        learning_rate=0.001,
    )

    # Evaluate
    metrics = evaluate_model(
        model=trained_model,
        dataset=dataset,
    )

    return trained_model, metrics

# Run with full tracking!
result_model, result_metrics = ml_pipeline(model=model)
print(f"Loss: {result_metrics['loss']:.4f}")
print(f"MAE: {result_metrics['mae']:.4f}")
```

## Complete Example with All Steps 🚀

```python
from zenml import pipeline
from mlpotion.integrations.zenml.tensorflow.steps import (
    load_data,
    optimize_data,
    train_model,
    evaluate_model,
    save_model,
    export_model,
    inspect_model,
)
import keras

@pipeline
def complete_ml_pipeline(model, train_path: str, test_path: str):
    """Complete MLOps pipeline with MLPotion + ZenML."""

    # Load training and test data
    train_dataset = load_data(
        file_path=train_path,
        batch_size=32,
        label_name="target",
    )
    test_dataset = load_data(
        file_path=test_path,
        batch_size=32,
        label_name="target",
    )

    # Optimize training dataset
    train_dataset = optimize_data(
        dataset=train_dataset,
        batch_size=32,
        cache=True,
        prefetch=True,
        shuffle_buffer_size=1000,
    )

    # Train model
    trained_model, history = train_model(
        model=model,
        dataset=train_dataset,
        epochs=50,
        learning_rate=0.001,
        validation_dataset=test_dataset,
    )

    # Evaluate on test data
    metrics = evaluate_model(
        model=trained_model,
        dataset=test_dataset,
    )

    # Inspect model architecture
    inspection = inspect_model(
        model=trained_model,
        include_layers=True,
        include_signatures=True,
    )

    # Save model
    save_path = save_model(
        model=trained_model,
        save_path="models/my_model",
    )

    # Export for serving
    export_path = export_model(
        model=trained_model,
        export_path="exports/my_model",
        export_format="keras",
    )

    return metrics, export_path

# Run the pipeline
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])

metrics, export_path = complete_ml_pipeline(
    model=model,
    train_path="data/train.csv",
    test_path="data/test.csv",
)
```

## Benefits You Get 🌟

- ✅ Automatic artifact versioning
- ✅ Full pipeline lineage tracking
- ✅ Experiment comparison
- ✅ Model registry integration
- ✅ Reproducible runs
- ✅ Caching of unchanged steps
- ✅ Step output tracking with metadata

## Viewing Pipeline Runs 🔍

```bash
# List all pipeline runs
zenml pipeline runs list

# View details of a specific run
zenml pipeline runs describe <RUN_ID>

# Compare different runs
zenml pipeline runs compare <RUN_ID_1> <RUN_ID_2>
```

See the [ZenML Integration Guide](../integrations/zenml.md) for complete documentation and examples with PyTorch and Keras!

---

<p align="center">
  <strong>Ready for production MLOps!</strong> 🚀
</p>
