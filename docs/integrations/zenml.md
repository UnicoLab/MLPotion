# ZenML Integration 🔄

Transform your MLPotion pipelines into production-ready MLOps workflows with ZenML!

**Important:** ZenML is just one integration example. MLPotion is designed to be **framework and orchestrator agnostic**. You can easily extend MLPotion to work with **Prefect**, **Airflow**, **Kubeflow**, or any other orchestration platform. Community contributions welcome - see [Contributing Guide](../contributing/overview.md)!

## Why ZenML? 🤔

- **Reproducibility**: Track every pipeline run with full lineage
- **Versioning**: Automatic artifact and model versioning
- **Collaboration**: Share pipelines with your team
- **Scalability**: Run on different compute backends
- **Observability**: Monitor pipeline health and performance

## Installation 📥

```bash
# TensorFlow + ZenML
poetry add mlpotion -E tensorflow -E zenml

# PyTorch + ZenML
poetry add mlpotion -E pytorch -E zenml

# Initialize ZenML (first time only)
zenml init
```

## Quick Example 🚀


```python
from zenml import pipeline, step
from mlpotion.integrations.zenml.tensorflow.steps import (
    load_data,
    train_model,
    evaluate_model,
)
import keras

# custom model init step
@step
def init_model() -> keras.Model:
    """Initialize the model.

    Note: When using label_name with load_data, the dataset returns
    (features_dict, labels) where features_dict is a dictionary.
    The model must accept dict inputs matching the CSV column names.
    """
    # Create model that accepts dict inputs (matching CSV columns)
    inputs = {
        "feature_1": keras.Input(shape=(1,), name="feature_1"),
        "feature_2": keras.Input(shape=(1,), name="feature_2"),
        "feature_3": keras.Input(shape=(1,), name="feature_3"),
    }
    concatenated = keras.layers.Concatenate()(list(inputs.values()))
    x = keras.layers.Dense(10, activation="relu")(concatenated)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

@pipeline
def ml_pipeline():
    """Simple ML pipeline with MLPotion + ZenML."""
    # Load data into tf.data.Dataset
    dataset = load_data(
        file_path="data.csv",
        batch_size=32,
        label_name="target",
    )
    # load model
    model = init_model()

    # Train model
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

    return metrics

# Run the pipeline
# Note: ZenML pipelines return a PipelineRunResponse, not the actual return values
run = ml_pipeline()

# Access step outputs from the pipeline run
# Get the step run by step name and load the artifact
evaluate_step = run.steps["evaluate_model"]
metrics = evaluate_step.output.load()  # Use .load() to get the actual value

# Now you can access the metrics dictionary
print(f"Loss: {metrics['loss']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
```

## Available ZenML Steps 📦

### TensorFlow Steps

```python
from mlpotion.integrations.zenml.tensorflow.steps import (
    load_data,          # Load CSV data into tf.data.Dataset
    optimize_data,      # Optimize dataset for training (caching, prefetch, etc.)
    transform_data,     # Transform data using model and save predictions
    train_model,        # Train TensorFlow/Keras model
    evaluate_model,     # Evaluate model performance
    save_model,         # Save model to disk
    load_model,         # Load model from disk
    export_model,       # Export model for serving (SavedModel, TFLite, etc.)
    inspect_model,      # Inspect model architecture
)
```

### PyTorch Steps

```python
from mlpotion.integrations.zenml.pytorch.steps import (
    load_csv_data,              # Load CSV into PyTorch DataLoader
    load_streaming_csv_data,    # Load large CSV as streaming DataLoader
    train_model,                # Train PyTorch model
    evaluate_model,             # Evaluate PyTorch model
    save_model,                 # Save PyTorch model (state_dict or full)
    load_model,                 # Load PyTorch model
    export_model,               # Export model (TorchScript, ONNX, state_dict)
)
```

### Keras Steps

```python
from mlpotion.integrations.zenml.keras.steps import (
    load_data,          # Load CSV into CSVSequence
    transform_data,     # Transform data with predictions
    train_model,        # Train Keras model
    evaluate_model,     # Evaluate Keras model
    save_model,         # Save Keras model
    load_model,         # Load Keras model
    export_model,       # Export Keras model
    inspect_model,      # Inspect Keras model
)
```

## Complete Production Pipeline 🚀

### TensorFlow Example

```python
from zenml import pipeline
from mlpotion.integrations.zenml.tensorflow.steps import (
    load_data,
    optimize_data,
    train_model,
    evaluate_model,
    save_model,
    export_model,
)
import keras

# custom model init step
@step
def init_model() -> keras.Model:
    """Initialize the model.

    Note: When using label_name with load_data, the dataset returns
    (features_dict, labels) where features_dict is a dictionary.
    The model must accept dict inputs matching the CSV column names.
    """
    # Create model that accepts dict inputs (matching CSV columns)
    inputs = {
        "feature_1": keras.Input(shape=(1,), name="feature_1"),
        "feature_2": keras.Input(shape=(1,), name="feature_2"),
        "feature_3": keras.Input(shape=(1,), name="feature_3"),
    }
    concatenated = keras.layers.Concatenate()(list(inputs.values()))
    x = keras.layers.Dense(10, activation="relu")(concatenated)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

@pipeline
def production_ml_pipeline(
    train_data: str,
    test_data: str,
    model_name: str,
    epochs: int = 50,
):
    """Full production ML pipeline for TensorFlow."""
    # Load data
    train_dataset = load_data(
        file_path=train_data,
        batch_size=32,
        label_name="target",
    )
    test_dataset = load_data(
        file_path=test_data,
        batch_size=32,
        label_name="target",
    )

    # Optimize datasets
    train_dataset = optimize_data(
        dataset=train_dataset,
        batch_size=32,
        cache=True,
        prefetch=True,
        shuffle_buffer_size=1000,
    )
    # init model
    model = init_model()

    # Train model
    trained_model, history = train_model(
        model=model,
        dataset=train_dataset,
        epochs=epochs,
        learning_rate=0.001,
        validation_dataset=test_dataset,
    )

    # Evaluate
    metrics = evaluate_model(
        model=trained_model,
        dataset=test_dataset,
    )

    # Save model
    save_path = save_model(
        model=trained_model,
        save_path=f"models/{model_name}",
    )

    # Export for serving
    export_path = export_model(
        model=trained_model,
        export_path=f"exports/{model_name}",
        export_format="keras",
    )

    return metrics, export_path


# Run the pipeline
# Note: ZenML pipelines return a PipelineRunResponse, not the actual return values
run = production_ml_pipeline(
    train_data="s3://bucket/train.csv",
    test_data="s3://bucket/test.csv",
    model_name="my-model-v1",
    epochs=50,
)

# Access step outputs from the pipeline run
# Use .load() to get the actual artifact values
evaluate_step = run.steps["evaluate_model"]
metrics = evaluate_step.output.load()

export_step = run.steps["export_model"]
export_path = export_step.output.load()

print(f"Metrics: {metrics}")
print(f"Export path: {export_path}")
```

### PyTorch Example

```python
from zenml import pipeline
from mlpotion.integrations.zenml.pytorch.steps import (
    load_csv_data,
    train_model,
    evaluate_model,
    export_model,
)
import torch.nn as nn

@step
def init_model() -> nn.Module:
    """Initialize the model."""
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    return model

@pipeline
def pytorch_ml_pipeline(
    train_data: str,
    test_data: str,
    model_name: str,
    epochs: int = 50,
):
    """Full production ML pipeline for PyTorch."""
    # Load data
    train_loader = load_csv_data(
        file_path=train_data,
        batch_size=32,
        label_name="target",
        shuffle=True,
        num_workers=4,
    )
    test_loader = load_csv_data(
        file_path=test_data,
        batch_size=32,
        label_name="target",
        shuffle=False,
    )
    # init model
    model = init_model()

    # Train model
    trained_model, train_metrics = train_model(
        model=model,
        dataloader=train_loader,
        epochs=epochs,
        learning_rate=0.001,
        device="cuda",
        validation_dataloader=test_loader,
    )

    # Evaluate
    eval_metrics = evaluate_model(
        model=trained_model,
        dataloader=test_loader,
        device="cuda",
    )

    # Export for serving
    export_path = export_model(
        model=trained_model,
        export_path=f"exports/{model_name}",
        export_format="torchscript",
        jit_mode="script",
    )

    return eval_metrics, export_path

# Run the pipeline
# Note: ZenML pipelines return a PipelineRunResponse, not the actual return values
run = pytorch_ml_pipeline(
    train_data="data/train.csv",
    test_data="data/test.csv",
    model_name="pytorch-model-v1",
    epochs=50,
)

# Access step outputs from the pipeline run
# Use .load() to get the actual artifact values
evaluate_step = run.steps["evaluate_model"]
eval_metrics = evaluate_step.output.load()

export_step = run.steps["export_model"]
export_path = export_step.output.load()

print(f"Evaluation metrics: {eval_metrics}")
print(f"Export path: {export_path}")
```


## Benefits of ZenML Integration 🌟

### 1. Automatic Tracking

Every pipeline run is tracked automatically:
- Input data versions
- Model versions
- Hyperparameters
- Training metrics
- Output artifacts

### 2. Artifact Caching

ZenML caches artifacts, so unchanged steps are skipped:

```python
# First run: All steps execute
result1 = ml_pipeline()

# Second run: Only changed steps execute
result2 = ml_pipeline()  # Much faster!
```

### 3. Experiment Comparison

Compare different runs:

```bash
# View all pipeline runs
zenml pipeline runs list

# Compare specific runs
zenml pipeline runs compare RUN_ID_1 RUN_ID_2
```

### 4. Model Registry

Automatically version and register models:

```python
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer

# Models are automatically registered
# Access them via ZenML's model registry
```

## Custom ZenML Steps 🔧

Create your own MLPotion-based ZenML steps:

```python
from zenml import step
from mlpotion.frameworks.tensorflow import TFModelTrainer
import keras

@step
def custom_train_step(
    model: keras.Model,
    dataset,
    epochs: int = 10,
    learning_rate: float = 0.001,
):
    """Custom training step with MLPotion."""
    trainer = TFModelTrainer()

    compile_params = {
        "optimizer": keras.optimizers.Adam(learning_rate=learning_rate),
        "loss": "mse",
        "metrics": ["mae"],
    }

    fit_params = {
        "epochs": epochs,
        "verbose": 1,
    }

    history = trainer.train(
        model=model,
        data=dataset,
        compile_params=compile_params,
        fit_params=fit_params,
    )
    return model, history
```

## Extending Beyond ZenML 🚀

**MLPotion is not limited to ZenML!** The same modular components work with any orchestrator:

### Prefect Integration Example

```python
from prefect import task, flow
from mlpotion.frameworks.tensorflow import TFCSVDataLoader, TFModelTrainer

@task
def load_data_task(file_path: str):
    loader = TFCSVDataLoader(file_pattern=file_path, label_name="target", batch_size=32)
    return loader.load()

@task
def train_model_task(model, dataset, epochs: int = 10):
    trainer = TFModelTrainer()
    compile_params = {"optimizer": "adam", "loss": "mse", "metrics": ["mae"]}
    fit_params = {"epochs": epochs}
    return trainer.train(model, dataset, compile_params, fit_params)

@flow
def ml_flow():
    dataset = load_data_task("data.csv")
    result = train_model_task(model, dataset, epochs=10)
    return result
```

### Airflow Integration Example

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from mlpotion.frameworks.tensorflow import TFCSVDataLoader, TFModelTrainer

def load_data(**context):
    loader = TFCSVDataLoader(file_pattern="data.csv", label_name="target", batch_size=32)
    dataset = loader.load()
    # Store dataset reference or materialize
    context['ti'].xcom_push(key='dataset', value=dataset)

def train_model(**context):
    dataset = context['ti'].xcom_pull(key='dataset')
    trainer = TFModelTrainer()
    # Training logic...

with DAG('ml_pipeline', ...) as dag:
    load_task = PythonOperator(task_id='load_data', python_callable=load_data)
    train_task = PythonOperator(task_id='train_model', python_callable=train_model)
    load_task >> train_task
```

**Community contributions welcome!** Want to add official support for your favorite orchestrator? See [Contributing Guide](../contributing/overview.md)!

## Next Steps 🗺️

- [ZenML Documentation](https://docs.zenml.io/) - Learn more about ZenML
- [API Reference →](../api/integrations/zenml.md) - Detailed API docs
- [Contributing Guide →](../contributing/overview.md) - Add new integrations

---

<p align="center">
  <strong>MLPotion: Built for extensibility, works with any orchestrator!</strong> 🔄
</p>
