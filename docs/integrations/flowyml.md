# FlowyML Integration

MLPotion provides a **first-class integration** with [FlowyML](https://github.com/UnicoLab/FlowyML),
an artifact-centric ML pipeline framework. Every MLPotion component (data loaders,
trainers, evaluators, exporters) is exposed as a fully-wired FlowyML `@step` that
returns **typed artifacts** — `Dataset`, `Model`, and `Metrics` — with automatic
metadata extraction, lineage tracking, and DAG resolution.

---

## Quick Start

```bash
# Install MLPotion with FlowyML support
pip install mlpotion[flowyml,keras]
```

### Train a Keras Model in 5 Lines

```python
from flowyml.core.context import Context
from mlpotion.integrations.flowyml.keras import (
    create_keras_training_pipeline,
)

ctx = Context(
    file_path="data/train.csv",
    label_name="target",
    batch_size=32,
    epochs=20,
    learning_rate=0.001,
    experiment_name="quick-start",
)

pipeline = create_keras_training_pipeline(context=ctx)
result = pipeline.run()
```

---

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    FlowyML Runtime                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ DAG      │  │ Caching  │  │ Retry    │  │ Resource │ │
│  │ Resolver │  │ Engine   │  │ Logic    │  │ Scheduler│ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
│                        │                                  │
│            ┌───────────┼───────────┐                      │
│            ▼           ▼           ▼                      │
│      ┌──────────┐ ┌──────────┐ ┌──────────┐             │
│      │  Keras   │ │ PyTorch  │ │ TF/Keras │  ← Steps    │
│      │  Steps   │ │  Steps   │ │  Steps   │             │
│      └──────────┘ └──────────┘ └──────────┘             │
│            │           │           │                      │
└────────────┼───────────┼───────────┼──────────────────────┘
             ▼           ▼           ▼
      ┌──────────────────────────────────────┐
      │        MLPotion Components           │
      │  CSVDataLoader · ModelTrainer ·      │
      │  ModelEvaluator · ModelExporter ·    │
      │  ModelPersistence · ModelInspector   │
      └──────────────────────────────────────┘
```

### FlowyML Features Used

| Feature | How We Use It |
|---------|--------------|
| **Artifact types** | Every step returns `Dataset`, `Model`, or `Metrics` with auto-metadata |
| **DAG wiring** | `inputs`/`outputs` on `@step` enable auto-resolution between steps |
| **Caching** | `cache="code_hash"` or `cache="input_hash"` skips unchanged steps |
| **Retry** | `retry=1` on training steps for transient failures |
| **Resource specs** | `ResourceRequirements(cpu, memory, gpu)` on GPU-intensive steps |
| **Execution groups** | `execution_group="training"` groups training + eval on same node |
| **Tags** | `tags={"framework": "keras"}` for filtering and observability |
| **Context injection** | `Context(...)` provides hyperparameters to all steps |
| **Conditional flows** | `If(condition=..., then_steps=[...])` for conditional deployment |
| **Scheduling** | `PipelineScheduler` with cron syntax for periodic retraining |
| **Checkpointing** | `enable_checkpointing=True` for resumable long-running pipelines |
| **Experiment tracking** | `FlowymlKerasCallback` auto-captures all training metrics live |
| **Lineage** | `parent=` links transformed datasets back to their source |

---

## Supported Frameworks

### Keras

```python
from mlpotion.integrations.flowyml.keras import (
    # Steps
    load_data,
    transform_data,
    train_model,
    evaluate_model,
    export_model,
    save_model,
    load_model,
    inspect_model,
    # Pipeline templates
    create_keras_training_pipeline,
    create_keras_full_pipeline,
    create_keras_evaluation_pipeline,
    create_keras_export_pipeline,
    create_keras_experiment_pipeline,
    create_keras_scheduled_pipeline,
)
```

### PyTorch

```python
from mlpotion.integrations.flowyml.pytorch import (
    # Steps
    load_csv_data,
    load_streaming_csv_data,
    train_model,
    evaluate_model,
    export_model,
    save_model,
    load_model,
    # Pipeline templates
    create_pytorch_training_pipeline,
    create_pytorch_full_pipeline,
    create_pytorch_evaluation_pipeline,
    create_pytorch_export_pipeline,
    create_pytorch_experiment_pipeline,
    create_pytorch_scheduled_pipeline,
)
```

### TensorFlow

```python
from mlpotion.integrations.flowyml.tensorflow import (
    # Steps
    load_data,
    optimize_data,
    transform_data,
    train_model,
    evaluate_model,
    export_model,
    save_model,
    load_model,
    inspect_model,
    # Pipeline templates
    create_tf_training_pipeline,
    create_tf_full_pipeline,
    create_tf_evaluation_pipeline,
    create_tf_export_pipeline,
    create_tf_experiment_pipeline,
    create_tf_scheduled_pipeline,
)
```

---

## Steps Reference

### Data Steps

Every framework provides a `load_data` step that returns a **Dataset** asset:

```python
from mlpotion.integrations.flowyml.keras import load_data

# Returns Dataset asset with auto-extracted metadata
dataset = load_data(
    file_path="data/train.csv",
    label_name="target",
    batch_size=32,
)

# Access the raw data
raw_sequence = dataset.data

# Access metadata
print(dataset.metadata.properties)
# {'source': 'data/train.csv', 'batch_size': 32, ...}
```

### Training Steps

All training steps return a **(Model, Metrics)** tuple:

```python
from mlpotion.integrations.flowyml.keras import train_model

model_asset, metrics_asset = train_model(
    model=my_keras_model,
    data=dataset,               # Accepts Dataset asset or raw data
    epochs=20,
    learning_rate=0.001,
    experiment_name="v1",       # Enables FlowyML experiment tracking
)

# Access the trained model
trained_model = model_asset.data

# Access metrics
print(metrics_asset.get_metric("loss"))
print(metrics_asset.values)     # All metrics as dict
```

Training steps include:

- **`execution_group="training"`** — groups training and evaluation on the same compute node
- **`ResourceRequirements`** — requests 2 CPU, 8Gi memory, 1 GPU
- **`retry=1`** — retries once on transient failure
- **`FlowymlKerasCallback`** — auto-attached when `experiment_name` is provided (Keras/TF)

### Evaluation Steps

Returns a **Metrics** asset:

```python
from mlpotion.integrations.flowyml.keras import evaluate_model

metrics = evaluate_model(
    model=model_asset,    # Accepts Model asset — auto-unwraps
    data=dataset,         # Accepts Dataset asset — auto-unwraps
)

print(metrics.get_metric("accuracy"))
print(metrics.values)
```

### Export / Save / Load Steps

All return **Model** assets:

```python
from mlpotion.integrations.flowyml.keras import export_model, save_model, load_model

# Export to format
exported = export_model(model=model_asset, export_path="models/prod/", export_format="keras")

# Save to disk
saved = save_model(model=model_asset, save_path="models/backup/model.keras")

# Load from disk
loaded = load_model(model_path="models/backup/model.keras")
```

### Inspect Steps (Keras, TensorFlow)

Returns a **Metrics** asset with model architecture details:

```python
from mlpotion.integrations.flowyml.keras import inspect_model

inspection = inspect_model(model=model_asset)

print(inspection.get_metric("name"))          # Model name
print(inspection.get_metric("parameters"))    # Parameter counts
```

---

## Pipeline Templates

Each framework provides **6 ready-to-use pipeline factories**. All accept a
`Context` object for injecting hyperparameters.

### 1. Training Pipeline

Basic load → train → evaluate:

```python
from flowyml.core.context import Context
from mlpotion.integrations.flowyml.keras import create_keras_training_pipeline

ctx = Context(
    file_path="data/train.csv",
    label_name="target",
    batch_size=32,
    epochs=20,
    learning_rate=0.001,
)

pipeline = create_keras_training_pipeline(
    name="classification_v1",
    context=ctx,
    project_name="my_project",
)
result = pipeline.run()
```

### 2. Full Pipeline

Complete lifecycle with data transformation and export:

```python
from mlpotion.integrations.flowyml.keras import create_keras_full_pipeline

ctx = Context(
    file_path="data/train.csv",
    label_name="target",
    batch_size=32,
    data_output_path="data/transformed/",
    epochs=50,
    learning_rate=0.001,
    experiment_name="full-run",
    export_path="models/production/",
    export_format="keras",
)

pipeline = create_keras_full_pipeline(
    context=ctx,
    enable_checkpointing=True,    # Resume on failure
)
result = pipeline.run()
```

### 3. Evaluation Pipeline

Evaluate an existing model against new data:

```python
from mlpotion.integrations.flowyml.keras import create_keras_evaluation_pipeline

ctx = Context(
    model_path="models/production/model.keras",
    file_path="data/test.csv",
    label_name="target",
    batch_size=64,
)

pipeline = create_keras_evaluation_pipeline(context=ctx)
result = pipeline.run()
```

### 4. Export Pipeline

Convert and persist a model:

```python
from mlpotion.integrations.flowyml.keras import create_keras_export_pipeline

ctx = Context(
    model_path="models/trained/model.keras",
    export_path="models/exported/",
    export_format="saved_model",
    save_path="models/backup/model.keras",
)

pipeline = create_keras_export_pipeline(context=ctx)
result = pipeline.run()
```

### 5. Experiment Pipeline (Conditional Deploy)

Train and auto-deploy if metrics exceed a threshold:

```python
from mlpotion.integrations.flowyml.keras import create_keras_experiment_pipeline

ctx = Context(
    file_path="data/train.csv",
    label_name="target",
    epochs=30,
    experiment_name="experiment-v1",
    export_path="models/production/",
    save_path="models/checkpoints/model.keras",
)

pipeline = create_keras_experiment_pipeline(
    context=ctx,
    deploy_threshold=0.85,        # Only deploy if accuracy ≥ 85%
    threshold_metric="accuracy",
)
result = pipeline.run()
```

### 6. Scheduled Retraining

Periodic retraining with cron scheduling:

```python
from mlpotion.integrations.flowyml.keras import create_keras_scheduled_pipeline

info = create_keras_scheduled_pipeline(
    context=ctx,
    schedule="0 2 * * 0",   # Every Sunday at 2 AM
    timezone="UTC",
)

pipeline = info["pipeline"]
scheduler = info["scheduler"]

# Run once now
result = pipeline.run()

# Or start the scheduler for automatic retraining
scheduler.start()
```

---

## Artifact Types

### Dataset

Wraps raw data (CSVSequence, DataLoader, tf.data.Dataset) with metadata:

```python
from flowyml import Dataset

# Created automatically by load_data steps
dataset = load_data(file_path="data/train.csv", batch_size=32)

assert isinstance(dataset, Dataset)
print(dataset.data)                    # Raw CSVSequence/DataLoader
print(dataset.metadata.properties)     # {'source': '...', 'batch_size': 32, ...}
```

### Model

Wraps trained models with auto-extracted architecture metadata:

```python
from flowyml import Model

# Created automatically by train_model, from_keras / from_pytorch
model_asset, _ = train_model(model=my_model, data=dataset, epochs=10)

assert isinstance(model_asset, Model)
print(model_asset.data)                # Raw keras.Model / nn.Module
print(model_asset.metadata.properties) # Auto-extracted: layers, params, etc.
```

### Metrics

Wraps numeric metrics and training history:

```python
from flowyml import Metrics

_, metrics = train_model(model=my_model, data=dataset, epochs=10)

assert isinstance(metrics, Metrics)
print(metrics.get_metric("loss"))          # Access single metric
print(metrics.values)                      # All metrics as dict
print(metrics.metadata.properties)         # Also stored in properties
```

---

## DAG Wiring

Steps declare `inputs` and `outputs` so FlowyML can auto-wire them:

```
load_data            →  outputs: ["dataset"]
train_model          →  inputs: ["dataset"],  outputs: ["model", "training_metrics"]
evaluate_model       →  inputs: ["model", "dataset"],  outputs: ["metrics"]
export_model         →  inputs: ["model"],  outputs: ["exported_model"]
```

When steps are added to a pipeline, FlowyML's DAG resolver automatically
connects matching output→input names, creating a dependency graph.

Steps also gracefully accept both **raw objects** and **FlowyML artifacts** as
input — they unwrap artifacts internally before passing data to the underlying
MLPotion components.

---

## GPU Resources & Execution Groups

Steps ship **without hardcoded resource requirements** so they work out-of-the-box
on any hardware. When you need GPU scheduling, simply pass `resources` and
`execution_group` to the `@step` decorator — FlowyML natively supports these:

```python
from flowyml.core.step import step
from flowyml.core.resources import ResourceRequirements, GPUConfig

# Option 1: Override when defining your own step
@step(
    name="my_train",
    resources=ResourceRequirements(
        cpu="4",
        memory="16Gi",
        gpu=GPUConfig(gpu_type="nvidia-a100", count=2),
    ),
    execution_group="training",
)
def my_train_model(...):
    ...

# Option 2: Use the predefined step and configure resources at the pipeline level
pipeline.add_step(train_model, resources=ResourceRequirements(...))
```

This design gives you **full flexibility** — the predefined steps remain
portable across CPU-only and GPU environments.

---

## Framework-Specific Notes

### Keras
- `FlowymlKerasCallback` auto-captures all training metrics live to the FlowyML dashboard
- Supports `experiment_name` and `project` parameters for experiment tracking
- `Model.from_keras()` auto-extracts layer counts, parameter counts, and optimizer info

### PyTorch
- Supports both `CSVDataset` and `StreamingCSVDataset` for memory-efficient loading
- `Model.from_pytorch()` auto-extracts module architecture
- Export supports `torchscript` and `onnx` formats with `sample_input` for tracing

### TensorFlow
- Includes `optimize_data` step for `tf.data.Dataset` optimization (prefetch, cache, shuffle)
- Includes `transform_data` step for model-based data transformation with CSV output
- Uses `Model.from_keras()` since TF models are Keras models under the hood
