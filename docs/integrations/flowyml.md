# FlowyML Integration

MLPotion provides a **first-class integration** with [FlowyML](https://github.com/UnicoLab/FlowyML),
an artifact-centric ML pipeline framework. Every MLPotion component (data loaders,
trainers, evaluators, exporters) is exposed as a fully-wired FlowyML `@step` that
returns **typed artifacts** — `Dataset`, `Model`, and `Metrics` — with automatic
metadata extraction, lineage tracking, and DAG resolution.

> **TL;DR** — Install with `pip install mlpotion[flowyml,keras]`, pick a pipeline
> template, pass a `Context`, and call `.run()`. The integration handles artifact
> wrapping, metadata, caching, retry, and DAG wiring for you.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Architecture](#architecture)
- [Reusable Steps](#reusable-steps)
  - [Step Design Principles](#step-design-principles)
  - [Keras Steps](#keras-steps)
  - [PyTorch Steps](#pytorch-steps)
  - [TensorFlow Steps](#tensorflow-steps)
  - [The `FlowyMLAdapter` (Generic Steps)](#the-flowymladapter-generic-steps)
- [Pipeline Templates](#pipeline-templates)
  - [Pipeline Design Principles](#pipeline-design-principles)
  - [1. Training Pipeline](#1-training-pipeline)
  - [2. Full Pipeline](#2-full-pipeline)
  - [3. Evaluation Pipeline](#3-evaluation-pipeline)
  - [4. Export Pipeline](#4-export-pipeline)
  - [5. Experiment Pipeline (Conditional Deploy)](#5-experiment-pipeline-conditional-deploy)
  - [6. Scheduled Pipeline](#6-scheduled-pipeline)
- [Composing Custom Pipelines](#composing-custom-pipelines)
- [Artifact Types](#artifact-types)
- [DAG Wiring](#dag-wiring)
- [Step Decorator Options](#step-decorator-options)
- [Caching Strategies](#caching-strategies)
- [GPU Resources & Execution Groups](#gpu-resources--execution-groups)
- [Context Injection](#context-injection)
- [Advanced Patterns](#advanced-patterns)
  - [Cross-Framework Reuse](#cross-framework-reuse)
  - [Combining Steps from Different Frameworks](#combining-steps-from-different-frameworks)
  - [Building a Custom Step from MLPotion Components](#building-a-custom-step-from-mlpotion-components)
  - [Conditional Flows](#conditional-flows)
  - [Lineage Tracking](#lineage-tracking)
- [Framework-Specific Notes](#framework-specific-notes)
- [API Reference](#api-reference)

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

## Installation

Install the integration alongside the framework you need:

```bash
# Keras
pip install mlpotion[flowyml,keras]

# PyTorch
pip install mlpotion[flowyml,pytorch]

# TensorFlow
pip install mlpotion[flowyml,tensorflow]

# All frameworks
pip install mlpotion[flowyml,keras,pytorch,tensorflow]
```

The `FLOWYML_AVAILABLE` flag is set at import time — if `flowyml` is not
installed, the integration module skips all imports gracefully:

```python
from mlpotion.integrations.flowyml import FLOWYML_AVAILABLE

if FLOWYML_AVAILABLE:
    from mlpotion.integrations.flowyml.keras import create_keras_training_pipeline
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

## Reusable Steps

Every step in the integration follows the same design contract:

1. **Accepts raw objects _or_ FlowyML artifacts** — steps auto-unwrap `Dataset.data`, `Model.data`, etc.
2. **Returns a typed artifact** — `Dataset`, `Model`, or `Metrics` with auto-extracted metadata.
3. **Declares `inputs`/`outputs`** — enables FlowyML DAG auto-wiring.
4. **Ships without hardcoded resource requirements** — portable across CPU and GPU environments.
5. **Is independently usable** — every step can be called standalone or composed into pipelines.

### Step Design Principles

```python
# Every step can be used in three ways:

# 1. Standalone — call it directly as a function
dataset = load_data(file_path="data/train.csv", batch_size=32)

# 2. In a pipeline — add it to a Pipeline object
pipeline = Pipeline("my_pipeline")
pipeline.add_step(load_data)

# 3. Via the adapter — wrap any MLPotion protocol component
step = FlowyMLAdapter.create_data_loader_step(my_custom_loader)
pipeline.add_step(step)
```

---

### Keras Steps

**Module:** `mlpotion.integrations.flowyml.keras`

```python
from mlpotion.integrations.flowyml.keras import (
    load_data,         # CSV → Dataset artifact
    transform_data,    # Dataset + Model → transformed Dataset artifact
    train_model,       # Model + Dataset → (Model, Metrics) artifacts
    evaluate_model,    # Model + Dataset → Metrics artifact
    export_model,      # Model → exported Model artifact
    save_model,        # Model → saved Model artifact
    load_model,        # path → Model artifact
    inspect_model,     # Model → Metrics artifact (architecture details)
)
```

#### `load_data`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | `str` | — | Glob pattern for CSV files |
| `batch_size` | `int` | `32` | Batch size for the CSVSequence |
| `label_name` | `str \| None` | `None` | Target column name |
| `column_names` | `list[str] \| None` | `None` | Columns to load (None = all) |
| `shuffle` | `bool` | `True` | Whether to shuffle data |
| `dtype` | `str` | `"float32"` | Data type for numeric conversion |

**Returns:** `Dataset` artifact with `source`, `batch_size`, `batches`, `label_name` metadata.

**Decorator config:** `outputs=["dataset"]`, `cache="code_hash"`, `tags={"framework": "keras"}`

```python
dataset = load_data(file_path="data/train.csv", batch_size=32, label_name="target")

# Access raw data
raw_sequence = dataset.data          # CSVSequence
print(dataset.metadata.properties)   # {'source': '...', 'batch_size': 32, ...}
```

#### `transform_data`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `Dataset` | — | Input Dataset artifact |
| `model` | `keras.Model` | — | Model for generating predictions |
| `data_output_path` | `str` | — | Output path for transformed CSV |
| `data_output_per_batch` | `bool` | `False` | One file per batch |
| `batch_size` | `int \| None` | `None` | Batch size override |
| `feature_names` | `list[str] \| None` | `None` | Feature names for output |
| `input_columns` | `list[str] \| None` | `None` | Input columns to pass to model |

**Returns:** `Dataset` artifact with **parent lineage** linked to the input dataset.

**Decorator config:** `inputs=["dataset"]`, `outputs=["transformed"]`, `cache="code_hash"`

#### `train_model`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `keras.Model` | — | Compiled Keras model |
| `data` | `CSVSequence \| Dataset` | — | Training data |
| `epochs` | `int` | `10` | Number of epochs |
| `learning_rate` | `float` | `0.001` | Learning rate |
| `verbose` | `int` | `1` | Keras verbosity level |
| `validation_data` | `CSVSequence \| Dataset \| None` | `None` | Validation data |
| `callbacks` | `list[Callback] \| None` | `None` | Extra Keras callbacks |
| `experiment_name` | `str \| None` | `None` | FlowyML experiment name |
| `project` | `str \| None` | `None` | FlowyML project name |
| `log_model` | `bool` | `True` | Log model artifact after training |

**Returns:** `tuple[Model, Metrics]` — Model via `Model.from_keras()` with auto-extracted architecture metadata; Metrics with training history.

**Decorator config:** `inputs=["dataset"]`, `outputs=["model", "training_metrics"]`, `cache=False`, `retry=1`

> **🔑 Key feature:** A `FlowymlKerasCallback` is **automatically attached** to
> capture all training metrics live to the FlowyML dashboard.

```python
model_asset, metrics_asset = train_model(
    model=my_keras_model,
    data=dataset,
    epochs=20,
    learning_rate=0.001,
    experiment_name="v1",
)

print(metrics_asset.get_metric("loss"))
print(metrics_asset.values)           # All metrics as dict
print(model_asset.metadata.properties)  # Auto-extracted: layers, params, etc.
```

#### `evaluate_model`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `keras.Model \| Model` | — | Trained model or Model artifact |
| `data` | `CSVSequence \| Dataset` | — | Evaluation data |
| `verbose` | `int` | `0` | Keras verbosity level |

**Returns:** `Metrics` artifact with evaluation results.

**Decorator config:** `inputs=["model", "dataset"]`, `outputs=["metrics"]`, `cache="input_hash"`

#### `export_model`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `keras.Model \| Model` | — | Model to export |
| `export_path` | `str` | — | Destination path |
| `export_format` | `str \| None` | `None` | Format: `"keras"`, `"saved_model"`, `"tflite"` |

**Returns:** `Model` artifact with export metadata.

**Decorator config:** `inputs=["model"]`, `outputs=["exported_model"]`, `cache="code_hash"`

#### `save_model`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `keras.Model \| Model` | — | Model to save |
| `save_path` | `str` | — | Destination file path |

**Returns:** `Model` artifact with save location metadata.

**Decorator config:** `inputs=["model"]`, `outputs=["saved_model"]`

#### `load_model`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | — | Path to saved model |
| `inspect` | `bool` | `False` | Log inspection info |

**Returns:** `Model` artifact wrapping the loaded Keras model.

**Decorator config:** `outputs=["model"]`, `cache="code_hash"`

#### `inspect_model`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `keras.Model \| Model` | — | Model to inspect |
| `include_layers` | `bool` | `True` | Include per-layer info |
| `include_signatures` | `bool` | `True` | Include I/O signatures |

**Returns:** `Metrics` artifact with model architecture details (name, parameter counts, layer info).

**Decorator config:** `inputs=["model"]`, `outputs=["inspection"]`

---

### PyTorch Steps

**Module:** `mlpotion.integrations.flowyml.pytorch`

```python
from mlpotion.integrations.flowyml.pytorch import (
    load_csv_data,           # CSV → Dataset artifact (standard)
    load_streaming_csv_data, # CSV → Dataset artifact (chunked streaming)
    train_model,             # Model + Dataset → (Model, Metrics) artifacts
    evaluate_model,          # Model + Dataset → Metrics artifact
    export_model,            # Model → exported Model artifact
    save_model,              # Model → saved Model artifact
    load_model,              # path → Model artifact
)
```

#### `load_csv_data`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | `str` | — | Glob pattern for CSV files |
| `batch_size` | `int` | `32` | Batch size |
| `label_name` | `str \| None` | `None` | Target column |
| `column_names` | `list[str] \| None` | `None` | Columns to load |
| `shuffle` | `bool` | `True` | Shuffle data |
| `num_workers` | `int` | `0` | DataLoader workers |
| `pin_memory` | `bool` | `False` | Pin memory for faster GPU transfer |
| `drop_last` | `bool` | `False` | Drop last incomplete batch |
| `dtype` | `str` | `"float32"` | Tensor data type |

**Returns:** `Dataset` artifact wrapping a PyTorch `DataLoader`.

**Decorator config:** `outputs=["dataset"]`, `cache="code_hash"`

#### `load_streaming_csv_data`

For datasets that **don't fit in memory** — uses chunked reading:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | `str` | — | Glob pattern for CSV files |
| `batch_size` | `int` | `32` | Batch size |
| `label_name` | `str \| None` | `None` | Target column |
| `column_names` | `list[str] \| None` | `None` | Columns to load |
| `num_workers` | `int` | `0` | DataLoader workers |
| `pin_memory` | `bool` | `False` | Pin memory for GPU |
| `chunksize` | `int` | `10000` | Rows per chunk |
| `dtype` | `str` | `"float32"` | Tensor data type |

**Returns:** `Dataset` artifact with `mode: "streaming"` metadata.

**Decorator config:** `outputs=["dataset"]`, `cache=False` (streaming data changes)

> **💡 When to use:** Choose `load_streaming_csv_data` when your dataset is too
> large for memory. Choose `load_csv_data` for standard in-memory workflows.

#### `train_model`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | PyTorch model |
| `data` | `DataLoader \| Dataset` | — | Training data |
| `epochs` | `int` | `10` | Number of epochs |
| `learning_rate` | `float` | `0.001` | Learning rate |
| `optimizer` | `str` | `"adam"` | Optimizer: `"adam"`, `"sgd"`, `"adamw"` |
| `loss_fn` | `str` | `"mse"` | Loss: `"mse"`, `"cross_entropy"` |
| `device` | `str` | `"cpu"` | Device: `"cpu"` or `"cuda"` |
| `validation_data` | `DataLoader \| Dataset \| None` | `None` | Validation data |
| `verbose` | `bool` | `True` | Log per-epoch metrics |
| `max_batches_per_epoch` | `int \| None` | `None` | Limit batches (debugging) |

**Returns:** `tuple[Model, Metrics]` — Model via `Model.from_pytorch()` with auto-extracted module architecture.

**Decorator config:** `inputs=["dataset"]`, `outputs=["model", "training_metrics"]`, `cache=False`, `retry=1`

#### `evaluate_model`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module \| Model` | — | Trained model or Model artifact |
| `data` | `DataLoader \| Dataset` | — | Evaluation data |
| `loss_fn` | `str` | `"mse"` | Loss function |
| `device` | `str` | `"cpu"` | Device |
| `verbose` | `bool` | `True` | Log metrics |
| `max_batches` | `int \| None` | `None` | Limit batches |

**Returns:** `Metrics` artifact.

**Decorator config:** `inputs=["model", "dataset"]`, `outputs=["metrics"]`, `cache="input_hash"`

#### `export_model`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module \| Model` | — | Model to export |
| `export_path` | `str` | — | Destination path |
| `export_format` | `str` | `"torchscript"` | Format: `"torchscript"`, `"onnx"` |
| `sample_input` | `torch.Tensor \| None` | `None` | Required for ONNX tracing |

**Returns:** `Model` artifact with export metadata.

**Decorator config:** `inputs=["model"]`, `outputs=["exported_model"]`, `cache="code_hash"`

#### `save_model` / `load_model`

Same contract as Keras — accept raw or artifact objects, return `Model` artifacts.

---

### TensorFlow Steps

**Module:** `mlpotion.integrations.flowyml.tensorflow`

```python
from mlpotion.integrations.flowyml.tensorflow import (
    load_data,         # CSV → Dataset artifact
    optimize_data,     # Dataset → optimized Dataset artifact (TF-specific)
    transform_data,    # Dataset + Model → transformed Dataset artifact
    train_model,       # Model + Dataset → (Model, Metrics) artifacts
    evaluate_model,    # Model + Dataset → Metrics artifact
    export_model,      # Model → exported Model artifact
    save_model,        # Model → saved Model artifact
    load_model,        # path → Model artifact
    inspect_model,     # Model → Metrics artifact
)
```

#### `optimize_data` *(TF-exclusive)*

Applies `tf.data.Dataset` optimization (prefetch, cache, shuffle) and returns
a new `Dataset` artifact with **parent lineage** linked to the input:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `tf.data.Dataset \| Dataset` | — | Input dataset |
| `batch_size` | `int` | `32` | Batch size |
| `shuffle_buffer_size` | `int \| None` | `None` | Shuffle buffer size |
| `prefetch` | `bool` | `True` | Enable prefetching |
| `cache` | `bool` | `False` | Enable dataset caching |

**Returns:** `Dataset` artifact with `parent=` lineage.

**Decorator config:** `inputs=["dataset"]`, `outputs=["optimized_dataset"]`, `cache="code_hash"`

```python
from mlpotion.integrations.flowyml.tensorflow import load_data, optimize_data

dataset = load_data(file_path="data/train.csv", batch_size=32)
optimized = optimize_data(dataset=dataset, prefetch=True, cache=True)

# Lineage is preserved
print(optimized.parent)  # Points back to `dataset`
```

> The remaining TF steps (`load_data`, `train_model`, `evaluate_model`, etc.)
> follow the same contract as Keras. The `train_model` step also auto-attaches
> `FlowymlKerasCallback` when `experiment_name` is provided.

---

### The `FlowyMLAdapter` (Generic Steps)

For **framework-agnostic** or **custom component** scenarios, the
`FlowyMLAdapter` wraps any MLPotion protocol-compliant component into a
FlowyML step:

```python
from mlpotion.integrations.flowyml import FlowyMLAdapter
```

#### `FlowyMLAdapter.create_data_loader_step()`

```python
step = FlowyMLAdapter.create_data_loader_step(
    loader,                     # Any DataLoader protocol implementation
    name="custom_load",         # Step name (default: "load_data")
    cache="code_hash",          # Caching strategy
    retry=0,                    # Retry count
    resources=None,             # ResourceRequirements
    tags={"env": "staging"},    # Metadata tags
)
```

**Returns:** A `Step` that calls `loader.load()` and wraps the result as a `Dataset` artifact.

#### `FlowyMLAdapter.create_training_step()`

```python
step = FlowyMLAdapter.create_training_step(
    trainer,                    # Any ModelTrainer protocol implementation
    name="custom_train",        # Step name (default: "train_model")
    cache=False,                # Default: no caching for training
    retry=1,                    # Retry once on failure
    resources=gpu_resources,    # GPU ResourceRequirements
    tags={"stage": "train"},
)
```

**Returns:** A `Step` that calls `trainer.train()` and wraps result as a `Model` artifact.

#### `FlowyMLAdapter.create_evaluation_step()`

```python
step = FlowyMLAdapter.create_evaluation_step(
    evaluator,                  # Any ModelEvaluator protocol implementation
    name="custom_eval",         # Step name (default: "evaluate_model")
    cache="input_hash",         # Default: cache by input hash
    retry=0,
    tags={"stage": "eval"},
)
```

**Returns:** A `Step` that calls `evaluator.evaluate()` and wraps result as a `Metrics` artifact.

#### Example: Custom Component → Pipeline

```python
from mlpotion.frameworks.keras.data.loaders import CSVDataLoader
from mlpotion.frameworks.keras.training.trainers import ModelTrainer
from mlpotion.frameworks.keras.evaluation.evaluators import ModelEvaluator
from mlpotion.integrations.flowyml import FlowyMLAdapter
from flowyml.core.pipeline import Pipeline

# Wrap MLPotion components
load_step = FlowyMLAdapter.create_data_loader_step(
    CSVDataLoader(file_pattern="data/*.csv", batch_size=64)
)
train_step = FlowyMLAdapter.create_training_step(ModelTrainer())
eval_step = FlowyMLAdapter.create_evaluation_step(ModelEvaluator())

# Build a pipeline
pipeline = Pipeline("custom_pipeline")
pipeline.add_step(load_step)
pipeline.add_step(train_step)
pipeline.add_step(eval_step)

result = pipeline.run()
```

---

## Pipeline Templates

Each framework provides **6 ready-to-use pipeline factories**. All accept a
`Context` object for injecting hyperparameters and return a configured `Pipeline`
(or `dict` with pipeline + scheduler for scheduled pipelines).

### Pipeline Design Principles

1. **Context-driven** — all hyperparameters flow through a single `Context` object.
2. **DAG-resolved** — steps are wired by their `inputs`/`outputs` declarations.
3. **Configurable** — toggle caching, checkpointing, versioning via factory args.
4. **Composable** — use them as-is or as templates for your own pipelines.

### Pipeline Availability Matrix

| Pipeline | Keras | PyTorch | TensorFlow |
|----------|-------|---------|------------|
| Training | `create_keras_training_pipeline` | `create_pytorch_training_pipeline` | `create_tf_training_pipeline` |
| Full | `create_keras_full_pipeline` | `create_pytorch_full_pipeline` | `create_tf_full_pipeline` |
| Evaluation | `create_keras_evaluation_pipeline` | `create_pytorch_evaluation_pipeline` | `create_tf_evaluation_pipeline` |
| Export | `create_keras_export_pipeline` | `create_pytorch_export_pipeline` | `create_tf_export_pipeline` |
| Experiment | `create_keras_experiment_pipeline` | `create_pytorch_experiment_pipeline` | `create_tf_experiment_pipeline` |
| Scheduled | `create_keras_scheduled_pipeline` | `create_pytorch_scheduled_pipeline` | `create_tf_scheduled_pipeline` |

---

### 1. Training Pipeline

Basic **load → train → evaluate** workflow.

**DAG:**
```
load_data → train_model → evaluate_model
```

**Factory arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `name` | `str` | `"<fw>_training"` | Pipeline name |
| `context` | `Context \| None` | `None` | Hyperparameters |
| `enable_cache` | `bool` | `True` | Enable step caching |
| `project_name` | `str \| None` | `None` | FlowyML project |
| `version` | `str \| None` | `None` | Pipeline version |

**Context parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `file_path` | ✅ | Path/glob to training CSV |
| `label_name` | ✅ | Target column name |
| `batch_size` | — | Batch size (default: 32) |
| `epochs` | — | Training epochs (default: 10) |
| `learning_rate` | — | Learning rate (default: 0.001) |
| `experiment_name` | — | FlowyML experiment tracking name |

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

**PyTorch equivalent** includes additional context params: `optimizer`, `loss_fn`, `device`:

```python
from mlpotion.integrations.flowyml.pytorch import create_pytorch_training_pipeline

ctx = Context(
    file_path="data/train.csv",
    label_name="target",
    batch_size=32,
    epochs=20,
    learning_rate=0.001,
    optimizer="adam",
    loss_fn="cross_entropy",
    device="cuda",
)

pipeline = create_pytorch_training_pipeline(context=ctx)
result = pipeline.run()
```

---

### 2. Full Pipeline

Complete lifecycle with data transformation/optimization and export.

**DAG (Keras):**
```
load_data → transform_data → train_model → evaluate_model → export_model
```

**DAG (TensorFlow):**
```
load_data → optimize_data → train_model → evaluate_model → export_model
```

**DAG (PyTorch):**
```
load_csv_data → train_model → evaluate_model → export_model → save_model
```

**Additional factory arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `enable_checkpointing` | `bool` | `True` | Resume on failure |

**Context parameters (superset):**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `file_path` | ✅ | Training data path |
| `label_name` | ✅ | Target column |
| `batch_size` | — | Batch size |
| `data_output_path` | — | Keras: transformed data output path |
| `shuffle_buffer_size` | — | TF: shuffle buffer size |
| `prefetch` | — | TF: enable prefetching |
| `epochs` | — | Training epochs |
| `learning_rate` | — | Learning rate |
| `experiment_name` | — | Experiment tracking name |
| `export_path` | — | Export destination |
| `export_format` | — | Export format |

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
    enable_checkpointing=True,
)
result = pipeline.run()
```

---

### 3. Evaluation Pipeline

Evaluate an existing model against new data.

**DAG:**
```
load_model → load_data → evaluate_model → inspect_model
```

> **Note:** PyTorch evaluation pipeline does not include `inspect_model`.

**Context parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `model_path` | ✅ | Path to saved model |
| `file_path` | ✅ | Evaluation data path |
| `label_name` | ✅ | Target column |
| `batch_size` | — | Batch size |

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

---

### 4. Export Pipeline

Convert and persist a model to a specified format.

**DAG:**
```
load_model → export_model, save_model
```

**Context parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `model_path` | ✅ | Path to trained model |
| `export_path` | ✅ | Export destination |
| `export_format` | — | Export format |
| `save_path` | — | Backup save path |

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

> **PyTorch** supports `export_format="torchscript"` or `"onnx"`.

---

### 5. Experiment Pipeline (Conditional Deploy)

Train and **auto-deploy only if metrics exceed a threshold**.

**DAG:**
```
load_data → train_model → evaluate_model
                                ↓
                       [if metric ≥ threshold]
                                ↓
                       export_model → save_model
```

**Additional factory arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `deploy_threshold` | `float` | `0.8` | Minimum metric value |
| `threshold_metric` | `str` | `"accuracy"` | Metric to check |

**Pipeline features enabled:**
- `enable_experiment_tracking=True`
- `enable_checkpointing=True`
- `enable_cache=False`

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
    deploy_threshold=0.85,
    threshold_metric="accuracy",
)
result = pipeline.run()
```

Under the hood, this uses FlowyML's `If` conditional flow:

```python
from flowyml.core.conditional import If

deploy_condition = If(
    condition=lambda metrics: metrics.get_metric("accuracy", 0) >= 0.85,
    then_steps=[export_model, save_model],
    name="deploy_if_accuracy_above_0.85",
)
pipeline.control_flows.append(deploy_condition)
```

---

### 6. Scheduled Pipeline

Periodic retraining with cron scheduling. Returns **both** the pipeline and
a configured scheduler:

**DAG:**
```
load_data → train_model → evaluate_model → export_model
```

**Additional factory arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `schedule` | `str` | `"0 2 * * 0"` | Cron expression |
| `timezone` | `str` | `"UTC"` | Timezone |

**Returns:** `dict[str, Any]` with `"pipeline"` and `"scheduler"` keys.

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

**Common cron patterns:**

| Pattern | Description |
|---------|-------------|
| `0 2 * * 0` | Every Sunday at 2 AM |
| `0 0 * * *` | Every day at midnight |
| `0 */6 * * *` | Every 6 hours |
| `0 0 1 * *` | First day of every month |

---

## Composing Custom Pipelines

You can mix and match **any steps** from the integration to build your own
pipeline. Steps are not locked to their pipeline templates:

### Example: Custom Train → Inspect → Export Pipeline

```python
from flowyml.core.context import Context
from flowyml.core.pipeline import Pipeline
from mlpotion.integrations.flowyml.keras import (
    load_data,
    train_model,
    inspect_model,
    export_model,
)

ctx = Context(
    file_path="data/train.csv",
    label_name="target",
    epochs=50,
    export_path="models/prod/",
)

pipeline = Pipeline(
    name="train_inspect_export",
    context=ctx,
    enable_cache=True,
    enable_checkpointing=True,
)

# Add only the steps you need
pipeline.add_step(load_data)
pipeline.add_step(train_model)
pipeline.add_step(inspect_model)   # Inspect architecture after training
pipeline.add_step(export_model)

result = pipeline.run()
```

### Example: Adding a Custom Step

```python
from flowyml.core.step import step
from flowyml import Metrics

@step(
    name="compute_custom_metrics",
    inputs=["model", "dataset"],
    outputs=["custom_metrics"],
    tags={"stage": "custom"},
)
def compute_custom_metrics(model, data):
    """Your custom metric computation logic."""
    # ... your code here ...
    return Metrics.create(
        metrics={"f1_score": 0.92, "precision": 0.95},
        name="custom_metrics",
    )

# Add it to any pipeline
pipeline.add_step(compute_custom_metrics)
```

### Example: Mixing Adapter and Pre-built Steps

```python
from mlpotion.integrations.flowyml import FlowyMLAdapter
from mlpotion.integrations.flowyml.keras import train_model, evaluate_model

# Use MyCustomLoader with the adapter
my_loader = MyCustomLoader(source="s3://bucket/data.csv")
load_step = FlowyMLAdapter.create_data_loader_step(my_loader, name="s3_load")

# Combine with pre-built steps
pipeline = Pipeline("hybrid_pipeline")
pipeline.add_step(load_step)          # Custom adapter step
pipeline.add_step(train_model)         # Pre-built Keras step
pipeline.add_step(evaluate_model)      # Pre-built Keras step

result = pipeline.run()
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

**Auto-extraction methods:**
- `Model.from_keras(model, ...)` — extracts layers, parameters, optimizer info
- `Model.from_pytorch(model, ...)` — extracts module architecture

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
transform_data       →  inputs: ["dataset"],  outputs: ["transformed"]
optimize_data (TF)   →  inputs: ["dataset"],  outputs: ["optimized_dataset"]
train_model          →  inputs: ["dataset"],   outputs: ["model", "training_metrics"]
evaluate_model       →  inputs: ["model", "dataset"],  outputs: ["metrics"]
export_model         →  inputs: ["model"],     outputs: ["exported_model"]
save_model           →  inputs: ["model"],     outputs: ["saved_model"]
load_model           →  outputs: ["model"]
inspect_model        →  inputs: ["model"],     outputs: ["inspection"]
```

When steps are added to a pipeline, FlowyML's DAG resolver automatically
connects matching output→input names, creating a dependency graph.

### Artifact Unwrapping

Steps also gracefully accept both **raw objects** and **FlowyML artifacts** as
input — they unwrap artifacts internally before passing data to the underlying
MLPotion components:

```python
# Both of these work identically:
evaluate_model(model=keras_model, data=csv_sequence)           # raw objects
evaluate_model(model=model_artifact, data=dataset_artifact)     # FlowyML artifacts
```

The unwrapping logic is:

```python
raw_model = model.data if isinstance(model, Model) else model
raw_data  = data.data  if isinstance(data, Dataset) else data
```

---

## Step Decorator Options

Every step in the integration uses FlowyML's `@step` decorator. Here is the
full set of options available to you when creating custom steps:

```python
from flowyml.core.step import step
from flowyml.core.resources import ResourceRequirements, GPUConfig

@step(
    name="my_step",                          # Unique step name
    inputs=["dataset"],                      # DAG input names
    outputs=["model", "metrics"],            # DAG output names
    cache="code_hash",                       # "code_hash", "input_hash", False
    retry=1,                                 # Retry count on failure
    resources=ResourceRequirements(          # Resource requirements
        cpu="4",
        memory="16Gi",
        gpu=GPUConfig(gpu_type="nvidia-a100", count=2),
    ),
    execution_group="training",              # Group steps on same node
    tags={"framework": "keras"},             # Metadata tags
)
def my_step(data: Dataset) -> tuple[Model, Metrics]:
    ...
```

| Option | Type | Description |
|--------|------|-------------|
| `name` | `str` | Unique step name within the pipeline |
| `inputs` | `list[str]` | DAG input artifact names |
| `outputs` | `list[str]` | DAG output artifact names |
| `cache` | `bool \| str \| Callable` | Caching strategy |
| `retry` | `int` | Number of retries on failure |
| `resources` | `ResourceRequirements` | CPU/memory/GPU requirements |
| `execution_group` | `str` | Group steps on the same compute node |
| `tags` | `dict[str, str]` | Metadata tags for filtering/observability |

---

## Caching Strategies

| Strategy | Value | Use Case | Used By |
|----------|-------|----------|---------|
| **Code hash** | `"code_hash"` | Skip if step code hasn't changed | `load_data`, `export_model`, `load_model` |
| **Input hash** | `"input_hash"` | Skip if inputs haven't changed | `evaluate_model` |
| **Disabled** | `False` | Always re-execute | `train_model`, `streaming_load` |

> **Rule of thumb:** Data loading and export steps use `code_hash` since they're
> deterministic. Evaluation uses `input_hash` since results depend on the model.
> Training always re-runs because model weights are non-deterministic.

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

## Context Injection

The `Context` object is the single entry-point for passing hyperparameters
to all steps in a pipeline. FlowyML resolves context values to matching
step parameters automatically:

```python
from flowyml.core.context import Context

ctx = Context(
    # Data loading params → resolved to load_data(file_path=, batch_size=, ...)
    file_path="data/train.csv",
    label_name="target",
    batch_size=32,

    # Training params → resolved to train_model(epochs=, learning_rate=, ...)
    epochs=20,
    learning_rate=0.001,
    experiment_name="v1",

    # Export params → resolved to export_model(export_path=, export_format=, ...)
    export_path="models/prod/",
    export_format="keras",
)
```

**How it works:** when a pipeline runs, FlowyML inspects each step's function
signature and matches context keys to parameter names. Any matching key is
injected as a keyword argument.

### Complete Context Parameter Reference

| Parameter | Steps That Use It | Framework |
|-----------|-------------------|-----------|
| `file_path` | `load_data`, `load_csv_data` | All |
| `label_name` | `load_data`, `load_csv_data` | All |
| `batch_size` | `load_data`, `load_csv_data`, `optimize_data` | All |
| `column_names` | `load_data`, `load_csv_data` | All |
| `shuffle` | `load_data`, `load_csv_data` | All |
| `dtype` | `load_data`, `load_csv_data` | All |
| `num_workers` | `load_csv_data` | PyTorch |
| `pin_memory` | `load_csv_data` | PyTorch |
| `chunksize` | `load_streaming_csv_data` | PyTorch |
| `shuffle_buffer_size` | `optimize_data` | TensorFlow |
| `prefetch` | `optimize_data` | TensorFlow |
| `data_output_path` | `transform_data` | Keras, TF |
| `epochs` | `train_model` | All |
| `learning_rate` | `train_model` | All |
| `verbose` | `train_model`, `evaluate_model` | All |
| `optimizer` | `train_model` | PyTorch |
| `loss_fn` | `train_model`, `evaluate_model` | PyTorch |
| `device` | `train_model`, `evaluate_model`, `load_model` | PyTorch |
| `experiment_name` | `train_model` | Keras, TF |
| `project` | `train_model` | Keras, TF |
| `log_model` | `train_model` | Keras, TF |
| `model_path` | `load_model` | All |
| `export_path` | `export_model` | All |
| `export_format` | `export_model` | All |
| `save_path` | `save_model` | All |
| `sample_input` | `export_model` | PyTorch |

---

## Advanced Patterns

### Cross-Framework Reuse

All pipeline templates follow the **same factory signature**, making it trivial
to swap frameworks:

```python
# The same Context works across frameworks
ctx = Context(
    file_path="data/train.csv",
    label_name="target",
    epochs=20,
)

# Swap just the import
from mlpotion.integrations.flowyml.keras import create_keras_training_pipeline
from mlpotion.integrations.flowyml.pytorch import create_pytorch_training_pipeline
from mlpotion.integrations.flowyml.tensorflow import create_tf_training_pipeline

# All three work with the same context
keras_pipe = create_keras_training_pipeline(context=ctx)
pytorch_pipe = create_pytorch_training_pipeline(context=ctx)
tf_pipe = create_tf_training_pipeline(context=ctx)
```

### Combining Steps from Different Frameworks

While not common, you can mix steps from different framework modules
in a single pipeline when the artifact types are compatible:

```python
from flowyml.core.pipeline import Pipeline
from mlpotion.integrations.flowyml.keras import load_data, train_model
from mlpotion.integrations.flowyml.keras import inspect_model

pipeline = Pipeline("multi_step")
pipeline.add_step(load_data)         # Keras data loading
pipeline.add_step(train_model)       # Keras training
pipeline.add_step(inspect_model)     # Keras inspection
# All compatible because they share the same artifact types
```

### Building a Custom Step from MLPotion Components

Create a new FlowyML step from scratch using any MLPotion component:

```python
from flowyml.core.step import step
from flowyml import Dataset, Model, Metrics
from mlpotion.frameworks.keras.data.loaders import CSVDataLoader
from mlpotion.frameworks.keras.training.trainers import ModelTrainer

@step(
    name="custom_train_with_augmentation",
    inputs=["dataset"],
    outputs=["model", "metrics"],
    retry=2,
    tags={"stage": "training", "augmentation": "enabled"},
)
def custom_train_with_augmentation(
    model,
    data: Dataset,
    epochs: int = 10,
    augment: bool = True,
) -> tuple[Model, Metrics]:
    """Custom training step with data augmentation."""
    raw_data = data.data if isinstance(data, Dataset) else data

    if augment:
        # Your custom augmentation logic
        raw_data = apply_augmentation(raw_data)

    trainer = ModelTrainer()
    result = trainer.train(model=model, dataset=raw_data, config=...)

    return (
        Model.from_keras(result.model, name="augmented_model"),
        Metrics.create(metrics=result.metrics, name="aug_metrics"),
    )
```

### Conditional Flows

Use FlowyML's `If` to add conditional logic to any pipeline:

```python
from flowyml.core.conditional import If
from mlpotion.integrations.flowyml.keras import export_model, save_model

# Deploy only if loss is below 0.1
deploy_condition = If(
    condition=lambda metrics: metrics.get_metric("loss", 1.0) < 0.1,
    then_steps=[export_model, save_model],
    name="deploy_if_loss_low",
)
pipeline.control_flows.append(deploy_condition)
```

### Lineage Tracking

Steps that transform data automatically link to their parent via
FlowyML's `parent=` parameter:

```python
# transform_data and optimize_data both preserve lineage
transformed = Dataset.create(
    data=output_data,
    name="transformed",
    parent=input_dataset,    # ← Lineage link
)

# Query lineage
print(transformed.parent)   # → original Dataset reference
```

---

## Framework-Specific Notes

### Keras

- `FlowymlKerasCallback` auto-captures all training metrics live to the FlowyML dashboard
- Supports `experiment_name` and `project` parameters for experiment tracking
- `Model.from_keras()` auto-extracts layer counts, parameter counts, and optimizer info
- Includes `transform_data` step for model-based data transformation with CSV output
- Full pipeline includes: `load_data → transform_data → train_model → evaluate_model → export_model`

### PyTorch

- Supports both `CSVDataset` and `StreamingCSVDataset` for memory-efficient loading
- `Model.from_pytorch()` auto-extracts module architecture
- Export supports `torchscript` and `onnx` formats with `sample_input` for tracing
- Configurable `device` parameter (`"cpu"` or `"cuda"`) on train and eval steps
- Optimizer selection: `"adam"`, `"sgd"`, `"adamw"`
- Full pipeline: `load_csv_data → train_model → evaluate_model → export_model → save_model`

### TensorFlow

- Includes `optimize_data` step for `tf.data.Dataset` optimization (prefetch, cache, shuffle)
- Includes `transform_data` step for model-based data transformation with CSV output
- Uses `Model.from_keras()` since TF models are Keras models under the hood
- Full pipeline: `load_data → optimize_data → train_model → evaluate_model → export_model`

---

## API Reference

### Steps by Framework

#### Keras (`mlpotion.integrations.flowyml.keras`)

| Step | Inputs | Outputs | Cache | Retry |
|------|--------|---------|-------|-------|
| `load_data` | — | `dataset` | `code_hash` | 0 |
| `transform_data` | `dataset` | `transformed` | `code_hash` | 0 |
| `train_model` | `dataset` | `model`, `training_metrics` | `False` | 1 |
| `evaluate_model` | `model`, `dataset` | `metrics` | `input_hash` | 0 |
| `export_model` | `model` | `exported_model` | `code_hash` | 0 |
| `save_model` | `model` | `saved_model` | — | 0 |
| `load_model` | — | `model` | `code_hash` | 0 |
| `inspect_model` | `model` | `inspection` | — | 0 |

#### PyTorch (`mlpotion.integrations.flowyml.pytorch`)

| Step | Inputs | Outputs | Cache | Retry |
|------|--------|---------|-------|-------|
| `load_csv_data` | — | `dataset` | `code_hash` | 0 |
| `load_streaming_csv_data` | — | `dataset` | `False` | 0 |
| `train_model` | `dataset` | `model`, `training_metrics` | `False` | 1 |
| `evaluate_model` | `model`, `dataset` | `metrics` | `input_hash` | 0 |
| `export_model` | `model` | `exported_model` | `code_hash` | 0 |
| `save_model` | `model` | `saved_model` | — | 0 |
| `load_model` | — | `model` | `code_hash` | 0 |

#### TensorFlow (`mlpotion.integrations.flowyml.tensorflow`)

| Step | Inputs | Outputs | Cache | Retry |
|------|--------|---------|-------|-------|
| `load_data` | — | `dataset` | `code_hash` | 0 |
| `optimize_data` | `dataset` | `optimized_dataset` | `code_hash` | 0 |
| `transform_data` | `dataset` | `transformed` | — | 0 |
| `train_model` | `dataset` | `model`, `training_metrics` | `False` | 1 |
| `evaluate_model` | `model`, `dataset` | `metrics` | `input_hash` | 0 |
| `export_model` | `model` | `exported_model` | `code_hash` | 0 |
| `save_model` | `model` | `saved_model` | — | 0 |
| `load_model` | — | `model` | `code_hash` | 0 |
| `inspect_model` | `model` | `inspection` | — | 0 |

#### Generic Adapter (`mlpotion.integrations.flowyml.FlowyMLAdapter`)

| Factory Method | Wraps | Returns |
|----------------|-------|---------|
| `create_data_loader_step(loader)` | `DataLoader` protocol | `Dataset` artifact |
| `create_training_step(trainer)` | `ModelTrainer` protocol | `Model` artifact |
| `create_evaluation_step(evaluator)` | `ModelEvaluator` protocol | `Metrics` artifact |

### Pipelines by Framework

| Pipeline Factory | DAG | Returns |
|-----------------|-----|---------|
| `create_<fw>_training_pipeline` | load → train → eval | `Pipeline` |
| `create_<fw>_full_pipeline` | load → [transform\|optimize] → train → eval → export | `Pipeline` |
| `create_<fw>_evaluation_pipeline` | load_model → load_data → eval [→ inspect] | `Pipeline` |
| `create_<fw>_export_pipeline` | load_model → export + save | `Pipeline` |
| `create_<fw>_experiment_pipeline` | load → train → eval → [if metric ≥ threshold] → export + save | `Pipeline` |
| `create_<fw>_scheduled_pipeline` | load → train → eval → export (+ scheduler) | `dict` |
