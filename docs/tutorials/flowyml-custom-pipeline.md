# Custom Pipelines with FlowyML 🧩

Learn how to **compose custom pipelines** by mixing and matching individual
MLPotion steps. Go beyond the pre-built templates — build exactly the workflow
your project needs.

**Time**: ~15 minutes
**Level**: Intermediate
**Prerequisites**: Completed the [FlowyML Quick Start](flowyml-quickstart.md)

---

## What We'll Build 🎯

Three custom pipelines that are **not** available as pre-built templates:

1. 🔍 **Train → Inspect → Export** — inspect architecture before deploying
2. 🔄 **Multi-Dataset Evaluation** — evaluate one model against multiple datasets
3. 🧪 **Custom Step Pipeline** — add your own business logic as a FlowyML step

---

## Pipeline 1: Train → Inspect → Export 🔍

Sometimes you want to inspect the model architecture *before* exporting.
No template for this? No problem — build it from individual steps.

```python
# custom_inspect_pipeline.py
import keras
from flowyml.core.context import Context
from flowyml.core.pipeline import Pipeline
from mlpotion.integrations.flowyml.keras import (
    load_data,
    train_model,
    inspect_model,
    export_model,
)


def create_model() -> keras.Model:
    model = keras.Sequential([
        keras.layers.Dense(256, activation="relu", input_shape=(4,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    ctx = Context(
        file_path="train.csv",
        label_name="price",
        batch_size=32,
        epochs=30,
        experiment_name="inspect-before-export",
        export_path="models/inspected/",
        export_format="keras",
    )

    # Build the custom pipeline from individual steps
    pipeline = Pipeline(
        name="train_inspect_export",
        context=ctx,
        enable_cache=True,
        enable_checkpointing=True,
    )

    pipeline.add_step(load_data)        # → Dataset artifact
    pipeline.add_step(train_model)      # → (Model, Metrics) artifacts
    pipeline.add_step(inspect_model)    # → Metrics artifact (architecture)
    pipeline.add_step(export_model)     # → Model artifact (exported)

    result = pipeline.run()
    print("✅ Custom pipeline complete!")


if __name__ == "__main__":
    main()
```

**DAG:**
```
load_data → train_model → inspect_model → export_model
```

FlowyML auto-wires this because:
- `train_model` outputs `"model"` → `inspect_model` takes input `"model"`
- `inspect_model` does NOT consume the model — it passes through
- `export_model` takes input `"model"` from `train_model`

---

## Pipeline 2: Multi-Step Evaluation 📊

Evaluate a single model against multiple data splits using separate
`load_data` calls and a shared `evaluate_model` step:

```python
# multi_eval_pipeline.py
from flowyml.core.context import Context
from flowyml.core.pipeline import Pipeline
from mlpotion.integrations.flowyml.keras import (
    load_data,
    load_model,
    evaluate_model,
)


def main():
    # Evaluate an existing model against test data
    ctx = Context(
        model_path="models/production/model.keras",
        file_path="test.csv",
        label_name="price",
        batch_size=64,
    )

    pipeline = Pipeline(
        name="comprehensive_evaluation",
        context=ctx,
        enable_cache=True,
    )

    pipeline.add_step(load_model)       # → Model artifact
    pipeline.add_step(load_data)        # → Dataset artifact
    pipeline.add_step(evaluate_model)   # → Metrics artifact

    result = pipeline.run()
    print("✅ Multi-evaluation complete!")


if __name__ == "__main__":
    main()
```

---

## Pipeline 3: Custom Step Pipeline 🧪

Add your **own business logic** as a proper FlowyML step, then compose
it with pre-built MLPotion steps:

```python
# custom_step_pipeline.py
import keras
from flowyml.core.step import step
from flowyml.core.context import Context
from flowyml.core.pipeline import Pipeline
from flowyml import Dataset, Model, Metrics
from mlpotion.integrations.flowyml.keras import (
    load_data,
    train_model,
    export_model,
)


# ─── Your Custom Step ─────────────────────────────────────
@step(
    name="validate_metrics",
    inputs=["metrics"],
    outputs=["validation_report"],
    tags={"stage": "validation", "custom": "true"},
)
def validate_metrics(
    metrics: Metrics,
    max_acceptable_loss: float = 1000.0,
    min_acceptable_mae: float = 200.0,
) -> Metrics:
    """Custom business logic: validate that metrics meet your criteria."""
    loss = metrics.get_metric("loss", float("inf"))
    mae = metrics.get_metric("mae", float("inf"))

    report = {
        "loss": loss,
        "mae": mae,
        "loss_acceptable": loss <= max_acceptable_loss,
        "mae_acceptable": mae <= min_acceptable_mae,
        "overall_pass": loss <= max_acceptable_loss and mae <= min_acceptable_mae,
    }

    return Metrics.create(
        metrics=report,
        name="validation_report",
        tags={"stage": "validation"},
        properties=report,
    )


# ─── Another Custom Step ──────────────────────────────────
@step(
    name="log_to_slack",
    inputs=["validation_report"],
    outputs=["notification_status"],
    tags={"stage": "notification"},
)
def log_to_slack(validation_report: Metrics) -> Metrics:
    """Send validation results to Slack (mock for demo)."""
    passed = validation_report.get_metric("overall_pass", False)
    status = "✅ PASSED" if passed else "❌ FAILED"

    print(f"\n📢 Slack notification: Model validation {status}")
    print(f"   Loss: {validation_report.get_metric('loss'):.2f}")
    print(f"   MAE:  {validation_report.get_metric('mae'):.2f}")

    return Metrics.create(
        metrics={"notified": True, "status": status},
        name="notification_status",
    )


def main():
    ctx = Context(
        file_path="train.csv",
        label_name="price",
        batch_size=32,
        epochs=20,
        learning_rate=0.001,
        export_path="models/validated/",
        export_format="keras",
    )

    pipeline = Pipeline(
        name="custom_validated_pipeline",
        context=ctx,
        enable_cache=True,
    )

    # Mix pre-built + custom steps
    pipeline.add_step(load_data)            # MLPotion step
    pipeline.add_step(train_model)          # MLPotion step
    pipeline.add_step(validate_metrics)     # Your custom step
    pipeline.add_step(log_to_slack)         # Your custom step
    pipeline.add_step(export_model)         # MLPotion step

    result = pipeline.run()
    print("\n✅ Custom validated pipeline complete!")


if __name__ == "__main__":
    main()
```

**DAG:**
```
load_data → train_model → validate_metrics → log_to_slack
                      ↘ export_model
```

---

## Using the FlowyMLAdapter for Custom Components 🔌

If you have a **custom data loader, trainer, or evaluator** that implements
MLPotion's protocol interface, wrap it with `FlowyMLAdapter`:

```python
from mlpotion.integrations.flowyml import FlowyMLAdapter
from mlpotion.core.protocols import DataLoader
from flowyml.core.pipeline import Pipeline


class S3DataLoader:
    """Custom loader that reads from S3."""
    def load(self):
        # Your S3 loading logic
        import pandas as pd
        return pd.read_csv("s3://my-bucket/data.csv")


# Wrap it as a FlowyML step
s3_load_step = FlowyMLAdapter.create_data_loader_step(
    S3DataLoader(),
    name="s3_data_load",
    cache="code_hash",
    tags={"source": "s3", "env": "production"},
)

# Use in a pipeline alongside pre-built steps
from mlpotion.integrations.flowyml.keras import train_model, evaluate_model

pipeline = Pipeline("s3_pipeline")
pipeline.add_step(s3_load_step)        # Your custom adapter step
pipeline.add_step(train_model)         # Pre-built MLPotion step
pipeline.add_step(evaluate_model)      # Pre-built MLPotion step

result = pipeline.run()
```

---

## Tips for Custom Pipelines 💡

### 1. Always Declare `inputs` and `outputs`

FlowyML needs these to resolve the DAG. If you omit them, steps won't wire
automatically:

```python
# ✅ Good — FlowyML can wire this
@step(name="my_step", inputs=["model"], outputs=["processed_model"])
def my_step(model: Model) -> Model: ...

# ❌ Bad — FlowyML can't auto-wire this
@step(name="my_step")
def my_step(model: Model) -> Model: ...
```

### 2. Accept Artifacts OR Raw Objects

Follow the pattern used by all MLPotion steps — unwrap artifacts if present:

```python
@step(name="my_step", inputs=["model", "dataset"], outputs=["result"])
def my_step(model, data):
    # Unwrap if needed
    raw_model = model.data if isinstance(model, Model) else model
    raw_data = data.data if isinstance(data, Dataset) else data
    # ...
```

### 3. Use Tags for Observability

Tags make it easy to filter and search steps in the FlowyML dashboard:

```python
@step(
    name="my_step",
    tags={
        "stage": "preprocessing",
        "framework": "keras",
        "team": "ml-platform",
        "priority": "high",
    },
)
```

### 4. Use `Metrics.create()` for Any Custom Metrics

Any dict of values can be wrapped as a `Metrics` artifact:

```python
from flowyml import Metrics

report = Metrics.create(
    metrics={"f1": 0.92, "precision": 0.95, "recall": 0.89},
    name="custom_classification_metrics",
    tags={"model_version": "v2"},
    properties={"threshold": 0.5},
)
```

---

## What You Learned 🎓

1. ✅ How to compose custom pipelines from individual steps
2. ✅ How to create your own custom FlowyML steps
3. ✅ How to mix pre-built and custom steps in the same pipeline
4. ✅ How to use `FlowyMLAdapter` for custom protocol components
5. ✅ Best practices for DAG wiring and artifact handling

## Next Steps 🚀

- [Experiment Tracking →](flowyml-experiment-tracking.md) — Conditional deploy + metrics thresholds
- [Scheduled Retraining →](flowyml-scheduled-retraining.md) — Cron pipelines
- [FlowyML Integration Guide →](../integrations/flowyml.md) — Full API reference

---

<p align="center">
  <strong>You're now a FlowyML pipeline architect!</strong> 🏗️
</p>
