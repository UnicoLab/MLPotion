# FlowyML Quick Start Tutorial 🚀

Build a **complete artifact-centric ML pipeline** in under 10 minutes using
MLPotion + FlowyML. Unlike a basic standalone pipeline, this approach gives
you typed artifacts, automatic metadata, DAG resolution, caching, and lineage
tracking — all out of the box.

**Time**: ~10 minutes
**Level**: Beginner
**Prerequisites**: Python 3.10+

---

## What We'll Build 🎯

A house-price regression pipeline that:

1. 📥 Loads CSV data → returns a **Dataset** artifact
2. 🎓 Trains a neural network → returns a **Model** + **Metrics** artifact
3. 📊 Evaluates performance → returns a **Metrics** artifact
4. 💾 Saves the model → returns a **Model** artifact with save path

All wired together as a FlowyML DAG with automatic dependency resolution.

---

## Step 1: Install 📦

```bash
pip install mlpotion[flowyml,keras]
```

---

## Step 2: Prepare Synthetic Data 📊

```python
# create_data.py
import pandas as pd
import numpy as np

np.random.seed(42)
n = 10_000

data = pd.DataFrame({
    "square_feet": np.random.randint(500, 5000, n),
    "bedrooms": np.random.randint(1, 6, n),
    "bathrooms": np.random.randint(1, 4, n),
    "age_years": np.random.randint(0, 100, n),
})

data["price"] = (
    data["square_feet"] * 200
    + data["bedrooms"] * 10_000
    + data["bathrooms"] * 15_000
    - data["age_years"] * 500
    + np.random.randn(n) * 50_000
)

split = int(0.8 * len(data))
data[:split].to_csv("train.csv", index=False)
data[split:].to_csv("test.csv", index=False)

print(f"✅ Created {split} training + {n - split} test samples")
```

```bash
python create_data.py
```

---

## Step 3: Build the Pipeline 🏗️

```python
# flowyml_pipeline.py
import keras
from flowyml.core.context import Context
from mlpotion.integrations.flowyml.keras import create_keras_training_pipeline


def create_model(input_dim: int = 4) -> keras.Model:
    """Create a simple regression model."""
    model = keras.Sequential([
        keras.layers.Dense(128, activation="relu", input_shape=(input_dim,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    # 1. Define all hyperparameters in a single Context
    ctx = Context(
        file_path="train.csv",
        label_name="price",
        batch_size=32,
        epochs=20,
        learning_rate=0.001,
        experiment_name="house-prices-v1",
    )

    # 2. Create the pipeline — one line!
    pipeline = create_keras_training_pipeline(
        name="house_price_training",
        context=ctx,
        project_name="house-prices",
    )

    # 3. Run it
    print("🚀 Running FlowyML pipeline...")
    result = pipeline.run()

    # 4. Access the artifacts
    print("\n✅ Pipeline complete!")
    print(f"   Steps executed: {len(result.steps)}")

    # The pipeline steps return typed artifacts:
    # - load_data   → Dataset artifact
    # - train_model → (Model, Metrics) artifacts
    # - evaluate    → Metrics artifact


if __name__ == "__main__":
    main()
```

---

## Step 4: Run the Pipeline 🏃

```bash
python flowyml_pipeline.py
```

Expected output:

```
🚀 Running FlowyML pipeline...
📦 Loaded dataset: 250 batches, batch_size=32, source=train.csv
🎯 Training complete: 20 epochs, metrics captured: ['loss', 'mae', ...]
📊 Evaluation: {'loss': 15234.56, 'mae': 98.21}

✅ Pipeline complete!
   Steps executed: 3
```

---

## Step 5: Understand What Happened 🔍

### The DAG That Ran

```
load_data (outputs: dataset)
    ↓
train_model (inputs: dataset → outputs: model, training_metrics)
    ↓
evaluate_model (inputs: model, dataset → outputs: metrics)
```

FlowyML **automatically resolved** the dependency graph from the `inputs`/`outputs`
declarations on each step.

### The Artifacts Created

| Step | Artifact | Type | Auto-Extracted Metadata |
|------|----------|------|------------------------|
| `load_data` | `dataset` | `Dataset` | source, batch_size, batches, label_name |
| `train_model` | `model` | `Model` | layers, parameters, optimizer info |
| `train_model` | `training_metrics` | `Metrics` | loss, mae, epochs, learning_rate |
| `evaluate_model` | `metrics` | `Metrics` | loss, mae (on eval data) |

### Caching In Action

Run the pipeline again — notice that `load_data` is **skipped** because its
`cache="code_hash"` detects no code changes:

```bash
python flowyml_pipeline.py
# 📦 load_data — CACHED ✓
# 🎯 Training complete...
```

---

## Step 6: Use Individual Steps Standalone 🧩

Every step can also be called directly — no pipeline required:

```python
from mlpotion.integrations.flowyml.keras import load_data, evaluate_model

# Load data standalone
dataset = load_data(file_path="test.csv", batch_size=64, label_name="price")
print(type(dataset))                    # <class 'flowyml.assets.dataset.Dataset'>
print(dataset.metadata.properties)      # {'source': 'test.csv', ...}

# Evaluate standalone
metrics = evaluate_model(model=trained_model, data=dataset)
print(metrics.get_metric("mae"))        # 102.34
```

---

## What You Learned 🎓

1. ✅ How to install MLPotion with FlowyML support
2. ✅ How to configure a pipeline with `Context`
3. ✅ How to use a pipeline template (`create_keras_training_pipeline`)
4. ✅ How FlowyML auto-resolves the DAG from step I/O declarations
5. ✅ How every step returns typed artifacts with metadata
6. ✅ How caching skips unchanged steps automatically
7. ✅ How to use individual steps standalone

## Next Steps 🚀

- [Custom Pipelines →](flowyml-custom-pipeline.md) — Compose your own step combinations
- [Experiment Tracking →](flowyml-experiment-tracking.md) — Conditional deploy + experiment comparison
- [Scheduled Retraining →](flowyml-scheduled-retraining.md) — Cron-based periodic pipelines
- [FlowyML Integration Guide →](../integrations/flowyml.md) — Full reference docs

---

<p align="center">
  <strong>Congratulations — you've built your first artifact-centric ML pipeline!</strong> 🎉
</p>
