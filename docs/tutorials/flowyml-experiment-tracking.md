# Experiment Tracking & Conditional Deployment 🧪

Build a **production-grade experiment pipeline** that trains a model,
tracks all metrics live, and **only deploys if the model exceeds a
quality threshold**. This is the pattern used in real MLOps workflows
to prevent bad models from reaching production.

**Time**: ~15 minutes
**Level**: Intermediate
**Prerequisites**: Completed the [FlowyML Quick Start](flowyml-quickstart.md)

---

## What We'll Build 🎯

An experiment pipeline that:

1. 📥 Loads training data
2. 🎓 Trains a model with **live metric capture** via `FlowymlKerasCallback`
3. 📊 Evaluates on test data
4. 🚦 **Conditionally deploys** only if accuracy ≥ 85%
5. 💾 Saves + exports the model (only if threshold met)

```
load_data → train_model → evaluate_model
                                ↓
                       [if accuracy ≥ 0.85]
                                ↓
                       export_model → save_model
```

---

## Option 1: Using the Pre-Built Template 🏭

MLPotion provides `create_keras_experiment_pipeline` that does exactly this:

```python
# experiment_template.py
import keras
from flowyml.core.context import Context
from mlpotion.integrations.flowyml.keras import create_keras_experiment_pipeline


def create_model() -> keras.Model:
    model = keras.Sequential([
        keras.layers.Dense(128, activation="relu", input_shape=(4,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),  # Binary classification
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    ctx = Context(
        # Data
        file_path="data/train.csv",
        label_name="is_fraud",
        batch_size=64,
        # Training
        epochs=50,
        learning_rate=0.001,
        experiment_name="fraud-detection-v3",
        project="fraud-detection",
        # Export (only used if threshold met)
        export_path="models/production/fraud_model/",
        save_path="models/checkpoints/fraud_model.keras",
    )

    pipeline = create_keras_experiment_pipeline(
        name="fraud_experiment",
        context=ctx,
        project_name="fraud-detection",
        deploy_threshold=0.85,        # Only deploy if accuracy ≥ 85%
        threshold_metric="accuracy",  # Which metric to check
    )

    result = pipeline.run()
    print("✅ Experiment complete!")


if __name__ == "__main__":
    main()
```

### What Happens Under the Hood

The template creates a `Pipeline` with:

- `enable_experiment_tracking=True` — all metrics are tracked
- `enable_checkpointing=True` — pipeline can resume on failure
- A `FlowymlKerasCallback` auto-attached to the training step
- An `If` conditional flow that gates the export steps

---

## Option 2: Build It Yourself 🏗️

For full control, build the conditional pipeline manually:

```python
# experiment_manual.py
import keras
from flowyml.core.context import Context
from flowyml.core.pipeline import Pipeline
from flowyml.core.conditional import If
from mlpotion.integrations.flowyml.keras import (
    load_data,
    train_model,
    evaluate_model,
    export_model,
    save_model,
)


def main():
    ctx = Context(
        file_path="data/train.csv",
        label_name="is_fraud",
        batch_size=64,
        epochs=50,
        experiment_name="fraud-detection-manual",
        export_path="models/production/",
        save_path="models/checkpoints/model.keras",
    )

    pipeline = Pipeline(
        name="fraud_experiment_manual",
        context=ctx,
        enable_cache=False,               # Don't cache — we want fresh runs
        enable_experiment_tracking=True,   # Track all metrics
        enable_checkpointing=True,        # Resume on failure
    )

    # Core training DAG
    pipeline.add_step(load_data)
    pipeline.add_step(train_model)
    pipeline.add_step(evaluate_model)

    # Conditional deployment gate
    deploy_gate = If(
        condition=lambda metrics: (
            metrics.get_metric("accuracy", 0) >= 0.85
            if hasattr(metrics, "get_metric")
            else metrics.get("accuracy", 0) >= 0.85
        ),
        then_steps=[export_model, save_model],
        name="deploy_if_accuracy_above_0.85",
    )
    pipeline.control_flows.append(deploy_gate)

    result = pipeline.run()

    # Check what happened
    print("✅ Experiment complete!")
    # The export/save steps only ran if accuracy ≥ 85%


if __name__ == "__main__":
    main()
```

---

## Understanding `FlowymlKerasCallback` 📈

When you provide `experiment_name` to `train_model`, a `FlowymlKerasCallback`
is **automatically attached** to your training loop. It captures:

| What | How |
|------|-----|
| **All epoch metrics** | `loss`, `accuracy`, `val_loss`, `val_accuracy`, etc. |
| **Per-batch metrics** | If enabled, captures granular training progress |
| **Model artifact** | Optionally logs the model itself after training |
| **Training metadata** | Epochs completed, learning rate, batch size |

```python
# The callback is auto-created inside train_model:
from flowyml.integrations.keras import FlowymlKerasCallback

callback = FlowymlKerasCallback(
    experiment_name="fraud-detection-v3",
    project="fraud-detection",
    log_model=True,
)
# This is attached to the Keras model.fit() call automatically
```

You can also **add your own callbacks** alongside the auto-attached one:

```python
from mlpotion.integrations.flowyml.keras import train_model

model_asset, metrics_asset = train_model(
    model=my_model,
    data=dataset,
    epochs=50,
    experiment_name="v3",
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10),
        keras.callbacks.ReduceLROnPlateau(factor=0.5),
    ],
)
# FlowymlKerasCallback is ADDED to your list, not replaced
```

---

## Multi-Metric Conditional Gates 🚦

You can create more complex conditions that check multiple metrics:

```python
from flowyml.core.conditional import If

# Gate on multiple metrics
deploy_gate = If(
    condition=lambda metrics: (
        metrics.get_metric("accuracy", 0) >= 0.85
        and metrics.get_metric("loss", float("inf")) < 0.3
        and metrics.get_metric("precision", 0) >= 0.80
    ),
    then_steps=[export_model, save_model],
    name="deploy_if_quality_sufficient",
)
```

---

## Cross-Framework Experiments 🔀

The same experiment pattern works across all three frameworks.
Just swap the imports:

=== "Keras"

    ```python
    from mlpotion.integrations.flowyml.keras import create_keras_experiment_pipeline

    pipeline = create_keras_experiment_pipeline(
        context=ctx,
        deploy_threshold=0.85,
        threshold_metric="accuracy",
    )
    ```

=== "PyTorch"

    ```python
    from mlpotion.integrations.flowyml.pytorch import create_pytorch_experiment_pipeline

    pipeline = create_pytorch_experiment_pipeline(
        context=ctx,
        deploy_threshold=0.85,
        threshold_metric="accuracy",
    )
    ```

=== "TensorFlow"

    ```python
    from mlpotion.integrations.flowyml.tensorflow import create_tf_experiment_pipeline

    pipeline = create_tf_experiment_pipeline(
        context=ctx,
        deploy_threshold=0.85,
        threshold_metric="accuracy",
    )
    ```

---

## Experiment Comparison 📊

After running multiple experiments, use FlowyML's built-in tools to compare:

```python
from flowyml.core.experiment import ExperimentTracker

tracker = ExperimentTracker(project="fraud-detection")

# List all experiments
experiments = tracker.list_experiments()
for exp in experiments:
    print(f"  {exp.name}: accuracy={exp.metrics.get('accuracy', '?')}")

# Compare two experiments
comparison = tracker.compare(
    experiment_a="fraud-detection-v2",
    experiment_b="fraud-detection-v3",
)
print(comparison.summary())
```

---

## Best Practices 💡

### 1. Always Name Your Experiments

Use descriptive, versioned names that you can search later:

```python
ctx = Context(
    experiment_name="fraud-detection-v3-lr0001-epochs50",
    project="fraud-detection",
)
```

### 2. Use Checkpointing for Long Training Runs

```python
pipeline = Pipeline(
    name="long_training",
    context=ctx,
    enable_checkpointing=True,  # Resume from last checkpoint
)
```

### 3. Set Conservative Initial Thresholds

Start with a lower threshold and tighten it as your model improves:

```python
# Start conservatively
pipeline = create_keras_experiment_pipeline(
    context=ctx,
    deploy_threshold=0.70,   # Low threshold initially
)

# After baseline is established
pipeline = create_keras_experiment_pipeline(
    context=ctx,
    deploy_threshold=0.90,   # Tighter threshold for v2
)
```

### 4. Combine with Scheduling

Auto-retrain and conditionally deploy on a schedule:

```python
from mlpotion.integrations.flowyml.keras import create_keras_scheduled_pipeline

info = create_keras_scheduled_pipeline(
    context=ctx,
    schedule="0 2 * * 0",  # Weekly Sunday 2 AM
)

# The scheduled pipeline will auto-deploy only good models
# if you add the conditional gate
```

---

## What You Learned 🎓

1. ✅ How to use the experiment pipeline template
2. ✅ How to build manual conditional deployment gates
3. ✅ How `FlowymlKerasCallback` auto-captures training metrics
4. ✅ How to create multi-metric quality gates
5. ✅ How to compare experiments across runs
6. ✅ Best practices for production experiment workflows

## Next Steps 🚀

- [Scheduled Retraining →](flowyml-scheduled-retraining.md) — Cron-based pipelines
- [Custom Pipelines →](flowyml-custom-pipeline.md) — Build your own step combinations
- [FlowyML Integration Guide →](../integrations/flowyml.md) — Full API reference

---

<p align="center">
  <strong>Your models now deploy themselves — only when they're good enough!</strong> 🚦
</p>
