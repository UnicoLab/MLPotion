# Scheduled Retraining Pipelines ⏰

Set up **automatic periodic retraining** for your ML models using FlowyML's
built-in scheduler. Keep your models fresh with new data without manual
intervention.

**Time**: ~10 minutes
**Level**: Intermediate
**Prerequisites**: Completed the [FlowyML Quick Start](flowyml-quickstart.md)

---

## What We'll Build 🎯

A scheduled pipeline that:

1. ⏰ Runs automatically on a cron schedule (e.g., every Sunday at 2 AM)
2. 📥 Loads the latest training data
3. 🎓 Retrains the model from scratch
4. 📊 Evaluates on the latest test data
5. 📤 Exports the new model for serving

All with `enable_cache=False` to ensure **fresh data** on every run.

---

## Option 1: Using the Pre-Built Template 🏭

The fastest way — one function call:

```python
# scheduled_pipeline.py
import keras
from flowyml.core.context import Context
from mlpotion.integrations.flowyml.keras import create_keras_scheduled_pipeline


def create_model() -> keras.Model:
    model = keras.Sequential([
        keras.layers.Dense(128, activation="relu", input_shape=(4,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    ctx = Context(
        file_path="data/latest/train.csv",   # Points to latest data
        label_name="price",
        batch_size=64,
        epochs=30,
        learning_rate=0.001,
        export_path="models/production/",
        export_format="keras",
    )

    # Create scheduled pipeline — returns dict with pipeline + scheduler
    info = create_keras_scheduled_pipeline(
        name="weekly_retraining",
        context=ctx,
        project_name="house-prices",
        schedule="0 2 * * 0",    # Every Sunday at 2 AM
        timezone="UTC",
    )

    pipeline = info["pipeline"]
    scheduler = info["scheduler"]

    # Option A: Run once immediately (for testing)
    print("🏃 Running pipeline once for testing...")
    result = pipeline.run()
    print(f"✅ Test run complete!")

    # Option B: Start the scheduler for automatic retraining
    print("\n⏰ Starting scheduler (Ctrl+C to stop)...")
    scheduler.start()


if __name__ == "__main__":
    main()
```

### What the Template Does

The `create_keras_scheduled_pipeline` factory:

1. Creates a `Pipeline` with `enable_cache=False` (fresh data each run)
2. Enables checkpointing for long-running jobs
3. Adds steps: `load_data → train_model → evaluate_model → export_model`
4. Creates and configures a `PipelineScheduler`
5. Returns both as a dict: `{"pipeline": ..., "scheduler": ...}`

---

## Option 2: Build It Manually 🏗️

For full control over the pipeline and scheduler:

```python
# scheduled_manual.py
from flowyml.core.context import Context
from flowyml.core.pipeline import Pipeline
from flowyml.core.scheduler import PipelineScheduler
from mlpotion.integrations.flowyml.keras import (
    load_data,
    train_model,
    evaluate_model,
    export_model,
    save_model,
)


def main():
    ctx = Context(
        file_path="data/latest/train.csv",
        label_name="price",
        batch_size=64,
        epochs=30,
        export_path="models/production/",
        save_path="models/archive/model_latest.keras",
    )

    # Build the pipeline
    pipeline = Pipeline(
        name="manual_scheduled_retraining",
        context=ctx,
        enable_cache=False,           # Always use fresh data
        enable_checkpointing=True,    # Resume on failure
        project_name="house-prices",
    )

    pipeline.add_step(load_data)
    pipeline.add_step(train_model)
    pipeline.add_step(evaluate_model)
    pipeline.add_step(export_model)
    pipeline.add_step(save_model)      # Also save a backup

    # Configure the scheduler
    scheduler = PipelineScheduler()
    scheduler.schedule(
        pipeline=pipeline,
        cron="0 2 * * 0",     # Every Sunday at 2 AM
        timezone="UTC",
    )

    # Start
    print("⏰ Scheduler registered. Starting...")
    scheduler.start()


if __name__ == "__main__":
    main()
```

---

## Cron Expression Reference 📅

The `schedule` parameter uses standard cron syntax:

```
┌───────── minute (0-59)
│ ┌─────── hour (0-23)
│ │ ┌───── day of month (1-31)
│ │ │ ┌─── month (1-12)
│ │ │ │ ┌─ day of week (0-6, Sunday=0)
│ │ │ │ │
* * * * *
```

### Common Patterns

| Schedule | Cron Expression | Use Case |
|----------|----------------|----------|
| Every Sunday 2 AM | `0 2 * * 0` | Weekly retraining |
| Every day midnight | `0 0 * * *` | Daily retraining |
| Every 6 hours | `0 */6 * * *` | Real-time model freshness |
| Every Monday & Thursday 3 AM | `0 3 * * 1,4` | Bi-weekly retraining |
| First day of month 1 AM | `0 1 1 * *` | Monthly retraining |
| Every 15 minutes | `*/15 * * * *` | Frequent updates (streaming data) |
| Weekdays only 6 AM | `0 6 * * 1-5` | Business-hours retraining |

---

## Cross-Framework Scheduled Pipelines 🔀

All three frameworks support scheduled pipelines with the same interface:

=== "Keras"

    ```python
    from mlpotion.integrations.flowyml.keras import create_keras_scheduled_pipeline

    info = create_keras_scheduled_pipeline(
        context=ctx,
        schedule="0 2 * * 0",
    )
    ```

=== "PyTorch"

    ```python
    from mlpotion.integrations.flowyml.pytorch import create_pytorch_scheduled_pipeline

    info = create_pytorch_scheduled_pipeline(
        context=ctx,
        schedule="0 2 * * 0",
    )
    ```

=== "TensorFlow"

    ```python
    from mlpotion.integrations.flowyml.tensorflow import create_tf_scheduled_pipeline

    info = create_tf_scheduled_pipeline(
        context=ctx,
        schedule="0 2 * * 0",
    )
    ```

---

## Combining Scheduling with Conditional Deployment 🚦

The most powerful pattern: **schedule retraining + only deploy good models**.

```python
from flowyml.core.context import Context
from flowyml.core.pipeline import Pipeline
from flowyml.core.scheduler import PipelineScheduler
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
        file_path="data/latest/train.csv",
        label_name="is_fraud",
        batch_size=64,
        epochs=50,
        experiment_name="fraud-scheduled",
        export_path="models/production/",
        save_path="models/archive/latest.keras",
    )

    pipeline = Pipeline(
        name="scheduled_conditional_retrain",
        context=ctx,
        enable_cache=False,
        enable_experiment_tracking=True,
        enable_checkpointing=True,
    )

    # Training DAG
    pipeline.add_step(load_data)
    pipeline.add_step(train_model)
    pipeline.add_step(evaluate_model)

    # Only deploy if accuracy ≥ 90%
    deploy_gate = If(
        condition=lambda m: m.get_metric("accuracy", 0) >= 0.90,
        then_steps=[export_model, save_model],
        name="deploy_if_accuracy_above_0.90",
    )
    pipeline.control_flows.append(deploy_gate)

    # Schedule weekly
    scheduler = PipelineScheduler()
    scheduler.schedule(pipeline=pipeline, cron="0 2 * * 0", timezone="UTC")

    print("⏰ Scheduled weekly retraining with quality gate")
    print("   → Models only deploy if accuracy ≥ 90%")
    scheduler.start()


if __name__ == "__main__":
    main()
```

**This gives you:**
- 🔄 Automatic weekly retraining
- 📈 Full experiment tracking on every run
- 🚦 Quality gate — bad models never reach production
- ♻️ Checkpointing — long runs resume on failure
- 📊 Metrics history across all scheduled runs

---

## Monitoring Scheduled Runs 📊

### Check Run History

```python
from flowyml.core.experiment import ExperimentTracker

tracker = ExperimentTracker(project="fraud-detection")
runs = tracker.list_experiments()

for run in runs:
    print(
        f"  {run.name} | "
        f"accuracy={run.metrics.get('accuracy', '?')} | "
        f"deployed={run.metrics.get('deployed', False)}"
    )
```

### Pipeline Logs

Each scheduled run produces its own logs with step-by-step output:

```
[2026-02-09 02:00:00] ⏰ Scheduled run started: weekly_retraining
[2026-02-09 02:00:01] 📦 Loaded dataset: 500 batches, source=data/latest/train.csv
[2026-02-09 02:03:45] 🎯 Training complete: 50 epochs
[2026-02-09 02:03:47] 📊 Evaluation: {accuracy: 0.93, loss: 0.12}
[2026-02-09 02:03:47] 🚦 Accuracy 0.93 ≥ 0.90 → deploying
[2026-02-09 02:03:48] 📤 Exported model to: models/production/
[2026-02-09 02:03:48] 💾 Saved model to: models/archive/latest.keras
[2026-02-09 02:03:48] ✅ Scheduled run complete!
```

---

## Production Deployment Tips 💡

### 1. Use a Process Manager

For production, run the scheduler with a process manager like `supervisord`:

```ini
# supervisord.conf
[program:model_retraining]
command=python scheduled_pipeline.py
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/retraining.err.log
stdout_logfile=/var/log/retraining.out.log
```

### 2. Point to Dynamic Data Sources

Don't hardcode file paths. Use symlinks or dynamic paths:

```python
ctx = Context(
    file_path="data/latest/train.csv",  # Symlinked to newest data
    # Or use a date pattern:
    # file_path=f"data/{datetime.now().strftime('%Y-%m')}/train.csv",
)
```

### 3. Always Enable Checkpointing

```python
pipeline = Pipeline(
    name="scheduled_retraining",
    context=ctx,
    enable_checkpointing=True,  # ← Always set this for scheduled pipelines
)
```

### 4. Combine with Alerting

Add a notification step for failed runs:

```python
@step(name="alert_on_failure", tags={"stage": "alerting"})
def alert_on_failure(metrics):
    if metrics.get_metric("accuracy", 0) < 0.70:
        send_slack_alert("⚠️ Model accuracy dropped below 70%!")
```

---

## What You Learned 🎓

1. ✅ How to create scheduled pipelines with cron syntax
2. ✅ How to build manual scheduled pipelines
3. ✅ Cron expression patterns for common schedules
4. ✅ How to combine scheduling with conditional deployment
5. ✅ How to monitor scheduled runs
6. ✅ Production deployment best practices

## Next Steps 🚀

- [Custom Pipelines →](flowyml-custom-pipeline.md) — Build your own step combinations
- [Experiment Tracking →](flowyml-experiment-tracking.md) — Conditional deploy + metrics
- [FlowyML Integration Guide →](../integrations/flowyml.md) — Full API reference

---

<p align="center">
  <strong>Your models now retrain themselves on schedule!</strong> ⏰🤖
</p>
