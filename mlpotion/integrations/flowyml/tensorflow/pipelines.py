"""Pre-built FlowyML pipeline templates for TensorFlow/Keras workflows.

Provides ready-to-run pipelines that wire MLPotion TensorFlow steps together
with proper DAG dependencies, context injection, experiment tracking,
and optional scheduling.

Available pipelines:

- **create_tf_training_pipeline**  — Load → Train → Evaluate
- **create_tf_full_pipeline**      — Load → Optimize → Transform → Train → Evaluate → Export
- **create_tf_evaluation_pipeline** — Load model + data → Evaluate → Inspect
- **create_tf_export_pipeline**    — Load model → Export + Save
- **create_tf_experiment_pipeline** — Full pipeline with conditional deploy
- **create_tf_scheduled_pipeline** — Scheduled retraining with cron
"""

from __future__ import annotations

from typing import Any

from flowyml.core.context import Context
from flowyml.core.pipeline import Pipeline

from mlpotion.integrations.flowyml.tensorflow.steps import (
    evaluate_model,
    export_model,
    inspect_model,
    load_data,
    load_model,
    optimize_data,
    save_model,
    train_model,
)


# ---------------------------------------------------------------------------
# 1. Basic Training Pipeline
# ---------------------------------------------------------------------------


def create_tf_training_pipeline(
    name: str = "tf_training",
    context: Context | None = None,
    enable_cache: bool = True,
    project_name: str | None = None,
    version: str | None = None,
) -> Pipeline:
    """Create a ready-to-run TensorFlow training pipeline.

    **DAG**: ``load_data → train_model → evaluate_model``

    Provide hyperparameters via the context object::

        from flowyml.core.context import Context

        ctx = Context(
            file_path="data/train.csv",
            label_name="target",
            batch_size=32,
            epochs=10,
            learning_rate=0.001,
            experiment_name="my-experiment",
        )

        pipeline = create_tf_training_pipeline(
            name="my_training",
            context=ctx,
            project_name="my_project",
        )
        result = pipeline.run()

    Args:
        name: Pipeline name.
        context: FlowyML Context with parameters to inject.
        enable_cache: Whether to enable step caching.
        project_name: Project to attach this pipeline to.
        version: Optional version string for versioned pipeline.

    Returns:
        Configured FlowyML Pipeline ready for ``.run()``.
    """
    pipeline = Pipeline(
        name=name,
        context=context,
        enable_cache=enable_cache,
        project_name=project_name,
        version=version,
    )

    pipeline.add_step(load_data)
    pipeline.add_step(train_model)
    pipeline.add_step(evaluate_model)

    return pipeline


# ---------------------------------------------------------------------------
# 2. Full Pipeline (Load → Optimize → Transform → Train → Evaluate → Export)
# ---------------------------------------------------------------------------


def create_tf_full_pipeline(
    name: str = "tf_full",
    context: Context | None = None,
    enable_cache: bool = True,
    enable_checkpointing: bool = True,
    project_name: str | None = None,
    version: str | None = None,
) -> Pipeline:
    """Create a full TensorFlow pipeline covering the entire ML lifecycle.

    **DAG**: ``load_data → optimize_data → train_model → evaluate_model → export_model``

    Includes data optimization (prefetch/cache/shuffle) and checkpointing
    for long-running training steps.

    Context parameters::

        ctx = Context(
            # Data loading
            file_path="data/train.csv",
            label_name="target",
            batch_size=32,
            # Optimization
            shuffle_buffer_size=10000,
            prefetch=True,
            cache=True,
            # Training
            epochs=50,
            learning_rate=0.001,
            experiment_name="full-run",
            project="my_project",
            # Export
            export_path="models/production/",
            export_format="saved_model",
        )

    Args:
        name: Pipeline name.
        context: FlowyML Context with parameters.
        enable_cache: Whether to enable step caching.
        enable_checkpointing: Whether to enable checkpointing.
        project_name: Project to attach this pipeline to.
        version: Optional version string.

    Returns:
        Configured FlowyML Pipeline ready for ``.run()``.
    """
    pipeline = Pipeline(
        name=name,
        context=context,
        enable_cache=enable_cache,
        enable_checkpointing=enable_checkpointing,
        project_name=project_name,
        version=version,
    )

    pipeline.add_step(load_data)
    pipeline.add_step(optimize_data)
    pipeline.add_step(train_model)
    pipeline.add_step(evaluate_model)
    pipeline.add_step(export_model)

    return pipeline


# ---------------------------------------------------------------------------
# 3. Evaluation-Only Pipeline
# ---------------------------------------------------------------------------


def create_tf_evaluation_pipeline(
    name: str = "tf_evaluation",
    context: Context | None = None,
    enable_cache: bool = True,
    project_name: str | None = None,
) -> Pipeline:
    """Create a pipeline for evaluating an existing TF/Keras model.

    **DAG**: ``load_model → load_data → evaluate_model → inspect_model``

    Useful for model validation, A/B testing, and periodic evaluation
    against new data without retraining.

    Context parameters::

        ctx = Context(
            model_path="models/production/model.keras",
            file_path="data/test.csv",
            label_name="target",
            batch_size=64,
        )

    Args:
        name: Pipeline name.
        context: FlowyML Context with parameters.
        enable_cache: Whether to enable step caching.
        project_name: Project to attach this pipeline to.

    Returns:
        Configured FlowyML Pipeline ready for ``.run()``.
    """
    pipeline = Pipeline(
        name=name,
        context=context,
        enable_cache=enable_cache,
        project_name=project_name,
    )

    pipeline.add_step(load_model)
    pipeline.add_step(load_data)
    pipeline.add_step(evaluate_model)
    pipeline.add_step(inspect_model)

    return pipeline


# ---------------------------------------------------------------------------
# 4. Export / Save Pipeline
# ---------------------------------------------------------------------------


def create_tf_export_pipeline(
    name: str = "tf_export",
    context: Context | None = None,
    project_name: str | None = None,
) -> Pipeline:
    """Create a pipeline for exporting and saving an existing model.

    **DAG**: ``load_model → export_model, save_model``

    Useful for converting a trained model to multiple formats
    (SavedModel, TFLite, Keras) and persisting to different locations.

    Context parameters::

        ctx = Context(
            model_path="models/trained/model.keras",
            export_path="models/exported/",
            export_format="saved_model",
            save_path="models/backup/model.keras",
        )

    Args:
        name: Pipeline name.
        context: FlowyML Context with parameters.
        project_name: Project to attach this pipeline to.

    Returns:
        Configured FlowyML Pipeline ready for ``.run()``.
    """
    pipeline = Pipeline(
        name=name,
        context=context,
        enable_cache=False,  # Always re-export
        project_name=project_name,
    )

    pipeline.add_step(load_model)
    pipeline.add_step(export_model)
    pipeline.add_step(save_model)

    return pipeline


# ---------------------------------------------------------------------------
# 5. Experiment Pipeline (with tracking & conditional deployment)
# ---------------------------------------------------------------------------


def create_tf_experiment_pipeline(
    name: str = "tf_experiment",
    context: Context | None = None,
    project_name: str | None = None,
    version: str | None = None,
    deploy_threshold: float = 0.8,
    threshold_metric: str = "accuracy",
) -> Pipeline:
    """Create a full experiment pipeline with conditional deployment.

    **DAG**::

        load_data → train_model → evaluate_model
                                       ↓
                              [if metric > threshold]
                                       ↓
                              export_model → save_model

    Integrates FlowyML experiment tracking and conditionally exports the
    model only if validation metrics exceed the given threshold.

    Context parameters::

        ctx = Context(
            file_path="data/train.csv",
            label_name="target",
            batch_size=32,
            epochs=30,
            learning_rate=0.001,
            experiment_name="experiment-v1",
            project="my_project",
            export_path="models/production/",
            save_path="models/checkpoints/model.keras",
        )

        pipeline = create_tf_experiment_pipeline(
            context=ctx,
            deploy_threshold=0.85,
            threshold_metric="accuracy",
        )
        result = pipeline.run()

    Args:
        name: Pipeline name.
        context: FlowyML Context with parameters.
        project_name: Project to attach this pipeline to.
        version: Optional version string.
        deploy_threshold: Minimum metric value to trigger deployment.
        threshold_metric: Which metric to check against the threshold.

    Returns:
        Configured FlowyML Pipeline ready for ``.run()``.
    """
    from flowyml.core.conditional import If

    pipeline = Pipeline(
        name=name,
        context=context,
        enable_cache=False,
        enable_experiment_tracking=True,
        enable_checkpointing=True,
        project_name=project_name,
        version=version,
    )

    # Core training DAG
    pipeline.add_step(load_data)
    pipeline.add_step(train_model)
    pipeline.add_step(evaluate_model)

    # Conditional deployment: only export if metric exceeds threshold
    deploy_condition = If(
        condition=lambda metrics: (
            metrics.get_metric(threshold_metric, 0) >= deploy_threshold
            if hasattr(metrics, "get_metric")
            else metrics.get(threshold_metric, 0) >= deploy_threshold
        ),
        then_steps=[export_model, save_model],
        name=f"deploy_if_{threshold_metric}_above_{deploy_threshold}",
    )
    pipeline.control_flows.append(deploy_condition)

    return pipeline


# ---------------------------------------------------------------------------
# 6. Scheduled Retraining Pipeline
# ---------------------------------------------------------------------------


def create_tf_scheduled_pipeline(
    name: str = "tf_scheduled_retraining",
    context: Context | None = None,
    project_name: str | None = None,
    schedule: str = "0 2 * * 0",
    timezone: str = "UTC",
) -> dict[str, Any]:
    """Create a scheduled retraining pipeline.

    Returns both the pipeline and a configured scheduler so you can
    register periodic retraining (e.g., weekly) with a single call.

    **DAG**: ``load_data → train_model → evaluate_model → export_model``

    Schedule format uses cron syntax (default: every Sunday at 2 AM)::

        pipeline_info = create_tf_scheduled_pipeline(
            context=ctx,
            project_name="my_project",
            schedule="0 2 * * 0",  # Weekly
        )

        # Access the components
        pipeline = pipeline_info["pipeline"]
        scheduler = pipeline_info["scheduler"]

        # Run once immediately
        result = pipeline.run()

        # Or start the scheduler for automatic retraining
        scheduler.start()

    Args:
        name: Pipeline name.
        context: FlowyML Context with parameters.
        project_name: Project to attach this pipeline to.
        schedule: Cron expression for scheduling.
        timezone: Timezone for the schedule.

    Returns:
        Dict with ``pipeline`` and ``scheduler`` keys.
    """
    from flowyml.core.scheduler import PipelineScheduler

    pipeline = Pipeline(
        name=name,
        context=context,
        enable_cache=False,  # Fresh data each run
        enable_checkpointing=True,
        project_name=project_name,
    )

    pipeline.add_step(load_data)
    pipeline.add_step(train_model)
    pipeline.add_step(evaluate_model)
    pipeline.add_step(export_model)

    # Configure scheduler
    scheduler = PipelineScheduler()
    scheduler.schedule(
        pipeline=pipeline,
        cron=schedule,
        timezone=timezone,
    )

    return {
        "pipeline": pipeline,
        "scheduler": scheduler,
    }
