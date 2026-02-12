"""Tests for FlowyML TensorFlow steps — unit tests + end-to-end pipeline tests.

Tests verify:
- Every step returns the correct FlowyML artifact type (Dataset, Model, Metrics)
- Artifact metadata is populated correctly
- DAG wiring (output names match downstream input names)
- Steps gracefully accept both raw objects and artifact-wrapped inputs
- End-to-end pipeline executes successfully with mocked data
"""

import pytest
from unittest.mock import MagicMock, patch

# Skip entire module if flowyml or tensorflow not available
flowyml = pytest.importorskip("flowyml")
tf = pytest.importorskip("tensorflow")

from flowyml import Dataset, Model, Metrics  # noqa: E402

# Pre-import so @patch can resolve the full module path
import mlpotion.integrations.flowyml.tensorflow.steps  # noqa: E402, F401


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def mock_tf_dataset():
    """Create a mock tf.data.Dataset."""
    ds = MagicMock()
    ds.batch_size = 32
    return ds


@pytest.fixture
def mock_keras_model():
    """Create a simple Keras model for testing."""
    import keras as k

    model = k.Sequential(
        [
            k.layers.Dense(16, activation="relu", input_shape=(10,)),
            k.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


@pytest.fixture
def mock_training_result(mock_keras_model):
    """Create a mock training result."""
    result = MagicMock()
    result.model = mock_keras_model
    result.metrics = {"loss": 0.3, "mae": 0.2}
    result.history = {"loss": [0.5, 0.3], "mae": [0.4, 0.2]}
    return result


@pytest.fixture
def mock_evaluation_result():
    """Create a mock evaluation result."""
    result = MagicMock()
    result.metrics = {"loss": 0.05, "mae": 0.02, "accuracy": 0.95}
    return result


# -------------------------------------------------------------------------
# Data Step Tests
# -------------------------------------------------------------------------


class TestLoadData:
    """Test load_data step returns a Dataset asset."""

    @patch("mlpotion.integrations.flowyml.tensorflow.steps.CSVDataLoader")
    @patch("mlpotion.integrations.flowyml.tensorflow.steps.DataLoadingConfig")
    def test_returns_dataset_asset(self, mock_config_cls, mock_loader_cls):
        """load_data must return a Dataset asset, not a raw tf.data.Dataset."""
        from mlpotion.integrations.flowyml.tensorflow.steps import load_data

        mock_config = MagicMock()
        mock_config.dict.return_value = {
            "file_pattern": "test.csv",
            "batch_size": 32,
            "label_name": "target",
            "column_names": None,
        }
        mock_config_cls.return_value = mock_config

        mock_instance = MagicMock()
        mock_instance.load.return_value = MagicMock()
        mock_loader_cls.return_value = mock_instance

        result = load_data(file_path="test.csv", batch_size=32)

        assert isinstance(result, Dataset), f"Expected Dataset, got {type(result)}"
        assert result.data is not None
        mock_config_cls.assert_called_once()
        mock_instance.load.assert_called_once()


class TestOptimizeData:
    """Test optimize_data step returns a Dataset asset."""

    @patch("mlpotion.integrations.flowyml.tensorflow.steps.DatasetOptimizer")
    @patch("mlpotion.integrations.flowyml.tensorflow.steps.DataOptimizationConfig")
    def test_returns_dataset_asset(self, mock_config_cls, mock_opt_cls):
        """optimize_data must return a Dataset asset."""
        from mlpotion.integrations.flowyml.tensorflow.steps import optimize_data

        mock_config = MagicMock()
        mock_config.dict.return_value = {
            "batch_size": 32,
            "shuffle_buffer_size": None,
            "prefetch": True,
            "cache": True,
        }
        mock_config_cls.return_value = mock_config

        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = MagicMock()
        mock_opt_cls.return_value = mock_optimizer

        mock_dataset = MagicMock()
        result = optimize_data(dataset=mock_dataset, cache=True)

        assert isinstance(result, Dataset), f"Expected Dataset, got {type(result)}"
        assert result is not None


class TestTransformData:
    """Test transform_data step returns a Dataset asset."""

    @patch("mlpotion.integrations.flowyml.tensorflow.steps.DataToCSVTransformer")
    @patch("mlpotion.integrations.flowyml.tensorflow.steps.DataTransformationConfig")
    def test_returns_dataset_asset(self, mock_config_cls, mock_transformer_cls):
        """transform_data must return a Dataset asset."""
        from mlpotion.integrations.flowyml.tensorflow.steps import transform_data

        mock_config_cls.return_value = MagicMock()
        mock_transformer_cls.return_value = MagicMock()

        mock_dataset = MagicMock()
        mock_model = MagicMock()

        result = transform_data(
            dataset=mock_dataset,
            model=mock_model,
            data_output_path="/output/data.csv",
        )

        assert isinstance(result, Dataset), f"Expected Dataset, got {type(result)}"
        assert result.name == "tf_transformed_data"
        assert result.metadata.properties["output_path"] == "/output/data.csv"


# -------------------------------------------------------------------------
# Training Step Tests
# -------------------------------------------------------------------------


class TestTrainModel:
    """Test train_model step returns (Model, Metrics) tuple."""

    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelTrainer")
    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelTrainingConfig")
    def test_returns_model_and_metrics(
        self,
        mock_config_cls,
        mock_trainer_cls,
        mock_tf_dataset,
        mock_keras_model,
        mock_training_result,
    ):
        """train_model must return (Model, Metrics) tuple."""
        from mlpotion.integrations.flowyml.tensorflow.steps import train_model

        mock_config_cls.return_value = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_training_result
        mock_trainer_cls.return_value = mock_trainer

        model_asset, metrics_asset = train_model(
            model=mock_keras_model,
            data=mock_tf_dataset,
            epochs=5,
        )

        assert isinstance(
            model_asset, Model
        ), f"Expected Model, got {type(model_asset)}"
        assert isinstance(
            metrics_asset, Metrics
        ), f"Expected Metrics, got {type(metrics_asset)}"

    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelTrainer")
    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelTrainingConfig")
    def test_model_asset_metadata(
        self,
        mock_config_cls,
        mock_trainer_cls,
        mock_tf_dataset,
        mock_keras_model,
        mock_training_result,
    ):
        """Check that Model asset name and framework are set correctly."""
        from mlpotion.integrations.flowyml.tensorflow.steps import train_model

        mock_config_cls.return_value = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_training_result
        mock_trainer_cls.return_value = mock_trainer

        model_asset, _ = train_model(
            model=mock_keras_model,
            data=mock_tf_dataset,
            epochs=10,
        )

        assert model_asset.name == "tf_trained_model"
        assert model_asset.framework == "keras"

    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelTrainer")
    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelTrainingConfig")
    def test_metrics_asset_properties(
        self,
        mock_config_cls,
        mock_trainer_cls,
        mock_tf_dataset,
        mock_keras_model,
        mock_training_result,
    ):
        """Metrics asset must contain training configuration."""
        from mlpotion.integrations.flowyml.tensorflow.steps import train_model

        mock_config_cls.return_value = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_training_result
        mock_trainer_cls.return_value = mock_trainer

        _, metrics_asset = train_model(
            model=mock_keras_model,
            data=mock_tf_dataset,
            epochs=5,
            learning_rate=0.001,
        )

        assert metrics_asset.name == "tf_training_metrics"
        props = metrics_asset.metadata.properties
        assert props["epochs"] == 5
        assert props["learning_rate"] == 0.001

    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelTrainer")
    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelTrainingConfig")
    @patch("mlpotion.integrations.flowyml.tensorflow.steps.FlowymlKerasCallback")
    def test_callback_attached_when_experiment_name(
        self,
        mock_cb_cls,
        mock_config_cls,
        mock_trainer_cls,
        mock_tf_dataset,
        mock_keras_model,
        mock_training_result,
    ):
        """FlowymlKerasCallback auto-attached when experiment_name is provided."""
        from mlpotion.integrations.flowyml.tensorflow.steps import train_model

        mock_config_cls.return_value = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_training_result
        mock_trainer_cls.return_value = mock_trainer

        train_model(
            model=mock_keras_model,
            data=mock_tf_dataset,
            epochs=3,
            experiment_name="test-experiment",
        )

        mock_cb_cls.assert_called_once()

    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelTrainer")
    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelTrainingConfig")
    def test_accepts_dataset_asset_input(
        self,
        mock_config_cls,
        mock_trainer_cls,
        mock_tf_dataset,
        mock_keras_model,
        mock_training_result,
    ):
        """train_model should accept a Dataset-wrapped input and unwrap it."""
        from mlpotion.integrations.flowyml.tensorflow.steps import train_model

        mock_config_cls.return_value = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_training_result
        mock_trainer_cls.return_value = mock_trainer

        wrapped_data = Dataset.create(data=mock_tf_dataset, name="test_ds")

        model_asset, metrics_asset = train_model(
            model=mock_keras_model,
            data=wrapped_data,
            epochs=3,
        )

        # Trainer should receive the raw dataset, not the Dataset wrapper
        call_kwargs = mock_trainer.train.call_args
        assert (
            call_kwargs.kwargs.get("dataset") is mock_tf_dataset
            or call_kwargs[1].get("dataset") is mock_tf_dataset
        )


# -------------------------------------------------------------------------
# Evaluation Step Tests
# -------------------------------------------------------------------------


class TestEvaluateModel:
    """Test evaluate_model step returns a Metrics asset."""

    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelEvaluator")
    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelEvaluationConfig")
    def test_returns_metrics_asset(
        self,
        mock_config_cls,
        mock_eval_cls,
        mock_tf_dataset,
        mock_keras_model,
        mock_evaluation_result,
    ):
        """evaluate_model must return a Metrics asset."""
        from mlpotion.integrations.flowyml.tensorflow.steps import evaluate_model

        mock_config_cls.return_value = MagicMock()
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = mock_evaluation_result
        mock_eval_cls.return_value = mock_evaluator

        result = evaluate_model(
            model=mock_keras_model,
            data=mock_tf_dataset,
        )

        assert isinstance(result, Metrics), f"Expected Metrics, got {type(result)}"
        assert result.name == "tf_evaluation_metrics"

    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelEvaluator")
    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelEvaluationConfig")
    def test_accepts_model_asset_input(
        self,
        mock_config_cls,
        mock_eval_cls,
        mock_tf_dataset,
        mock_keras_model,
        mock_evaluation_result,
    ):
        """evaluate_model should unwrap a Model asset to get the raw model."""
        from mlpotion.integrations.flowyml.tensorflow.steps import evaluate_model

        mock_config_cls.return_value = MagicMock()
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = mock_evaluation_result
        mock_eval_cls.return_value = mock_evaluator

        model_asset = Model.from_keras(mock_keras_model, name="test_model")
        data_asset = Dataset.create(data=mock_tf_dataset, name="test_ds")

        result = evaluate_model(model=model_asset, data=data_asset)

        assert isinstance(result, Metrics)
        # Verify evaluator received the raw model, not the wrapper
        call_kwargs = mock_evaluator.evaluate.call_args
        raw_model_passed = call_kwargs.kwargs.get("model") or call_kwargs[1].get(
            "model"
        )
        assert raw_model_passed is mock_keras_model


# -------------------------------------------------------------------------
# Export / Save / Load / Inspect Step Tests
# -------------------------------------------------------------------------


class TestExportModel:
    """Test export_model step returns a Model asset."""

    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelExporter")
    def test_returns_model_asset(self, mock_exp_cls, mock_keras_model):
        """export_model must return a Model asset."""
        from mlpotion.integrations.flowyml.tensorflow.steps import export_model

        mock_exp_cls.return_value = MagicMock()

        result = export_model(
            model=mock_keras_model,
            export_path="/exported/model",
            export_format="keras",
        )

        assert isinstance(result, Model), f"Expected Model, got {type(result)}"
        assert result.name == "tf_exported_model"
        assert result.data is mock_keras_model

    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelExporter")
    def test_accepts_model_asset_input(self, mock_exp_cls, mock_keras_model):
        """export_model should unwrap a Model asset."""
        from mlpotion.integrations.flowyml.tensorflow.steps import export_model

        mock_exp_cls.return_value = MagicMock()
        model_asset = Model.from_keras(mock_keras_model, name="trained")

        result = export_model(
            model=model_asset,
            export_path="/export/model",
        )

        assert isinstance(result, Model)


class TestSaveModel:
    """Test save_model step returns a Model asset."""

    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelPersistence")
    def test_returns_model_asset(self, mock_persist_cls, mock_keras_model):
        """save_model must return a Model asset."""
        from mlpotion.integrations.flowyml.tensorflow.steps import save_model

        mock_persist_cls.return_value = MagicMock()

        result = save_model(
            model=mock_keras_model,
            save_path="/saved/model.keras",
        )

        assert isinstance(result, Model), f"Expected Model, got {type(result)}"
        assert result.name == "tf_saved_model"


class TestLoadModel:
    """Test load_model step returns a Model asset."""

    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelPersistence")
    def test_returns_model_asset(self, mock_persist_cls, mock_keras_model):
        """load_model must return a Model asset wrapping the loaded model."""
        from mlpotion.integrations.flowyml.tensorflow.steps import load_model

        mock_persist = MagicMock()
        mock_persist.load.return_value = (mock_keras_model, None)
        mock_persist_cls.return_value = mock_persist

        result = load_model(model_path="/saved/model.keras")

        assert isinstance(result, Model), f"Expected Model, got {type(result)}"
        assert result.name == "tf_loaded_model"
        assert result.data is mock_keras_model
        mock_persist.load.assert_called_once_with(inspect=False)


class TestInspectModel:
    """Test inspect_model step returns a Metrics asset."""

    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelInspector")
    def test_returns_metrics_asset(self, mock_insp_cls, mock_keras_model):
        """inspect_model must return a Metrics asset."""
        from mlpotion.integrations.flowyml.tensorflow.steps import inspect_model

        mock_inspector = MagicMock()
        mock_inspector.inspect.return_value = {
            "name": "sequential",
            "parameters": {"total": 177, "trainable": 177},
        }
        mock_insp_cls.return_value = mock_inspector

        result = inspect_model(model=mock_keras_model)

        assert isinstance(result, Metrics), f"Expected Metrics, got {type(result)}"
        assert result.name == "tf_model_inspection"
        assert result.data["name"] == "sequential"
        assert result.data["parameters"]["total"] == 177


# -------------------------------------------------------------------------
# DAG Wiring Tests
# -------------------------------------------------------------------------


class TestDAGWiring:
    """Verify that @step output names wire correctly to downstream inputs."""

    def test_train_outputs_match_evaluate_inputs(self):
        """train_model outputs must include 'model' to match evaluate_model inputs."""
        from mlpotion.integrations.flowyml.tensorflow.steps import (
            train_model,
            evaluate_model,
        )

        train_outputs = set(train_model.outputs)
        eval_inputs = set(evaluate_model.inputs)

        assert "model" in train_outputs, (
            f"train_model outputs={train_outputs} must include 'model' "
            f"to wire into evaluate_model inputs={eval_inputs}"
        )
        assert "model" in eval_inputs

    def test_load_output_matches_train_input(self):
        """load_data output 'dataset' must match train_model input 'dataset'."""
        from mlpotion.integrations.flowyml.tensorflow.steps import (
            load_data,
            train_model,
        )

        load_outputs = set(load_data.outputs)
        train_inputs = set(train_model.inputs)

        assert "dataset" in load_outputs
        assert "dataset" in train_inputs

    def test_step_decorator_tags(self):
        """All steps should have framework=tensorflow in tags."""
        from mlpotion.integrations.flowyml.tensorflow import steps

        step_functions = [
            steps.load_data,
            steps.optimize_data,
            steps.transform_data,
            steps.train_model,
            steps.evaluate_model,
            steps.export_model,
            steps.save_model,
            steps.load_model,
            steps.inspect_model,
        ]

        for fn in step_functions:
            assert (
                fn.tags.get("framework") == "tensorflow"
            ), f"{fn.name} missing framework=tensorflow tag"


# -------------------------------------------------------------------------
# Pipeline Template Tests
# -------------------------------------------------------------------------


class TestTFPipelineTemplates:
    """Test the pre-built TensorFlow pipeline templates."""

    @patch("mlpotion.integrations.flowyml.tensorflow.pipelines.Pipeline")
    def test_training_pipeline_has_3_steps(self, mock_pipeline_cls):
        """Training pipeline: load → train → evaluate = 3 steps."""
        from mlpotion.integrations.flowyml.tensorflow.pipelines import (
            create_tf_training_pipeline,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.name = "tf_training"
        mock_pipeline_cls.return_value = mock_pipeline

        _pipeline = create_tf_training_pipeline()
        assert mock_pipeline.add_step.call_count == 3

    @patch("mlpotion.integrations.flowyml.tensorflow.pipelines.Pipeline")
    def test_full_pipeline_has_5_steps(self, mock_pipeline_cls):
        """Full pipeline: load → optimize → train → evaluate → export = 5 steps."""
        from mlpotion.integrations.flowyml.tensorflow.pipelines import (
            create_tf_full_pipeline,
        )

        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        create_tf_full_pipeline()
        assert mock_pipeline.add_step.call_count == 5

    @patch("mlpotion.integrations.flowyml.tensorflow.pipelines.Pipeline")
    def test_evaluation_pipeline_has_4_steps(self, mock_pipeline_cls):
        """Evaluation pipeline: load_model → load_data → evaluate → inspect = 4 steps."""
        from mlpotion.integrations.flowyml.tensorflow.pipelines import (
            create_tf_evaluation_pipeline,
        )

        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        create_tf_evaluation_pipeline()
        assert mock_pipeline.add_step.call_count == 4

    @patch("mlpotion.integrations.flowyml.tensorflow.pipelines.Pipeline")
    def test_export_pipeline_has_3_steps(self, mock_pipeline_cls):
        """Export pipeline: load_model → export → save = 3 steps."""
        from mlpotion.integrations.flowyml.tensorflow.pipelines import (
            create_tf_export_pipeline,
        )

        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        create_tf_export_pipeline()
        assert mock_pipeline.add_step.call_count == 3

    @patch("mlpotion.integrations.flowyml.tensorflow.pipelines.Pipeline")
    def test_pipeline_passes_context(self, mock_pipeline_cls):
        """Pipeline must pass context to Pipeline constructor."""
        from mlpotion.integrations.flowyml.tensorflow.pipelines import (
            create_tf_training_pipeline,
        )
        from flowyml.core.context import Context

        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        ctx = Context(file_path="data.csv", epochs=5)
        create_tf_training_pipeline(name="test", context=ctx, project_name="proj")

        call_kwargs = mock_pipeline_cls.call_args.kwargs
        assert call_kwargs["name"] == "test"
        assert call_kwargs["context"] is ctx
        assert call_kwargs["project_name"] == "proj"


# -------------------------------------------------------------------------
# End-to-End Integration Test
# -------------------------------------------------------------------------


class TestEndToEndPipeline:
    """End-to-end test exercising the full pipeline with mocked components.

    Simulates: load_data → train_model → evaluate_model → export_model → save_model
    """

    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelPersistence")
    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelExporter")
    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelEvaluator")
    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelEvaluationConfig")
    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelTrainer")
    @patch("mlpotion.integrations.flowyml.tensorflow.steps.ModelTrainingConfig")
    @patch("mlpotion.integrations.flowyml.tensorflow.steps.CSVDataLoader")
    @patch("mlpotion.integrations.flowyml.tensorflow.steps.DataLoadingConfig")
    def test_full_pipeline_flow(
        self,
        mock_load_config_cls,
        mock_loader_cls,
        mock_train_config_cls,
        mock_trainer_cls,
        mock_eval_config_cls,
        mock_evaluator_cls,
        mock_exporter_cls,
        mock_persist_cls,
        mock_keras_model,
    ):
        """Execute load → train → evaluate → export → save with artifact passing."""
        from mlpotion.integrations.flowyml.tensorflow.steps import (
            load_data,
            train_model,
            evaluate_model,
            export_model,
            save_model,
        )

        # ---- Step 1: Load Data ----
        mock_load_config = MagicMock()
        mock_load_config.dict.return_value = {
            "file_pattern": "data/train.csv",
            "batch_size": 32,
            "label_name": "target",
            "column_names": None,
        }
        mock_load_config_cls.return_value = mock_load_config

        mock_tf_ds = MagicMock()
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_tf_ds
        mock_loader_cls.return_value = mock_loader

        dataset = load_data(
            file_path="data/train.csv", batch_size=32, label_name="target"
        )
        assert isinstance(dataset, Dataset)
        assert dataset.data is mock_tf_ds

        # ---- Step 2: Train Model ----
        mock_train_config_cls.return_value = MagicMock()
        mock_result = MagicMock()
        mock_result.model = mock_keras_model
        mock_result.metrics = {"loss": 0.1, "mae": 0.05}
        mock_result.history = {"loss": [0.5, 0.3, 0.1]}
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_result
        mock_trainer_cls.return_value = mock_trainer

        model_asset, metrics_asset = train_model(
            model=mock_keras_model,
            data=dataset,  # <-- passing Dataset asset
            epochs=10,
            learning_rate=0.001,
        )

        assert isinstance(model_asset, Model)
        assert isinstance(metrics_asset, Metrics)
        assert metrics_asset.data["epochs_completed"] == 10

        # ---- Step 3: Evaluate Model ----
        mock_eval_config_cls.return_value = MagicMock()
        mock_eval_result = MagicMock()
        mock_eval_result.metrics = {"accuracy": 0.96, "loss": 0.08}
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = mock_eval_result
        mock_evaluator_cls.return_value = mock_evaluator

        eval_metrics = evaluate_model(
            model=model_asset,
            data=dataset,
        )

        assert isinstance(eval_metrics, Metrics)
        assert eval_metrics.data["accuracy"] == 0.96

        # ---- Step 4: Export Model ----
        mock_exporter_cls.return_value = MagicMock()

        exported = export_model(
            model=model_asset,
            export_path="/models/production/model",
            export_format="keras",
        )

        assert isinstance(exported, Model)
        assert exported.name == "tf_exported_model"

        # ---- Step 5: Save Model ----
        mock_persist_cls.return_value = MagicMock()

        saved = save_model(
            model=model_asset,
            save_path="/models/backup/model.keras",
        )

        assert isinstance(saved, Model)
        assert saved.name == "tf_saved_model"
