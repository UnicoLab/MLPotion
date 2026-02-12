"""Tests for FlowyML Keras steps — unit tests + end-to-end pipeline tests.

Tests verify:
- Every step returns the correct FlowyML artifact type (Dataset, Model, Metrics)
- Artifact metadata is populated correctly
- DAG wiring (output names match downstream input names)
- Steps gracefully accept both raw objects and artifact-wrapped inputs
- End-to-end pipeline executes successfully with dummy data
"""

import pytest
from unittest.mock import MagicMock, patch

# Skip entire module if flowyml or keras not available
flowyml = pytest.importorskip("flowyml")
keras = pytest.importorskip("keras")

from flowyml import Dataset, Model, Metrics  # noqa: E402

# Pre-import so @patch decorators can resolve the full module path
import mlpotion.integrations.flowyml.keras.steps  # noqa: E402, F401


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def mock_csv_sequence():
    """Create a mock CSVSequence."""
    seq = MagicMock()
    seq.__len__ = MagicMock(return_value=10)
    seq.batch_size = 32
    return seq


@pytest.fixture
def mock_keras_model():
    """Create a simple Keras model for testing."""
    model = keras.Sequential(
        [
            keras.layers.Dense(16, activation="relu", input_shape=(10,)),
            keras.layers.Dense(1),
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

    @patch("mlpotion.integrations.flowyml.keras.steps.CSVDataLoader")
    @patch("mlpotion.integrations.flowyml.keras.steps.DataLoadingConfig")
    def test_returns_dataset_asset(self, mock_config_cls, mock_loader_cls):
        """load_data must return a Dataset asset, not a raw CSVSequence."""
        from mlpotion.integrations.flowyml.keras.steps import load_data

        mock_config = MagicMock()
        mock_config.dict.return_value = {
            "file_pattern": "data/*.csv",
            "batch_size": 32,
            "label_name": "target",
            "column_names": None,
            "shuffle": True,
            "dtype": "float32",
        }
        mock_config_cls.return_value = mock_config

        mock_sequence = MagicMock()
        mock_sequence.__len__ = MagicMock(return_value=5)
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_sequence
        mock_loader_cls.return_value = mock_loader

        result = load_data(
            file_path="data/*.csv",
            batch_size=32,
            label_name="target",
        )

        # Must be a Dataset asset
        assert isinstance(result, Dataset), f"Expected Dataset, got {type(result)}"
        # The raw data should be accessible via .data
        assert result.data is mock_sequence

    @patch("mlpotion.integrations.flowyml.keras.steps.CSVDataLoader")
    @patch("mlpotion.integrations.flowyml.keras.steps.DataLoadingConfig")
    def test_dataset_metadata(self, mock_config_cls, mock_loader_cls):
        """Check that Dataset metadata is correctly populated."""
        from mlpotion.integrations.flowyml.keras.steps import load_data

        mock_config = MagicMock()
        mock_config.dict.return_value = {
            "file_pattern": "train.csv",
            "batch_size": 64,
            "label_name": "price",
            "column_names": None,
            "shuffle": False,
            "dtype": "float32",
        }
        mock_config_cls.return_value = mock_config

        mock_sequence = MagicMock()
        mock_sequence.__len__ = MagicMock(return_value=20)
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_sequence
        mock_loader_cls.return_value = mock_loader

        result = load_data(
            file_path="train.csv",
            batch_size=64,
            label_name="price",
            shuffle=False,
        )

        assert result.name == "keras_csv_dataset"
        props = result.metadata.properties
        assert props["source"] == "train.csv"
        assert props["batch_size"] == 64
        assert props["label_name"] == "price"
        assert props["framework"] == "keras"


# -------------------------------------------------------------------------
# Transform Step Tests
# -------------------------------------------------------------------------


class TestTransformData:
    """Test transform_data step returns a Dataset asset with lineage."""

    @patch("mlpotion.integrations.flowyml.keras.steps.CSVDataTransformer")
    @patch("mlpotion.integrations.flowyml.keras.steps.DataTransformationConfig")
    def test_returns_dataset_with_lineage(
        self, mock_config_cls, mock_transformer_cls, mock_csv_sequence, mock_keras_model
    ):
        """transform_data must return a Dataset asset linked to parent."""
        from mlpotion.integrations.flowyml.keras.steps import transform_data

        mock_config_cls.return_value = MagicMock()
        mock_transformer_cls.return_value = MagicMock()

        # Wrap input as a Dataset asset to verify lineage
        parent_dataset = Dataset.create(
            data=mock_csv_sequence,
            name="input_dataset",
        )

        result = transform_data(
            dataset=parent_dataset,
            model=mock_keras_model,
            data_output_path="/output/transformed.csv",
        )

        assert isinstance(result, Dataset), f"Expected Dataset, got {type(result)}"
        assert result.name == "keras_transformed_data"
        assert result.metadata.properties["output_path"] == "/output/transformed.csv"

    @patch("mlpotion.integrations.flowyml.keras.steps.CSVDataTransformer")
    @patch("mlpotion.integrations.flowyml.keras.steps.DataTransformationConfig")
    def test_accepts_raw_sequence(
        self, mock_config_cls, mock_transformer_cls, mock_csv_sequence, mock_keras_model
    ):
        """transform_data should also accept raw CSVSequence (not wrapped)."""
        from mlpotion.integrations.flowyml.keras.steps import transform_data

        mock_config_cls.return_value = MagicMock()
        mock_transformer_cls.return_value = MagicMock()

        result = transform_data(
            dataset=mock_csv_sequence,
            model=mock_keras_model,
            data_output_path="/output/data.csv",
        )

        assert isinstance(result, Dataset)


# -------------------------------------------------------------------------
# Training Step Tests
# -------------------------------------------------------------------------


class TestTrainModel:
    """Test train_model step returns (Model, Metrics) tuple."""

    @patch("mlpotion.integrations.flowyml.keras.steps.ModelTrainer")
    @patch("mlpotion.integrations.flowyml.keras.steps.ModelTrainingConfig")
    def test_returns_model_and_metrics(
        self,
        mock_config_cls,
        mock_trainer_cls,
        mock_csv_sequence,
        mock_keras_model,
        mock_training_result,
    ):
        """train_model must return (Model, Metrics) tuple."""
        from mlpotion.integrations.flowyml.keras.steps import train_model

        mock_config_cls.return_value = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_training_result
        mock_trainer_cls.return_value = mock_trainer

        model_asset, metrics_asset = train_model(
            model=mock_keras_model,
            data=mock_csv_sequence,
            epochs=5,
            learning_rate=0.01,
        )

        assert isinstance(
            model_asset, Model
        ), f"Expected Model, got {type(model_asset)}"
        assert isinstance(
            metrics_asset, Metrics
        ), f"Expected Metrics, got {type(metrics_asset)}"

    @patch("mlpotion.integrations.flowyml.keras.steps.ModelTrainer")
    @patch("mlpotion.integrations.flowyml.keras.steps.ModelTrainingConfig")
    def test_model_asset_metadata(
        self,
        mock_config_cls,
        mock_trainer_cls,
        mock_csv_sequence,
        mock_keras_model,
        mock_training_result,
    ):
        """Check that Model asset name and framework are set correctly."""
        from mlpotion.integrations.flowyml.keras.steps import train_model

        mock_config_cls.return_value = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_training_result
        mock_trainer_cls.return_value = mock_trainer

        model_asset, _ = train_model(
            model=mock_keras_model,
            data=mock_csv_sequence,
            epochs=10,
        )

        assert model_asset.name == "keras_trained_model"
        assert model_asset.framework == "keras"
        # The raw Keras model must be accessible
        assert model_asset.data is mock_keras_model

    @patch("mlpotion.integrations.flowyml.keras.steps.ModelTrainer")
    @patch("mlpotion.integrations.flowyml.keras.steps.ModelTrainingConfig")
    def test_metrics_asset_properties(
        self,
        mock_config_cls,
        mock_trainer_cls,
        mock_csv_sequence,
        mock_keras_model,
        mock_training_result,
    ):
        """Metrics asset must contain training configuration."""
        from mlpotion.integrations.flowyml.keras.steps import train_model

        mock_config_cls.return_value = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_training_result
        mock_trainer_cls.return_value = mock_trainer

        _, metrics_asset = train_model(
            model=mock_keras_model,
            data=mock_csv_sequence,
            epochs=5,
            learning_rate=0.001,
        )

        assert metrics_asset.name == "keras_training_metrics"
        props = metrics_asset.metadata.properties
        assert props["epochs"] == 5
        assert props["learning_rate"] == 0.001

    @patch("mlpotion.integrations.flowyml.keras.steps.ModelTrainer")
    @patch("mlpotion.integrations.flowyml.keras.steps.ModelTrainingConfig")
    def test_accepts_dataset_asset_input(
        self,
        mock_config_cls,
        mock_trainer_cls,
        mock_csv_sequence,
        mock_keras_model,
        mock_training_result,
    ):
        """train_model should accept a Dataset-wrapped input and unwrap it."""
        from mlpotion.integrations.flowyml.keras.steps import train_model

        mock_config_cls.return_value = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_training_result
        mock_trainer_cls.return_value = mock_trainer

        wrapped_data = Dataset.create(data=mock_csv_sequence, name="test_ds")

        model_asset, metrics_asset = train_model(
            model=mock_keras_model,
            data=wrapped_data,
            epochs=3,
        )

        # Trainer should receive the raw sequence, not the Dataset wrapper
        call_kwargs = mock_trainer.train.call_args
        assert (
            call_kwargs.kwargs.get("dataset") is mock_csv_sequence
            or call_kwargs[1].get("dataset") is mock_csv_sequence
        )

    @patch("mlpotion.integrations.flowyml.keras.steps.ModelTrainer")
    @patch("mlpotion.integrations.flowyml.keras.steps.ModelTrainingConfig")
    @patch("mlpotion.integrations.flowyml.keras.steps.FlowymlKerasCallback")
    def test_callback_attached_when_experiment_name(
        self,
        mock_cb_cls,
        mock_config_cls,
        mock_trainer_cls,
        mock_csv_sequence,
        mock_keras_model,
        mock_training_result,
    ):
        """FlowymlKerasCallback auto-attached when experiment_name is provided."""
        from mlpotion.integrations.flowyml.keras.steps import train_model

        mock_config_cls.return_value = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_training_result
        mock_trainer_cls.return_value = mock_trainer

        train_model(
            model=mock_keras_model,
            data=mock_csv_sequence,
            epochs=3,
            experiment_name="my-exp",
            project="my-project",
        )

        mock_cb_cls.assert_called_once_with(
            experiment_name="my-exp",
            project="my-project",
            log_model=True,
        )


# -------------------------------------------------------------------------
# Evaluation Step Tests
# -------------------------------------------------------------------------


class TestEvaluateModel:
    """Test evaluate_model step returns a Metrics asset."""

    @patch("mlpotion.integrations.flowyml.keras.steps.ModelEvaluator")
    @patch("mlpotion.integrations.flowyml.keras.steps.ModelEvaluationConfig")
    def test_returns_metrics_asset(
        self,
        mock_config_cls,
        mock_eval_cls,
        mock_csv_sequence,
        mock_keras_model,
        mock_evaluation_result,
    ):
        """evaluate_model must return a Metrics asset."""
        from mlpotion.integrations.flowyml.keras.steps import evaluate_model

        mock_config_cls.return_value = MagicMock()
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = mock_evaluation_result
        mock_eval_cls.return_value = mock_evaluator

        result = evaluate_model(
            model=mock_keras_model,
            data=mock_csv_sequence,
        )

        assert isinstance(result, Metrics), f"Expected Metrics, got {type(result)}"
        assert result.name == "keras_evaluation_metrics"

    @patch("mlpotion.integrations.flowyml.keras.steps.ModelEvaluator")
    @patch("mlpotion.integrations.flowyml.keras.steps.ModelEvaluationConfig")
    def test_accepts_model_asset_input(
        self,
        mock_config_cls,
        mock_eval_cls,
        mock_csv_sequence,
        mock_keras_model,
        mock_evaluation_result,
    ):
        """evaluate_model should unwrap a Model asset to get the raw model."""
        from mlpotion.integrations.flowyml.keras.steps import evaluate_model

        mock_config_cls.return_value = MagicMock()
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = mock_evaluation_result
        mock_eval_cls.return_value = mock_evaluator

        model_asset = Model.from_keras(mock_keras_model, name="test_model")
        data_asset = Dataset.create(data=mock_csv_sequence, name="test_ds")

        result = evaluate_model(model=model_asset, data=data_asset)

        assert isinstance(result, Metrics)
        # Verify evaluator received the raw model, not the wrapper
        call_kwargs = mock_evaluator.evaluate.call_args
        raw_model_passed = call_kwargs.kwargs.get("model") or call_kwargs[1].get(
            "model"
        )
        assert raw_model_passed is mock_keras_model


# -------------------------------------------------------------------------
# Export / Save / Load Step Tests
# -------------------------------------------------------------------------


class TestExportModel:
    """Test export_model step returns a Model asset."""

    @patch("mlpotion.integrations.flowyml.keras.steps.ModelExporter")
    def test_returns_model_asset(self, mock_exp_cls, mock_keras_model):
        """export_model must return a Model asset."""
        from mlpotion.integrations.flowyml.keras.steps import export_model

        mock_exp_cls.return_value = MagicMock()

        result = export_model(
            model=mock_keras_model,
            export_path="/exported/model.keras",
            export_format="keras",
        )

        assert isinstance(result, Model), f"Expected Model, got {type(result)}"
        assert result.name == "keras_exported_model"
        assert result.data is mock_keras_model

    @patch("mlpotion.integrations.flowyml.keras.steps.ModelExporter")
    def test_accepts_model_asset_input(self, mock_exp_cls, mock_keras_model):
        """export_model should unwrap a Model asset."""
        from mlpotion.integrations.flowyml.keras.steps import export_model

        mock_exp_cls.return_value = MagicMock()
        model_asset = Model.from_keras(mock_keras_model, name="trained")

        result = export_model(
            model=model_asset,
            export_path="/export/model",
        )

        assert isinstance(result, Model)


class TestSaveModel:
    """Test save_model step returns a Model asset."""

    @patch("mlpotion.integrations.flowyml.keras.steps.ModelPersistence")
    def test_returns_model_asset(self, mock_persist_cls, mock_keras_model):
        """save_model must return a Model asset."""
        from mlpotion.integrations.flowyml.keras.steps import save_model

        mock_persist_cls.return_value = MagicMock()

        result = save_model(
            model=mock_keras_model,
            save_path="/saved/model.keras",
        )

        assert isinstance(result, Model), f"Expected Model, got {type(result)}"
        assert result.name == "keras_saved_model"


class TestLoadModel:
    """Test load_model step returns a Model asset."""

    @patch("mlpotion.integrations.flowyml.keras.steps.ModelPersistence")
    def test_returns_model_asset(self, mock_persist_cls, mock_keras_model):
        """load_model must return a Model asset wrapping the loaded model."""
        from mlpotion.integrations.flowyml.keras.steps import load_model

        mock_persist = MagicMock()
        mock_persist.load.return_value = (mock_keras_model, None)
        mock_persist_cls.return_value = mock_persist

        result = load_model(model_path="/saved/model.keras")

        assert isinstance(result, Model), f"Expected Model, got {type(result)}"
        assert result.name == "keras_loaded_model"
        assert result.data is mock_keras_model
        mock_persist.load.assert_called_once_with(inspect=False)


class TestInspectModel:
    """Test inspect_model step returns a Metrics asset."""

    @patch("mlpotion.integrations.flowyml.keras.steps.ModelInspector")
    def test_returns_metrics_asset(self, mock_insp_cls, mock_keras_model):
        """inspect_model must return a Metrics asset."""
        from mlpotion.integrations.flowyml.keras.steps import inspect_model

        mock_inspector = MagicMock()
        mock_inspector.inspect.return_value = {
            "name": "sequential",
            "parameters": {"total": 177, "trainable": 177},
        }
        mock_insp_cls.return_value = mock_inspector

        result = inspect_model(model=mock_keras_model)

        assert isinstance(result, Metrics), f"Expected Metrics, got {type(result)}"
        assert result.name == "keras_model_inspection"
        assert result.data["name"] == "sequential"
        assert result.data["parameters"]["total"] == 177


# -------------------------------------------------------------------------
# DAG Wiring Tests
# -------------------------------------------------------------------------


class TestDAGWiring:
    """Verify that @step output names wire correctly to downstream inputs."""

    def test_train_outputs_match_evaluate_inputs(self):
        """train_model outputs must include 'model' to match evaluate_model inputs."""
        from mlpotion.integrations.flowyml.keras.steps import (
            train_model,
            evaluate_model,
        )

        train_outputs = set(train_model.outputs)
        eval_inputs = set(evaluate_model.inputs)

        # 'model' must be in train_model outputs AND evaluate_model inputs
        assert "model" in train_outputs, (
            f"train_model outputs={train_outputs} must include 'model' "
            f"to wire into evaluate_model inputs={eval_inputs}"
        )
        assert "model" in eval_inputs

    def test_load_output_matches_train_input(self):
        """load_data output 'dataset' must match train_model input 'dataset'."""
        from mlpotion.integrations.flowyml.keras.steps import load_data, train_model

        load_outputs = set(load_data.outputs)
        train_inputs = set(train_model.inputs)

        assert "dataset" in load_outputs
        assert "dataset" in train_inputs

    def test_step_decorator_tags(self):
        """All steps should have framework=keras in tags."""
        from mlpotion.integrations.flowyml.keras import steps

        step_functions = [
            steps.load_data,
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
                fn.tags.get("framework") == "keras"
            ), f"{fn.name} missing framework=keras tag"


# -------------------------------------------------------------------------
# Pipeline Template Tests
# -------------------------------------------------------------------------


class TestKerasPipelineTemplates:
    """Test the pre-built Keras pipeline templates."""

    @patch("mlpotion.integrations.flowyml.keras.pipelines.Pipeline")
    def test_training_pipeline_has_3_steps(self, mock_pipeline_cls):
        """Training pipeline: load → train → evaluate = 3 steps."""
        from mlpotion.integrations.flowyml.keras.pipelines import (
            create_keras_training_pipeline,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.name = "keras_training"
        mock_pipeline_cls.return_value = mock_pipeline

        _pipeline = create_keras_training_pipeline()
        assert mock_pipeline.add_step.call_count == 3

    @patch("mlpotion.integrations.flowyml.keras.pipelines.Pipeline")
    def test_full_pipeline_has_5_steps(self, mock_pipeline_cls):
        """Full pipeline: load → transform → train → evaluate → export = 5 steps."""
        from mlpotion.integrations.flowyml.keras.pipelines import (
            create_keras_full_pipeline,
        )

        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        create_keras_full_pipeline()
        assert mock_pipeline.add_step.call_count == 5

    @patch("mlpotion.integrations.flowyml.keras.pipelines.Pipeline")
    def test_evaluation_pipeline_has_4_steps(self, mock_pipeline_cls):
        """Evaluation pipeline: load_model → load_data → evaluate → inspect = 4 steps."""
        from mlpotion.integrations.flowyml.keras.pipelines import (
            create_keras_evaluation_pipeline,
        )

        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        create_keras_evaluation_pipeline()
        assert mock_pipeline.add_step.call_count == 4

    @patch("mlpotion.integrations.flowyml.keras.pipelines.Pipeline")
    def test_export_pipeline_has_3_steps(self, mock_pipeline_cls):
        """Export pipeline: load_model → export → save = 3 steps."""
        from mlpotion.integrations.flowyml.keras.pipelines import (
            create_keras_export_pipeline,
        )

        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        create_keras_export_pipeline()
        assert mock_pipeline.add_step.call_count == 3

    @patch("mlpotion.integrations.flowyml.keras.pipelines.Pipeline")
    def test_pipeline_passes_context(self, mock_pipeline_cls):
        """Pipeline must pass context to Pipeline constructor."""
        from mlpotion.integrations.flowyml.keras.pipelines import (
            create_keras_training_pipeline,
        )
        from flowyml.core.context import Context

        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        ctx = Context(file_path="data.csv", epochs=5)
        create_keras_training_pipeline(name="test", context=ctx, project_name="proj")

        call_kwargs = mock_pipeline_cls.call_args.kwargs
        assert call_kwargs["name"] == "test"
        assert call_kwargs["context"] is ctx
        assert call_kwargs["project_name"] == "proj"


# -------------------------------------------------------------------------
# End-to-End Integration Test
# -------------------------------------------------------------------------


class TestEndToEndPipeline:
    """End-to-end test exercising the full pipeline with mocked MLPotion components.

    This simulates a full training workflow:
      load_data → train_model → evaluate_model → export_model → save_model
    verifying that artifact types flow correctly between steps.
    """

    @patch("mlpotion.integrations.flowyml.keras.steps.ModelPersistence")
    @patch("mlpotion.integrations.flowyml.keras.steps.ModelExporter")
    @patch("mlpotion.integrations.flowyml.keras.steps.ModelEvaluator")
    @patch("mlpotion.integrations.flowyml.keras.steps.ModelTrainer")
    @patch("mlpotion.integrations.flowyml.keras.steps.CSVDataLoader")
    @patch("mlpotion.integrations.flowyml.keras.steps.DataLoadingConfig")
    @patch("mlpotion.integrations.flowyml.keras.steps.ModelTrainingConfig")
    @patch("mlpotion.integrations.flowyml.keras.steps.ModelEvaluationConfig")
    def test_full_pipeline_flow(
        self,
        mock_eval_config_cls,
        mock_train_config_cls,
        mock_load_config_cls,
        mock_loader_cls,
        mock_trainer_cls,
        mock_evaluator_cls,
        mock_exporter_cls,
        mock_persist_cls,
        mock_keras_model,
    ):
        """Execute load → train → evaluate → export → save with artifact passing."""
        from mlpotion.integrations.flowyml.keras.steps import (
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
            "shuffle": True,
            "dtype": "float32",
        }
        mock_load_config_cls.return_value = mock_load_config

        mock_sequence = MagicMock()
        mock_sequence.__len__ = MagicMock(return_value=100)
        mock_sequence.batch_size = 32
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_sequence
        mock_loader_cls.return_value = mock_loader

        dataset = load_data(
            file_path="data/train.csv",
            batch_size=32,
            label_name="target",
        )
        assert isinstance(dataset, Dataset)
        assert dataset.data is mock_sequence

        # ---- Step 2: Train Model ----
        mock_train_config_cls.return_value = MagicMock()
        mock_result = MagicMock()
        mock_result.model = mock_keras_model
        mock_result.metrics = {"loss": 0.1, "mae": 0.05}
        mock_result.history = {"loss": [0.5, 0.3, 0.1], "mae": [0.3, 0.1, 0.05]}
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_result
        mock_trainer_cls.return_value = mock_trainer

        # Pass the Dataset asset — train_model should unwrap it
        model_asset, metrics_asset = train_model(
            model=mock_keras_model,
            data=dataset,  # <-- passing Dataset asset, not raw sequence
            epochs=10,
            learning_rate=0.001,
        )

        assert isinstance(model_asset, Model)
        assert isinstance(metrics_asset, Metrics)
        assert model_asset.data is mock_keras_model
        assert metrics_asset.data["epochs_completed"] == 10

        # Verify the trainer received the RAW sequence (unwrapped)
        train_call = mock_trainer.train.call_args
        assert train_call.kwargs["dataset"] is mock_sequence

        # ---- Step 3: Evaluate Model ----
        mock_eval_config_cls.return_value = MagicMock()
        mock_eval_result = MagicMock()
        mock_eval_result.metrics = {"loss": 0.08, "mae": 0.04, "accuracy": 0.96}
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = mock_eval_result
        mock_evaluator_cls.return_value = mock_evaluator

        # Pass Model+Dataset assets — evaluate_model should unwrap both
        eval_metrics = evaluate_model(
            model=model_asset,  # <-- Model asset
            data=dataset,  # <-- Dataset asset
        )

        assert isinstance(eval_metrics, Metrics)
        assert eval_metrics.data["accuracy"] == 0.96

        # Verify evaluator received RAW objects
        eval_call = mock_evaluator.evaluate.call_args
        assert eval_call.kwargs["model"] is mock_keras_model
        assert eval_call.kwargs["dataset"] is mock_sequence

        # ---- Step 4: Export Model ----
        mock_exporter_cls.return_value = MagicMock()

        exported = export_model(
            model=model_asset,  # <-- Model asset
            export_path="/models/production/model.keras",
            export_format="keras",
        )

        assert isinstance(exported, Model)
        assert exported.name == "keras_exported_model"

        # ---- Step 5: Save Model ----
        mock_persist_cls.return_value = MagicMock()

        saved = save_model(
            model=model_asset,  # <-- Model asset
            save_path="/models/backup/model.keras",
        )

        assert isinstance(saved, Model)
        assert saved.name == "keras_saved_model"

    @patch("mlpotion.integrations.flowyml.keras.steps.ModelInspector")
    @patch("mlpotion.integrations.flowyml.keras.steps.ModelEvaluator")
    @patch("mlpotion.integrations.flowyml.keras.steps.ModelPersistence")
    @patch("mlpotion.integrations.flowyml.keras.steps.CSVDataLoader")
    @patch("mlpotion.integrations.flowyml.keras.steps.DataLoadingConfig")
    @patch("mlpotion.integrations.flowyml.keras.steps.ModelEvaluationConfig")
    def test_evaluation_pipeline_flow(
        self,
        mock_eval_config_cls,
        mock_load_config_cls,
        mock_loader_cls,
        mock_persist_cls,
        mock_evaluator_cls,
        mock_inspector_cls,
        mock_keras_model,
    ):
        """End-to-end evaluation pipeline: load_model → load_data → evaluate → inspect."""
        from mlpotion.integrations.flowyml.keras.steps import (
            load_model,
            load_data,
            evaluate_model,
            inspect_model,
        )

        # ---- Load Model ----
        mock_persist = MagicMock()
        mock_persist.load.return_value = (mock_keras_model, None)
        mock_persist_cls.return_value = mock_persist

        model_asset = load_model(model_path="/models/model.keras")
        assert isinstance(model_asset, Model)

        # ---- Load Data ----
        mock_load_config = MagicMock()
        mock_load_config.dict.return_value = {
            "file_pattern": "data/test.csv",
            "batch_size": 64,
            "label_name": "target",
            "column_names": None,
            "shuffle": False,
            "dtype": "float32",
        }
        mock_load_config_cls.return_value = mock_load_config
        mock_sequence = MagicMock()
        mock_sequence.__len__ = MagicMock(return_value=50)
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_sequence
        mock_loader_cls.return_value = mock_loader

        dataset = load_data(
            file_path="data/test.csv", batch_size=64, label_name="target"
        )
        assert isinstance(dataset, Dataset)

        # ---- Evaluate ----
        mock_eval_config_cls.return_value = MagicMock()
        mock_eval_result = MagicMock()
        mock_eval_result.metrics = {"accuracy": 0.92, "loss": 0.12}
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = mock_eval_result
        mock_evaluator_cls.return_value = mock_evaluator

        eval_metrics = evaluate_model(model=model_asset, data=dataset)
        assert isinstance(eval_metrics, Metrics)

        # ---- Inspect ----
        mock_inspector = MagicMock()
        mock_inspector.inspect.return_value = {
            "name": "sequential",
            "parameters": {"total": 500, "trainable": 500},
            "layers": [{"name": "dense", "output_shape": [16]}],
        }
        mock_inspector_cls.return_value = mock_inspector

        inspection = inspect_model(model=model_asset)
        assert isinstance(inspection, Metrics)
        assert inspection.data["name"] == "sequential"
