"""Tests for FlowyML PyTorch steps — unit tests + end-to-end pipeline tests.

Tests verify:
- Every step returns the correct FlowyML artifact type (Dataset, Model, Metrics)
- Artifact metadata is populated correctly
- DAG wiring (output names match downstream input names)
- Steps gracefully accept both raw objects and artifact-wrapped inputs
- End-to-end pipeline executes successfully with mocked data
"""

import pytest
from unittest.mock import MagicMock, patch

# Skip entire module if flowyml or torch not available
flowyml = pytest.importorskip("flowyml")
torch = pytest.importorskip("torch")

from flowyml import Dataset, Model, Metrics  # noqa: E402

# Pre-import so @patch can resolve the full module path
import mlpotion.integrations.flowyml.pytorch.steps  # noqa: E402, F401


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def mock_dataloader():
    """Create a mock DataLoader."""
    dl = MagicMock()
    dl.batch_size = 32
    dl.__len__ = MagicMock(return_value=10)
    return dl


@pytest.fixture
def mock_pytorch_model():
    """Create a simple PyTorch model for testing."""
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
    )
    return model


@pytest.fixture
def mock_training_result(mock_pytorch_model):
    """Create a mock training result."""
    result = MagicMock()
    result.model = mock_pytorch_model
    result.metrics = {"loss": 0.3}
    result.history = {"loss": [0.5, 0.3]}
    return result


@pytest.fixture
def mock_evaluation_result():
    """Create a mock evaluation result."""
    result = MagicMock()
    result.metrics = {"accuracy": 0.90, "loss": 0.1}
    return result


# -------------------------------------------------------------------------
# Data Step Tests
# -------------------------------------------------------------------------


class TestLoadCSVData:
    """Test load_csv_data step returns a Dataset asset."""

    @patch("mlpotion.integrations.flowyml.pytorch.steps.CSVDataLoader")
    @patch("mlpotion.integrations.flowyml.pytorch.steps.DataLoadingConfig")
    @patch("mlpotion.integrations.flowyml.pytorch.steps.CSVDataset")
    def test_returns_dataset_asset(
        self, mock_dataset_cls, mock_config_cls, mock_loader_cls
    ):
        """load_csv_data must return a Dataset asset, not a raw DataLoader."""
        from mlpotion.integrations.flowyml.pytorch.steps import load_csv_data

        mock_config = MagicMock()
        mock_config.dict.return_value = {
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
        }
        mock_config_cls.return_value = mock_config

        mock_dataset = MagicMock()
        mock_dataset_cls.return_value = mock_dataset

        mock_loader = MagicMock()
        mock_loader.load.return_value = MagicMock()
        mock_loader_cls.return_value = mock_loader

        result = load_csv_data(file_path="data/*.csv", batch_size=32)

        assert isinstance(result, Dataset), f"Expected Dataset, got {type(result)}"
        assert result.data is not None
        mock_dataset_cls.assert_called_once()
        mock_loader.load.assert_called_once_with(mock_dataset)


class TestLoadStreamingCSVData:
    """Test load_streaming_csv_data step returns a Dataset asset."""

    @patch("mlpotion.integrations.flowyml.pytorch.steps.CSVDataLoader")
    @patch("mlpotion.integrations.flowyml.pytorch.steps.DataLoadingConfig")
    @patch("mlpotion.integrations.flowyml.pytorch.steps.StreamingCSVDataset")
    def test_returns_dataset_asset(
        self, mock_dataset_cls, mock_config_cls, mock_loader_cls
    ):
        """load_streaming_csv_data must return a Dataset asset."""
        from mlpotion.integrations.flowyml.pytorch.steps import load_streaming_csv_data

        mock_config = MagicMock()
        mock_config.dict.return_value = {
            "batch_size": 64,
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
        }
        mock_config_cls.return_value = mock_config

        mock_dataset = MagicMock()
        mock_dataset_cls.return_value = mock_dataset

        mock_loader = MagicMock()
        mock_loader.load.return_value = MagicMock()
        mock_loader_cls.return_value = mock_loader

        result = load_streaming_csv_data(file_path="data/*.csv", batch_size=64)

        assert isinstance(result, Dataset), f"Expected Dataset, got {type(result)}"
        assert result is not None


# -------------------------------------------------------------------------
# Training Step Tests
# -------------------------------------------------------------------------


class TestTrainModel:
    """Test train_model step returns (Model, Metrics) tuple."""

    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelTrainer")
    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelTrainingConfig")
    def test_returns_model_and_metrics(
        self,
        mock_config_cls,
        mock_trainer_cls,
        mock_dataloader,
        mock_pytorch_model,
        mock_training_result,
    ):
        """train_model must return (Model, Metrics) tuple."""
        from mlpotion.integrations.flowyml.pytorch.steps import train_model

        mock_config_cls.return_value = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_training_result
        mock_trainer_cls.return_value = mock_trainer

        model_asset, metrics_asset = train_model(
            model=mock_pytorch_model,
            data=mock_dataloader,
            epochs=5,
            learning_rate=0.01,
        )

        assert isinstance(
            model_asset, Model
        ), f"Expected Model, got {type(model_asset)}"
        assert isinstance(
            metrics_asset, Metrics
        ), f"Expected Metrics, got {type(metrics_asset)}"

    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelTrainer")
    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelTrainingConfig")
    def test_model_asset_metadata(
        self,
        mock_config_cls,
        mock_trainer_cls,
        mock_dataloader,
        mock_pytorch_model,
        mock_training_result,
    ):
        """Check that Model asset name and framework are set correctly."""
        from mlpotion.integrations.flowyml.pytorch.steps import train_model

        mock_config_cls.return_value = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_training_result
        mock_trainer_cls.return_value = mock_trainer

        model_asset, _ = train_model(
            model=mock_pytorch_model,
            data=mock_dataloader,
            epochs=10,
        )

        assert model_asset.name == "pytorch_trained_model"
        assert model_asset.framework == "pytorch"

    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelTrainer")
    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelTrainingConfig")
    def test_metrics_asset_properties(
        self,
        mock_config_cls,
        mock_trainer_cls,
        mock_dataloader,
        mock_pytorch_model,
        mock_training_result,
    ):
        """Metrics asset must contain training configuration."""
        from mlpotion.integrations.flowyml.pytorch.steps import train_model

        mock_config_cls.return_value = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_training_result
        mock_trainer_cls.return_value = mock_trainer

        _, metrics_asset = train_model(
            model=mock_pytorch_model,
            data=mock_dataloader,
            epochs=5,
            learning_rate=0.001,
        )

        assert metrics_asset.name == "pytorch_training_metrics"
        props = metrics_asset.metadata.properties
        assert props["epochs"] == 5
        assert props["learning_rate"] == 0.001

    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelTrainer")
    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelTrainingConfig")
    def test_accepts_dataset_asset_input(
        self,
        mock_config_cls,
        mock_trainer_cls,
        mock_dataloader,
        mock_pytorch_model,
        mock_training_result,
    ):
        """train_model should accept a Dataset-wrapped input and unwrap it."""
        from mlpotion.integrations.flowyml.pytorch.steps import train_model

        mock_config_cls.return_value = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_training_result
        mock_trainer_cls.return_value = mock_trainer

        wrapped_data = Dataset.create(data=mock_dataloader, name="test_ds")

        model_asset, metrics_asset = train_model(
            model=mock_pytorch_model,
            data=wrapped_data,
            epochs=3,
        )

        # Trainer should receive the raw dataloader, not the Dataset wrapper
        call_kwargs = mock_trainer.train.call_args
        assert (
            call_kwargs.kwargs.get("dataloader") is mock_dataloader
            or call_kwargs[1].get("dataloader") is mock_dataloader
        )


# -------------------------------------------------------------------------
# Evaluation Step Tests
# -------------------------------------------------------------------------


class TestEvaluateModel:
    """Test evaluate_model step returns a Metrics asset."""

    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelEvaluator")
    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelEvaluationConfig")
    def test_returns_metrics_asset(
        self,
        mock_config_cls,
        mock_eval_cls,
        mock_dataloader,
        mock_pytorch_model,
        mock_evaluation_result,
    ):
        """evaluate_model must return a Metrics asset."""
        from mlpotion.integrations.flowyml.pytorch.steps import evaluate_model

        mock_config_cls.return_value = MagicMock()
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = mock_evaluation_result
        mock_eval_cls.return_value = mock_evaluator

        result = evaluate_model(
            model=mock_pytorch_model,
            data=mock_dataloader,
        )

        assert isinstance(result, Metrics), f"Expected Metrics, got {type(result)}"
        assert result.name == "pytorch_evaluation_metrics"

    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelEvaluator")
    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelEvaluationConfig")
    def test_accepts_model_asset_input(
        self,
        mock_config_cls,
        mock_eval_cls,
        mock_dataloader,
        mock_pytorch_model,
        mock_evaluation_result,
    ):
        """evaluate_model should unwrap a Model asset to get the raw model."""
        from mlpotion.integrations.flowyml.pytorch.steps import evaluate_model

        mock_config_cls.return_value = MagicMock()
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = mock_evaluation_result
        mock_eval_cls.return_value = mock_evaluator

        model_asset = Model.from_pytorch(mock_pytorch_model, name="test_model")
        data_asset = Dataset.create(data=mock_dataloader, name="test_ds")

        result = evaluate_model(model=model_asset, data=data_asset)

        assert isinstance(result, Metrics)
        # Verify evaluator received the raw model, not the wrapper
        call_kwargs = mock_evaluator.evaluate.call_args
        raw_model_passed = call_kwargs.kwargs.get("model") or call_kwargs[1].get(
            "model"
        )
        assert raw_model_passed is mock_pytorch_model


# -------------------------------------------------------------------------
# Export / Save / Load Step Tests
# -------------------------------------------------------------------------


class TestExportModel:
    """Test export_model step returns a Model asset."""

    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelExporter")
    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelExportConfig")
    def test_returns_model_asset(
        self, mock_config_cls, mock_exp_cls, mock_pytorch_model
    ):
        """export_model must return a Model asset."""
        from mlpotion.integrations.flowyml.pytorch.steps import export_model

        mock_config_cls.return_value = MagicMock()
        mock_exp_cls.return_value = MagicMock()

        result = export_model(
            model=mock_pytorch_model,
            export_path="/exported/model.pt",
            export_format="torchscript",
        )

        assert isinstance(result, Model), f"Expected Model, got {type(result)}"
        assert result.name == "pytorch_exported_model"
        assert result.data is mock_pytorch_model

    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelExporter")
    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelExportConfig")
    def test_accepts_model_asset_input(
        self, mock_config_cls, mock_exp_cls, mock_pytorch_model
    ):
        """export_model should unwrap a Model asset."""
        from mlpotion.integrations.flowyml.pytorch.steps import export_model

        mock_config_cls.return_value = MagicMock()
        mock_exp_cls.return_value = MagicMock()
        model_asset = Model.from_pytorch(mock_pytorch_model, name="trained")

        result = export_model(
            model=model_asset,
            export_path="/export/model.pt",
        )

        assert isinstance(result, Model)


class TestSaveModel:
    """Test save_model step returns a Model asset."""

    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelPersistence")
    def test_returns_model_asset(self, mock_persist_cls, mock_pytorch_model):
        """save_model must return a Model asset."""
        from mlpotion.integrations.flowyml.pytorch.steps import save_model

        mock_persist_cls.return_value = MagicMock()

        result = save_model(
            model=mock_pytorch_model,
            save_path="/saved/model.pt",
        )

        assert isinstance(result, Model), f"Expected Model, got {type(result)}"
        assert result.name == "pytorch_saved_model"


class TestLoadModel:
    """Test load_model step returns a Model asset."""

    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelPersistence")
    def test_returns_model_asset(self, mock_persist_cls, mock_pytorch_model):
        """load_model must return a Model asset wrapping the loaded model."""
        from mlpotion.integrations.flowyml.pytorch.steps import load_model

        mock_persist = MagicMock()
        mock_persist.load.return_value = mock_pytorch_model
        mock_persist_cls.return_value = mock_persist

        result = load_model(model_path="/saved/model.pt")

        assert isinstance(result, Model), f"Expected Model, got {type(result)}"
        assert result.name == "pytorch_loaded_model"
        assert result.data is mock_pytorch_model
        mock_persist.load.assert_called_once()


# -------------------------------------------------------------------------
# DAG Wiring Tests
# -------------------------------------------------------------------------


class TestDAGWiring:
    """Verify that @step output names wire correctly to downstream inputs."""

    def test_train_outputs_match_evaluate_inputs(self):
        """train_model outputs must include 'model' to match evaluate_model inputs."""
        from mlpotion.integrations.flowyml.pytorch.steps import (
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
        """load_csv_data output 'dataset' must match train_model input 'dataset'."""
        from mlpotion.integrations.flowyml.pytorch.steps import (
            load_csv_data,
            train_model,
        )

        load_outputs = set(load_csv_data.outputs)
        train_inputs = set(train_model.inputs)

        assert "dataset" in load_outputs
        assert "dataset" in train_inputs

    def test_step_decorator_tags(self):
        """All steps should have framework=pytorch in tags."""
        from mlpotion.integrations.flowyml.pytorch import steps

        step_functions = [
            steps.load_csv_data,
            steps.load_streaming_csv_data,
            steps.train_model,
            steps.evaluate_model,
            steps.export_model,
            steps.save_model,
            steps.load_model,
        ]

        for fn in step_functions:
            assert (
                fn.tags.get("framework") == "pytorch"
            ), f"{fn.name} missing framework=pytorch tag"


# -------------------------------------------------------------------------
# Pipeline Template Tests
# -------------------------------------------------------------------------


class TestPyTorchPipelineTemplates:
    """Test the pre-built PyTorch pipeline templates."""

    @patch("mlpotion.integrations.flowyml.pytorch.pipelines.Pipeline")
    def test_training_pipeline_has_3_steps(self, mock_pipeline_cls):
        """Training pipeline: load → train → evaluate = 3 steps."""
        from mlpotion.integrations.flowyml.pytorch.pipelines import (
            create_pytorch_training_pipeline,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.name = "pytorch_training"
        mock_pipeline_cls.return_value = mock_pipeline

        _pipeline = create_pytorch_training_pipeline()
        assert mock_pipeline.add_step.call_count == 3

    @patch("mlpotion.integrations.flowyml.pytorch.pipelines.Pipeline")
    def test_full_pipeline_has_5_steps(self, mock_pipeline_cls):
        """Full pipeline: load → train → evaluate → export → save = 5 steps."""
        from mlpotion.integrations.flowyml.pytorch.pipelines import (
            create_pytorch_full_pipeline,
        )

        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        create_pytorch_full_pipeline()
        assert mock_pipeline.add_step.call_count == 5

    @patch("mlpotion.integrations.flowyml.pytorch.pipelines.Pipeline")
    def test_evaluation_pipeline_has_3_steps(self, mock_pipeline_cls):
        """Evaluation pipeline: load_model → load_data → evaluate = 3 steps."""
        from mlpotion.integrations.flowyml.pytorch.pipelines import (
            create_pytorch_evaluation_pipeline,
        )

        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        create_pytorch_evaluation_pipeline()
        assert mock_pipeline.add_step.call_count == 3

    @patch("mlpotion.integrations.flowyml.pytorch.pipelines.Pipeline")
    def test_export_pipeline_has_3_steps(self, mock_pipeline_cls):
        """Export pipeline: load_model → export → save = 3 steps."""
        from mlpotion.integrations.flowyml.pytorch.pipelines import (
            create_pytorch_export_pipeline,
        )

        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        create_pytorch_export_pipeline()
        assert mock_pipeline.add_step.call_count == 3

    @patch("mlpotion.integrations.flowyml.pytorch.pipelines.Pipeline")
    def test_pipeline_passes_context(self, mock_pipeline_cls):
        """Pipeline must pass context to Pipeline constructor."""
        from mlpotion.integrations.flowyml.pytorch.pipelines import (
            create_pytorch_training_pipeline,
        )
        from flowyml.core.context import Context

        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        ctx = Context(file_path="data.csv", epochs=5)
        create_pytorch_training_pipeline(name="test", context=ctx, project_name="proj")

        call_kwargs = mock_pipeline_cls.call_args.kwargs
        assert call_kwargs["name"] == "test"
        assert call_kwargs["context"] is ctx
        assert call_kwargs["project_name"] == "proj"


# -------------------------------------------------------------------------
# End-to-End Integration Test
# -------------------------------------------------------------------------


class TestEndToEndPipeline:
    """End-to-end test exercising the full pipeline with mocked components.

    Simulates: load_csv_data → train_model → evaluate_model → export_model → save_model
    """

    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelPersistence")
    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelExporter")
    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelExportConfig")
    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelEvaluator")
    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelEvaluationConfig")
    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelTrainer")
    @patch("mlpotion.integrations.flowyml.pytorch.steps.ModelTrainingConfig")
    @patch("mlpotion.integrations.flowyml.pytorch.steps.CSVDataLoader")
    @patch("mlpotion.integrations.flowyml.pytorch.steps.DataLoadingConfig")
    @patch("mlpotion.integrations.flowyml.pytorch.steps.CSVDataset")
    def test_full_pipeline_flow(
        self,
        mock_dataset_cls,
        mock_load_config_cls,
        mock_loader_cls,
        mock_train_config_cls,
        mock_trainer_cls,
        mock_eval_config_cls,
        mock_evaluator_cls,
        mock_export_config_cls,
        mock_exporter_cls,
        mock_persist_cls,
        mock_pytorch_model,
    ):
        """Execute load → train → evaluate → export → save with artifact passing."""
        from mlpotion.integrations.flowyml.pytorch.steps import (
            load_csv_data,
            train_model,
            evaluate_model,
            export_model,
            save_model,
        )

        # ---- Step 1: Load Data ----
        mock_load_config = MagicMock()
        mock_load_config.dict.return_value = {
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
        }
        mock_load_config_cls.return_value = mock_load_config

        mock_csv_dataset = MagicMock()
        mock_dataset_cls.return_value = mock_csv_dataset

        mock_dataloader = MagicMock()
        mock_dataloader.batch_size = 32
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataloader
        mock_loader_cls.return_value = mock_loader

        dataset = load_csv_data(file_path="data/train.csv", batch_size=32)
        assert isinstance(dataset, Dataset)
        assert dataset.data is mock_dataloader

        # ---- Step 2: Train Model ----
        mock_train_config_cls.return_value = MagicMock()
        mock_result = MagicMock()
        mock_result.model = mock_pytorch_model
        mock_result.metrics = {"loss": 0.1}
        mock_result.history = {"loss": [0.5, 0.3, 0.1]}
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_result
        mock_trainer_cls.return_value = mock_trainer

        model_asset, metrics_asset = train_model(
            model=mock_pytorch_model,
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
        mock_export_config_cls.return_value = MagicMock()
        mock_exporter_cls.return_value = MagicMock()

        exported = export_model(
            model=model_asset,
            export_path="/models/production/model.pt",
            export_format="torchscript",
        )

        assert isinstance(exported, Model)
        assert exported.name == "pytorch_exported_model"

        # ---- Step 5: Save Model ----
        mock_persist_cls.return_value = MagicMock()

        saved = save_model(
            model=model_asset,
            save_path="/models/backup/model.pt",
        )

        assert isinstance(saved, Model)
        assert saved.name == "pytorch_saved_model"
