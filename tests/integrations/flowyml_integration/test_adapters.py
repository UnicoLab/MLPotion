"""Tests for the generic FlowyMLAdapter."""

import pytest
from unittest.mock import MagicMock


# Skip if flowyml not available
flowyml = pytest.importorskip("flowyml")


class TestFlowyMLAdapter:
    """Test FlowyMLAdapter factory methods."""

    def test_create_data_loader_step(self):
        """Test wrapping a DataLoader into a FlowyML step."""
        from mlpotion.integrations.flowyml.adapters import FlowyMLAdapter

        # Mock a DataLoader
        mock_loader = MagicMock()
        mock_loader.load.return_value = [1, 2, 3]

        step = FlowyMLAdapter.create_data_loader_step(mock_loader)

        # Verify it's a FlowyML Step
        from flowyml.core.step import Step

        assert isinstance(step, Step)
        assert step.name == "load_data"
        assert step.tags == {"component": "data_loader"}

    def test_create_data_loader_step_custom_config(self):
        """Test data loader step with custom configuration."""
        from mlpotion.integrations.flowyml.adapters import FlowyMLAdapter

        mock_loader = MagicMock()
        mock_loader.load.return_value = [1, 2, 3]

        step = FlowyMLAdapter.create_data_loader_step(
            mock_loader,
            name="custom_loader",
            cache=False,
            retry=3,
            tags={"env": "staging"},
        )

        assert step.name == "custom_loader"
        assert step.cache is False
        assert step.retry == 3
        assert step.tags == {"env": "staging"}

    def test_create_training_step(self):
        """Test wrapping a ModelTrainer into a FlowyML step."""
        from mlpotion.integrations.flowyml.adapters import FlowyMLAdapter

        mock_trainer = MagicMock()
        result = MagicMock()
        result.model = "trained_model"
        result.history = {"loss": [0.5, 0.3]}
        mock_trainer.train.return_value = result

        step = FlowyMLAdapter.create_training_step(mock_trainer)

        from flowyml.core.step import Step

        assert isinstance(step, Step)
        assert step.name == "train_model"
        assert step.cache is False  # Training should not cache by default
        assert step.tags == {"component": "model_trainer"}

    def test_create_evaluation_step(self):
        """Test wrapping a ModelEvaluator into a FlowyML step."""
        from mlpotion.integrations.flowyml.adapters import FlowyMLAdapter

        mock_evaluator = MagicMock()
        result = MagicMock()
        result.metrics = {"accuracy": 0.95, "loss": 0.05}
        mock_evaluator.evaluate.return_value = result

        step = FlowyMLAdapter.create_evaluation_step(mock_evaluator)

        from flowyml.core.step import Step

        assert isinstance(step, Step)
        assert step.name == "evaluate_model"
        assert step.cache == "input_hash"  # Evaluation should cache by input
        assert step.tags == {"component": "model_evaluator"}

    def test_create_training_step_with_resources(self):
        """Test training step with GPU resource requirements."""
        from mlpotion.integrations.flowyml.adapters import FlowyMLAdapter

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = MagicMock(model="m", history={})

        try:
            from flowyml.core.resources import ResourceRequirements, GPUConfig

            resources = ResourceRequirements(
                gpu=GPUConfig(count=1, gpu_type="nvidia-tesla-t4")
            )
        except ImportError:
            resources = {"gpu": {"count": 1}}

        step = FlowyMLAdapter.create_training_step(
            mock_trainer,
            resources=resources,
        )

        assert step.resources is not None


class TestFlowyMLAdapterImports:
    """Test that the integration module imports correctly."""

    def test_import_adapter(self):
        """Test top-level import."""
        from mlpotion.integrations.flowyml import FlowyMLAdapter

        assert FlowyMLAdapter is not None

    def test_import_flowyml_available(self):
        """Test FLOWYML_AVAILABLE flag."""
        from mlpotion.integrations.flowyml import FLOWYML_AVAILABLE

        assert FLOWYML_AVAILABLE is True
