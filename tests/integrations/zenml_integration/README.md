# ZenML Integration Tests

This directory contains integration tests for ZenML steps across all supported frameworks (TensorFlow, Keras, PyTorch).

**⚠️ Important:** This directory is named `zenml_integration` (not `zenml`) to avoid Python module shadowing issues with the actual `zenml` package.

## Prerequisites

To run these tests, you need to install ZenML:

```bash
# Install with specific framework
poetry install --extras tensorflow-zenml
poetry install --extras pytorch-zenml

# Or install ZenML only
poetry install --extras zenml

# Or install all dependencies
poetry install --extras all
```

## Running the Tests

### Run all ZenML integration tests:
```bash
pytest tests/integrations/zenml_integration/ -v
```

### Run tests for a specific framework:
```bash
# TensorFlow
pytest tests/integrations/zenml_integration/tensorflow/ -v

# Keras
pytest tests/integrations/zenml_integration/keras/ -v

# PyTorch
pytest tests/integrations/zenml_integration/pytorch/ -v
```

### Skip tests when ZenML is not installed:
The tests will automatically skip if ZenML is not installed. You'll see output like:
```
SKIPPED [1] tests/integrations/zenml/tensorflow/test_steps.py: ZenML not installed
```

### Run only integration tests:
```bash
pytest -m integration
```

### Run without ZenML tests:
```bash
pytest -m "not integration"
```

## Test Structure

Each framework has comprehensive tests covering:

### TensorFlow Tests
- `test_load_data_step` - Load CSV data into TensorFlow datasets
- `test_optimize_data_step` - Optimize datasets for training
- `test_train_model_step` - Train TensorFlow/Keras models
- `test_evaluate_model_step` - Evaluate model performance
- `test_save_and_load_model_steps` - Model persistence
- `test_export_model_step` - Export models to various formats
- `test_inspect_model_step` - Model inspection and metadata
- `test_transform_data_step` - Data transformation with model inference

### Keras Tests
- `test_load_data_step` - Load CSV data into Keras Sequences
- `test_train_model_step` - Train Keras models
- `test_evaluate_model_step` - Evaluate Keras models
- `test_save_and_load_model_steps` - Model persistence
- `test_export_model_step` - Export Keras models
- `test_inspect_model_step` - Model inspection
- `test_transform_data_step` - Data transformation

### PyTorch Tests
- `test_load_csv_data_step` - Load CSV data into PyTorch DataLoaders
- `test_load_streaming_csv_data_step` - Load large CSV files as streaming datasets
- `test_train_model_step` - Train PyTorch models
- `test_evaluate_model_step` - Evaluate PyTorch models
- `test_save_and_load_model_steps` - Model persistence (state_dict)
- `test_export_model_step_state_dict` - Export as state_dict
- `test_export_model_step_torchscript` - Export as TorchScript
- `test_train_with_validation_step` - Training with validation data

## Test Features

All tests:
- Use temporary directories for test data and models
- Initialize ZenML client in `setUpClass` when available
- Create sample CSV data with configurable features
- Test complete ML workflows from data loading to model deployment
- Include proper error handling and logging
- Follow the `TestBase` pattern for consistent test structure

## ZenML Client Initialization

The tests attempt to initialize a ZenML client in the `setUpClass` method:

```python
if ZENML_AVAILABLE:
    try:
        client = Client()
        logger.info(f"ZenML client initialized: {client.active_stack_model.name}")
    except Exception as e:
        logger.warning(f"Could not initialize ZenML client: {e}")
```

If ZenML is not installed or the client cannot be initialized, the tests will skip gracefully.

## Troubleshooting

### ZenML not installed
If you see `ModuleNotFoundError: No module named 'zenml'`, install it:
```bash
poetry install --extras zenml
```

### ZenML client initialization fails
If ZenML is installed but the client fails to initialize, you may need to initialize a ZenML repository:
```bash
zenml init
```

### Tests are skipped
This is expected behavior when ZenML is not installed. The tests use `@pytest.mark.skipif` to skip when dependencies are missing.
