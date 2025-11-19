# MLPotion Examples

This directory contains practical examples demonstrating how to use MLPotion with different ML frameworks, both standalone and integrated with ZenML pipelines.

## Directory Structure

```
examples/
├── data/
│   └── sample.csv          # Sample dataset for all examples
├── keras/
│   ├── basic_usage.py      # Keras standalone example
│   └── zenml_pipeline.py   # Keras with ZenML orchestration
├── pytorch/
│   ├── basic_usage.py      # PyTorch standalone example
│   └── zenml_pipeline.py   # PyTorch with ZenML orchestration
├── tensorflow/
│   ├── basic_usage.py      # TensorFlow standalone example
│   └── zenml_pipeline.py   # TensorFlow with ZenML orchestration
└── standalone/
    └── ...                 # Framework-agnostic examples
```

## Quick Start

### 1. Standalone Examples (Without ZenML)

These examples demonstrate the core MLPotion workflow without any orchestration framework:

#### Keras
```bash
python examples/keras/basic_usage.py
```

Features demonstrated:
- Load data from CSV
- Create and compile a Keras model
- Train the model
- Evaluate performance
- Save and load models

#### PyTorch
```bash
python examples/pytorch/basic_usage.py
```

Features demonstrated:
- Load data from CSV using PyTorchCSVDataset
- Create DataLoaders with PyTorchDataLoaderFactory
- Define and train a PyTorch model
- Evaluate model performance
- Save and load model state

#### TensorFlow
```bash
python examples/tensorflow/basic_usage.py
```

Features demonstrated:
- Load data from CSV with TFCSVDataLoader
- Optimize datasets for performance
- Build and compile TensorFlow models
- Train and evaluate models
- Save and export for serving

### 2. ZenML Pipeline Examples

These examples show how to orchestrate MLPotion components in reproducible ZenML pipelines:

#### Prerequisites
```bash
# Install ZenML
pip install zenml

# Initialize ZenML (first time only)
zenml init

# For testing without full stack setup
export ZENML_RUN_SINGLE_STEPS_WITHOUT_STACK=true
```

#### Keras Pipeline
```bash
python examples/keras/zenml_pipeline.py
```

Pipeline steps:
1. Load data
2. Create model
3. Create training config
4. Train model
5. Evaluate model
6. Save model
7. Export for serving

#### PyTorch Pipeline
```bash
python examples/pytorch/zenml_pipeline.py
```

Pipeline steps:
1. Load CSV data
2. Create PyTorch model
3. Create training config
4. Train model
5. Evaluate model
6. Save model (state_dict)
7. Export as TorchScript

#### TensorFlow Pipeline
```bash
python examples/tensorflow/zenml_pipeline.py
```

Pipeline steps:
1. Load data from CSV
2. Optimize dataset
3. Create TensorFlow model
4. Create training config
5. Train model
6. Evaluate model
7. Save model
8. Export as SavedModel

## Common Patterns

### Data Loading

**Keras:**
```python
from mlpotion.frameworks.keras import KerasCSVDataLoader

loader = KerasCSVDataLoader(
    file_pattern="examples/data/sample.csv",
    label_name="target",
    batch_size=8,
    shuffle=True,
)
dataset = loader.load()
```

**PyTorch:**
```python
from mlpotion.frameworks.pytorch import PyTorchCSVDataset, PyTorchDataLoaderFactory

dataset = PyTorchCSVDataset(
    file_pattern="examples/data/sample.csv",
    label_name="target",
)
factory = PyTorchDataLoaderFactory(batch_size=8, shuffle=True)
dataloader = factory.load(dataset)
```

**TensorFlow:**
```python
from mlpotion.frameworks.tensorflow import TFCSVDataLoader, TFDatasetOptimizer

loader = TFCSVDataLoader(
    file_pattern="examples/data/sample.csv",
    label_name="target",
)
dataset = loader.load()

optimizer = TFDatasetOptimizer(batch_size=8, shuffle_buffer_size=100)
dataset = optimizer.optimize(dataset)
```

### Training

All frameworks use a similar training pattern:

```python
from mlpotion.frameworks.[framework] import [Framework]ModelTrainer, [Framework]TrainingConfig

trainer = [Framework]ModelTrainer()
config = [Framework]TrainingConfig(
    epochs=10,
    batch_size=8,
    learning_rate=0.001,
    verbose=1,
)
result = trainer.train(model, dataset, config)
```

### Evaluation

```python
from mlpotion.frameworks.[framework] import [Framework]ModelEvaluator

evaluator = [Framework]ModelEvaluator()
eval_result = evaluator.evaluate(model, dataset, config)
print(eval_result.metrics)
```

### Model Persistence

```python
from mlpotion.frameworks.[framework] import [Framework]ModelPersistence

persistence = [Framework]ModelPersistence()

# Save
persistence.save(model, "/path/to/model", save_format="...")

# Load
loaded_model = persistence.load("/path/to/model")
```

## Sample Data

The `examples/data/sample.csv` file contains synthetic regression data with:
- 10 features (feature_0 through feature_9)
- 1 target variable
- 50 samples

This dataset is used across all examples for consistency.

## ZenML Integration Benefits

Using ZenML pipelines provides:

1. **Reproducibility**: Track all pipeline runs with versioned artifacts
2. **Experiment Tracking**: Compare different configurations and results
3. **Collaboration**: Share pipelines with team members
4. **Scalability**: Run pipelines on different compute backends
5. **Artifact Caching**: Skip unchanged steps in subsequent runs

## Customization

Each example is designed to be easily customizable:

1. **Change model architecture**: Modify the model creation code
2. **Adjust hyperparameters**: Update the training configuration
3. **Use your own data**: Replace the file path with your CSV file
4. **Add validation split**: Set `validation_split` in training config
5. **Enable early stopping**: Configure callbacks in training config

## Troubleshooting

### Import Errors
Make sure MLPotion is installed:
```bash
pip install -e .
```

### ZenML Errors
If you encounter ZenML initialization errors:
```bash
export ZENML_RUN_SINGLE_STEPS_WITHOUT_STACK=true
```

### Data Loading Issues
Ensure the CSV file exists and has the correct format:
```bash
ls -la examples/data/sample.csv
head examples/data/sample.csv
```

## Next Steps

- Explore the [MLPotion documentation](../README.md)
- Check out the [test suite](../tests/) for more usage examples
- Read about [ZenML best practices](https://docs.zenml.io/)
- Customize the examples for your own use cases

## Contributing

Found an issue or want to add a new example? Please open an issue or PR!
