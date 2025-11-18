# PyTorch API Reference 📖

Complete API reference for MLPotion's PyTorch components.

!!! info "Auto-Generated Documentation"
    This page will be automatically populated with API documentation from the source code once the package is fully documented. For now, refer to the [PyTorch Guide](../../frameworks/pytorch.md) and inline docstrings.

!!! note "Extensibility"
    These components are built using protocol-based design, making MLPotion easy to extend. Want to add new data sources, training methods, or integrations? See [Contributing Guide](../../contributing/overview.md).

## Data Loading

### CSVDataset

**Module**: `mlpotion.frameworks.pytorch.data.datasets`

**Inherits**: `torch.utils.data.Dataset`

PyTorch Dataset for loading CSV files.

**Constructor Parameters**:
- `file_pattern: str` - File path or glob pattern
- `column_names: list[str] | None` - Column names (auto-detect if None)
- `label_name: str | None` - Column name for labels
- `dtype: torch.dtype` - Data type (default: torch.float32)
- `normalize: bool` - Normalize features
- `transform: Callable | None` - Custom transform function

**Methods**:
- `__len__() -> int` - Return number of samples
- `__getitem__(idx: int) -> tuple[torch.Tensor, torch.Tensor]` - Get sample at index

**Example**:
```python
from mlpotion.frameworks.pytorch import CSVDataset
import torch

dataset = CSVDataset(
    file_pattern="data.csv",
    label_name="target",
    dtype=torch.float32,
)
print(f"Dataset size: {len(dataset)}")
features, label = dataset[0]
```

### CSVDataLoader

**Module**: `mlpotion.frameworks.pytorch.data.loaders`

**Implements**: `DataLoader[torch.utils.data.DataLoader]` protocol

Factory for creating PyTorch DataLoaders.

**Constructor Parameters**:
- `batch_size: int` - Batch size
- `shuffle: bool` - Shuffle data
- `num_workers: int` - Number of worker processes
- `pin_memory: bool` - Pin memory for GPU
- `drop_last: bool` - Drop incomplete batches
- And more...

**Methods**:
- `load(dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader` - Create DataLoader from dataset

**Example**:
```python
from mlpotion.frameworks.pytorch import CSVDataset, CSVDataLoader

dataset = CSVDataset(file_pattern="data.csv", label_name="target")
loader_factory = CSVDataLoader(
    batch_size=32,
    shuffle=True,
    num_workers=4,
)
dataloader = loader_factory.load(dataset)
```

## Training

### ModelTrainer

**Module**: `mlpotion.frameworks.pytorch.training.trainers`

**Implements**: `ModelTrainer[nn.Module, DataLoader]` protocol (partially)

Train PyTorch models.

**Methods**:
- `train(model: nn.Module, dataloader: DataLoader, config: ModelTrainingConfig, validation_dataloader: DataLoader | None = None) -> TrainingResult`
  - `model`: PyTorch model to train
  - `dataloader`: Training DataLoader
  - `config`: Training configuration
  - `validation_dataloader`: Optional validation DataLoader
  - **Returns**: TrainingResult with trained model, history, metrics, best_epoch

**Example**:
```python
from mlpotion.frameworks.pytorch import ModelTrainer, ModelTrainingConfig

trainer = ModelTrainer()
config = ModelTrainingConfig(
    epochs=10,
    learning_rate=0.001,
    optimizer="adam",
    loss_fn="mse",
    device="cuda",
)
result = trainer.train(model, train_loader, config, validation_dataloader=val_loader)
print(f"Final loss: {result.metrics['loss']:.4f}")
print(f"Best epoch: {result.best_epoch}")
```

## Evaluation

### ModelEvaluator

**Module**: `mlpotion.frameworks.pytorch.evaluation.evaluators`

**Implements**: `ModelEvaluator[nn.Module, DataLoader]` protocol (partially)

Evaluate PyTorch models.

**Methods**:
- `evaluate(model: nn.Module, dataloader: DataLoader, config: ModelEvaluationConfig) -> EvaluationResult`
  - `model`: PyTorch model to evaluate
  - `dataloader`: Evaluation DataLoader
  - `config`: Evaluation configuration
  - **Returns**: EvaluationResult with metrics and evaluation time

**Example**:
```python
from mlpotion.frameworks.pytorch import ModelEvaluator, ModelEvaluationConfig

evaluator = ModelEvaluator()
config = ModelEvaluationConfig(
    batch_size=32,
    device="cuda",
    verbose=1,
)
result = evaluator.evaluate(model, test_loader, config)
print(f"Loss: {result.metrics['loss']:.4f}")
```

## Persistence

### ModelPersistence

**Module**: `mlpotion.frameworks.pytorch.deployment.persistence`

**Implements**: `ModelPersistence[nn.Module]` protocol (partially)

Save and load PyTorch models.

**Constructor Parameters**:
- `path: str` - Path to save/load model
- `model: nn.Module | None` - Model to save (required for save operations)

**Methods**:
- `save(save_full_model: bool = False) -> None` - Save model state_dict or full model
  - `save_full_model`: If True, saves entire model; if False, saves state_dict only
- `load(model_class: type[nn.Module] | None = None, map_location: str = "cpu", strict: bool = True) -> nn.Module`
  - `model_class`: Model class for loading state_dict (required if state_dict was saved)
  - `map_location`: Device to load to
  - `strict`: Strict state_dict loading
  - **Returns**: Loaded PyTorch model (note: does not return tuple like TF/Keras)

**Example**:
```python
from mlpotion.frameworks.pytorch import ModelPersistence

# Save state_dict
persistence = ModelPersistence(path="models/my_model.pt", model=model)
persistence.save(save_full_model=False)

# Load state_dict (requires model class)
persistence = ModelPersistence(path="models/my_model.pt")
loaded_model = persistence.load(model_class=MyModel, map_location="cuda")

# Save full model
persistence = ModelPersistence(path="models/my_model_full.pt", model=model)
persistence.save(save_full_model=True)

# Load full model
persistence = ModelPersistence(path="models/my_model_full.pt")
loaded_model = persistence.load()
```

## Export

### ModelExporter

**Module**: `mlpotion.frameworks.pytorch.deployment.exporters`

**Implements**: `ModelExporter[nn.Module]` protocol (partially)

Export PyTorch models for deployment.

**Methods**:
- `export(model: nn.Module, config: ModelExportConfig) -> ExportResult`
  - `model`: PyTorch model to export
  - `config`: Export configuration with path, format, and options
  - **Returns**: ExportResult with export_path, format, and metadata

**Example**:
```python
from mlpotion.frameworks.pytorch import ModelExporter, ModelExportConfig
import torch

# Export as TorchScript
config = ModelExportConfig(
    export_path="exports/model.pt",
    format="torchscript",
    jit_mode="script",
)
exporter = ModelExporter()
result = exporter.export(model, config)

# Export as ONNX
config = ModelExportConfig(
    export_path="exports/model.onnx",
    format="onnx",
    example_input=torch.randn(1, 10),
    input_names=["input"],
    output_names=["output"],
)
result = exporter.export(model, config)
print(f"Exported to: {result.export_path}")
```

---

<p align="center">
  <em>See the <a href="../../frameworks/pytorch/">PyTorch Guide</a> for usage examples</em>
</p>
