# Keras API Reference 📖

Complete API reference for MLPotion's Keras components.

!!! info "Auto-Generated Documentation"
    This page will be automatically populated with API documentation from the source code once the package is fully documented. For now, refer to the [Keras Guide](../../frameworks/keras.md) and inline docstrings.

!!! note "Extensibility"
    These components are built using protocol-based design, making MLPotion easy to extend. Want to add new data sources, training methods, or integrations? See [Contributing Guide](../../contributing/overview.md).

## Data Loading

### CSVDataLoader

**Module**: `mlpotion.frameworks.keras.data.loaders`

**Implements**: `DataLoader[CSVSequence]` protocol

Load data from CSV files into Keras CSVSequence.

**Constructor Parameters**:
- `file_pattern: str` - File path or glob pattern
- `label_name: str | None` - Column name for labels
- `column_names: list[str] | None` - Column names (auto-detect if None)
- `batch_size: int` - Batch size
- `shuffle: bool` - Shuffle data
- `dtype: str` - Data type (default: "float32")
- And more...

**Methods**:
- `load() -> CSVSequence` - Load and return a Keras Sequence (no parameters)

**Example**:
```python
from mlpotion.frameworks.keras import CSVDataLoader

loader = CSVDataLoader(
    file_pattern="data.csv",
    label_name="target",
    batch_size=32,
    shuffle=True,
)
sequence = loader.load()  # Returns CSVSequence
```

### CSVSequence

**Module**: `mlpotion.frameworks.keras.data.loaders`

**Inherits**: `keras.utils.Sequence`

Keras Sequence for batched CSV data loading.

**Methods**:
- `__len__() -> int` - Return number of batches
- `__getitem__(idx: int) -> tuple[np.ndarray, np.ndarray]` - Get batch at index

## Training

### ModelTrainer

**Module**: `mlpotion.frameworks.keras.training.trainers`

**Implements**: `ModelTrainer[keras.Model, CSVSequence]` protocol (partially)

Train Keras models.

**Methods**:
- `train(model: keras.Model, data: CSVSequence, compile_params: dict, fit_params: dict) -> keras.callbacks.History`
  - `model`: Keras model to train
  - `data`: Training sequence
  - `compile_params`: Parameters for model.compile() (optimizer, loss, metrics)
  - `fit_params`: Parameters for model.fit() (epochs, validation_data, callbacks, etc.)
  - **Returns**: Keras History object with training history

**Example**:
```python
from mlpotion.frameworks.keras import ModelTrainer

trainer = ModelTrainer()
compile_params = {
    "optimizer": keras.optimizers.Adam(learning_rate=0.001),
    "loss": "mse",
    "metrics": ["mae"],
}
fit_params = {
    "epochs": 10,
    "verbose": 1,
    "validation_data": val_sequence,
}
history = trainer.train(model, train_sequence, compile_params, fit_params)
```

## Evaluation

### ModelEvaluator

**Module**: `mlpotion.frameworks.keras.evaluation.evaluators`

**Implements**: `ModelEvaluator[keras.Model, CSVSequence]` protocol (partially)

Evaluate Keras models.

**Methods**:
- `evaluate(model: keras.Model, data: CSVSequence, eval_params: dict) -> dict[str, float]`
  - `model`: Keras model to evaluate
  - `data`: Evaluation sequence
  - `eval_params`: Parameters for model.evaluate() (verbose, etc.)
  - **Returns**: Dictionary of metric names to values

**Example**:
```python
from mlpotion.frameworks.keras import ModelEvaluator

evaluator = ModelEvaluator()
eval_params = {"verbose": 1}
metrics = evaluator.evaluate(model, test_sequence, eval_params)
print(f"Loss: {metrics['loss']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
```

## Persistence

### ModelPersistence

**Module**: `mlpotion.frameworks.keras.deployment.persistence`

**Implements**: `ModelPersistence[keras.Model]` protocol

Save and load Keras models.

**Constructor Parameters**:
- `path: str` - Path to save/load model
- `model: keras.Model | None` - Model to save (required for save operations)

**Methods**:
- `save() -> None` - Save model to the configured path
- `load(inspect: bool = True) -> tuple[keras.Model, dict[str, Any]]` - Load model from disk
  - `inspect`: Whether to inspect the model on load
  - **Returns**: Tuple of (loaded model, inspection metadata dictionary)

**Example**:
```python
from mlpotion.frameworks.keras import ModelPersistence

# Save
persistence = ModelPersistence(path="models/my_model", model=model)
persistence.save()

# Load - returns tuple!
persistence = ModelPersistence(path="models/my_model")
loaded_model, metadata = persistence.load(inspect=True)
print(f"Model inputs: {metadata['inputs']}")
print(f"Model outputs: {metadata['outputs']}")
print(f"Total params: {metadata['total_params']}")
```

## Export

### ModelExporter

**Module**: `mlpotion.frameworks.keras.deployment.exporters`

**Implements**: `ModelExporter[keras.Model]` protocol (partially)

Export Keras models for deployment.

**Methods**:
- `export(model: keras.Model, path: str, export_format: str | None = None) -> None`
  - `model`: Keras model to export
  - `path`: Export path
  - `export_format`: Export format (optional, defaults to Keras format)
  - **Returns**: None

**Example**:
```python
from mlpotion.frameworks.keras import ModelExporter

exporter = ModelExporter()
exporter.export(
    model=model,
    path="exports/my_model",
    export_format="keras",
)
```

## Model Inspection

### ModelInspector

**Module**: `mlpotion.frameworks.keras.models.inspection`

**Implements**: `ModelInspector[keras.Model]` protocol

Inspect Keras model structure.

**Constructor Parameters**:
- `include_layers: bool` - Include layer information
- `include_signatures: bool` - Include input/output signatures

**Methods**:
- `inspect(model: keras.Model) -> dict[str, Any]` - Inspect model and return metadata
  - **Returns**: Dictionary with model information (inputs, outputs, layers, params, etc.)

**Example**:
```python
from mlpotion.frameworks.keras import ModelInspector

inspector = ModelInspector(
    include_layers=True,
    include_signatures=True,
)
info = inspector.inspect(model)
print(f"Input shape: {info['inputs']}")
print(f"Output shape: {info['outputs']}")
print(f"Total parameters: {info['total_params']}")
```

---

<p align="center">
  <em>See the <a href="../../frameworks/keras/">Keras Guide</a> for usage examples</em>
</p>
