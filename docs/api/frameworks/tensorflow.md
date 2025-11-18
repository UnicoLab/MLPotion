# TensorFlow API Reference 📖

Complete API reference for MLPotion's TensorFlow components.

!!! info "Auto-Generated Documentation"
    This page will be automatically populated with API documentation from the source code once the package is fully documented. For now, refer to the [TensorFlow Guide](../../frameworks/tensorflow.md) and inline docstrings.

!!! note "Extensibility"
    These components are built using protocol-based design, making MLPotion easy to extend. Want to add new data sources, training methods, or integrations? See [Contributing Guide](../../contributing/overview.md).

## Data Loading

### CSVDataLoader

**Module**: `mlpotion.frameworks.tensorflow.data.loaders`

**Implements**: `DataLoader[tf.data.Dataset]` protocol

Load data from CSV files into `tf.data.Dataset`.

**Constructor Parameters**:
- `file_pattern: str` - File path or glob pattern
- `label_name: str` - Column name for labels
- `column_names: list[str] | None` - Column names (auto-detect if None)
- `batch_size: int` - Batch size
- `shuffle: bool` - Whether to shuffle data
- `shuffle_buffer_size: int | None` - Shuffle buffer size
- `prefetch: bool` - Enable prefetching
- `cache: bool` - Cache dataset in memory
- And more...

**Methods**:
- `load() -> tf.data.Dataset` - Load and return a TensorFlow dataset (no parameters)

**Example**:
```python
from mlpotion.frameworks.tensorflow import CSVDataLoader

loader = CSVDataLoader(
    file_pattern="data.csv",
    label_name="target",
    batch_size=32,
)
dataset = loader.load()  # Returns tf.data.Dataset
```

### DatasetOptimizer

**Module**: `mlpotion.frameworks.tensorflow.data.optimizers`

**Implements**: `DatasetOptimizer[tf.data.Dataset]` protocol

Optimize `tf.data.Dataset` for performance.

**Constructor Parameters**:
- `batch_size: int | None` - Batch size
- `shuffle: bool` - Whether to shuffle
- `shuffle_buffer_size: int | None` - Shuffle buffer size
- `cache: bool` - Cache dataset in memory
- `prefetch: bool` - Enable prefetching
- And more...

**Methods**:
- `optimize(dataset: tf.data.Dataset) -> tf.data.Dataset` - Optimize the dataset

## Training

### ModelTrainer

**Module**: `mlpotion.frameworks.tensorflow.training.trainers`

**Implements**: `ModelTrainer[keras.Model, tf.data.Dataset]` protocol (partially - uses simpler signature)

Train TensorFlow/Keras models.

**Methods**:
- `train(model: keras.Model, data: tf.data.Dataset, compile_params: dict, fit_params: dict) -> keras.callbacks.History`
  - `model`: Keras model to train
  - `data`: Training dataset
  - `compile_params`: Parameters for model.compile() (optimizer, loss, metrics)
  - `fit_params`: Parameters for model.fit() (epochs, validation_data, callbacks, etc.)
  - **Returns**: Keras History object with training history

**Example**:
```python
from mlpotion.frameworks.tensorflow import ModelTrainer

trainer = ModelTrainer()
compile_params = {
    "optimizer": keras.optimizers.Adam(learning_rate=0.001),
    "loss": "mse",
    "metrics": ["mae"],
}
fit_params = {
    "epochs": 10,
    "verbose": 1,
    "validation_data": val_dataset,
}
history = trainer.train(model, dataset, compile_params, fit_params)
```

## Evaluation

### ModelEvaluator

**Module**: `mlpotion.frameworks.tensorflow.evaluation.evaluators`

**Implements**: `ModelEvaluator[keras.Model, tf.data.Dataset]` protocol (partially)

Evaluate TensorFlow/Keras models.

**Methods**:
- `evaluate(model: keras.Model, data: tf.data.Dataset, eval_params: dict) -> dict[str, float]`
  - `model`: Keras model to evaluate
  - `data`: Evaluation dataset
  - `eval_params`: Parameters for model.evaluate() (verbose, etc.)
  - **Returns**: Dictionary of metric names to values

**Example**:
```python
from mlpotion.frameworks.tensorflow import ModelEvaluator

evaluator = ModelEvaluator()
eval_params = {"verbose": 1}
metrics = evaluator.evaluate(model, test_dataset, eval_params)
print(f"Loss: {metrics['loss']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
```

## Persistence

### ModelPersistence

**Module**: `mlpotion.frameworks.tensorflow.deployment.persistence`

**Implements**: `ModelPersistence[keras.Model]` protocol

Save and load TensorFlow/Keras models.

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
from mlpotion.frameworks.tensorflow import ModelPersistence

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

**Module**: `mlpotion.frameworks.tensorflow.deployment.exporters`

**Implements**: `ModelExporter[keras.Model]` protocol (partially)

Export TensorFlow models for serving.

**Methods**:
- `export(model: keras.Model, path: str, export_format: str = "keras") -> None`
  - `model`: Keras model to export
  - `path`: Export path
  - `export_format`: Export format ("keras", "tf", "h5")
  - **Returns**: None

**Example**:
```python
from mlpotion.frameworks.tensorflow import ModelExporter

exporter = ModelExporter()
exporter.export(
    model=model,
    path="exports/my_model",
    export_format="keras",
)
```

## Model Inspection

### ModelInspector

**Module**: `mlpotion.frameworks.tensorflow.models.inspection`

**Implements**: `ModelInspector[keras.Model]` protocol

Inspect TensorFlow/Keras model structure.

**Constructor Parameters**:
- `include_layers: bool` - Include layer information
- `include_signatures: bool` - Include input/output signatures

**Methods**:
- `inspect(model: keras.Model) -> dict[str, Any]` - Inspect model and return metadata
  - **Returns**: Dictionary with model information (inputs, outputs, layers, params, etc.)

**Example**:
```python
from mlpotion.frameworks.tensorflow import ModelInspector

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
  <em>See the <a href="../../frameworks/tensorflow/">TensorFlow Guide</a> for usage examples</em>
</p>
