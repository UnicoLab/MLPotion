# ZenML Integration API Reference 📖

Complete API reference for MLPotion's ZenML integration.

!!! info "Auto-Generated Documentation"
    This page will be automatically populated with API documentation from the source code once the package is fully documented. For now, refer to the [ZenML Integration Guide](../../integrations/zenml.md) and inline docstrings.

## TensorFlow Steps

### tf_data_loader_step

**Module**: `mlpotion.integrations.zenml.tensorflow.steps`

ZenML step for loading TensorFlow data.

**Parameters**:
- `file_pattern: str` - File path or pattern
- `label_name: str` - Label column name
- `batch_size: int` - Batch size
- And more...

**Returns**: `tf.data.Dataset`

### tf_train_step

**Module**: `mlpotion.integrations.zenml.tensorflow.steps`

ZenML step for training TensorFlow models.

**Parameters**:
- `dataset: tf.data.Dataset` - Training dataset
- `model: tf.keras.Model | None` - Model to train
- `epochs: int` - Number of epochs
- `learning_rate: float` - Learning rate
- And more...

**Returns**: `TrainingResult`

### tf_evaluate_step

**Module**: `mlpotion.integrations.zenml.tensorflow.steps`

ZenML step for evaluating TensorFlow models.

**Parameters**:
- `model: tf.keras.Model` - Model to evaluate
- `dataset: tf.data.Dataset` - Evaluation dataset
- And more...

**Returns**: `EvaluationResult`

### tf_export_step

**Module**: `mlpotion.integrations.zenml.tensorflow.steps`

ZenML step for exporting TensorFlow models.

**Parameters**:
- `model: tf.keras.Model` - Model to export
- `export_path: str` - Export location
- `format: str` - Export format
- And more...

**Returns**: `ExportResult`

## PyTorch Steps

### pytorch_data_loader_step

**Module**: `mlpotion.integrations.zenml.pytorch.steps`

ZenML step for loading PyTorch data.

### pytorch_train_step

**Module**: `mlpotion.integrations.zenml.pytorch.steps`

ZenML step for training PyTorch models.

### pytorch_evaluate_step

**Module**: `mlpotion.integrations.zenml.pytorch.steps`

ZenML step for evaluating PyTorch models.

## Keras Steps

### keras_data_loader_step

**Module**: `mlpotion.integrations.zenml.keras.steps`

ZenML step for loading Keras data.

### keras_train_step

**Module**: `mlpotion.integrations.zenml.keras.steps`

ZenML step for training Keras models.

### keras_evaluate_step

**Module**: `mlpotion.integrations.zenml.keras.steps`

ZenML step for evaluating Keras models.

## Materializers

### TensorFlowModelMaterializer

**Module**: `mlpotion.integrations.zenml.tensorflow.materializers`

ZenML materializer for TensorFlow/Keras models. Handles artifact serialization and versioning.

### PyTorchModelMaterializer

**Module**: `mlpotion.integrations.zenml.pytorch.materializers`

ZenML materializer for PyTorch models. Handles artifact serialization and versioning.

---

<p align="center">
  <em>See the <a href="../../integrations/zenml/">ZenML Integration Guide</a> for usage examples</em>
</p>
