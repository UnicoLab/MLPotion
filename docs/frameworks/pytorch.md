# PyTorch Guide 🔥

Complete guide to using MLPotion with PyTorch - the researcher's favorite framework!

## Why PyTorch + MLPotion? 🤔

- **Research-Friendly**: Dynamic computation graphs, easy debugging
- **Pythonic**: Feels like native Python code
- **Flexible**: Full control over training loops
- **Ecosystem**: Huge community, extensive libraries
- **MLPotion Benefits**: Type-safe, modular components with consistent APIs

## Installation 📥

```bash
poetry add mlpotion -E pytorch
```

This installs:
- `torch>=2.0`
- `torchvision>=0.16`
- All PyTorch-specific MLPotion components

## Quick Example 🚀

```python
from mlpotion.frameworks.pytorch import (
    CSVDataset,
    CSVDataLoader,
    ModelTrainer,
    ModelTrainingConfig,
)
import torch.nn as nn

# Load data
dataset = CSVDataset("data.csv", label_name="target")
factory = CSVDataLoader(batch_size=32, shuffle=True)
dataloader = factory.load(dataset)

# Create model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

model = SimpleModel()

# Train
trainer = ModelTrainer()
config = ModelTrainingConfig(
    epochs=10,
    learning_rate=0.001,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
result = trainer.train(model, dataloader, config)

print(f"Final loss: {result.metrics['loss']:.4f}")
```

## Data Loading 📊

### CSV Dataset

```python
from mlpotion.frameworks.pytorch import CSVDataset

dataset = CSVDataset(
    file_pattern="data.csv",        # File path or pattern
    label_name="target",            # Label column name
    column_names=None,              # Auto-detect or specify
    dtype=torch.float32,            # Data type
)

# Use like any PyTorch dataset
print(f"Dataset size: {len(dataset)}")
features, label = dataset[0]
```

### DataLoader Factory

```python
from mlpotion.frameworks.pytorch import CSVDataLoader

factory = CSVDataLoader(
    batch_size=32,
    shuffle=True,
    num_workers=4,                  # Parallel data loading
    pin_memory=True,                # Faster GPU transfer
    drop_last=False,
    persistent_workers=True,        # Keep workers alive
)

# Create dataloaders
train_loader = factory.load(train_dataset)
val_loader = factory.load(val_dataset)
test_loader = factory.load(test_dataset)
```

## Model Training 🎓

### Basic Training

```python
from mlpotion.frameworks.pytorch import ModelTrainer, ModelTrainingConfig

config = ModelTrainingConfig(
    epochs=10,
    learning_rate=0.001,
    device="cuda",
    optimizer="adam",
    loss_fn="mse",
    verbose=True,
)

trainer = ModelTrainer()
result = trainer.train(model, train_loader, config)

print(f"Training time: {result.training_time:.2f}s")
print(f"Final loss: {result.metrics['loss']:.4f}")
```

### Advanced Training

```python
import torch.nn as nn
import torch.optim as optim

config = ModelTrainingConfig(
    # Training parameters
    epochs=100,
    learning_rate=0.001,
    device="cuda" if torch.cuda.is_available() else "cpu",

    # Optimizer
    optimizer="adamw",  # or "adam", "sgd", "rmsprop"
    optimizer_kwargs={
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
    },
)

result = trainer.train(model, train_loader, config, val_loader=val_loader)
```

### Custom Training Loop

```python
from mlpotion.frameworks.pytorch import ModelTrainer
import torch

class CustomTrainer(ModelTrainer):
    def training_step(self, model, batch, device):
        """Custom training step logic."""
        features, labels = batch
        features, labels = features.to(device), labels.to(device)

        # Forward pass
        predictions = model(features)
        loss = self.criterion(predictions, labels)

        # Add custom regularization
        l2_reg = sum(p.pow(2).sum() for p in model.parameters())
        loss = loss + 0.001 * l2_reg

        return loss

    def validation_step(self, model, batch, device):
        """Custom validation step logic."""
        features, labels = batch
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            predictions = model(features)
            loss = self.criterion(predictions, labels)

        return loss

# Use custom trainer
custom_trainer = CustomTrainer()
result = custom_trainer.train(model, train_loader, config)
```

## Model Evaluation 📊

```python
from mlpotion.frameworks.pytorch import ModelEvaluator, ModelEvaluationConfig

config = ModelEvaluationConfig(
    device="cuda",
    batch_size=32,
    metrics=["mse", "mae"],
    verbose=True,
)

evaluator = ModelEvaluator()
result = evaluator.evaluate(model, test_loader, config)

print(f"Test loss: {result.metrics['loss']:.4f}")
print(f"Test MAE: {result.metrics['mae']:.4f}")
```

## Model Persistence 💾

```python
from mlpotion.frameworks.pytorch import ModelPersistence

persistence = ModelPersistence(path="models/my_model.pth", model=model)

# Save model (state_dict - recommended)
persistence.save()

# Save full model
persistence.save(save_full_model=True)

# Load model
loader = ModelPersistence(path="models/my_model.pth")
loaded_model, metadata = loader.load(
    model_class=SimpleModel,  # Need model class for state_dict
)

# Load full model (auto-detected)
loader_full = ModelPersistence(path="models/my_model_full.pth")
loaded_model_full, _ = loader_full.load()
```

## Model Export 📤

### TorchScript

```python
from mlpotion.frameworks.pytorch import ModelExporter, ModelExportConfig

exporter = ModelExporter()

config = ModelExportConfig(
    format="torchscript",
    method="trace",  # or "script"
    example_inputs=torch.randn(1, 10),  # For tracing
)

result = exporter.export(model, "exports/model.pt", config)
print(f"Exported to: {result.export_path}")
```

### ONNX

```python
config = ModelExportConfig(
    format="onnx",
    input_names=["features"],
    output_names=["predictions"],
    dynamic_axes={"features": {0: "batch_size"}},
    opset_version=14,
)

result = exporter.export(model, "exports/model.onnx", config)
```

## Common Patterns 🎯

### Pattern: Train-Val-Test Pipeline

```python
# Load data
train_dataset = CSVDataset("train.csv", label_name="target")
val_dataset = CSVDataset("val.csv", label_name="target")
test_dataset = CSVDataset("test.csv", label_name="target")

# Create dataloaders
factory = CSVDataLoader(batch_size=32, shuffle=True, num_workers=4)
train_loader = factory.load(train_dataset)
val_loader = factory.load(val_dataset)
test_loader = factory.load(test_dataset)

# Train with validation
trainer = ModelTrainer()
config = ModelTrainingConfig(
    epochs=50,
    learning_rate=0.001,
    early_stopping=True,
    early_stopping_patience=10,
)

result = trainer.train(model, train_loader, config, val_loader=val_loader)

# Evaluate on test set
evaluator = ModelEvaluator()
test_metrics = evaluator.evaluate(result.model, test_loader, config)

print(f"Best epoch: {result.best_epoch}")
print(f"Test loss: {test_metrics.metrics['loss']:.4f}")
```

### Pattern: Multi-GPU Training

```python
import torch
import torch.nn as nn

# Wrap model for multi-GPU
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    device = "cuda"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)

# Train as usual
config = ModelTrainingConfig(
    epochs=50,
    learning_rate=0.001,
    device=device,
)

result = trainer.train(model, train_loader, config)
```

### Pattern: Mixed Precision Training

```python
config = ModelTrainingConfig(
    epochs=50,
    learning_rate=0.001,
    use_amp=True,  # Enable automatic mixed precision
    device="cuda",
)

result = trainer.train(model, train_loader, config)
```

## Best Practices 💡

1. **Use DataLoaders**: Always use `CSVDataLoader` for efficient loading
2. **Enable num_workers**: Set `num_workers>0` for parallel data loading
3. **Pin Memory**: Use `pin_memory=True` for faster GPU transfer
4. **AMP Training**: Enable mixed precision for faster training
5. **Gradient Clipping**: Prevent exploding gradients with `clip_grad_norm`
6. **State Dict**: Save models as state_dict for better compatibility

## Next Steps 🚀

- [TensorFlow Guide →](tensorflow.md) - Compare with TensorFlow
- [ZenML Integration →](../integrations/zenml.md) - Add MLOps
- [API Reference →](../api/frameworks/pytorch.md) - Detailed API docs

---

<p align="center">
  <strong>PyTorch + MLPotion = Research + Production!</strong> 🔥
</p>
