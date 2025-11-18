# Quick Start ⚡

Welcome to the fastest way to get productive with MLPotion! In 5 minutes, you'll have your first ML pipeline running.

## Your First Pipeline in 5 Minutes ⏱️

Let's build a simple regression pipeline with TensorFlow. Same principles apply to PyTorch and Keras!

### Step 1: Install MLPotion (30 seconds)

```bash
pip install mlpotion[tensorflow]
```

### Step 2: Prepare Your Data (1 minute)

Create a simple CSV file or use your own:

```python
# create_data.py
import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'feature_1': np.random.randn(n_samples),
    'feature_2': np.random.randn(n_samples),
    'feature_3': np.random.randn(n_samples),
    'target': np.random.randn(n_samples)
})

data.to_csv('data.csv', index=False)
print("✅ Data created!")
```

Run it:

```bash
python create_data.py
```

### Step 3: Load and Explore Data (1 minute)

```python
from mlpotion.frameworks.tensorflow import CSVDataLoader

# Create a data loader
loader = CSVDataLoader(
    file_pattern="data.csv",
    label_name="target",
    batch_size=32,
    shuffle=True,
)

# Load the dataset
dataset = loader.load()
print("✅ Data loaded!")

# Peek at the data
for batch in dataset.take(1):
    features, labels = batch
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
```

Output:

```
✅ Data loaded!
Features shape: (32, 3)
Labels shape: (32,)
```

### Step 4: Create and Train a Model (2 minutes)

```python
import tensorflow as tf
from mlpotion.frameworks.tensorflow import (
    ModelTrainer,
    ModelTrainingConfig,
)

# Create a simple model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

model = create_model()

# Configure training
config = ModelTrainingConfig(
    epochs=10,
    learning_rate=0.001,
    optimizer_type="adam",
    loss="mse",
    metrics=["mae"],
    verbose=1,
)

# Train the model
trainer = ModelTrainer()
result = trainer.train(model, dataset, config)

print(f"✅ Training complete!")
print(f"Final loss: {result.metrics['loss']:.4f}")
print(f"Final MAE: {result.metrics['mae']:.4f}")
```

### Step 5: Evaluate and Save (1 minute)

```python
from mlpotion.frameworks.tensorflow import (
    ModelEvaluator,
    ModelPersistence,
)

# Evaluate the model
evaluator = ModelEvaluator()
eval_result = evaluator.evaluate(result.model, dataset, config)

print(f"✅ Evaluation complete!")
print(f"Test MAE: {eval_result.metrics['mae']:.4f}")

# Save the model
persistence = ModelPersistence(
    path="my_model",
    model=result.model,
)
persistence.save()

print("✅ Model saved to 'my_model' directory!")
```

### Complete Script

Here's everything together:

```python
"""quickstart_example.py - Your first MLPotion pipeline!"""
import numpy as np
import pandas as pd
import tensorflow as tf
from mlpotion.frameworks.tensorflow import (
    CSVDataLoader,
    ModelTrainer,
    ModelEvaluator,
    ModelPersistence,
    ModelTrainingConfig,
)

# 1. Create sample data
print("📊 Creating sample data...")
np.random.seed(42)
data = pd.DataFrame({
    'feature_1': np.random.randn(1000),
    'feature_2': np.random.randn(1000),
    'feature_3': np.random.randn(1000),
    'target': np.random.randn(1000)
})
data.to_csv('data.csv', index=False)

# 2. Load data
print("📥 Loading data...")
loader = CSVDataLoader(
    file_pattern="data.csv",
    label_name="target",
    batch_size=32,
    shuffle=True,
)
dataset = loader.load()

# 3. Create model
print("🏗️ Creating model...")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# 4. Train model
print("🎓 Training model...")
config = ModelTrainingConfig(
    epochs=10,
    learning_rate=0.001,
    optimizer_type="adam",
    loss="mse",
    metrics=["mae"],
)

trainer = ModelTrainer()
result = trainer.train(model, dataset, config)

print(f"\n✅ Training complete!")
print(f"   Final loss: {result.metrics['loss']:.4f}")
print(f"   Final MAE: {result.metrics['mae']:.4f}")

# 5. Evaluate
print("\n📊 Evaluating model...")
evaluator = ModelEvaluator()
eval_result = evaluator.evaluate(result.model, dataset, config)

print(f"✅ Evaluation complete!")
print(f"   Test MAE: {eval_result.metrics['mae']:.4f}")

# 6. Save model
print("\n💾 Saving model...")
persistence = ModelPersistence(
    path="my_model",
    model=result.model,
)
persistence.save()

print("\n🎉 All done! Your first MLPotion pipeline is complete!")
```

Run it:

```bash
python quickstart_example.py
```

## Quick Start with PyTorch 🔥

Prefer PyTorch? Here's the same pipeline:

```python
"""quickstart_pytorch.py - PyTorch version"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mlpotion.frameworks.pytorch import (
    CSVDataset,
    CSVDataLoader,
    ModelTrainer,
    ModelEvaluator,
    ModelTrainingConfig,
)

# 1. Create sample data
print("📊 Creating sample data...")
np.random.seed(42)
data = pd.DataFrame({
    'feature_1': np.random.randn(1000),
    'feature_2': np.random.randn(1000),
    'feature_3': np.random.randn(1000),
    'target': np.random.randn(1000)
})
data.to_csv('data.csv', index=False)

# 2. Load data
print("📥 Loading data...")
dataset = CSVDataset(
    file_pattern="data.csv",
    label_name="target",
)

factory = CSVDataLoader(batch_size=32, shuffle=True)
dataloader = factory.load(dataset)

# 3. Create model
print("🏗️ Creating model...")
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

model = SimpleModel()

# 4. Train model
print("🎓 Training model...")
config = ModelTrainingConfig(
    epochs=10,
    learning_rate=0.001,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

trainer = ModelTrainer()
result = trainer.train(model, dataloader, config)

print(f"\n✅ Training complete!")
print(f"   Final loss: {result.metrics['loss']:.4f}")

# 5. Evaluate
print("\n📊 Evaluating model...")
evaluator = ModelEvaluator()
eval_result = evaluator.evaluate(result.model, dataloader, config)

print(f"✅ Evaluation complete!")
print(f"   Test loss: {eval_result.metrics['loss']:.4f}")

print("\n🎉 PyTorch pipeline complete!")
```

## Quick Start with Keras 🎨

And the Keras version:

```python
"""quickstart_keras.py - Keras version"""
import numpy as np
import pandas as pd
import keras
from mlpotion.frameworks.keras import (
    CSVDataLoader,
    ModelTrainer,
    ModelEvaluator,
    ModelTrainingConfig,
)

# 1. Create sample data
print("📊 Creating sample data...")
np.random.seed(42)
data = pd.DataFrame({
    'feature_1': np.random.randn(1000),
    'feature_2': np.random.randn(1000),
    'feature_3': np.random.randn(1000),
    'target': np.random.randn(1000)
})
data.to_csv('data.csv', index=False)

# 2. Load data
print("📥 Loading data...")
loader = CSVDataLoader(
    file_pattern="data.csv",
    label_name="target",
    batch_size=32,
    shuffle=True,
)
dataset = loader.load()

# 3. Create model
print("🏗️ Creating model...")
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(3,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

# 4. Train model
print("🎓 Training model...")
config = ModelTrainingConfig(
    epochs=10,
    learning_rate=0.001,
    optimizer_type="adam",
    loss="mse",
    metrics=["mae"],
)

trainer = ModelTrainer()
result = trainer.train(model, dataset, config)

print(f"\n✅ Training complete!")
print(f"   Final loss: {result.metrics['loss']:.4f}")
print(f"   Final MAE: {result.metrics['mae']:.4f}")

print("\n🎉 Keras pipeline complete!")
```

## Understanding What Just Happened 🤔

Let's break down what you just built:

### 1. Data Loading (The Easy Part)

```python
loader = CSVDataLoader(
    file_pattern="data.csv",
    label_name="target",
    batch_size=32,
)
dataset = loader.load()
```

**What it does:**

- Reads your CSV file
- Separates features from labels
- Batches data for training
- Handles shuffling and caching

**Why it's cool:**

- Same API across all frameworks
- Optimized for performance
- Handles edge cases automatically

### 2. Training Configuration (The Type-Safe Part)

```python
config = ModelTrainingConfig(
    epochs=10,
    learning_rate=0.001,
    optimizer_type="adam",
    loss="mse",
    metrics=["mae"],
)
```

**What it does:**

- Defines all training parameters
- Type-checked at runtime
- Validated before training

**Why it's cool:**

- Your IDE autocompletes everything
- Catch errors before running
- Consistent across frameworks

### 3. Training (The Simple Part)

```python
trainer = ModelTrainer()
result = trainer.train(model, dataset, config)
```

**What it does:**

- Compiles your model
- Runs training loop
- Tracks metrics and history
- Returns a rich result object

**Why it's cool:**

- One line to train
- Consistent interface
- Detailed results

## Common Patterns 🎯

### Pattern 1: Train-Val-Test Split

```python
from mlpotion.frameworks.tensorflow import CSVDataLoader

# Load different splits
train_loader = CSVDataLoader(file_pattern="train.csv", label_name="target")
val_loader = CSVDataLoader(file_pattern="val.csv", label_name="target")
test_loader = CSVDataLoader(file_pattern="test.csv", label_name="target")

train_data = train_loader.load()
val_data = val_loader.load()
test_data = test_loader.load()

# Train with validation
result = trainer.train(
    model,
    train_data,
    config,
    validation_dataset=val_data
)

# Evaluate on test set
test_metrics = evaluator.evaluate(result.model, test_data, config)
```

### Pattern 2: Early Stopping

```python
config = ModelTrainingConfig(
    epochs=100,  # Set high
    learning_rate=0.001,
    early_stopping=True,
    early_stopping_patience=10,
    early_stopping_monitor="val_loss",
)

result = trainer.train(model, train_data, config, validation_dataset=val_data)
print(f"Best epoch: {result.best_epoch}")
```

### Pattern 3: Model Checkpointing

```python
from mlpotion.frameworks.tensorflow import ModelPersistence

# During training, save checkpoints
for epoch in range(10):
    result = trainer.train(model, dataset, config)
    if epoch % 2 == 0:  # Save every 2 epochs
        persistence = ModelPersistence(
            path=f"checkpoint_epoch_{epoch}",
            model=result.model,
        )
        persistence.save()
```

### Pattern 4: Hyperparameter Tuning

```python
learning_rates = [0.0001, 0.001, 0.01]
best_mae = float('inf')
best_model = None

for lr in learning_rates:
    config = ModelTrainingConfig(
        epochs=10,
        learning_rate=lr,
        verbose=0,  # Quiet mode
    )

    result = trainer.train(model, dataset, config)

    if result.metrics['mae'] < best_mae:
        best_mae = result.metrics['mae']
        best_model = result.model
        print(f"✨ New best! LR={lr}, MAE={best_mae:.4f}")

persistence = ModelPersistence(path="best_model", model=best_model)
persistence.save()
```

## What's Next? 🗺️

You've built your first pipeline! Here are your next steps:

### Beginner Path 🌱

1. **[Core Concepts →](concepts.md)** - Understand the architecture
2. **[Framework Guide →](../frameworks/tensorflow.md)** - Deep dive into your framework
3. **[Basic Tutorial →](../tutorials/basic-pipeline.md)** - Build a complete project

### Intermediate Path 🌿

1. **[ZenML Integration →](../integrations/zenml.md)** - Add MLOps superpowers
2. **[Advanced Training →](../tutorials/advanced-training.md)** - Custom callbacks, mixed precision
3. **[Multi-Framework →](../tutorials/multi-framework.md)** - Switch between frameworks

### Advanced Path 🌳

1. **[Custom Components →](../tutorials/custom-components.md)** - Build your own atoms
2. **[Production Deployment →](../tutorials/deployment.md)** - Ship to production
3. **[API Reference →](../api/core.md)** - Deep technical docs

## Need Help? 🆘

- **Questions?** Check the [FAQ](../faq.md)
- **Issues?** [GitHub Issues](https://github.com/UnicoLab/MLPotion/issues)
- **Ideas?** [Discussions](https://github.com/UnicoLab/MLPotion/discussions)

## Quick Reference Card 📇

```python
# Import patterns (UNIFIED API - same names across all frameworks!)
from mlpotion.frameworks.tensorflow import (
    CSVDataLoader,           # Load CSV data
    DatasetOptimizer,        # Optimize datasets
    ModelTrainer,            # Train models
    ModelEvaluator,          # Evaluate models
    ModelPersistence,        # Save/load models
    ModelExporter,           # Export for serving
    ModelTrainingConfig,     # Training configuration
    ModelEvaluationConfig,   # Evaluation configuration
    ModelExportConfig,       # Export configuration
)

# Basic workflow
loader = CSVDataLoader(...)        # 1. Load data
dataset = loader.load()

trainer = ModelTrainer()           # 2. Train
result = trainer.train(model, dataset, config)

evaluator = ModelEvaluator()       # 3. Evaluate
metrics = evaluator.evaluate(result.model, dataset, config)

persistence = ModelPersistence(    # 4. Save
    path="path/",
    model=result.model,
)
persistence.save()
```

---

<p align="center">
  <strong>Congratulations! You're now brewing ML magic!</strong> 🧪✨<br>
  <em>Ready for more? Check out the tutorials!</em>
</p>
