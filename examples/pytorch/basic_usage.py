"""Basic PyTorch usage WITHOUT ZenML."""

import torch
import torch.nn as nn

from mlpotion.frameworks.pytorch import (
    PyTorchCSVDataset,
    PyTorchDataLoaderFactory,
    PyTorchModelEvaluator,
    PyTorchModelTrainer,
    PyTorchTrainingConfig,
)


class SimpleModel(nn.Module):
    """Simple feedforward neural network."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def main() -> None:
    """Run basic PyTorch training pipeline."""
    print("=" * 60)
    print("MLPotion - PyTorch Basic Usage")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading data...")
    dataset = PyTorchCSVDataset(
        file_pattern="examples/data/sample.csv",
        label_name="target",
    )
    print(f"Dataset size: {len(dataset)}")

    # 2. Create DataLoader
    print("\n2. Creating DataLoader...")
    factory = PyTorchDataLoaderFactory(batch_size=32, shuffle=True)
    dataloader = factory.create(dataset)

    # 3. Create model
    print("\n3. Creating model...")
    model = SimpleModel(input_dim=10, hidden_dim=64)
    print(model)

    # 4. Train model
    print("\n4. Training model...")
    trainer = PyTorchModelTrainer()
    config = PyTorchTrainingConfig(
        epochs=5,
        learning_rate=0.001,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=1,
    )
    result = trainer.train(model, dataloader, config)

    print(f"\nTraining completed!")
    print(f"Training time: {result.training_time:.2f}s")
    print(f"Final loss: {result.metrics['loss']:.4f}")

    # 5. Evaluate model
    print("\n5. Evaluating model...")
    evaluator = PyTorchModelEvaluator()
    eval_result = evaluator.evaluate(model, dataloader, config)

    print(f"Evaluation metrics: {eval_result.metrics}")

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()