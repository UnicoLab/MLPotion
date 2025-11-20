"""Basic PyTorch usage WITHOUT ZenML.

This example demonstrates the core MLPotion PyTorch workflow:
1. Load data from CSV
2. Create a PyTorch model
3. Train the model
4. Evaluate the model
5. Save and export the model
"""

import torch
import torch.nn as nn

from mlpotion.frameworks.pytorch import (
    CSVDataset,
    CSVDataLoader,
    ModelEvaluator,
    ModelPersistence,
    ModelTrainer,
    ModelTrainingConfig,
)


class SimpleModel(nn.Module):
    """Simple feedforward neural network.

    Args:
        input_dim: Number of input features.
        hidden_dim: Size of hidden layer.
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)


def main() -> None:
    """Run basic PyTorch training pipeline."""
    print("=" * 60)
    print("MLPotion - PyTorch Basic Usage")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading data...")
    dataset = CSVDataset(
        file_pattern="examples/data/sample.csv",
        label_name="target",
    )
    print(f"Dataset size: {len(dataset)}")

    # 2. Create DataLoader
    print("\n2. Creating DataLoader...")
    factory = CSVDataLoader(batch_size=8, shuffle=True)
    dataloader = factory.load(dataset)

    # 3. Create model
    print("\n3. Creating model...")
    model = SimpleModel(input_dim=10, hidden_dim=64)
    print(model)

    # 4. Train model
    print("\n4. Training model...")
    trainer = ModelTrainer()
    config = ModelTrainingConfig(
        epochs=10,
        learning_rate=0.001,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=1,
    )
    result = trainer.train(
        model=model,
        dataloader=dataloader,
        config=config,
    )

    print("\nTraining completed!")
    print(f"Training time: {result.training_time:.2f}s")
    print(f"Final loss: {result.metrics['loss']:.4f}")

    # 5. Evaluate model
    print("\n5. Evaluating model...")
    evaluator = ModelEvaluator()
    eval_result = evaluator.evaluate(model, dataloader, config)

    print(f"Evaluation completed in {eval_result.evaluation_time:.2f}s")
    print("Evaluation metrics:")
    for metric_name, metric_value in eval_result.metrics.items():
        print(f"  - {metric_name}: {metric_value:.4f}")

    # 6. Save model
    print("\n6. Saving model...")
    persistence = ModelPersistence(
        path="/tmp/pytorch_model.pth",
        model=model,
    )
    model_path = "/tmp/pytorch_model.pth"
    persistence.save()
    print(f"Model saved to: {model_path}")

    # 7. Load model
    print("\n7. Loading model...")
    loaded_model, metadata = persistence.load()
    print(f"Model loaded successfully: {type(loaded_model)}")

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
