"""PyTorch training pipeline WITH ZenML orchestration.

This example demonstrates how to use MLPotion's PyTorch components
within a ZenML pipeline for reproducible and tracked ML workflows.

Requirements:
    pip install zenml

Setup:
    zenml init  # Initialize ZenML repository
    export ZENML_RUN_SINGLE_STEPS_WITHOUT_STACK=true  # For testing without full stack
"""

import torch
import torch.nn as nn
from zenml import pipeline, step

from mlpotion.frameworks.pytorch import ModelTrainingConfig
from mlpotion.integrations.zenml.pytorch.steps import (
    evaluate_model,
    export_model,
    load_csv_data,
    save_model,
    train_model,
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


@step
def create_model() -> nn.Module:
    """Create a PyTorch model.

    Returns:
        PyTorch model ready for training.
    """
    model = SimpleModel(input_dim=10, hidden_dim=64)
    return model


@pipeline(enable_cache=False)
def pytorch_training_pipeline(
    file_path: str = "examples/data/sample.csv",
    label_name: str = "target",
    model_save_path: str = "/tmp/pytorch_model.pth",
    export_path: str = "/tmp/pytorch_model_export.pt",
):
    """Complete PyTorch training pipeline with ZenML.

    This pipeline orchestrates the entire ML workflow:
    1. Load data from CSV
    2. Create and configure model
    3. Train model
    4. Evaluate model
    5. Save model
    6. Export model for deployment

    Args:
        file_path: Path to CSV data file.
        label_name: Name of the target column.
        model_save_path: Path to save the trained model.
        export_path: Path to export the model for serving.
    """
    # Step 1: Load data
    dataloader = load_csv_data(
        file_path=file_path,
        label_name=label_name,
        batch_size=8,
        shuffle=True,
    )

    # Step 2: Create model and config
    model = create_model()

    # Step 3: Train model
    _config_train = {
        "epochs": 10,
        "learning_rate": 0.001,
        "verbose": 1,
    }
    model, metrics = train_model(
        model=model,
        dataloader=dataloader,
        **_config_train,
    )

    # Step 4: Evaluate model
    evaluation_metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
    )

    # Step 5: Save model
    save_model(
        model=model,
        save_path=model_save_path,
    )

    # # Step 6: Export model for serving (TorchScript)
    export_model(
        model=model,
        export_path=export_path,
        export_format="torchscript",
    )
    return None
    # return trained_model, training_metrics, evaluation_metrics


if __name__ == "__main__":
    """Run the PyTorch ZenML pipeline."""
    print("=" * 60)
    print("MLPotion - PyTorch ZenML Pipeline")
    print("=" * 60)

    # Initialize ZenML (if not already initialized)
    try:
        from zenml.client import Client
        client = Client()
        print(f"✅ ZenML initialized. Active stack: {client.active_stack_model.name}")
    except Exception as e:
        print(f"⚠️  ZenML client error: {e}")
        print("Run 'zenml init' if you haven't already")

    # Run the pipeline
    print("\nRunning ZenML pipeline...")
    result = pytorch_training_pipeline()

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")

