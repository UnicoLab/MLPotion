import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mlpotion.frameworks.pytorch.data.transformers import (
    PyTorchDataToCSVTransformer,
)


class _SimpleTensorDataset(Dataset):
    """Small supervised dataset returning (features, labels) as tensors.

    Features: [[0], [1], ..., [n-1]]
    Labels:   [[0], [2], ..., [2*(n-1)]]
    """

    def __init__(self, n: int = 10) -> None:
        super().__init__()
        self.features = torch.arange(n, dtype=torch.float32).unsqueeze(1)
        self.labels = 2 * self.features

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class _NamedFeatureDataset(Dataset):
    """Dataset exposing `_feature_cols` for name inference.

    Features: 2D tensor with two columns.
    No labels; transformer sees batches as pure feature tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self._feature_cols = ["a", "b"]
        self._data = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._data[idx]


class _DoubleLinearModel(nn.Module):
    """Simple 1D → 1D linear model that approximates y = 2x."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)
        with torch.no_grad():
            self.linear.weight.fill_(2.0)
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _IdentityModel(nn.Module):
    """Model that returns inputs as outputs (for shape / naming tests)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TestPyTorchDataToCSVTransformer(unittest.TestCase):
    # ------------------------------------------------------------------ #
    # End-to-end: single CSV output (data_output_per_batch=False)
    # ------------------------------------------------------------------ #
    def test_transform_full_csv_from_explicit_dataloader_and_model(self) -> None:
        """transform() should write a single CSV with features + pred column."""
        dataset = _SimpleTensorDataset(n=10)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        model = _DoubleLinearModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "preds.csv")

            transformer = PyTorchDataToCSVTransformer(
                dataloader=dataloader,
                model=model,
                data_output_path=out_path,
                data_output_per_batch=False,
                device="cpu",
            )

            # Simple config-like object with data_output_path attribute
            config = SimpleNamespace(data_output_path=out_path)

            result = transformer.transform(
                dataloader=None,
                model=None,
                config=config,
            )

            self.assertEqual(result.data_output_path, out_path)
            self.assertTrue(Path(out_path).exists())

            df = pd.read_csv(out_path)
            # Expect 10 rows: one per sample
            self.assertEqual(len(df), 10)

            # Generic feature naming: feature_0 + pred
            self.assertIn("feature_0", df.columns)
            self.assertIn("pred", df.columns)

            # Check feature / prediction values
            np.testing.assert_allclose(
                df["feature_0"].to_numpy(),
                np.arange(10, dtype=np.float32),
            )
            np.testing.assert_allclose(
                df["pred"].to_numpy(),
                2.0 * np.arange(10, dtype=np.float32),
                rtol=1e-5,
            )

    # ------------------------------------------------------------------ #
    # Per-batch CSV output (data_output_per_batch=True)
    # ------------------------------------------------------------------ #
    def test_transform_per_batch_writes_multiple_csv_files(self) -> None:
        """transform() with data_output_per_batch=True should write batch_*.csv files."""
        dataset = _SimpleTensorDataset(n=5)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        model = _DoubleLinearModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Using a directory-like path (no suffix)
            out_dir = os.path.join(tmpdir, "batches")

            transformer = PyTorchDataToCSVTransformer(
                dataloader=dataloader,
                model=model,
                data_output_path=out_dir,
                data_output_per_batch=True,
                device="cpu",
            )

            config = SimpleNamespace(data_output_path=out_dir)

            _ = transformer.transform(
                dataloader=None,
                model=None,
                config=config,
            )

            out_dir_path = Path(out_dir)
            self.assertTrue(out_dir_path.exists())
            batch_files = sorted(out_dir_path.glob("batch_*.csv"))

            # 5 samples, batch_size=2 -> 3 batches: 2, 2, 1
            self.assertEqual(len(batch_files), 3)

            # Check that total rows across all batch CSVs equals dataset size
            total_rows = 0
            for f in batch_files:
                batch_df = pd.read_csv(f)
                total_rows += len(batch_df)
                # Expect at least feature_0 and pred columns
                self.assertIn("feature_0", batch_df.columns)
                self.assertIn("pred", batch_df.columns)

            self.assertEqual(total_rows, 5)

    # ------------------------------------------------------------------ #
    # Feature name inference from dataset._feature_cols
    # ------------------------------------------------------------------ #
    def test_feature_names_inferred_from_dataset_feature_cols(self) -> None:
        """If dataset exposes `_feature_cols`, transformer should use them as column names."""
        dataset = _NamedFeatureDataset()
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        model = _IdentityModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "named_features.csv")

            transformer = PyTorchDataToCSVTransformer(
                dataloader=dataloader,
                model=model,
                data_output_path=out_path,
                data_output_per_batch=False,
                device="cpu",
            )

            config = SimpleNamespace(data_output_path=out_path)

            _ = transformer.transform(
                dataloader=None,
                model=None,
                config=config,
            )

            df = pd.read_csv(out_path)

            # `dataset._feature_cols` = ["a", "b"] should be used as feature names
            self.assertIn("a", df.columns)
            self.assertIn("b", df.columns)

            # Identity model: preds have shape (N, 2) -> pred_0, pred_1
            self.assertIn("pred_0", df.columns)
            self.assertIn("pred_1", df.columns)

            # Check that features and pred_* align with underlying data
            expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            np.testing.assert_allclose(df[["a", "b"]].to_numpy(), expected)
            np.testing.assert_allclose(
                df[["pred_0", "pred_1"]].to_numpy(), expected, rtol=1e-5
            )


if __name__ == "__main__":
    unittest.main()
