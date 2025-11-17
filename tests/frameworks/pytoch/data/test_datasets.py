import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mlpotion.core.exceptions import DataLoadingError
from mlpotion.frameworks.pytorch.data.datasets import (
    PyTorchCSVDataset,
    StreamingPyTorchCSVDataset,
)
from tests.core import TestBase  # provides temp_dir, setUp/tearDown


class TestPyTorchCSVDataset(TestBase):
    def setUp(self) -> None:
        super().setUp()

        # Dedicated relative directory for these tests
        self.data_dir = Path(f"{self.temp_dir.name}_pytorch")
        self.data_dir.mkdir(exist_ok=True)

        # Synthetic CSV content: 2 files, 5 rows each
        self.n_files = 2
        self.rows_per_file = 5
        self.total_rows = self.n_files * self.rows_per_file

        rng = np.random.default_rng(42)
        self.df1 = pd.DataFrame(
            {
                "feature_0": rng.normal(size=self.rows_per_file),
                "feature_1": rng.normal(size=self.rows_per_file),
                "target": rng.integers(0, 2, size=self.rows_per_file),
            }
        )
        self.df2 = pd.DataFrame(
            {
                "feature_0": rng.normal(size=self.rows_per_file),
                "feature_1": rng.normal(size=self.rows_per_file),
                "target": rng.integers(0, 2, size=self.rows_per_file),
            }
        )

        self.file1 = self.data_dir / "part1.csv"
        self.file2 = self.data_dir / "part2.csv"
        self.df1.to_csv(self.file1, index=False)
        self.df2.to_csv(self.file2, index=False)

        # Glob pattern that matches only our test CSVs
        self.pattern = f"{self.data_dir.as_posix()}/*.csv"

    # ------------------------------------------------------------------ #
    # Construction & basic properties
    # ------------------------------------------------------------------ #
    def test_init_raises_if_no_files_found(self) -> None:
        """PyTorchCSVDataset should raise DataLoadingError for empty pattern."""
        with self.assertRaises(DataLoadingError):
            _ = PyTorchCSVDataset(file_pattern="no_such_dir_123/*.csv")

    def test_len_and_getitem_without_labels(self) -> None:
        """Dataset without labels returns only feature tensors."""
        dataset = PyTorchCSVDataset(
            file_pattern=self.pattern,
            column_names=["feature_0", "feature_1", "target"],
            label_name=None,
            dtype=torch.float32,
        )

        self.assertEqual(len(dataset), self.total_rows)

        x0 = dataset[0]
        self.assertIsInstance(x0, torch.Tensor)
        # 3 selected columns: feature_0, feature_1, target
        self.assertEqual(x0.shape[0], 3)
        self.assertEqual(x0.dtype, torch.float32)

    def test_len_and_getitem_with_labels(self) -> None:
        """Dataset with label_name returns (features, label) tuples."""
        dataset = PyTorchCSVDataset(
            file_pattern=self.pattern,
            column_names=["feature_0", "feature_1", "target"],
            label_name="target",
            dtype=torch.float32,
        )

        self.assertEqual(len(dataset), self.total_rows)

        x0, y0 = dataset[0]
        self.assertIsInstance(x0, torch.Tensor)
        self.assertIsInstance(y0, torch.Tensor)

        # feature_0, feature_1 → 2 features
        self.assertEqual(x0.shape[0], 2)
        # Don't over-specify label shape; just ensure it's numeric and non-empty
        self.assertEqual(y0.dtype, torch.float32)
        self.assertGreaterEqual(y0.numel(), 1)

    def test_column_selection_reduces_features(self) -> None:
        """column_names should control which features are exposed."""
        dataset = PyTorchCSVDataset(
            file_pattern=self.pattern,
            column_names=["feature_0"],  # single feature
            label_name=None,
        )

        x0 = dataset[0]
        self.assertEqual(x0.shape[0], 1)

    def test_column_selection_with_label_keeps_requested_features(self) -> None:
        """column_names + label_name should keep requested features and label."""
        dataset = PyTorchCSVDataset(
            file_pattern=self.pattern,
            column_names=["feature_0", "feature_1", "target"],
            label_name="target",
        )

        x0, y0 = dataset[0]
        # Two feature columns (label removed from features)
        self.assertEqual(x0.shape[0], 2)
        self.assertIsInstance(y0, torch.Tensor)

    def test_column_selection_missing_raises(self) -> None:
        """Missing requested columns should raise DataLoadingError."""
        with self.assertRaises(DataLoadingError):
            _ = PyTorchCSVDataset(
                file_pattern=self.pattern,
                column_names=["does_not_exist"],
                label_name=None,
            )

    def test_label_column_missing_raises(self) -> None:
        """Missing label column should raise DataLoadingError."""
        with self.assertRaises(DataLoadingError):
            _ = PyTorchCSVDataset(
                file_pattern=self.pattern,
                column_names=["feature_0", "feature_1"],
                label_name="missing_label",
            )

    def test_custom_dtype_is_respected(self) -> None:
        """dtype parameter should control tensor dtype."""
        dataset = PyTorchCSVDataset(
            file_pattern=self.pattern,
            column_names=["feature_0", "feature_1", "target"],
            label_name="target",
            dtype=torch.float64,
        )

        x0, y0 = dataset[0]
        self.assertEqual(x0.dtype, torch.float64)
        self.assertEqual(y0.dtype, torch.float64)


class TestStreamingPyTorchCSVDataset(TestBase):
    def setUp(self) -> None:
        super().setUp()

        # Dedicated relative directory for streaming tests
        base = Path(f"{self.temp_dir.name}_pytorch_stream")
        base.mkdir(exist_ok=True)
        self.stream_dir = base

        # Synthetic CSV content: 2 files, 4 rows each
        self.n_files = 2
        self.rows_per_file = 4
        self.total_rows = self.n_files * self.rows_per_file

        rng = np.random.default_rng(7)
        self.df1 = pd.DataFrame(
            {
                "feature_0": rng.normal(size=self.rows_per_file),
                "feature_1": rng.normal(size=self.rows_per_file),
                "target": rng.integers(0, 3, size=self.rows_per_file),
            }
        )
        self.df2 = pd.DataFrame(
            {
                "feature_0": rng.normal(size=self.rows_per_file),
                "feature_1": rng.normal(size=self.rows_per_file),
                "target": rng.integers(0, 3, size=self.rows_per_file),
            }
        )

        # Name them with a "good" prefix so we can pattern-match only these
        self.file1 = self.stream_dir / "stream_good1.csv"
        self.file2 = self.stream_dir / "stream_good2.csv"
        self.df1.to_csv(self.file1, index=False)
        self.df2.to_csv(self.file2, index=False)

        # Pattern matching only the good streaming CSVs
        self.stream_pattern = f"{self.stream_dir.as_posix()}/stream_good*.csv"

    # ------------------------------------------------------------------ #
    # Construction & validation
    # ------------------------------------------------------------------ #
    def test_init_raises_if_no_files_found(self) -> None:
        """StreamingPyTorchCSVDataset should raise if no files found."""
        with self.assertRaises(DataLoadingError):
            _ = StreamingPyTorchCSVDataset(file_pattern="no_stream_dir_123/*.csv")

    # ------------------------------------------------------------------ #
    # Iteration without labels
    # ------------------------------------------------------------------ #
    def test_streaming_without_labels_yields_features_only(self) -> None:
        """Streaming dataset without label_name should yield only feature tensors."""
        dataset = StreamingPyTorchCSVDataset(
            file_pattern=self.stream_pattern,
            column_names=["feature_0", "feature_1"],
            label_name=None,
            chunksize=2,
            dtype=torch.float32,
        )

        samples = list(dataset)
        # Only 2 good files: 2 * 4 = 8 rows
        self.assertEqual(len(samples), self.total_rows)

        x0 = samples[0]
        self.assertIsInstance(x0, torch.Tensor)
        self.assertEqual(x0.shape[0], 2)
        self.assertEqual(x0.dtype, torch.float32)

    # ------------------------------------------------------------------ #
    # Iteration with labels
    # ------------------------------------------------------------------ #
    def test_streaming_with_labels_yields_feature_label_tuples(self) -> None:
        """Streaming dataset with label_name yields (x, y) samples."""
        dataset = StreamingPyTorchCSVDataset(
            file_pattern=self.stream_pattern,
            column_names=["feature_0", "feature_1", "target"],
            label_name="target",
            chunksize=3,
            dtype=torch.float32,
        )

        samples = list(dataset)
        self.assertEqual(len(samples), self.total_rows)

        x0, y0 = samples[0]
        self.assertIsInstance(x0, torch.Tensor)
        self.assertIsInstance(y0, torch.Tensor)
        self.assertEqual(x0.shape[0], 2)  # 2 feature columns
        # label shape may be scalar or 1D; just check it has at least 1 element
        self.assertGreaterEqual(y0.numel(), 1)

    def test_streaming_raises_when_label_missing_in_file(self) -> None:
        """If label_name is requested but missing from a file's columns, raise."""
        # Create a file missing the "target" label column
        bad_df = pd.DataFrame(
            {
                "feature_0": np.arange(3, dtype=float),
                "feature_1": np.arange(3, dtype=float),
            }
        )
        bad_file = self.stream_dir / "stream_bad.csv"
        bad_df.to_csv(bad_file, index=False)

        # Pattern that matches only the bad file
        bad_pattern = bad_file.as_posix()

        dataset = StreamingPyTorchCSVDataset(
            file_pattern=bad_pattern,
            column_names=None,
            label_name="target",
            chunksize=2,
        )

        with self.assertRaises(DataLoadingError):
            _ = list(dataset)

    def test_streaming_respects_chunksize(self) -> None:
        """Streaming should yield all rows regardless of chunksize."""
        dataset_small_chunk = StreamingPyTorchCSVDataset(
            file_pattern=self.stream_pattern,
            column_names=["feature_0", "feature_1"],
            label_name=None,
            chunksize=1,
        )
        dataset_large_chunk = StreamingPyTorchCSVDataset(
            file_pattern=self.stream_pattern,
            column_names=["feature_0", "feature_1"],
            label_name=None,
            chunksize=10,
        )

        samples_small = list(dataset_small_chunk)
        samples_large = list(dataset_large_chunk)

        self.assertEqual(len(samples_small), self.total_rows)
        self.assertEqual(len(samples_large), self.total_rows)

    def test_streaming_custom_dtype_is_respected(self) -> None:
        """dtype parameter should control tensor dtype in streaming dataset."""
        dataset = StreamingPyTorchCSVDataset(
            file_pattern=self.stream_pattern,
            column_names=["feature_0", "feature_1", "target"],
            label_name="target",
            chunksize=2,
            dtype=torch.float64,
        )

        x0, y0 = next(iter(dataset))
        self.assertEqual(x0.dtype, torch.float64)
        self.assertEqual(y0.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
