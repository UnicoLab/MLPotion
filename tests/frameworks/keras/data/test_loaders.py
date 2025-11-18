import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from loguru import logger

from mlpotion.core.exceptions import DataLoadingError
from mlpotion.frameworks.keras.data.loaders import CSVSequence, CSVDataLoader
from tests.core import TestBase  # provides temp_dir, setUp/tearDown


class TestCSVSequence(TestBase):
    def setUp(self) -> None:
        super().setUp()
        logger.info(f"Setting up CSVSequence tests in {self.temp_dir}")

        self.n_samples = 10
        self.n_features = 3
        self.batch_size = 4

        rng = np.random.default_rng(42)
        self.features = rng.normal(
            size=(self.n_samples, self.n_features)
        ).astype("float32")
        self.labels = rng.integers(0, 2, size=(self.n_samples,)).astype("float32")

    def test_len_and_getitem_with_labels(self) -> None:
        """CSVSequence with labels should batch correctly and return (x, y)."""
        logger.info("Testing CSVSequence __len__ and __getitem__ with labels")

        seq = CSVSequence(
            features=self.features,
            labels=self.labels,
            batch_size=self.batch_size,
            shuffle=False,
        )

        expected_len = int(np.ceil(self.n_samples / self.batch_size))
        self.assertEqual(len(seq), expected_len)

        total_seen = 0
        for idx in range(len(seq)):
            batch_x, batch_y = seq[idx]
            logger.info(f"Batch {idx}: x.shape={batch_x.shape}, y.shape={batch_y.shape}")

            self.assertIsInstance(batch_x, np.ndarray)
            self.assertIsInstance(batch_y, np.ndarray)
            self.assertEqual(batch_x.shape[1], self.n_features)
            self.assertEqual(batch_x.shape[0], batch_y.shape[0])

            total_seen += batch_x.shape[0]

        self.assertEqual(total_seen, self.n_samples)

    def test_getitem_without_labels_returns_only_features(self) -> None:
        """CSVSequence without labels should return only x."""
        logger.info("Testing CSVSequence __getitem__ without labels")

        seq = CSVSequence(
            features=self.features,
            labels=None,
            batch_size=self.batch_size,
            shuffle=False,
        )

        batch_x = seq[0]
        self.assertIsInstance(batch_x, np.ndarray)
        self.assertEqual(batch_x.shape[1], self.n_features)

    def test_invalid_features_dimension_raises(self) -> None:
        """Features must be 2D; otherwise CSVSequence raises ValueError."""
        logger.info("Testing CSVSequence with invalid features dimension")

        bad_features = np.zeros((5,), dtype="float32")  # 1D

        with self.assertRaises(ValueError):
            CSVSequence(
                features=bad_features,
                labels=None,
            )

    def test_labels_length_mismatch_raises(self) -> None:
        """Features and labels length mismatch should raise ValueError."""
        logger.info("Testing CSVSequence with length-mismatched labels")

        bad_labels = np.zeros((self.n_samples + 1,), dtype="float32")

        with self.assertRaises(ValueError):
            CSVSequence(
                features=self.features,
                labels=bad_labels,
            )

    def test_on_epoch_end_shuffles_indices_when_enabled(self) -> None:
        """on_epoch_end should call np.random.shuffle when shuffle=True."""
        logger.info("Testing CSVSequence.on_epoch_end with shuffle=True")

        seq = CSVSequence(
            features=self.features,
            labels=self.labels,
            batch_size=self.batch_size,
            shuffle=True,
        )

        with patch("numpy.random.shuffle") as mock_shuffle:
            seq.on_epoch_end()
            mock_shuffle.assert_called_once()

    def test_on_epoch_end_does_not_shuffle_when_disabled(self) -> None:
        """on_epoch_end should not call np.random.shuffle when shuffle=False."""
        logger.info("Testing CSVSequence.on_epoch_end with shuffle=False")

        seq = CSVSequence(
            features=self.features,
            labels=self.labels,
            batch_size=self.batch_size,
            shuffle=False,
        )

        with patch("numpy.random.shuffle") as mock_shuffle:
            seq.on_epoch_end()
            mock_shuffle.assert_not_called()


class TestCSVDataLoader(TestBase):
    def setUp(self) -> None:
        super().setUp()
        logger.info(f"Setting up CSVDataLoader tests in {self.temp_dir}")

        # IMPORTANT: TestBase likely chdirs into temp_dir already.
        # We will write files *inside* that directory, and use a RELATIVE pattern.
        self.n_per_file = 5
        self.n_features = 2

        data1 = pd.DataFrame(
            {
                "feature_0": np.arange(self.n_per_file, dtype="float32"),
                "feature_1": np.arange(self.n_per_file, dtype="float32") + 100,
                "target": np.arange(self.n_per_file, dtype="float32") % 2,
            }
        )
        data2 = pd.DataFrame(
            {
                "feature_0": np.arange(self.n_per_file, dtype="float32") + 10,
                "feature_1": np.arange(self.n_per_file, dtype="float32") + 200,
                "target": (np.arange(self.n_per_file, dtype="float32") + 1) % 2,
            }
        )

        # Write files into the temp directory
        self.file1 = self.temp_dir / "data_1.csv"
        self.file2 = self.temp_dir / "data_2.csv"

        data1.to_csv(self.file1, index=False)
        data2.to_csv(self.file2, index=False)

        # Use absolute pattern pointing to temp directory
        self.file_pattern = str(self.temp_dir / "data_*.csv")

    # ------------------------------------------------------------------ #
    # Basic loading with labels
    # ------------------------------------------------------------------ #
    def test_load_basic_with_labels(self) -> None:
        """CSVDataLoader.load should return a CSVSequence with correct size."""
        logger.info("Testing CSVDataLoader.load with labels")

        loader = CSVDataLoader(
            file_pattern=self.file_pattern,
            label_name="target",
            batch_size=4,
            shuffle=False,
            dtype="float32",
        )

        seq = loader.load()
        self.assertIsInstance(seq, CSVSequence)

        total_rows = self.n_per_file * 2
        expected_len = int(np.ceil(total_rows / 4))
        self.assertEqual(len(seq), expected_len)

        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []

        for idx in range(len(seq)):
            x_batch, y_batch = seq[idx]
            xs.append(x_batch)
            ys.append(y_batch)

        x_all = np.concatenate(xs, axis=0)
        y_all = np.concatenate(ys, axis=0)

        self.assertEqual(x_all.shape[0], total_rows)
        self.assertEqual(x_all.shape[1], self.n_features)
        self.assertEqual(y_all.shape[0], total_rows)

    # ------------------------------------------------------------------ #
    # Loading without labels
    # ------------------------------------------------------------------ #
    def test_load_without_labels(self) -> None:
        """If label_name=None, CSVSequence should return only features (all columns)."""
        logger.info("Testing CSVDataLoader.load without labels")

        loader = CSVDataLoader(
            file_pattern=self.file_pattern,
            label_name=None,
            batch_size=3,
            shuffle=False,
            dtype="float32",
        )

        seq = loader.load()
        self.assertIsInstance(seq, CSVSequence)

        batch_x = seq[0]
        self.assertIsInstance(batch_x, np.ndarray)

        # We have feature_0, feature_1, target → 3 numeric columns
        expected_num_features = self.n_features + 1  # +1 for 'target'
        self.assertEqual(batch_x.shape[1], expected_num_features)


    # ------------------------------------------------------------------ #
    # Column selection
    # ------------------------------------------------------------------ #
    def test_column_selection_reduces_feature_dimension(self) -> None:
        """column_names should restrict feature columns properly."""
        logger.info("Testing CSVDataLoader with column_names selection")

        loader = CSVDataLoader(
            file_pattern=self.file_pattern,
            column_names=["feature_0"],
            label_name="target",
            batch_size=4,
            shuffle=False,
            dtype="float32",
        )

        seq = loader.load()
        x_batch, y_batch = seq[0]

        self.assertEqual(x_batch.shape[1], 1)  # only feature_0
        self.assertEqual(x_batch.shape[0], 4)
        self.assertEqual(y_batch.shape[0], 4)

    # ------------------------------------------------------------------ #
    # Error cases
    # ------------------------------------------------------------------ #
    def test_missing_files_raises_dataloadingerror(self) -> None:
        """No files matching pattern should raise DataLoadingError."""
        logger.info("Testing CSVDataLoader with missing files")

        loader = CSVDataLoader(
            file_pattern="no_such_file_*.csv",
            label_name="target",
        )

        with self.assertRaises(DataLoadingError):
            loader.load()

    def test_missing_label_column_raises_dataloadingerror(self) -> None:
        """Missing label column should raise DataLoadingError."""
        logger.info("Testing CSVDataLoader with missing label column")

        loader = CSVDataLoader(
            file_pattern=self.file_pattern,
            label_name="not_a_column",
        )

        with self.assertRaises(DataLoadingError):
            loader.load()

    def test_missing_feature_column_raises_dataloadingerror(self) -> None:
        """Requesting non-existent feature columns should raise DataLoadingError."""
        logger.info("Testing CSVDataLoader with invalid column_names")

        loader = CSVDataLoader(
            file_pattern=self.file_pattern,
            column_names=["feature_0", "does_not_exist"],
            label_name="target",
        )

        with self.assertRaises(DataLoadingError):
            loader.load()

    # ------------------------------------------------------------------ #
    # Internal helpers (sanity)
    # ------------------------------------------------------------------ #
    def test_get_files_returns_both_csv_files(self) -> None:
        """_get_files should find all CSV files matching the pattern."""
        logger.info("Testing CSVDataLoader._get_files")

        loader = CSVDataLoader(
            file_pattern=self.file_pattern,
            label_name="target",
        )

        files = loader._get_files()
        # _get_files returns paths relative to CWD; compare by name
        self.assertEqual(
            sorted(f.name for f in files),
            sorted([self.file1.name, self.file2.name]),
        )


if __name__ == "__main__":
    unittest.main()
