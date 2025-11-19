"""Keras data loaders."""
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from keras.utils import Sequence
from loguru import logger

from mlpotion.core.exceptions import DataLoadingError
from mlpotion.core.protocols import DataLoader
from mlpotion.utils import trycatch


class CSVSequence(Sequence):
    """Keras Sequence for CSV data backed by NumPy arrays.

    This class implements the `keras.utils.Sequence` interface, allowing it to be used directly
    with `model.fit()`, `model.evaluate()`, and `model.predict()`. It handles data batching
    and shuffling efficiently in memory.

    Attributes:
        features (np.ndarray): The feature data as a 2D NumPy array.
        labels (np.ndarray | None): The label data as a NumPy array, or None if not available.
        batch_size (int): The size of each data batch.
        shuffle (bool): Whether to shuffle the data at the end of each epoch.
        dtype (np.dtype | str): The data type of the features and labels.

    Example:
        ```python
        from mlpotion.frameworks.keras.data.loaders import CSVSequence
        import numpy as np

        # Create dummy data
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        # Create sequence
        sequence = CSVSequence(
            features=X,
            labels=y,
            batch_size=32,
            shuffle=True
        )

        # Use in training
        model.fit(sequence, epochs=5)
        ```
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray | None,
        batch_size: int = 32,
        shuffle: bool = True,
        dtype: np.dtype | str = "float32",
    ) -> None:
        if features.ndim != 2:
            raise ValueError(
                f"features must be 2D (n_samples, n_features), got shape {features.shape}"
            )

        if labels is not None and len(labels) != len(features):
            raise ValueError(
                f"features and labels must have same length, "
                f"got {len(features)} != {len(labels)}"
            )

        self._features = features.astype(dtype, copy=False)
        self._labels = labels.astype(dtype, copy=False) if labels is not None else None
        self._batch_size = int(batch_size)
        self._shuffle = bool(shuffle)
        self._indices = np.arange(len(self._features))

        if self._shuffle:
            np.random.shuffle(self._indices)

        logger.info(
            "Initialized CSVSequence: "
            f"n_samples={len(self._features)}, "
            f"batch_size={self._batch_size}, "
            f"shuffle={self._shuffle}, "
            f"labels={'yes' if self._labels is not None else 'no'}"
        )

    def __len__(self) -> int:
        """Number of batches per epoch."""
        n_samples = len(self._features)
        return int(np.ceil(n_samples / self._batch_size))

    def __getitem__(self, idx: int) -> Any:
        """Get batch by index.

        Returns:
            - (x_batch, y_batch) if labels are available
            - x_batch otherwise
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Batch index out of range: {idx}")

        start = idx * self._batch_size
        end = min(start + self._batch_size, len(self._features))
        batch_idx = self._indices[start:end]

        x_batch = self._features[batch_idx]
        if self._labels is not None:
            y_batch = self._labels[batch_idx]
            return x_batch, y_batch

        return x_batch

    def on_epoch_end(self) -> None:
        """Shuffle indices between epochs if enabled."""
        if self._shuffle:
            np.random.shuffle(self._indices)


@dataclass(slots=True)
class CSVDataLoader(DataLoader[CSVSequence]):
    """Loader for CSV files into a Keras-ready Sequence.

    This class provides a high-level interface for loading data from CSV files into a
    `CSVSequence` compatible with Keras models. It handles file globbing, reading,
    column selection, and label separation.

    Attributes:
        file_pattern (str): Glob pattern matching the CSV files to load (e.g., "data/*.csv").
        batch_size (int): Number of samples per batch in the resulting sequence.
        column_names (list[str] | None): List of column names to use as features. If None, all columns except the label are used.
        label_name (str | None): Name of the column to use as labels. If None, no labels are returned (inference mode).
        shuffle (bool): Whether to shuffle the data between epochs.
        dtype (np.dtype | str): Data type to use for the loaded data (default: "float32").

    Example:
        ```python
        import keras
        from mlpotion.frameworks.keras.data.loaders import CSVDataLoader

        # Define a simple model
        model = keras.Sequential([
            keras.layers.Dense(10, input_shape=(5,), activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')

        # Create loader
        loader = CSVDataLoader(
            file_pattern="data/train_*.csv",
            label_name="target_class",
            batch_size=64,
            shuffle=True
        )

        # Load data
        train_sequence = loader.load()

        # Train model
        model.fit(train_sequence, epochs=10)
        ```
    """

    file_pattern: str
    batch_size: int = 32
    column_names: list[str] | None = None
    label_name: str | None = None
    shuffle: bool = True
    dtype: np.dtype | str = "float32"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @trycatch(
        error=DataLoadingError,
        success_msg="✅ Successfully created Keras CSV Sequence",
    )
    def load(self) -> CSVSequence:
        """Load CSV files and return a CSVSequence.

        Returns:
            CSVSequence that can be passed directly to `model.fit(...)`.

        Raises:
            DataLoadingError: If files cannot be found or read.
        """
        files = self._get_files()
        df = self._load_dataframe(files)
        df = self._select_columns(df)

        features_np, labels_np = self._split_features_labels(df)

        logger.info(
            "Creating CSVSequence: n_samples={n}, n_features={d}, labels={labels}",
            n=features_np.shape[0],
            d=features_np.shape[1],
            labels="yes" if labels_np is not None else "no",
        )

        sequence = CSVSequence(
            features=features_np,
            labels=labels_np,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            dtype=self.dtype,
        )
        return sequence

    # ------------------------------------------------------------------ #
    # Validation helpers
    # ------------------------------------------------------------------ #
    def _get_files(self) -> list[Path]:
        """Return sorted list of files matching the pattern."""
        file_paths = sorted(glob(self.file_pattern))
        if not file_paths:
            raise DataLoadingError(
                f"No CSV files found matching pattern: {self.file_pattern}"
            )
        files = [Path(f) for f in file_paths]
        logger.info(
            "Found {count} CSV file(s) matching pattern: {pattern}",
            count=len(files),
            pattern=self.file_pattern,
        )
        return files

    def _load_dataframe(self, files: list[Path]) -> pd.DataFrame:
        """Load CSVs into a single DataFrame."""
        try:
            logger.info(f"Loading {len(files)} CSV file(s) into pandas DataFrame...")
            dfs = [pd.read_csv(f) for f in files]
            df = pd.concat(dfs, ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            raise DataLoadingError(f"Failed to load CSV files: {exc!s}") from exc

        logger.info(
            "Loaded {rows} rows from {files} file(s).",
            rows=len(df),
            files=len(files),
        )
        return df

    def _select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optionally restrict DataFrame to the requested feature columns."""
        if not self.column_names:
            return df

        missing = [c for c in self.column_names if c not in df.columns]
        if missing:
            raise DataLoadingError(
                f"Requested columns {missing} not found in CSV columns {list(df.columns)}"
            )

        logger.info(
            "Selecting subset of columns: {cols}",
            cols=self.column_names,
        )
        return df[self.column_names + ([self.label_name] if self.label_name else [])]

    def _split_features_labels(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Split DataFrame into features and optional labels."""
        if self.label_name is None:
            # All numeric columns treated as features
            features_df = df
            labels_np: np.ndarray | None = None
        else:
            if self.label_name not in df.columns:
                raise DataLoadingError(
                    f"Label column '{self.label_name}' not found in {list(df.columns)}"
                )

            labels_np = df[self.label_name].to_numpy()
            features_df = df.drop(columns=[self.label_name])

        # Convert features to numeric array (best-effort)
        try:
            features_np = features_df.to_numpy(dtype=self.dtype, copy=False)
        except TypeError:
            # For older pandas versions or mixed dtypes; fallback to float32
            features_np = features_df.to_numpy(dtype="float32", copy=False)

        if labels_np is not None:
            # Attempt to cast labels to numeric dtype if possible
            try:
                labels_np = labels_np.astype(self.dtype, copy=False)
            except TypeError:
                logger.warning(
                    "Unable to cast labels to dtype {dtype}; leaving as original.",
                    dtype=self.dtype,
                )

        return features_np, labels_np

