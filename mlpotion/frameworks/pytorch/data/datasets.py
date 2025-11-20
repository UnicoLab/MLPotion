from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset, IterableDataset

from mlpotion.core.exceptions import DataLoadingError


@dataclass(slots=True)
class CSVDataset(Dataset[tuple[torch.Tensor, torch.Tensor] | torch.Tensor]):
    """PyTorch Dataset for CSV files with on-demand tensor conversion.

    This class loads CSV data into memory (using Pandas) and provides a map-style PyTorch Dataset.
    It supports filtering columns, separating labels, and efficient on-demand tensor conversion
    to minimize memory usage.

    Attributes:
        file_pattern (str): Glob pattern matching the CSV files to load.
        column_names (list[str] | None): Specific columns to load. If None, all columns are loaded.
        label_name (str | None): Name of the column to use as the label. If None, no labels are returned.
        dtype (torch.dtype): The data type for the features (default: `torch.float32`).

    Example:
        ```python
        from mlpotion.frameworks.pytorch import CSVDataset
        from torch.utils.data import DataLoader

        # Create dataset
        dataset = CSVDataset(
            file_pattern="data/train_*.csv",
            label_name="target_class",
            column_names=["feature1", "feature2", "target_class"]
        )

        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Iterate
        for features, labels in dataloader:
            print(features.shape, labels.shape)
        ```
    """

    file_pattern: str
    column_names: list[str] | None = None
    label_name: str | None = None
    dtype: torch.dtype = torch.float32

    # Internal fields
    _features_df: pd.DataFrame | None = field(default=None, init=False)
    _labels: np.ndarray | None = field(default=None, init=False)
    _feature_cols: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Eagerly load CSV files into a DataFrame and validate configuration."""
        try:
            files = self._resolve_files()
            df = self._load_dataframe(files)
            df = self._select_columns(df)
            self._split_features_labels(df)

            logger.info(
                "Initialized CSVDataset with "
                "n_rows={rows}, n_features={features}, labels={labels}",
                rows=len(self._features_df) if self._features_df is not None else 0,
                features=len(self._feature_cols),
                labels="yes" if self._labels is not None else "no",
            )
        except DataLoadingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DataLoadingError(f"Failed to load CSV dataset: {exc!s}") from exc

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _resolve_files(self) -> list[Path]:
        """Find CSV files matching the pattern."""
        # Check if file_pattern is an absolute path or contains wildcards
        pattern_path = Path(self.file_pattern)

        if pattern_path.is_absolute():
            # For absolute paths without wildcards, check if file exists directly
            if "*" not in self.file_pattern and "?" not in self.file_pattern:
                if not pattern_path.exists():
                    raise DataLoadingError(f"File not found: {self.file_pattern}")
                files = [pattern_path]
            else:
                # For absolute paths with wildcards, use parent directory
                parent = pattern_path.parent
                pattern = pattern_path.name
                files = sorted(parent.glob(pattern))
        else:
            # For relative paths, use current directory
            files = sorted(Path().glob(self.file_pattern))

        if not files:
            raise DataLoadingError(f"No files found: {self.file_pattern}")
        logger.info(
            "Found {count} CSV file(s) matching pattern: {pattern}",
            count=len(files),
            pattern=self.file_pattern,
        )
        return files

    def _load_dataframe(self, files: list[Path]) -> pd.DataFrame:
        """Load CSV files into a single DataFrame."""
        logger.info(f"Loading {len(files)} CSV file(s) into pandas DataFrame...")
        dfs: list[pd.DataFrame] = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        logger.info(
            "Loaded {rows} rows from {files} file(s).",
            rows=len(df),
            files=len(files),
        )
        return df

    def _select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optionally restrict to the requested columns (excluding label)."""
        if not self.column_names:
            return df

        missing = [c for c in self.column_names if c not in df.columns]
        if missing:
            raise DataLoadingError(
                f"Requested columns {missing} not in CSV columns {list(df.columns)}"
            )
        logger.info(
            "Selecting subset of feature columns: {cols}",
            cols=self.column_names,
        )
        return df[self.column_names + ([self.label_name] if self.label_name else [])]

    def _split_features_labels(self, df: pd.DataFrame) -> None:
        """Split DataFrame into features and labels (if configured)."""
        self._dtype = self.dtype  # mirror Keras/PyTorch loaders style

        if self.label_name:
            if self.label_name not in df.columns:
                raise DataLoadingError(
                    f"Label column '{self.label_name}' not found in {list(df.columns)}"
                )
            self._labels = df[self.label_name].to_numpy()
            self._features_df = df.drop(columns=[self.label_name])
        else:
            self._labels = None
            self._features_df = df

        if self._features_df is None:
            raise DataLoadingError(
                "Internal error: features DataFrame is None after split."
            )

        self._feature_cols = list(self._features_df.columns)

    # ------------------------------------------------------------------ #
    # Dataset protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        """Return dataset length."""
        if self._features_df is None:
            return 0
        return len(self._features_df)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Get item at index.

        Args:
            idx: Global row index.

        Returns:
            (features, label) tuple if labels exist, else just features.
        """
        if self._features_df is None:
            raise IndexError("Dataset is empty or not properly initialized.")

        row = self._features_df.iloc[idx]

        # Convert to numpy array and then to tensor
        features_np: np.ndarray = row.to_numpy(dtype="float32", copy=False)
        features = torch.as_tensor(features_np, dtype=self._dtype)

        if self._labels is not None:
            label_val = self._labels[idx]
            label = torch.as_tensor(label_val, dtype=self._dtype)
            return features, label

        return features


@dataclass(slots=True)
class StreamingCSVDataset(
    IterableDataset[tuple[torch.Tensor, torch.Tensor] | torch.Tensor]
):
    """Streaming PyTorch IterableDataset for large CSV files.

    This dataset is designed for datasets that are too large to fit in memory. It reads CSV files
    in chunks (using Pandas) and streams samples one by one. It is compatible with PyTorch's
    `IterableDataset` interface.

    Attributes:
        file_pattern (str): Glob pattern matching the CSV files to load.
        column_names (list[str] | None): Specific columns to load.
        label_name (str | None): Name of the label column.
        chunksize (int): Number of rows to read into memory at a time per file.
        dtype (torch.dtype): The data type for the features.

    Example:
        ```python
        from mlpotion.frameworks.pytorch import StreamingCSVDataset
        from torch.utils.data import DataLoader

        # Create streaming dataset
        dataset = StreamingCSVDataset(
            file_pattern="data/large_dataset_*.csv",
            label_name="target",
            chunksize=10000
        )

        # Create DataLoader (shuffle must be False for IterableDataset)
        dataloader = DataLoader(dataset, batch_size=64)

        for features, labels in dataloader:
            # Train model...
            pass
        ```
    """

    file_pattern: str
    column_names: list[str] | None = None
    label_name: str | None = None
    chunksize: int = 1024
    dtype: torch.dtype = torch.float32

    files: list[Path] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Resolve files eagerly and log basic configuration."""
        self.files = self._resolve_files()
        logger.info(
            "Initialized StreamingCSVDataset with {n_files} file(s), "
            "chunksize={chunksize}, label_name={label}",
            n_files=len(self.files),
            chunksize=self.chunksize,
            label=self.label_name,
        )

    def _resolve_files(self) -> list[Path]:
        """Find CSV files matching the pattern."""
        # Check if file_pattern is an absolute path or contains wildcards
        pattern_path = Path(self.file_pattern)

        if pattern_path.is_absolute():
            # For absolute paths without wildcards, check if file exists directly
            if "*" not in self.file_pattern and "?" not in self.file_pattern:
                if not pattern_path.exists():
                    raise DataLoadingError(f"File not found: {self.file_pattern}")
                files = [pattern_path]
            else:
                # For absolute paths with wildcards, use parent directory
                parent = pattern_path.parent
                pattern = pattern_path.name
                files = sorted(parent.glob(pattern))
        else:
            # For relative paths, use current directory
            files = sorted(Path().glob(self.file_pattern))

        if not files:
            raise DataLoadingError(f"No files found: {self.file_pattern}")
        logger.info(
            "Found {count} CSV file(s) matching pattern: {pattern}",
            count=len(files),
            pattern=self.file_pattern,
        )
        return files

    def __iter__(
        self,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """Yield samples one by one across all CSV files."""
        for file_path in self.files:
            logger.info(f"Streaming CSV file: {file_path}")
            try:
                # Use pandas chunked reading
                chunk_iter = pd.read_csv(
                    file_path,
                    usecols=self.column_names,
                    chunksize=self.chunksize,
                )
            except TypeError:
                # If usecols=None is not accepted by some pandas version
                chunk_iter = pd.read_csv(
                    file_path,
                    chunksize=self.chunksize,
                )

            for chunk_df in chunk_iter:
                # Validate label column if needed
                if self.label_name:
                    if self.label_name not in chunk_df.columns:
                        raise DataLoadingError(
                            f"Label column '{self.label_name}' not found in "
                            f"file {file_path} (columns: {list(chunk_df.columns)})"
                        )
                    labels_np = chunk_df[self.label_name].to_numpy()
                    features_df = chunk_df.drop(columns=[self.label_name])
                else:
                    labels_np = None
                    features_df = chunk_df

                # Convert whole chunk to numpy once
                features_np = features_df.to_numpy(dtype="float32", copy=False)

                if labels_np is not None:
                    for row_idx in range(features_np.shape[0]):
                        x = torch.as_tensor(
                            features_np[row_idx],
                            dtype=self.dtype,
                        )
                        y = torch.as_tensor(labels_np[row_idx], dtype=self.dtype)
                        yield x, y
                else:
                    for row_idx in range(features_np.shape[0]):
                        x = torch.as_tensor(
                            features_np[row_idx],
                            dtype=self.dtype,
                        )
                        yield x
