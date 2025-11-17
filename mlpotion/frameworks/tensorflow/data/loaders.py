"""TensorFlow data loaders."""

from pathlib import Path
from typing import Any, Callable

import tensorflow as tf
from loguru import logger

from mlpotion.core.exceptions import DataLoadingError
from mlpotion.core.protocols import DataLoader
from mlpotion.utils import trycatch


class TFCSVDataLoader(DataLoader[tf.data.Dataset]):
    """Load CSV files into TensorFlow datasets.

    Args:
        file_pattern: Glob pattern pointing to CSV files.
        batch_size: Batch size for the dataset.
        column_names: Columns to load from CSV (None = infer all).
        label_name: Name of the label column (None = no labels).
        map_fn: Mapping function to apply to the dataset.
        config: Extra keyword arguments forwarded to
            `tf.data.experimental.make_csv_dataset`. `num_epochs` can also be
            provided here, but will be extracted and validated.

    Example:
        ```python
        from mlpotion.frameworks.tensorflow.data.loaders import TFCSVDataLoader
        from mlpotion.core.config import DataLoadingConfig

        # setup config with validation
        config = DataLoadingConfig(
            file_pattern="data/*.csv",
            label_name="target",
            num_epochs=1,
            shuffle=True,
        )
        # create loader with config
        loader = TFCSVDataLoader(config=config)
        # load dataset
        dataset = loader.load()
        # use dataset
        for features, labels in dataset:
            ...
        ```
    """

    def __init__(
        self,
        file_pattern: str,
        batch_size: int = 32,
        column_names: list[str] | None = None,
        label_name: str | None = None,
        map_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.file_pattern = file_pattern
        self.column_names = column_names
        self.label_name = label_name
        self.batch_size = batch_size
        self.map_fn = map_fn

        # set default config
        _default_config = {"ignore_errors": True, "num_epochs": 1}
        self.config: dict[str, Any] = dict(config or _default_config)

        # Extract and validate num_epochs *once* so we don't risk duplicating kwargs
        self.num_epochs = self._extract_and_validate_num_epochs()

        self._validate_files_exist()
        self._validate_finite_dataset()

        logger.info(
            "{class_name} initialized with attrs: {attrs}",
            class_name=self.__class__.__name__,
            attrs=vars(self),
        )

    # -------------------------------------------------------------------------
    # Validation helpers
    # -------------------------------------------------------------------------

    def _extract_and_validate_num_epochs(self) -> int:
        """Extract `num_epochs` from config and validate it.

        Returns:
            Validated `num_epochs` value.

        Raises:
            DataLoadingError: If `num_epochs` is invalid.
        """
        # Pop to avoid passing it twice to make_csv_dataset
        num_epochs = self.config.pop("num_epochs", 1)

        if not isinstance(num_epochs, int):
            raise DataLoadingError(
                f"num_epochs must be an integer, got {type(num_epochs)!r}"
            )

        if num_epochs <= 0:
            raise DataLoadingError(
                f"num_epochs must be >= 1 for a finite dataset, got {num_epochs}"
            )

        return num_epochs

    def _validate_files_exist(self) -> None:
        """Validate that the files matching the pattern exist.

        Raises:
            DataLoadingError: If no file matches the glob pattern.
        """
        files = list(Path().glob(self.file_pattern))
        if not files:
            raise DataLoadingError(
                f"No files found matching pattern: {self.file_pattern}"
            )

        logger.info(
            "Found {count} files matching pattern: {pattern}",
            count=len(files),
            pattern=self.file_pattern,
        )

    def _validate_finite_dataset(self) -> None:
        """Validate that the dataset will be finite.

        Raises:
            DataLoadingError: If the dataset is effectively infinite.
        """
        # With current design, we guarantee num_epochs >= 1 in _extract...
        # but we keep this guard here as a sanity check and for future changes.
        if self.num_epochs <= 0:
            raise DataLoadingError(
                f"Dataset must be finite; got num_epochs={self.num_epochs}"
            )

        logger.info("Dataset is finite with num_epochs={num_epochs}", num_epochs=self.num_epochs)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @trycatch(
        error=DataLoadingError,
        success_msg="✅ Successfully loaded dataset",
    )
    def load(self) -> tf.data.Dataset:
        """Load CSV files into a TensorFlow dataset.

        Returns:
            tf.data.Dataset: Features and, if configured, labels.

        Raises:
            DataLoadingError: If loading fails or configuration is invalid.

        Example:
            ```python
            loader = TFCSVDataLoader(
                file_pattern="data/*.csv",
                label_name="target",
                config={"num_epochs": 1, "shuffle": True},
            )
            dataset = loader.load()
            for features, labels in dataset:
                ...
            ```
        """
        dataset = tf.data.experimental.make_csv_dataset(
            file_pattern=self.file_pattern,
            batch_size=self.batch_size,
            label_name=self.label_name,
            column_names=self.column_names,
            num_epochs=self.num_epochs,  # extracted and validated
            **self.config,
        )

        if self.map_fn:
            logger.info("Applying mapping function to dataset")
            dataset = dataset.map(self.map_fn)

        return dataset


class TFRecordDataLoader(DataLoader[tf.data.Dataset]):
    """Loader for TFRecord files into tf.data.Dataset.

    Args:
        file_pattern: Glob pattern pointing to one or more .tfrecord files.
        batch_size: Batch size for the dataset.
        parse_fn: Optional callable to parse one serialized example into the
            desired output (features, labels or whatever).
        element_spec_json: Optional path or dict describing the element_spec (for nested structure).
        config: Extra config dict. Recognised keys:
            - num_parallel_reads: int or tf.data.AUTOTUNE
            - shuffle_buffer_size: int
            - prefetch_buffer_size: int
            - drop_remainder: bool
            - repeat_count: Optional[int] (number of epochs, None = infinite)
            - compression_type: Optional[str] (“”, “GZIP”, “ZLIB”)
    """

    def __init__(
        self,
        file_pattern: str,
        batch_size: int = 32,
        column_names: list[str] | None = None,
        label_name: str | None = None,
        map_fn: Callable[[tf.Tensor], Any] | None = None,
        element_spec_json: str | dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.file_pattern = file_pattern
        self.batch_size = batch_size
        self.map_fn = map_fn
        self.element_spec_json = element_spec_json
        self.column_names = column_names
        self.label_name = label_name

        # set config
        self.config = config or {}

        # validate files exist
        self._validate_files_exist()

        logger.info(
            "{class_name} initialized with attrs: {attrs}",
            class_name=self.__class__.__name__,
            attrs=vars(self),
        )

    def _validate_files_exist(self) -> None:
        """Validate that the files matching the pattern exist."""
        files = list(Path().glob(self.file_pattern))
        if not files:
            raise DataLoadingError(
                f"No TFRecord files found matching pattern: {self.file_pattern}"
            )
        logger.info(
            "Found {count} TFRecord files matching pattern: {pattern}",
            count=len(files),
            pattern=self.file_pattern,
        )

    def _apply_column_label_selection(
        self,
        example: dict[str, tf.Tensor],
    ) -> tuple[Any, Any] | Any:
        """Select subset of columns and optionally separate label."""
        # If no label_name and no column_names, return whole example dict
        if self.column_names is None and self.label_name is None:
            return example

        # Extract features dict
        features: dict[str, tf.Tensor] = {}
        for key, tensor in example.items():
            if key == self.label_name:
                continue
            if self.column_names is None or key in self.column_names:
                features[key] = tensor

        # Extract label if requested
        if self.label_name is not None:
            if self.label_name not in example:
                raise DataLoadingError(
                    f"label_name '{self.label_name}' not found among features: {list(example.keys())}"
                )
            label = example[self.label_name]
            return features, label

        # No label requested: just return features dict
        return features

    def _get_files_matching_pattern(self) -> list[str]:
        """Get files matching the pattern.
        
        Returns:
            list[str]: List of files matching the pattern.
        """
        return tf.data.Dataset.list_files(self.file_pattern, shuffle=False)


    @trycatch(
        error=DataLoadingError,
        success_msg="✅ Successfully loaded TFRecord dataset",
    )
    def load(self) -> tf.data.Dataset:
        """Load TFRecord files into a tf.data.Dataset.

        Returns:
            tf.data.Dataset: Parsed and optionally mapped dataset of (features, label) or features only.
        Raises:
            DataLoadingError: on failure.
        """
        filenames = self._get_files_matching_pattern()
        
        ds = tf.data.TFRecordDataset(
            filenames=filenames,
            compression_type=self.config.get("compression_type", ""),
            buffer_size=self.config.get("buffer_size", None),
            num_parallel_reads=self.config.get("num_parallel_reads", tf.data.AUTOTUNE),
        )

        # Optionally repeat
        if "repeat_count" in self.config:
            ds = ds.repeat(self.config["repeat_count"])

        # Apply column/label selection
        ds = ds.map(
            self._apply_column_label_selection,
            num_parallel_calls=self.config.get("num_parallel_reads", tf.data.AUTOTUNE),
        )

        # Shuffle if requested
        if "shuffle_buffer_size" in self.config:
            ds = ds.shuffle(self.config["shuffle_buffer_size"])

        # Batch
        ds = ds.batch(self.batch_size, drop_remainder=self.config.get("drop_remainder", False))

        # Prefetch
        ds = ds.prefetch(self.config.get("prefetch_buffer_size", tf.data.AUTOTUNE))

        # Apply mapping function
        if self.map_fn:
            logger.info("Applying mapping function to dataset")
            ds = ds.map(self.map_fn)

        return ds