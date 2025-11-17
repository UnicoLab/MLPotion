from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import keras
from keras.utils import Sequence
from loguru import logger

from mlpotion.frameworks.keras.config import ModelLoadingConfig, KerasCSVTransformationConfig

from mlpotion.core.exceptions import DataTransformationError
from mlpotion.core.protocols import DataTransformer
from mlpotion.core.results import TransformationResult
from mlpotion.frameworks.keras.data.loaders import CSVDataLoader, CSVSequence
from mlpotion.frameworks.keras.models.inspection import KerasModelInspector
from mlpotion.frameworks.keras.deployment.persistence import KerasModelPersistence
from mlpotion.frameworks.keras.utils.formatter import KerasPredictionFormatter
from mlpotion.utils import trycatch


@dataclass(slots=True)
class CSVDataTransformer(DataTransformer[CSVSequence, keras.Model]):
    """Transform tabular data to CSV using a Keras model (no TensorFlow).

    This class is the Keras/pandas analogue of TFDataToCSVTransformer:

    - Resolves data from:
        * a `CSVDataLoader` (recommended),
        * an explicit `dataset` (Sequence / DataFrame / ndarray / iterable of batches),
        * the `dataset` argument to `transform(...)` (fallback).
    - Resolves model from:
        * an attached `keras.Model`,
        * a `KerasModelPersistence` instance (path-based),
        * the `model` argument to `transform(...)` (fallback).
    - Optionally inspects the model via `KerasModelInspector` to derive
      input names and use them as `input_columns`.
    - Iterates over batches, runs `model.predict(...)`, and saves:
        * one CSV per batch (when `data_output_per_batch=True`), or
        * one concatenated CSV.

    Supported dataset types:
        - `CSVSequence` or any `keras.utils.Sequence`
        - `pandas.DataFrame`
        - `np.ndarray`
        - Iterable of pre-batched objects: `x_batch` or `(x_batch, y_batch)`

    Args:
        data_loader: Optional CSVDataLoader to create a `CSVSequence`.
        dataset: Optional dataset/sequence/iterable to transform.
        model: Optional compiled Keras model.
        model_persistence: Optional KerasModelPersistence to load the model and
            inspection metadata from disk.
        model_loading_config: Optional model loading config (if used in your
            project; currently only used to resolve path from it).
        model_path: Optional explicit path to a saved Keras model.
        batch_size: Optional batch size when chunking DataFrame/ndarray inputs.
            Ignored for `Sequence` datasets (already batched).
        data_output_path: Path to output CSV (file) or directory (when saving
            per-batch CSV files).
        data_output_per_batch: If True, writes `batch_00000.csv`, `batch_00001.csv`,
            etc. If False, concatenates all batches and writes a single CSV.
        feature_names: Optional feature names for ndarray batches. If not set,
            uses `feature_0`, `feature_1`, ....
        input_columns: Optional explicit list of columns to feed into the model.
            If None, they will be derived from model inspection (input names).

    Example:
        ```python
        data_loader = CSVDataLoader(
            file_pattern="data/train_*.csv",
            label_name="target",
            batch_size=64,
            shuffle=False,
        )

        transformer = CSVDataTransformer(
            data_loader=data_loader,
            model_path="models/my_model.keras",
            data_output_path="outputs/preds.csv",
            data_output_per_batch=False,
        )

        result = transformer.transform(
            dataset=None,
            model=None,
            config=DataTransformationConfig(data_output_path="outputs/preds.csv"),
        )
        print(result.data_output_path)
        ```
    """

    # Data sources
    data_loader: CSVDataLoader | None = None
    dataset: Any | None = None

    # Model sources
    model: keras.Model | None = None
    model_persistence: KerasModelPersistence | None = None
    model_loading_config: ModelLoadingConfig | None = None
    model_path: str | None = None

    # Transformation / IO configuration
    batch_size: int | None = None
    data_output_path: str | None = None
    data_output_per_batch: bool = False

    # Feature naming / model input selection
    feature_names: list[str] | None = None
    input_columns: list[str] | None = None

    # Prediction formatting utility
    prediction_formatter: KerasPredictionFormatter = field(
        default_factory=KerasPredictionFormatter
    )

    # Cached inspection metadata (e.g. from KerasModelPersistence)
    _model_inspection: dict[str, Any] | None = field(default=None, init=False)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @trycatch(
        error=DataTransformationError,
        success_msg="✅ Data transformed and saved to {config.data_output_path}",
    )
    def transform(
        self,
        dataset: Any,
        model: keras.Model,
        config: KerasCSVTransformationConfig,
        **kwargs: Any,
    ) -> TransformationResult:
        """Transform dataset using a Keras model and save predictions as CSV.

        The arguments are mainly for protocol compatibility; resolution is:

        - Dataset:
            1. `self.data_loader.load()` if `data_loader` is set,
            2. `self.dataset` if set,
            3. `dataset` argument as fallback.

        - Model:
            1. `self.model` if already set,
            2. `self.model_persistence.load(inspect=True)` if given,
               or `model_path` / `model_loading_config` path via a temporary
               `KerasModelPersistence`,
            3. `model` argument as fallback, with inspection run via
               `KerasModelInspector`.

        The model’s input names (from inspection) are used to filter the
        DataFrame columns passed into `model.predict(...)`, unless
        `input_columns` is explicitly specified.
        """
        resolved_dataset = self._load_data(fallback_dataset=dataset)
        resolved_model = self._load_model_and_inspection(fallback_model=model)
        output_path = self._resolve_output_path(config=config)

        # If user did not explicitly set input_columns, derive them from inspection
        if self.input_columns is None:
            self.input_columns = self._derive_input_columns_from_inspection()

        logger.info(
            "Starting CSV data transformation: "
            f"output_path={output_path}, per_batch={self.data_output_per_batch}"
        )

        batches = self._transform_data(
            data=resolved_dataset,
            model=resolved_model,
        )

        if self.data_output_per_batch:
            self._save_data_per_batch(batches)
        else:
            self._save_full_data_from_batches(batches)

        return TransformationResult(data_output_path=output_path)

    # ------------------------------------------------------------------ #
    # Data / model loading
    # ------------------------------------------------------------------ #
    def _load_data(self, fallback_dataset: Any | None = None) -> Any:
        """Resolve dataset from loader, instance attribute, or argument."""
        if self.data_loader is not None:
            logger.info("Loading dataset via CSVDataLoader")
            dataset = self.data_loader.load()
            self.dataset = dataset
            return dataset

        if self.dataset is not None:
            logger.info("Using dataset stored on transformer instance")
            return self.dataset

        if fallback_dataset is not None:
            logger.info("Using dataset passed to transform()")
            self.dataset = fallback_dataset
            return fallback_dataset

        raise DataTransformationError(
            "No dataset provided. Set `data_loader`, `dataset`, or pass "
            "`dataset` to transform()."
        )

    def _load_model_and_inspection(
        self,
        fallback_model: keras.Model | None = None,
    ) -> keras.Model:
        """Resolve model and (if possible) inspection metadata."""
        # 1) Already attached model + maybe cached inspection
        if self.model is not None:
            logger.info("Using model stored on transformer instance")
            if self._model_inspection is None:
                logger.info("Inspecting attached model with KerasModelInspector")
                inspector = KerasModelInspector()
                self._model_inspection = inspector.inspect(self.model)
            return self.model

        # 2) Use KerasModelPersistence if provided
        if self.model_persistence is not None:
            logger.info("Loading model via KerasModelPersistence (inspect=True)")
            model, inspection = self.model_persistence.load(inspect=True)
            self.model = model
            self._model_inspection = inspection
            return model

        # 3) Use model_path / model_loading_config via a temporary persistence
        resolved_path = self._resolve_model_path()
        if resolved_path is not None:
            logger.info(f"Loading model from path via KerasModelPersistence: {resolved_path}")
            persistence = KerasModelPersistence(path=resolved_path)
            model, inspection = persistence.load(inspect=True)
            self.model = model
            self._model_inspection = inspection
            return model

        # 4) Fallback: use model passed to transform()
        if fallback_model is not None:
            logger.info("Using model passed to transform()")
            self.model = fallback_model
            logger.info("Inspecting fallback model with KerasModelInspector")
            inspector = KerasModelInspector()
            self._model_inspection = inspector.inspect(fallback_model)
            return fallback_model

        raise DataTransformationError(
            "No model provided. Set `model`, `model_persistence`, `model_path`, "
            "`model_loading_config`, or pass `model` to transform()."
        )

    def _resolve_model_path(self) -> str | None:
        """Resolve model path from explicit attribute or ModelLoadingConfig."""
        if self.model_path is not None:
            return self.model_path

        if self.model_loading_config is not None:
            path = getattr(self.model_loading_config, "model_path", None)
            if path is not None:
                return str(path)

        return None

    def _resolve_output_path(self, config: KerasCSVTransformationConfig) -> str:
        """Determine and store the data_output_path used for saving results."""
        if self.data_output_path is not None:
            return self.data_output_path

        path = getattr(config, "data_output_path", None)
        if path is None:
            raise DataTransformationError(
                "data_output_path must be set either on transformer or in config."
            )

        self.data_output_path = str(path)
        return self.data_output_path

    # ------------------------------------------------------------------ #
    # Model inspection → input columns
    # ------------------------------------------------------------------ #
    def _derive_input_columns_from_inspection(self) -> list[str] | None:
        """Derive input column names from inspector metadata.

        Uses `inspection['input_names']` if available. Tensor names often look
        like `"input_1:0"`, so we strip trailing `":0"` etc. We **do not**
        check against actual DataFrame columns here; intersection is handled
        later in `_restrict_batch_columns`.
        """
        if not self._model_inspection:
            logger.info("No inspection metadata available; using all columns as inputs.")
            return None

        raw_names = self._model_inspection.get("input_names") or []
        if not raw_names:
            logger.info(
                "Inspection metadata has no 'input_names'; using all columns as inputs."
            )
            return None

        def _normalize(name: str) -> str:
            # Strip TensorFlow-style suffix if present, but works fine in pure Keras too
            if ":" in name:
                return name.split(":", maxsplit=1)[0]
            return name

        normalized = [_normalize(str(n)) for n in raw_names]
        logger.info("Derived input_columns from model inspection: {cols}", cols=normalized)
        return normalized or None

    # ------------------------------------------------------------------ #
    # Batch iteration logic
    # ------------------------------------------------------------------ #
    def _iter_batches(self, data: Any) -> Iterator[Any]:
        """Yield batches from various supported dataset types."""
        # Keras Sequence (already batched)
        if isinstance(data, Sequence):
            logger.info(
                "Iterating over Keras Sequence with {n_batches} batches",
                n_batches=len(data),
            )
            for idx in range(len(data)):
                yield data[idx]
            return

        # pandas DataFrame
        if isinstance(data, pd.DataFrame):
            logger.info(
                "Iterating over pandas DataFrame with {n_rows} rows",
                n_rows=len(data),
            )
            if not self.batch_size or self.batch_size <= 0:
                yield data
                return

            for start in range(0, len(data), self.batch_size):
                yield data.iloc[start : start + self.batch_size]
            return

        # NumPy array
        if isinstance(data, np.ndarray):
            logger.info(
                "Iterating over NumPy array with shape {shape}",
                shape=data.shape,
            )
            x = data
            if x.ndim == 1:
                x = x.reshape(-1, 1)

            if not self.batch_size or self.batch_size <= 0:
                yield x
                return

            for start in range(0, x.shape[0], self.batch_size):
                yield x[start : start + self.batch_size]
            return

        # Generic iterable of (possibly pre-batched) items
        if isinstance(data, Iterable):
            logger.info("Iterating over generic iterable of batches")
            for batch in data:
                yield batch
            return

        raise DataTransformationError(
            f"Unsupported dataset type for transformation: {type(data)!r}"
        )

    # ------------------------------------------------------------------ #
    # Core transform helpers
    # ------------------------------------------------------------------ #
    def _transform_data(
        self,
        data: Any,
        model: keras.Model,
    ) -> Iterator[pd.DataFrame]:
        """Transform all batches in the dataset and yield DataFrames."""
        for idx, batch in enumerate(self._iter_batches(data)):
            logger.debug(f"Transforming batch {idx}")
            yield self._transform_batch(batch=batch, model=model)

    def _transform_batch(self, batch: Any, model: keras.Model) -> pd.DataFrame:
        """Transform a single batch to a DataFrame with predictions."""
        if isinstance(batch, tuple) and len(batch) >= 1:
            x_batch = batch[0]
        else:
            x_batch = batch

        df_features = self._batch_to_dataframe(x_batch)
        df_inputs = self._restrict_batch_columns(df_features)

        logger.debug(
            "Running model.predict on batch of shape {shape}",
            shape=df_inputs.shape,
        )

        x_np = df_inputs.to_numpy()
        preds = model.predict(x_np, verbose=0)

        # Delegate all prediction-shape handling to KerasPredictionFormatter
        df_with_preds = self.prediction_formatter.format(df_inputs, preds)
        return df_with_preds

    def _batch_to_dataframe(self, x_batch: Any) -> pd.DataFrame:
        """Convert a single batch of features into a pandas DataFrame."""
        if isinstance(x_batch, pd.DataFrame):
            return x_batch.copy()

        if isinstance(x_batch, dict):
            return pd.DataFrame(x_batch)

        x_np = np.asarray(x_batch)
        if x_np.ndim == 1:
            x_np = x_np.reshape(-1, 1)

        n_features = x_np.shape[1]
        if self.feature_names is not None:
            if len(self.feature_names) != n_features:
                raise DataTransformationError(
                    f"feature_names length ({len(self.feature_names)}) does not "
                    f"match number of features ({n_features})"
                )
            columns = list(self.feature_names)
        else:
            columns = [f"feature_{i}" for i in range(n_features)]

        return pd.DataFrame(x_np, columns=columns)

    def _restrict_batch_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restrict DataFrame to model input columns, if specified.

        If `input_columns` is None, all columns are kept as inputs. If set,
        we intersect with DataFrame columns and preserve order.

        Raises:
            DataTransformationError: If none of the requested input columns
            are present in the DataFrame.
        """
        if not self.input_columns:
            logger.debug("No input_columns specified; using all DataFrame columns as inputs.")
            return df

        requested = list(self.input_columns)
        present = [c for c in requested if c in df.columns]

        if not present:
            logger.error(
                "None of the requested input columns are present in the batch DataFrame."
            )
            raise DataTransformationError(
                f"Expected any of {sorted(requested)}, but got DataFrame columns "
                f"{sorted(df.columns)}"
            )

        logger.info("Restricting batch to input columns: {cols}", cols=present)
        return df[present]

    # ------------------------------------------------------------------ #
    # Saving helpers
    # ------------------------------------------------------------------ #
    def _save_data_per_batch(
        self,
        batches: Iterable[pd.DataFrame],
    ) -> None:
        """Save transformed data batch-by-batch with unique index ranges."""
        if self.data_output_path is None:
            raise DataTransformationError("data_output_path is not set.")

        base_path = Path(self.data_output_path)
        global_row_idx = 0

        for batch_idx, batch_df in enumerate(batches):
            if batch_df.empty:
                logger.warning(f"Skipping empty batch {batch_idx}")
                continue

            batch_df = batch_df.reset_index(drop=True)
            batch_len = len(batch_df)
            batch_df.index = range(global_row_idx, global_row_idx + batch_len)
            global_row_idx += batch_len

            logger.info(
                f"Saving batch {batch_idx} with shape {batch_df.shape}, "
                f"index range [{batch_df.index[0]}, {batch_df.index[-1]}]"
            )
            self._save_single_batch(
                batch_df=batch_df,
                batch_idx=batch_idx,
                base_path=base_path,
            )

    def _save_single_batch(
        self,
        batch_df: pd.DataFrame,
        batch_idx: int,
        base_path: Path,
    ) -> None:
        """Persist a single batch DataFrame as `batch_00000.csv`, etc."""
        if base_path.suffix:  # if user gave a file, use its parent as directory
            out_dir = base_path.parent
        else:
            out_dir = base_path

        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"batch_{batch_idx:05d}.csv"

        batch_df.to_csv(out_file, index=False)
        logger.info("Saved batch {idx} to {path}", idx=batch_idx, path=out_file)

    def _save_full_data_from_batches(
        self,
        batches: Iterable[pd.DataFrame],
    ) -> pd.DataFrame:
        """Concatenate all batches and save as a single CSV file."""
        if self.data_output_path is None:
            raise DataTransformationError("data_output_path is not set.")

        all_batches = list(batches)
        if not all_batches:
            logger.warning("No batches to save; returning empty DataFrame.")
            full_df = pd.DataFrame()
        else:
            full_df = pd.concat(all_batches, ignore_index=True)

        out_path = Path(self.data_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving full dataset with shape {shape}", shape=full_df.shape)
        full_df.to_csv(out_path, index=False)

        return full_df
