"""PyTorch data transformation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, Dataset, IterableDataset

from mlpotion.core.exceptions import DataTransformationError
from mlpotion.core.protocols import DataTransformer
from mlpotion.core.results import TransformationResult
from mlpotion.frameworks.pytorch.config import (
    DataTransformationConfig,
    DataLoadingConfig,
    ModelLoadingConfig,
)
from mlpotion.frameworks.pytorch.data.loaders import (
    PyTorchCSVDataset,
    StreamingPyTorchCSVDataset,
    PyTorchDataLoaderFactory,
)
from mlpotion.frameworks.pytorch.deployment.persistence import PyTorchModelPersistence
from mlpotion.utils import trycatch


# --------------------------------------------------------------------------- #
# Simple prediction → DataFrame helper
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class PyTorchPredictionFormatter:
    """Attach PyTorch model predictions to a pandas DataFrame.

    - If predictions are 1D → single column ``pred``.
    - If predictions are 2D with shape (N, 1) → single column ``pred``.
    - If predictions are 2D with shape (N, C) → columns ``pred_0 .. pred_{C-1}``.

    You can always swap this out for something fancier (e.g. argmax, probs).
    """

    prefix: str = "pred"

    def format(self, df_inputs: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
        """Return a new DataFrame with prediction columns appended.

        Args:
            df_inputs: Input features as a DataFrame.
            preds: Raw model predictions as a numpy array.

        Returns:
            DataFrame with one or more prediction columns appended.
        """
        if not isinstance(preds, np.ndarray):
            preds = np.asarray(preds)

        if preds.ndim == 1:
            df_inputs[self.prefix] = preds
            return df_inputs

        if preds.ndim == 2:
            if preds.shape[1] == 1:
                df_inputs[self.prefix] = preds[:, 0]
            else:
                for i in range(preds.shape[1]):
                    df_inputs[f"{self.prefix}_{i}"] = preds[:, i]
            return df_inputs

        logger.warning(
            "Predictions have unexpected ndim={ndim}; flattening last dims.",
            ndim=preds.ndim,
        )
        batch = preds.shape[0]
        flat = preds.reshape(batch, -1)
        for i in range(flat.shape[1]):
            df_inputs[f"{self.prefix}_{i}"] = flat[:, i]
        return df_inputs


# --------------------------------------------------------------------------- #
# Main transformer
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class PyTorchDataToCSVTransformer(
    DataTransformer[DataLoader[Any] | None, nn.Module | None]
):
    """Transform PyTorch DataLoader batches to CSV using a PyTorch model.

    This is the PyTorch analogue of your TF/Keras CSV transformers:

    - Resolves data from:
        * a :class:`DataLoadingConfig` → :class:`PyTorchCSVDataset` or
          :class:`StreamingPyTorchCSVDataset` + :class:`PyTorchDataLoaderFactory`,
        * an explicit :class:`Dataset` / :class:`IterableDataset`,
        * a pre-built :class:`DataLoader` passed to :meth:`transform`.

    - Resolves model from:
        * an attached :class:`nn.Module`,
        * a :class:`ModelLoadingConfig` / path via :class:`PyTorchModelPersistence`,
        * a model instance passed to :meth:`transform`.

    - Uses :class:`PyTorchPredictionFormatter` to attach predictions to each batch
      as columns in a :class:`pandas.DataFrame`.

    - Saves either:
        * one CSV per batch (``data_output_per_batch=True``), or
        * one concatenated CSV (``data_output_per_batch=False``).

    Expected DataLoader batch shapes:
        - ``(features, labels)`` or ``features`` where:

            * ``features`` is ``torch.Tensor[batch, ...]``.
            * ``labels`` is ignored for prediction purposes but can be
              present in the underlying dataset.

    Args:
        data_loading_config:
            Optional configuration object for building CSV-based datasets.
        dataset:
            Optional PyTorch :class:`Dataset` or :class:`IterableDataset`.
        dataloader:
            Optional pre-built :class:`DataLoader`. If provided, the
            loader is used as-is and no internal factory is applied.
        model_loading_config:
            Optional model-loading configuration (e.g. path, model class).
        model:
            Optional PyTorch model instance.
        model_input_feature_names:
            Optional ordered list of feature names used when building the
            feature DataFrame. If not provided, and the dataset exposes a
            ``_feature_cols`` attribute (like :class:`PyTorchCSVDataset`),
            those are used instead; otherwise generic names are generated.
        data_output_path:
            Base path for CSV output. Can be a directory (for per-batch
            mode) or a file path (for full concatenated CSV).
        data_output_per_batch:
            If True, write per-batch CSVs under ``data_output_path``;
            otherwise write a single concatenated CSV.
        device:
            Device on which to run the model (e.g. ``"cpu"``, ``"cuda"``).

        # DataLoader configuration (used when we build it ourselves)
        batch_size:
            Batch size for :class:`PyTorchDataLoaderFactory`.
        shuffle:
            Shuffle flag for :class:`PyTorchDataLoaderFactory`.
        num_workers:
            Number of worker processes.
        pin_memory:
            Whether to pin memory in DataLoader.
        drop_last:
            Whether to drop the last incomplete batch.
        persistent_workers:
            Persistent worker flag (PyTorch >= 1.7; ignored if ``num_workers == 0``).
        prefetch_factor:
            Prefetch factor per worker.
    """

    # Data sources
    data_loading_config: DataLoadingConfig | None = None
    dataset: Dataset[Any] | IterableDataset[Any] | None = None
    dataloader: DataLoader[Any] | None = None

    # Model sources
    model_loading_config: ModelLoadingConfig | None = None
    model: nn.Module | None = None

    # Optional explicit feature names
    model_input_feature_names: list[str] | None = None

    # Transformation / IO configuration
    data_output_path: str | None = None
    data_output_per_batch: bool = False
    device: str | torch.device = "cpu"

    # DataLoader factory configuration
    batch_size: int = 10_000
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    persistent_workers: bool | None = None
    prefetch_factor: int | None = None

    # Internal
    _prediction_formatter: PyTorchPredictionFormatter = field(
        default_factory=PyTorchPredictionFormatter,
        init=False,
    )
    _feature_names: list[str] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._validate_data_sources()
        self._validate_model_sources()

    # ------------------------------------------------------------------ #
    # Validation helpers
    # ------------------------------------------------------------------ #
    def _validate_data_sources(self) -> None:
        """Validate that we don't have conflicting data sources."""
        # At most one of (data_loading_config, dataset, dataloader) should be primary.
        provided = [
            self.data_loading_config is not None,
            self.dataset is not None,
            self.dataloader is not None,
        ]
        if sum(provided) == 0:
            raise DataTransformationError(
                "At least one of data_loading_config, dataset, or dataloader "
                "must be provided."
            )
        if sum(provided) > 1:
            logger.info(
                "Multiple data sources provided; precedence is: "
                "dataloader > dataset > data_loading_config."
            )

    def _validate_model_sources(self) -> None:
        """Validate that we have either a model or a model loading config."""
        if self.model is None and self.model_loading_config is None:
            raise DataTransformationError(
                "Either a PyTorch model or a ModelLoadingConfig must be provided."
            )
        if self.model is not None and self.model_loading_config is not None:
            logger.info(
                "Both model and model_loading_config provided; using the attached model."
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @trycatch(
        error=DataTransformationError,
        success_msg="✅ Data transformed and saved to {config.data_output_path}",
    )
    def transform(
        self,
        dataloader: DataLoader[Any] | None,
        model: nn.Module | None,
        config: DataTransformationConfig,
        **_: Any,
    ) -> TransformationResult:
        """Transform batches from a DataLoader and save predictions as CSV.

        Resolution order:

        - DataLoader:
            1. If `dataloader` arg is not None → use it directly.
            2. Else, if `self.dataloader` is set → use it.
            3. Else, if `self.dataset` is set → wrap it via
               :class:`PyTorchDataLoaderFactory`.
            4. Else, if `data_loading_config` is set → create a CSV dataset
               then wrap via :class:`PyTorchDataLoaderFactory`.

        - Model:
            1. If `model` arg is not None → use it.
            2. Else, if `self.model` is set → use it.
            3. Else, use :class:`PyTorchModelPersistence` with
               :class:`ModelLoadingConfig`.

        Example:
            ```python
            transformer = PyTorchDataToCSVTransformer(
                data_loading_config=data_cfg,
                model_loading_config=model_cfg,
                data_output_per_batch=False,
                batch_size=4096,
                device="cuda",
            )

            result = transformer.transform(
                dataloader=None,
                model=None,
                config=transform_cfg,
            )
            print(result.data_output_path)
            ```
        """
        resolved_loader = self._load_data(fallback_loader=dataloader)
        resolved_model = self._load_model(fallback_model=model)
        output_path = self._resolve_output_path(config=config)

        logger.info(
            "Starting PyTorch data transformation: "
            "output_path={output_path}, per_batch={per_batch}",
            output_path=output_path,
            per_batch=self.data_output_per_batch,
        )

        resolved_model.eval()
        resolved_model.to(self.device)

        batches = self._transform_data(
            dataloader=resolved_loader,
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
    def _load_data(
        self,
        fallback_loader: DataLoader[Any] | None = None,
    ) -> DataLoader[Any]:
        """Resolve a DataLoader from all configured sources."""
        # 1) Explicit loader argument
        if fallback_loader is not None:
            logger.info("Using DataLoader passed to transform().")
            self.dataloader = fallback_loader
            self._infer_feature_names_from_loader(fallback_loader)
            return fallback_loader

        # 2) Loader stored on instance
        if self.dataloader is not None:
            logger.info("Using DataLoader stored on transformer instance.")
            self._infer_feature_names_from_loader(self.dataloader)
            return self.dataloader

        # 3) Dataset stored on instance → DataLoader
        if self.dataset is not None:
            logger.info("Wrapping Dataset/IterableDataset stored on instance in DataLoader.")
            loader = self._build_dataloader_from_dataset(self.dataset)
            self.dataloader = loader
            return loader

        # 4) DataLoadingConfig → CSV dataset → DataLoader
        if self.data_loading_config is not None:
            logger.info("Building dataset from DataLoadingConfig for PyTorch CSV loader.")
            cfg = (
                self.data_loading_config.model_dump()
                if hasattr(self.data_loading_config, "model_dump")
                else self.data_loading_config.__dict__
            )
            streaming = cfg.get("streaming", False)
            dataset_kwargs = {
                "file_pattern": cfg["file_pattern"],
                "column_names": cfg.get("column_names"),
                "label_name": cfg.get("label_name"),
            }

            if streaming:
                dataset = StreamingPyTorchCSVDataset(
                    chunksize=cfg.get("chunksize", 1024),
                    **dataset_kwargs,
                )
            else:
                dataset = PyTorchCSVDataset(**dataset_kwargs)

            self.dataset = dataset
            loader = self._build_dataloader_from_dataset(dataset)
            self.dataloader = loader
            return loader

        raise DataTransformationError(
            "No data source resolved. Provide dataloader, dataset, or data_loading_config."
        )

    def _build_dataloader_from_dataset(
        self,
        dataset: Dataset[Any] | IterableDataset[Any],
    ) -> DataLoader[Any]:
        """Wrap a Dataset/IterableDataset in a DataLoader via the factory."""
        factory = PyTorchDataLoaderFactory[Any](
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )
        loader = factory.load(dataset)
        self._infer_feature_names_from_dataset(dataset)
        return loader

    def _load_model(
        self,
        fallback_model: nn.Module | None = None,
    ) -> nn.Module:
        """Resolve the PyTorch model from sources and persistence."""
        # 1) Explicit model argument
        if fallback_model is not None:
            logger.info("Using model passed to transform().")
            self.model = fallback_model
            return fallback_model

        # 2) Attached model
        if self.model is not None:
            logger.info("Using model stored on transformer instance.")
            return self.model

        # 3) ModelLoadingConfig → PyTorchModelPersistence
        if self.model_loading_config is not None:
            model_path = getattr(self.model_loading_config, "model_path", None)
            if model_path is None:
                raise DataTransformationError(
                    "ModelLoadingConfig must define `model_path` for PyTorch."
                )

            logger.info(
                "Loading model via PyTorchModelPersistence from: {path}",
                path=model_path,
            )

            persistence = PyTorchModelPersistence(path=model_path)

            # Optional: model class + kwargs from config for state_dict loading
            model_class = getattr(self.model_loading_config, "model_class", None)
            model_kwargs = getattr(self.model_loading_config, "model_kwargs", None)

            model = persistence.load(
                model_class=model_class,
                model_kwargs=model_kwargs,
                map_location=self.device,
            )
            self.model = model
            return model

        raise DataTransformationError(
            "No model provided. Set `model`, `model_loading_config`, or pass "
            "`model` to transform()."
        )

    def _resolve_output_path(self, config: DataTransformationConfig) -> str:
        """Resolve output path from instance attribute or config."""
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
    # Feature naming helpers
    # ------------------------------------------------------------------ #
    def _infer_feature_names_from_dataset(
        self,
        dataset: Dataset[Any] | IterableDataset[Any],
    ) -> None:
        """Derive feature names from the underlying dataset when possible."""
        if self.model_input_feature_names is not None:
            self._feature_names = list(self.model_input_feature_names)
            return

        if hasattr(dataset, "_feature_cols"):
            cols = getattr(dataset, "_feature_cols")
            if isinstance(cols, Iterable):
                self._feature_names = [str(c) for c in cols]
                logger.info(
                    "Inferred feature names from dataset._feature_cols: {cols}",
                    cols=self._feature_names,
                )
                return

        self._feature_names = None
        logger.info("No explicit feature names; using generic column names.")

    def _infer_feature_names_from_loader(self, loader: DataLoader[Any]) -> None:
        """Best-effort inference of feature names from loader.dataset."""
        if self.model_input_feature_names is not None:
            self._feature_names = list(self.model_input_feature_names)
            return

        dataset = getattr(loader, "dataset", None)
        if dataset is not None:
            self._infer_feature_names_from_dataset(dataset)
        else:
            self._feature_names = None

    # ------------------------------------------------------------------ #
    # Batch iteration and transformation
    # ------------------------------------------------------------------ #
    def _iter_batches(
        self,
        dataloader: DataLoader[Any],
    ) -> Iterator[torch.Tensor]:
        """Yield feature tensors from a DataLoader.

        Handles both:
            - loader yielding ``(features, labels)``
            - loader yielding ``features`` only
        """
        for batch in dataloader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                features = batch[0]
            else:
                features = batch

            if not isinstance(features, torch.Tensor):
                raise DataTransformationError(
                    "PyTorchDataToCSVTransformer expects batches where the first "
                    "element is a torch.Tensor. Got type: "
                    f"{type(features)!r}"
                )
            yield features

    def _transform_data(
        self,
        dataloader: DataLoader[Any],
        model: nn.Module,
    ) -> Iterator[pd.DataFrame]:
        """Transform all batches in the DataLoader and yield DataFrames."""
        device = torch.device(self.device)

        with torch.no_grad():
            for idx, features in enumerate(self._iter_batches(dataloader)):
                logger.debug("Transforming batch {idx}", idx=idx)
                features = features.to(device)
                preds = model(features)
                df = self._transform_batch(
                    features=features,
                    preds=preds,
                )
                yield df

    def _transform_batch(
        self,
        features: torch.Tensor,
        preds: torch.Tensor,
    ) -> pd.DataFrame:
        """Transform a single feature batch + predictions into a DataFrame."""
        df_features = self._batch_tensor_to_dataframe(features)
        preds_np = preds.detach().cpu().numpy()
        df_with_preds = self._prediction_formatter.format(df_features, preds_np)
        return df_with_preds

    def _batch_tensor_to_dataframe(self, features: torch.Tensor) -> pd.DataFrame:
        """Convert a feature tensor to a pandas DataFrame."""
        if features.dim() == 1:
            features = features.unsqueeze(1)

        if features.dim() > 2:
            batch = features.shape[0]
            features = features.view(batch, -1)

        arr = features.detach().cpu().numpy()
        n_features = arr.shape[1]

        if self._feature_names and len(self._feature_names) == n_features:
            columns = list(self._feature_names)
        else:
            if self._feature_names and len(self._feature_names) != n_features:
                logger.warning(
                    "Feature name count ({n_names}) does not match tensor feature "
                    "dimension ({n_features}); falling back to generic names.",
                    n_names=len(self._feature_names),
                    n_features=n_features,
                )
            columns = [f"feature_{i}" for i in range(n_features)]

        return pd.DataFrame(arr, columns=columns)

    # ------------------------------------------------------------------ #
    # Saving helpers
    # ------------------------------------------------------------------ #
    def _save_data_per_batch(
        self,
        batches: Iterable[pd.DataFrame],
    ) -> None:
        """Save transformed data batch-by-batch with a global unique index."""
        if self.data_output_path is None:
            raise DataTransformationError("data_output_path is not set.")

        base_path = Path(self.data_output_path)
        global_row_idx = 0

        for batch_idx, batch_df in enumerate(batches):
            if batch_df.empty:
                logger.warning("Skipping empty batch {idx}", idx=batch_idx)
                continue

            batch_df = batch_df.reset_index(drop=True)
            batch_len = len(batch_df)
            batch_df.index = range(global_row_idx, global_row_idx + batch_len)
            global_row_idx += batch_len

            logger.info(
                "Saving batch {batch_idx} with shape {shape}, index range "
                "[{start}, {end}]",
                batch_idx=batch_idx,
                shape=batch_df.shape,
                start=batch_df.index[0],
                end=batch_df.index[-1],
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
        if base_path.suffix:  # user gave a file → use its parent as directory
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
