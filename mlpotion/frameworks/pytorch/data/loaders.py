from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from loguru import logger
from torch.utils.data import DataLoader, Dataset, IterableDataset

T_co = TypeVar("T_co", covariant=True)


@dataclass(slots=True)
class PyTorchDataLoaderFactory(Generic[T_co]):
    """Factory for creating PyTorch DataLoaders.

    This is a small convenience wrapper around :class:`torch.utils.data.DataLoader`
    that:

    - Supports both map-style :class:`Dataset` and :class:`IterableDataset`.
    - Explicitly disables shuffling for iterable datasets (and logs a warning
      if ``shuffle=True`` was requested).
    - Only applies worker-related options when they are valid (e.g.
      ``persistent_workers`` & ``prefetch_factor`` only when ``num_workers > 0``).

    Args:
        batch_size: Batch size to use for the DataLoader.
        shuffle: Whether to shuffle data (ignored for IterableDataset).
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to pin memory (useful for CUDA training).
        drop_last: Whether to drop the last incomplete batch.
        persistent_workers: Whether to use persistent worker processes
            (PyTorch >= 1.7; ignored if ``num_workers == 0``).
        prefetch_factor: Number of batches loaded in advance by each worker
            (only relevant when ``num_workers > 0``).

    Example:
        ```python
        from torch.utils.data import TensorDataset
        import torch

        x = torch.randn(100, 32)
        y = torch.randint(0, 2, (100,))
        dataset = TensorDataset(x, y)

        factory = PyTorchDataLoaderFactory(batch_size=16, shuffle=True)
        train_loader = factory.create(dataset)

        for batch_x, batch_y in train_loader:
            ...
        ```
    """

    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    persistent_workers: bool | None = None
    prefetch_factor: int | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load(
        self,
        dataset: Dataset[T_co] | IterableDataset[T_co],
    ) -> DataLoader[T_co]:
        """Load a configured :class:`DataLoader` from a dataset.

        This method is aware of :class:`IterableDataset` vs map-style
        :class:`Dataset` and will:

        - Disable shuffling for iterable datasets (with a warning if
          ``shuffle=True`` was requested).
        - Apply worker-related options only when valid.

        Args:
            dataset: PyTorch :class:`Dataset` or :class:`IterableDataset`.

        Returns:
            Configured :class:`torch.utils.data.DataLoader` instance.
        """
        is_iterable = isinstance(dataset, IterableDataset)
        effective_shuffle = self._resolve_shuffle(is_iterable=is_iterable)

        loader_kwargs = self._build_loader_kwargs(
            dataset=dataset,
            shuffle=effective_shuffle,
            is_iterable=is_iterable,
        )

        logger.info(
            "Creating DataLoader with config: "
            "batch_size={batch_size}, shuffle={shuffle}, "
            "num_workers={num_workers}, pin_memory={pin_memory}, "
            "drop_last={drop_last}, persistent_workers={persistent_workers}, "
            "prefetch_factor={prefetch_factor}, dataset_type={dtype}",
            batch_size=self.batch_size,
            shuffle=effective_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            dtype="IterableDataset" if is_iterable else "Dataset",
        )

        return DataLoader(**loader_kwargs)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _resolve_shuffle(self, is_iterable: bool) -> bool:
        """Determine effective shuffle flag for the given dataset type."""
        if not is_iterable:
            return self.shuffle

        if self.shuffle:
            logger.warning(
                "Shuffle=True requested but dataset is an IterableDataset. "
                "PyTorch ignores shuffle for iterable datasets; "
                "forcing shuffle=False."
            )
        return False

    def _build_loader_kwargs(
        self,
        dataset: Dataset[T_co] | IterableDataset[T_co],
        shuffle: bool,
        is_iterable: bool,
    ) -> dict[str, Any]:
        """Build kwargs dictionary for :class:`DataLoader` construction."""
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
        }

        # Only set these on modern PyTorch and when workers are actually used.
        if self.num_workers > 0:
            if self.persistent_workers is not None:
                kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.prefetch_factor

        return kwargs
