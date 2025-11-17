import unittest

import torch
from torch.utils.data import Dataset, IterableDataset, RandomSampler, SequentialSampler

from mlpotion.frameworks.pytorch.data.loaders import PyTorchDataLoaderFactory


class _SimpleDataset(Dataset[int]):
    """Small map-style dataset for testing."""

    def __init__(self, n: int = 10) -> None:
        self._data: list[int] = list(range(n))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> int:
        return self._data[idx]


class _SimpleIterableDataset(IterableDataset[int]):
    """Small iterable dataset for testing."""

    def __init__(self, n: int = 10) -> None:
        super().__init__()
        self._n = n

    def __iter__(self):
        for i in range(self._n):
            yield i


class TestPyTorchDataLoaderFactory(unittest.TestCase):
    # ------------------------------------------------------------------ #
    # Map-style Dataset behaviour
    # ------------------------------------------------------------------ #
    def test_create_with_map_dataset_shuffle_true_uses_random_sampler(self) -> None:
        """With shuffle=True and a map-style Dataset, RandomSampler should be used."""
        dataset = _SimpleDataset(n=10)
        factory = PyTorchDataLoaderFactory[int](
            batch_size=4,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

        loader = factory.create(dataset)

        self.assertEqual(loader.batch_size, 4)
        self.assertTrue(loader.drop_last)
        # For map-style datasets, DataLoader exposes a sampler
        self.assertIsInstance(loader.sampler, RandomSampler)

    def test_create_with_map_dataset_shuffle_false_uses_sequential_sampler(self) -> None:
        """With shuffle=False and a map-style Dataset, SequentialSampler should be used."""
        dataset = _SimpleDataset(n=10)
        factory = PyTorchDataLoaderFactory[int](
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

        loader = factory.create(dataset)

        self.assertEqual(loader.batch_size, 4)
        self.assertIsInstance(loader.sampler, SequentialSampler)

    # ------------------------------------------------------------------ #
    # IterableDataset behaviour
    # ------------------------------------------------------------------ #
    def test_resolve_shuffle_for_iterable_forces_false(self) -> None:
        """_resolve_shuffle should force shuffle=False for IterableDataset."""
        factory = PyTorchDataLoaderFactory[int](shuffle=True)
        # Directly test helper to avoid depending on internal DataLoader details
        effective = factory._resolve_shuffle(is_iterable=True)
        self.assertFalse(effective)

    def test_create_with_iterable_dataset_iterates_all_items(self) -> None:
        """create() with IterableDataset should still produce a working DataLoader."""
        dataset = _SimpleIterableDataset(n=10)
        factory = PyTorchDataLoaderFactory[int](batch_size=4, shuffle=True)

        loader = factory.create(dataset)
        # Items will be batched; each batch is a tensor of shape (batch_size,)
        all_items: list[int] = []
        for batch in loader:
            # batch is a tensor; extend with its contents
            all_items.extend(batch.tolist())

        self.assertEqual(len(all_items), 10)
        self.assertListEqual(sorted(all_items), list(range(10)))

    # ------------------------------------------------------------------ #
    # Worker-related options in _build_loader_kwargs
    # ------------------------------------------------------------------ #
    def test_build_loader_kwargs_omits_worker_options_when_num_workers_zero(self) -> None:
        """persistent_workers & prefetch_factor should not be set when num_workers == 0."""
        dataset = _SimpleDataset()
        factory = PyTorchDataLoaderFactory[int](
            batch_size=8,
            shuffle=True,
            num_workers=0,
            persistent_workers=True,
            prefetch_factor=3,
        )

        kwargs = factory._build_loader_kwargs(
            dataset=dataset,
            shuffle=True,
            is_iterable=False,
        )

        self.assertEqual(kwargs["batch_size"], 8)
        self.assertEqual(kwargs["shuffle"], True)
        self.assertEqual(kwargs["num_workers"], 0)
        self.assertNotIn("persistent_workers", kwargs)
        self.assertNotIn("prefetch_factor", kwargs)

    def test_build_loader_kwargs_sets_worker_options_when_num_workers_positive(self) -> None:
        """persistent_workers & prefetch_factor should be passed when num_workers > 0."""
        dataset = _SimpleDataset()
        factory = PyTorchDataLoaderFactory[int](
            batch_size=8,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=4,
        )

        kwargs = factory._build_loader_kwargs(
            dataset=dataset,
            shuffle=False,
            is_iterable=False,
        )

        self.assertEqual(kwargs["batch_size"], 8)
        self.assertEqual(kwargs["shuffle"], False)
        self.assertEqual(kwargs["num_workers"], 2)
        self.assertIn("persistent_workers", kwargs)
        self.assertIn("prefetch_factor", kwargs)
        self.assertTrue(kwargs["persistent_workers"])
        self.assertEqual(kwargs["prefetch_factor"], 4)

    # ------------------------------------------------------------------ #
    # Sanity check: batches shape/content for map-style dataset
    # ------------------------------------------------------------------ #
    def test_create_with_map_dataset_produces_correct_batch_shapes(self) -> None:
        """DataLoader produced by factory should yield batched tensors of expected shape."""
        dataset = _SimpleDataset(n=9)
        factory = PyTorchDataLoaderFactory[int](batch_size=4, shuffle=False)

        loader = factory.create(dataset)
        batches = list(loader)

        # 9 elements with batch_size=4 -> 3 batches: 4, 4, 1
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0].shape[0], 4)
        self.assertEqual(batches[1].shape[0], 4)
        self.assertEqual(batches[2].shape[0], 1)

        # Check that the content is sequential since shuffle=False
        concatenated = torch.cat(batches).tolist()
        self.assertListEqual(concatenated, list(range(9)))


if __name__ == "__main__":
    unittest.main()
