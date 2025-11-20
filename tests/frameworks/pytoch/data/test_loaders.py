import os
import tempfile
import unittest

import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset, RandomSampler, SequentialSampler

from mlpotion.frameworks.pytorch.data.loaders import (
    CSVDataset,
    StreamingCSVDataset,
    CSVDataLoader,
)


# --------------------------------------------------------------------------- #
# Simple in-memory datasets for DataLoaderFactory tests
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Tests for CSVDataLoader
# --------------------------------------------------------------------------- #
class TestCSVDataLoader(unittest.TestCase):
    # ------------------------------------------------------------------ #
    # Map-style Dataset behaviour
    # ------------------------------------------------------------------ #
    def test_load_with_map_dataset_shuffle_true_uses_random_sampler(self) -> None:
        """With shuffle=True and a map-style Dataset, RandomSampler should be used."""
        dataset = _SimpleDataset(n=10)
        factory = CSVDataLoader[int](
            batch_size=4,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

        loader = factory.load(dataset)

        self.assertEqual(loader.batch_size, 4)
        self.assertTrue(loader.drop_last)
        # For map-style datasets, DataLoader exposes a sampler
        self.assertIsInstance(loader.sampler, RandomSampler)

    def test_load_with_map_dataset_shuffle_false_uses_sequential_sampler(self) -> None:
        """With shuffle=False and a map-style Dataset, SequentialSampler should be used."""
        dataset = _SimpleDataset(n=10)
        factory = CSVDataLoader[int](
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

        loader = factory.load(dataset)

        self.assertEqual(loader.batch_size, 4)
        self.assertIsInstance(loader.sampler, SequentialSampler)

    # ------------------------------------------------------------------ #
    # IterableDataset behaviour
    # ------------------------------------------------------------------ #
    def test_resolve_shuffle_for_iterable_forces_false(self) -> None:
        """_resolve_shuffle should force shuffle=False for IterableDataset."""
        factory = CSVDataLoader[int](shuffle=True)
        # Directly test helper to avoid depending on internal DataLoader details
        effective = factory._resolve_shuffle(is_iterable=True)
        self.assertFalse(effective)

    def test_load_with_iterable_dataset_iterates_all_items(self) -> None:
        """load() with IterableDataset should still produce a working DataLoader."""
        dataset = _SimpleIterableDataset(n=10)
        factory = CSVDataLoader[int](batch_size=4, shuffle=True)

        loader = factory.load(dataset)
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
    def test_build_loader_kwargs_omits_worker_options_when_num_workers_zero(
        self,
    ) -> None:
        """persistent_workers & prefetch_factor should not be set when num_workers == 0."""
        dataset = _SimpleDataset()
        factory = CSVDataLoader[int](
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

    def test_build_loader_kwargs_sets_worker_options_when_num_workers_positive(
        self,
    ) -> None:
        """persistent_workers & prefetch_factor should be passed when num_workers > 0."""
        dataset = _SimpleDataset()
        factory = CSVDataLoader[int](
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
    def test_load_with_map_dataset_produces_correct_batch_shapes(self) -> None:
        """DataLoader produced by factory should yield batched tensors of expected shape."""
        dataset = _SimpleDataset(n=9)
        factory = CSVDataLoader[int](batch_size=4, shuffle=False)

        loader = factory.load(dataset)
        batches = list(loader)

        # 9 elements with batch_size=4 -> 3 batches: 4, 4, 1
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0].shape[0], 4)
        self.assertEqual(batches[1].shape[0], 4)
        self.assertEqual(batches[2].shape[0], 1)

        # Check that the content is sequential since shuffle=False
        concatenated = torch.cat(batches).tolist()
        self.assertListEqual(concatenated, list(range(9)))


# --------------------------------------------------------------------------- #
# Tests for CSVDataset
# --------------------------------------------------------------------------- #
class TestCSVDataset(unittest.TestCase):
    def _create_temp_csvs(self, tmpdir: str) -> list[str]:
        """Create a couple of small CSV files in tmpdir and return their paths."""
        data1 = pd.DataFrame(
            {
                "f1": [1.0, 2.0],
                "f2": [10.0, 20.0],
                "label": [0.0, 1.0],
            }
        )
        data2 = pd.DataFrame(
            {
                "f1": [3.0, 4.0],
                "f2": [30.0, 40.0],
                "label": [0.0, 1.0],
            }
        )
        paths: list[str] = []
        p1 = os.path.join(tmpdir, "part1.csv")
        p2 = os.path.join(tmpdir, "part2.csv")
        data1.to_csv(p1, index=False)
        data2.to_csv(p2, index=False)
        paths.extend([p1, p2])
        return paths

    def test_pytorch_csv_dataset_loads_files_and_splits_features_and_labels(
        self,
    ) -> None:
        """CSVDataset should load CSVs, expose correct length and tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_temp_csvs(tmpdir)
            pattern = os.path.join(tmpdir, "part*.csv")

            dataset = CSVDataset(
                file_pattern=pattern,
                label_name="label",
            )

            # 2 files × 2 rows each = 4 rows total
            self.assertEqual(len(dataset), 4)
            self.assertListEqual(dataset._feature_cols, ["f1", "f2"])
            self.assertIsNotNone(dataset._features_df)
            self.assertIsNotNone(dataset._labels)

            # Fetch first sample; expect (features, label)
            item0 = dataset[0]
            self.assertIsInstance(item0, tuple)
            x0, y0 = item0
            self.assertIsInstance(x0, torch.Tensor)
            self.assertIsInstance(y0, torch.Tensor)
            self.assertEqual(x0.shape, (2,))  # two feature columns
            # label is scalar → 0-dim tensor
            self.assertEqual(y0.shape, torch.Size([]))

    def test_pytorch_csv_dataset_without_label_returns_only_features(self) -> None:
        """If no label_name is given, __getitem__ should return only feature tensor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_temp_csvs(tmpdir)
            pattern = os.path.join(tmpdir, "part*.csv")

            dataset = CSVDataset(
                file_pattern=pattern,
                label_name=None,
            )

            self.assertEqual(len(dataset), 4)
            item0 = dataset[0]
            self.assertIsInstance(item0, torch.Tensor)
            # All columns (f1, f2, label) are treated as features when label_name=None
            self.assertEqual(item0.shape, (3,))

    def test_pytorch_csv_dataset_column_selection_and_missing_column_error(
        self,
    ) -> None:
        """column_names should restrict features and raise for missing columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_temp_csvs(tmpdir)
            pattern = os.path.join(tmpdir, "part*.csv")

            # Column selection
            dataset = CSVDataset(
                file_pattern=pattern,
                column_names=["f1"],
                label_name="label",
            )
            self.assertListEqual(dataset._feature_cols, ["f1"])
            x0, _ = dataset[0]
            self.assertEqual(x0.shape, (1,))

            # Missing requested column
            with self.assertRaises(Exception):
                _ = CSVDataset(
                    file_pattern=pattern,
                    column_names=["does_not_exist"],
                    label_name="label",
                )

    def test_pytorch_csv_dataset_raises_on_no_files_found(self) -> None:
        """Dataset should raise if the glob pattern matches no files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pattern = os.path.join(tmpdir, "nonexistent_*.csv")
            with self.assertRaises(Exception):
                _ = CSVDataset(
                    file_pattern=pattern,
                    label_name="label",
                )


# --------------------------------------------------------------------------- #
# Tests for StreamingCSVDataset
# --------------------------------------------------------------------------- #
class TestStreamingCSVDataset(unittest.TestCase):
    def _create_temp_csvs(self, tmpdir: str) -> list[str]:
        data1 = pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0],
                "f2": [10.0, 20.0, 30.0],
                "label": [0.0, 1.0, 0.0],
            }
        )
        data2 = pd.DataFrame(
            {
                "f1": [4.0, 5.0],
                "f2": [40.0, 50.0],
                "label": [1.0, 0.0],
            }
        )
        p1 = os.path.join(tmpdir, "s1.csv")
        p2 = os.path.join(tmpdir, "s2.csv")
        data1.to_csv(p1, index=False)
        data2.to_csv(p2, index=False)
        return [p1, p2]

    def test_streaming_dataset_yields_all_samples_with_labels(self) -> None:
        """StreamingCSVDataset should stream all rows across all files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_temp_csvs(tmpdir)
            pattern = os.path.join(tmpdir, "s*.csv")

            dataset = StreamingCSVDataset(
                file_pattern=pattern,
                label_name="label",
                chunksize=2,  # test chunked reading
            )

            samples = list(dataset)
            # 3 + 2 rows = 5 total samples
            self.assertEqual(len(samples), 5)

            for sample in samples:
                self.assertIsInstance(sample, tuple)
                x, y = sample
                self.assertIsInstance(x, torch.Tensor)
                self.assertIsInstance(y, torch.Tensor)
                self.assertEqual(x.shape, (2,))  # two feature columns
                # y is scalar → 0-dim tensor
                self.assertEqual(y.shape, torch.Size([]))

    def test_streaming_dataset_without_label_yields_only_features(self) -> None:
        """If label_name is None, streaming dataset should yield only feature tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_temp_csvs(tmpdir)
            pattern = os.path.join(tmpdir, "s*.csv")

            dataset = StreamingCSVDataset(
                file_pattern=pattern,
                label_name=None,
                chunksize=3,
            )

            samples = list(dataset)
            self.assertEqual(len(samples), 5)
            for x in samples:
                self.assertIsInstance(x, torch.Tensor)
                # All columns (f1, f2, label) are treated as features when label_name=None
                self.assertEqual(x.shape, (3,))

    def test_streaming_dataset_raises_when_label_missing_in_file(self) -> None:
        """If label_name is configured but missing in CSV, we should raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a CSV without 'label' column
            df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
            path = os.path.join(tmpdir, "no_label.csv")
            df.to_csv(path, index=False)
            pattern = os.path.join(tmpdir, "no_label.csv")

            dataset = StreamingCSVDataset(
                file_pattern=pattern,
                label_name="label",
                chunksize=2,
            )

            with self.assertRaises(Exception):
                _ = list(dataset)

    def test_streaming_dataset_raises_on_no_files_found(self) -> None:
        """Streaming dataset should raise if file pattern matches no files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pattern = os.path.join(tmpdir, "nothing_here_*.csv")
            with self.assertRaises(Exception):
                _ = StreamingCSVDataset(
                    file_pattern=pattern,
                    label_name="label",
                )


if __name__ == "__main__":
    unittest.main()
