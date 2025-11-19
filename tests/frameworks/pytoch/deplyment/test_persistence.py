import unittest

import torch
import torch.nn as nn

from mlpotion.core.exceptions import ModelPersistenceError
from mlpotion.frameworks.pytorch.deployment.persistence import ModelPersistence
from tests.core import TestBase  # provides temp_dir, setUp/tearDown


class SmallNet(nn.Module):
    """Simple toy model for persistence tests."""

    def __init__(self, in_features: int = 4, out_features: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestModelPersistence(TestBase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(123)
        self.model = SmallNet(in_features=4, out_features=2)

    # ------------------------------------------------------------------ #
    # save(): state_dict
    # ------------------------------------------------------------------ #
    def test_save_state_dict_creates_file_and_loads_back(self) -> None:
        """Saving a state_dict and loading it back should preserve weights."""
        path = self.temp_dir / "model_state.pth"

        # Save state_dict
        persistence_save = ModelPersistence(path=path, model=self.model)
        persistence_save.save(save_full_model=False)

        self.assertTrue(path.exists(), f"Expected state_dict file at {path}")

        # Load into a fresh model instance
        persistence_load = ModelPersistence(path=path)
        loaded_model, metadata = persistence_load.load(model_class=SmallNet)

        self.assertIsInstance(loaded_model, SmallNet)

        # Check that parameters match
        for p_orig, p_loaded in zip(self.model.parameters(), loaded_model.parameters()):
            self.assertTrue(
                torch.allclose(p_orig, p_loaded),
                "Loaded parameters do not match original state_dict",
            )

    # ------------------------------------------------------------------ #
    # save(): full model
    # ------------------------------------------------------------------ #
    def test_save_full_model_and_load_back(self) -> None:
        """Saving a full model and loading it back should return nn.Module."""
        path = self.temp_dir / "model_full.pt"

        persistence_save = ModelPersistence(path=path, model=self.model)
        persistence_save.save(save_full_model=True)

        self.assertTrue(path.exists(), f"Expected full model file at {path}")

        persistence_load = ModelPersistence(path=path)
        loaded_model, metadata = persistence_load.load()

        self.assertIsInstance(loaded_model, nn.Module)
        self.assertIsInstance(loaded_model, SmallNet)

    # ------------------------------------------------------------------ #
    # save(): error when model is missing
    # ------------------------------------------------------------------ #
    def test_save_without_model_attached_raises(self) -> None:
        """Calling save without an attached model should raise ModelPersistenceError."""
        path = self.temp_dir / "no_model.pth"
        persistence = ModelPersistence(path=path)

        with self.assertRaises(ModelPersistenceError):
            persistence.save()

    # ------------------------------------------------------------------ #
    # load(): error when file does not exist
    # ------------------------------------------------------------------ #
    def test_load_missing_file_raises(self) -> None:
        """Loading from a non-existing path should raise ModelPersistenceError."""
        path = self.temp_dir / "missing.pth"
        persistence = ModelPersistence(path=path)

        with self.assertRaises(ModelPersistenceError):
            _ = persistence.load(model_class=SmallNet)

    # ------------------------------------------------------------------ #
    # load(): state_dict requires model_class if no attached model
    # ------------------------------------------------------------------ #
    def test_load_state_dict_requires_model_class_when_no_attached_model(self) -> None:
        """State_dict checkpoints require model_class if no model is attached."""
        path = self.temp_dir / "requires_model_class.pth"
        torch.save(self.model.state_dict(), path)

        persistence = ModelPersistence(path=path)

        with self.assertRaises(ModelPersistenceError):
            _ = persistence.load()

    # ------------------------------------------------------------------ #
    # load(): uses attached model if present for state_dict
    # ------------------------------------------------------------------ #
    def test_load_uses_attached_model_for_state_dict(self) -> None:
        """When model is attached, load() should reuse it for state_dict loading."""
        path = self.temp_dir / "attached_model.pth"

        # Save checkpoint directly as state_dict
        original_state = self.model.state_dict()
        torch.save(original_state, path)

        # New model with different random weights
        torch.manual_seed(999)
        attached_model = SmallNet(in_features=4, out_features=2)

        persistence = ModelPersistence(path=path, model=attached_model)
        loaded_model, metadata = persistence.load()  # no model_class → uses attached

        self.assertIs(loaded_model, attached_model)

        # Loaded model should now match original state_dict
        for (name_orig, p_orig), (name_loaded, p_loaded) in zip(
            original_state.items(), loaded_model.state_dict().items()
        ):
            self.assertEqual(name_orig, name_loaded)
            self.assertTrue(
                torch.allclose(p_orig, p_loaded),
                f"Parameter {name_orig} does not match after loading into attached model",
            )

    # ------------------------------------------------------------------ #
    # load(): wrapped checkpoint with 'model_state_dict'
    # ------------------------------------------------------------------ #
    def test_load_uses_wrapped_model_state_dict_if_present(self) -> None:
        """If checkpoint has 'model_state_dict', it should be used as state_dict."""
        path = self.temp_dir / "wrapped_state_dict.pth"

        state_dict = self.model.state_dict()
        checkpoint = {"model_state_dict": state_dict, "epoch": 3}
        torch.save(checkpoint, path)

        persistence = ModelPersistence(path=path)
        loaded_model, metadata = persistence.load(model_class=SmallNet)

        self.assertIsInstance(loaded_model, SmallNet)

        for (name_orig, p_orig), (name_loaded, p_loaded) in zip(
            state_dict.items(), loaded_model.state_dict().items()
        ):
            self.assertEqual(name_orig, name_loaded)
            self.assertTrue(
                torch.allclose(p_orig, p_loaded),
                f"Parameter {name_orig} does not match from wrapped state_dict",
            )

    # ------------------------------------------------------------------ #
    # load(): unsupported checkpoint type
    # ------------------------------------------------------------------ #
    def test_load_unsupported_checkpoint_type_raises(self) -> None:
        """Non-module and non-dict checkpoint should raise ModelPersistenceError."""
        path = self.temp_dir / "unsupported_checkpoint.pth"

        # Save a simple string as checkpoint
        torch.save("not a model or dict", path)

        persistence = ModelPersistence(path=path)

        with self.assertRaises(ModelPersistenceError):
            _ = persistence.load()

    # ------------------------------------------------------------------ #
    # load(): _instantiate_model raises on bad kwargs
    # ------------------------------------------------------------------ #
    def test_load_fails_if_model_kwargs_do_not_match_constructor(self) -> None:
        """Bad model_kwargs should result in ModelPersistenceError."""
        path = self.temp_dir / "bad_kwargs.pth"

        torch.save(self.model.state_dict(), path)

        persistence = ModelPersistence(path=path)

        with self.assertRaises(ModelPersistenceError):
            _ = persistence.load(model_class=SmallNet, model_kwargs={"unknown_arg": 123})


if __name__ == "__main__":
    unittest.main()
