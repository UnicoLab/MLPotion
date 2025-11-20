import os
import unittest
import pytest
from pathlib import Path

import torch
import torch.nn as nn

from mlpotion.core.exceptions import ExportError
from mlpotion.core.results import ExportResult
from mlpotion.frameworks.pytorch.config import ModelExportConfig
from mlpotion.frameworks.pytorch.deployment.exporters import ModelExporter
from tests.core import TestBase  # provides temp_dir, setUp/tearDown
import importlib.util

onnx_available = importlib.util.find_spec("onnx") is not None


class SimpleLinearModel(nn.Module):
    """Tiny helper model used in tests."""

    def __init__(self, in_features: int = 4, out_features: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestModelExporter(TestBase):
    def setUp(self) -> None:
        super().setUp()
        self.exporter = ModelExporter()
        self.model = SimpleLinearModel(in_features=4, out_features=2)
        self.example_input = torch.randn(1, 4)  # batch_size=1, in_features=4

    # ------------------------------------------------------------------ #
    # TorchScript exports
    # ------------------------------------------------------------------ #
    @pytest.mark.skip(
        reason="TorchScript script mode may fail with 'could not get source code' in test environments"
    )
    def test_export_torchscript_script_creates_pt_file(self) -> None:
        """TorchScript script mode should create a .pt file and return ExportResult."""
        out_base = self.temp_dir / "script_model"
        config = ModelExportConfig(
            export_path=str(out_base),
            format="torchscript",
            device="cpu",
            jit_mode="script",
        )

        result = self.exporter.export(self.model, config)

        self.assertIsInstance(result, ExportResult)
        self.assertTrue(result.export_path.endswith(".pt"))

        out_path = Path(result.export_path)
        self.assertTrue(out_path.exists(), f"TorchScript file not found at {out_path}")

        # Optional sanity check: load back as TorchScript
        loaded = torch.jit.load(str(out_path))
        self.assertIsInstance(loaded, torch.jit.RecursiveScriptModule)

    def test_export_torchscript_trace_requires_example_input(self) -> None:
        """TorchScript trace mode should fail without example_input."""
        out_base = self.temp_dir / "trace_model"
        config = ModelExportConfig(
            export_path=str(out_base),
            format="torchscript",
            device="cpu",
            jit_mode="trace",
            example_input=None,  # missing
        )

        with self.assertRaises(ExportError):
            _ = self.exporter.export(self.model, config)

    def test_export_torchscript_trace_creates_pt_file(self) -> None:
        """TorchScript trace mode should use example_input and create .pt file."""
        out_base = self.temp_dir / "trace_model"
        config = ModelExportConfig(
            export_path=str(out_base),
            format="torchscript",
            device="cpu",
            jit_mode="trace",
            example_input=self.example_input,
        )

        result = self.exporter.export(self.model, config)

        self.assertIsInstance(result, ExportResult)
        self.assertTrue(result.export_path.endswith(".pt"))

        out_path = Path(result.export_path)
        self.assertTrue(out_path.exists(), f"TorchScript file not found at {out_path}")

        # Load traced model
        loaded = torch.jit.load(str(out_path))
        self.assertIsInstance(loaded, torch.jit.RecursiveScriptModule)

    # ------------------------------------------------------------------ #
    # ONNX exports
    # ------------------------------------------------------------------ #
    @pytest.mark.skipif(not onnx_available, reason="ONNX not installed")
    def test_export_onnx_requires_example_input(self) -> None:
        """ONNX export must fail if example_input is not provided."""
        out_base = self.temp_dir / "onnx_model"
        config = ModelExportConfig(
            export_path=str(out_base),
            format="onnx",
            device="cpu",
            example_input=None,
        )

        with self.assertRaises(ExportError):
            _ = self.exporter.export(self.model, config)

    @pytest.mark.skipif(not onnx_available, reason="ONNX not installed")
    def test_export_onnx_creates_onnx_file(self) -> None:
        """ONNX export should create a .onnx file."""
        out_base = self.temp_dir / "onnx_model"
        config = ModelExportConfig(
            export_path=str(out_base),
            format="onnx",
            device="cpu",
            example_input=self.example_input,
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
        )

        result = self.exporter.export(self.model, config)

        self.assertIsInstance(result, ExportResult)
        self.assertTrue(result.export_path.endswith(".onnx"))

        out_path = Path(result.export_path)
        self.assertTrue(out_path.exists(), f"ONNX file not found at {out_path}")
        self.assertGreater(os.path.getsize(out_path), 0)

    # ------------------------------------------------------------------ #
    # state_dict exports
    # ------------------------------------------------------------------ #
    def test_export_state_dict_creates_pth_file(self) -> None:
        """state_dict export should create a .pth file that can be loaded."""
        out_base = self.temp_dir / "weights"
        config = ModelExportConfig(
            export_path=str(out_base),
            format="state_dict",
            device="cpu",
        )

        result = self.exporter.export(self.model, config)

        self.assertIsInstance(result, ExportResult)
        self.assertTrue(result.export_path.endswith(".pth"))

        out_path = Path(result.export_path)
        self.assertTrue(out_path.exists(), f"state_dict file not found at {out_path}")

        state = torch.load(str(out_path), map_location="cpu")
        self.assertIsInstance(state, dict)
        self.assertSetEqual(set(state.keys()), set(self.model.state_dict().keys()))

    # ------------------------------------------------------------------ #
    # Error handling
    # ------------------------------------------------------------------ #
    def test_export_unknown_format_raises_export_error(self) -> None:
        """Unknown export format should raise ExportError."""
        out_base = self.temp_dir / "unknown"
        config = ModelExportConfig(
            export_path=str(out_base),
            format="weird_format",
            device="cpu",
        )

        with self.assertRaises(ExportError):
            _ = self.exporter.export(self.model, config)

    # ------------------------------------------------------------------ #
    # _move_to_device helper
    # ------------------------------------------------------------------ #
    def test_move_to_device_moves_nested_tensors(self) -> None:
        """_move_to_device should recurse through nested structures."""
        device = torch.device("cpu")

        nested = {
            "a": torch.zeros(1),
            "b": [
                torch.ones(1),
                {"c": torch.randn(1)},
            ],
            "d": "not_a_tensor",
        }

        moved = self.exporter._move_to_device(nested, device=device)

        # Structure preserved
        self.assertIsInstance(moved, dict)
        self.assertIn("a", moved)
        self.assertIn("b", moved)
        self.assertIn("d", moved)

        self.assertIsInstance(moved["a"], torch.Tensor)
        self.assertEqual(moved["a"].device.type, "cpu")

        self.assertIsInstance(moved["b"], list)
        self.assertIsInstance(moved["b"][0], torch.Tensor)
        self.assertEqual(moved["b"][0].device.type, "cpu")

        self.assertIsInstance(moved["b"][1], dict)
        self.assertIsInstance(moved["b"][1]["c"], torch.Tensor)
        self.assertEqual(moved["b"][1]["c"].device.type, "cpu")

        # Non-tensors are unchanged
        self.assertEqual(moved["d"], "not_a_tensor")


if __name__ == "__main__":
    unittest.main()
