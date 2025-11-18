from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger

from mlpotion.core.exceptions import ExportError
from mlpotion.core.results import ExportResult
from mlpotion.frameworks.pytorch.config import ModelExportConfig
from mlpotion.core.protocols import ModelExporter as ModelExporterProtocol
from mlpotion.utils import trycatch
from mlpotion.core.exceptions import ModelExporterError


@dataclass(slots=True)
class ModelExporter(ModelExporterProtocol[nn.Module]):
    """Export PyTorch models to TorchScript, ONNX, or state_dict formats.

    Supported formats (`config.format`):

        • `torchscript`
        • `onnx`
        • `state_dict`

    Example:
        ```python
        from mlpotion.frameworks.pytorch import ModelExporter

        exporter = ModelExporter()

        config = ModelExportConfig(
            export_path="models/model",
            format="torchscript",
            jit_mode="script",
        )
        result = exporter.export(model, config)
        ```
    """

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @trycatch(
        error=ModelExporterError,
        success_msg="✅ Successfully Exported model",
    )
    def export(
        self,
        model: nn.Module,
        config: ModelExportConfig,
    ) -> ExportResult:
        """Main entry point for exporting a model.

        Args:
            model: PyTorch model to export.
            config: Export options.

        Returns:
            ExportResult containing final path + metadata.

        Raises:
            ExportError: If any export stage fails.
        """
        try:
            export_root = Path(config.export_path)
            export_root.parent.mkdir(parents=True, exist_ok=True)

            fmt = config.format.lower()
            device_str = getattr(config, "device", "cpu")
            device = torch.device(device_str)

            logger.info(
                "Exporting PyTorch model "
                f"[format={fmt}, device={device_str}, target={export_root}]"
            )

            model = model.to(device)
            model.eval()

            # Dispatch
            if fmt == "torchscript":
                final_path = self._export_torchscript(
                    model=model,
                    export_root=export_root,
                    config=config,
                    device=device,
                )
            elif fmt == "onnx":
                final_path = self._export_onnx(
                    model=model,
                    export_root=export_root,
                    config=config,
                    device=device,
                )
            elif fmt == "state_dict":
                final_path = self._export_state_dict(
                    model=model,
                    export_root=export_root,
                )
            else:
                raise ExportError(f"Unknown export format: {config.format!r}")

            logger.success(f"Model successfully exported → {final_path}")

            metadata: dict[str, Any] = {
                "model_type": "pytorch",
                "format": fmt,
                "device": device_str,
            }

            return ExportResult(
                export_path=str(final_path),
                format=fmt,
                config=config,
                metadata=metadata,
            )

        except ExportError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExportError(f"Export failed: {exc!s}") from exc

    # ------------------------------------------------------------------ #
    # TorchScript
    # ------------------------------------------------------------------ #
    def _export_torchscript(
        self,
        model: nn.Module,
        export_root: Path,
        config: ModelExportConfig,
        device: torch.device,
    ) -> Path:
        """Export model as TorchScript (.pt)."""
        jit_mode = getattr(config, "jit_mode", "script").lower()
        example_input = getattr(config, "example_input", None)

        out_path = (
            export_root
            if export_root.suffix == ".pt"
            else export_root.with_suffix(".pt")
        )

        logger.info(f"[TorchScript] mode={jit_mode}, output={out_path}")

        if jit_mode == "trace":
            if example_input is None:
                raise ExportError(
                    "TorchScript trace mode requires `config.example_input`."
                )
            example_input = self._move_to_device(example_input, device)
            scripted = torch.jit.trace(model, example_input)
        else:
            scripted = torch.jit.script(model)

        scripted.save(str(out_path))
        return out_path

    # ------------------------------------------------------------------ #
    # ONNX
    # ------------------------------------------------------------------ #
    def _export_onnx(
        self,
        model: nn.Module,
        export_root: Path,
        config: ModelExportConfig,
        device: torch.device,
    ) -> Path:
        """Export model as ONNX (.onnx)."""
        example_input = getattr(config, "example_input", None)
        if example_input is None:
            raise ExportError(
                "ONNX export requires `config.example_input` "
                "(one example batch or tensor)."
            )

        example_input = self._move_to_device(example_input, device)

        out_path = (
            export_root
            if export_root.suffix == ".onnx"
            else export_root.with_suffix(".onnx")
        )

        input_names = getattr(config, "input_names", None) or ["input"]
        output_names = getattr(config, "output_names", None) or ["output"]
        dynamic_axes = getattr(config, "dynamic_axes", None)
        opset_version = getattr(config, "opset_version", 13)

        logger.info(
            f"[ONNX] output={out_path}, opset={opset_version}, "
            f"inputs={input_names}, outputs={output_names}, dyn_axes={dynamic_axes}"
        )

        torch.onnx.export(
            model,
            example_input,
            str(out_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        return out_path

    # ------------------------------------------------------------------ #
    # state_dict
    # ------------------------------------------------------------------ #
    def _export_state_dict(
        self,
        model: nn.Module,
        export_root: Path,
    ) -> Path:
        """Export model.parameters() as state_dict (.pth)."""
        out_path = (
            export_root
            if export_root.suffix == ".pth"
            else export_root.with_suffix(".pth")
        )

        logger.info(f"[state_dict] Saving model.state_dict() → {out_path}")
        torch.save(model.state_dict(), str(out_path))

        return out_path

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _move_to_device(self, obj: Any, device: torch.device) -> Any:
        """Recursively move tensors in any nested structure to the given device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True)

        if isinstance(obj, (list, tuple)):
            return type(obj)(self._move_to_device(o, device) for o in obj)

        if isinstance(obj, dict):
            return {k: self._move_to_device(v, device) for k, v in obj.items()}

        return obj
