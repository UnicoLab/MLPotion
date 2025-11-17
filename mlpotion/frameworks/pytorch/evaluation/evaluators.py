"""PyTorch model evaluation."""

import logging
import time
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mlpotion.core.exceptions import EvaluationError
from mlpotion.core.results import EvaluationResult
from mlpotion.frameworks.pytorch.config import PyTorchEvaluationConfig

logger = logging.getLogger(__name__)


class PyTorchModelEvaluator:
    """Generic evaluator for PyTorch models.

    This evaluator makes minimal assumptions about the model and data:

    - If a batch is `(inputs, targets)`, it uses `loss_fn(outputs, targets)`.
    - If a batch is just `inputs`, it uses `loss_fn(outputs, inputs)`
      (useful for autoencoders / unsupervised setups).
    - Any `nn.Module` is supported as long as its outputs/targets are compatible
      with the configured loss function.

    The loss function can be:
        - a string key (e.g. "mse", "cross_entropy")
        - an `nn.Module` instance
        - a callable `(outputs, targets) -> loss_tensor`

    Optional config attribute:
        - max_batches: Optional[int]
          If present and not None, evaluation will stop after this many batches.
    """

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader[Any],
        config: PyTorchEvaluationConfig,
    ) -> EvaluationResult:
        """Evaluate a PyTorch model.

        Args:
            model: Model to evaluate.
            dataloader: Evaluation DataLoader.
            config: Evaluation configuration.

        Returns:
            EvaluationResult with metrics, config, and evaluation time.

        Raises:
            EvaluationError: If evaluation fails.
        """
        try:
            device_str = getattr(config, "device", "cpu")
            logger.info("Starting PyTorch model evaluation...")
            logger.info(
                f"Config: device={device_str}, loss_fn={getattr(config, 'loss_fn', 'mse')}"
            )

            device = torch.device(device_str)
            model = model.to(device)
            model.eval()

            criterion = self._create_loss_fn(config)

            # Support optional max_batches on the config
            max_batches = getattr(config, "max_batches", None)

            total_loss = 0.0
            num_batches = 0
            start_time = time.time()

            with torch.no_grad():
                for batch in dataloader:
                    inputs, targets = self._prepare_batch(batch, device=device)

                    outputs = model(inputs)

                    if targets is not None:
                        loss = criterion(outputs, targets)
                    else:
                        loss = criterion(outputs, inputs)

                    total_loss += float(loss.item())
                    num_batches += 1

                    if max_batches is not None and num_batches >= max_batches:
                        logger.info(
                            f"Reached max_batches={max_batches}; "
                            "stopping evaluation early."
                        )
                        break

            if num_batches == 0:
                raise EvaluationError("Evaluation dataloader yielded no batches.")

            avg_loss = total_loss / num_batches
            evaluation_time = time.time() - start_time

            metrics = {"loss": float(avg_loss)}

            logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
            logger.info(f"Metrics: {metrics}")

            return EvaluationResult(
                metrics=metrics,
                config=config,
                evaluation_time=evaluation_time,
            )

        except EvaluationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"Evaluation failed: {exc!s}") from exc

    # ------------------------------------------------------------------ #
    # Loss / batch helpers
    # ------------------------------------------------------------------ #
    def _create_loss_fn(
        self,
        config: PyTorchEvaluationConfig,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Create loss function from config.

        Supports:
            - string key (e.g. "mse", "cross_entropy")
            - nn.Module instance
            - callable taking (outputs, targets)
        """
        loss_cfg = getattr(config, "loss_fn", None)

        if isinstance(loss_cfg, nn.Module):
            return loss_cfg

        if callable(loss_cfg):
            return loss_cfg  # type: ignore[return-value]

        name = (str(loss_cfg) if loss_cfg is not None else "mse").lower()

        loss_map: dict[str, nn.Module] = {
            "mse": nn.MSELoss(),
            "l1": nn.L1Loss(),
            "bce": nn.BCELoss(),
            "bce_with_logits": nn.BCEWithLogitsLoss(),
            "cross_entropy": nn.CrossEntropyLoss(),
            "smooth_l1": nn.SmoothL1Loss(),
        }

        if name not in loss_map:
            logger.warning(
                f"Unknown loss_fn '{name}', falling back to MSELoss(). "
                f"Available: {list(loss_map.keys())}"
            )
        return loss_map.get(name, nn.MSELoss())

    def _prepare_batch(
        self,
        batch: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepare batch: move to device, split into (inputs, targets).

        Supported batch formats:
            - (inputs, targets)
            - [inputs, targets]
            - inputs only
            - inputs as list/tuple of tensors (stacked into a batch)
        """
        # Split batch into inputs / targets
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            inputs, targets = batch
        else:
            inputs = batch
            targets = None

        # Move to device
        inputs = self._move_to_device(inputs, device)
        if targets is not None:
            targets = self._move_to_device(targets, device)

        # If inputs is list/tuple of tensors -> stack into a single tensor
        if isinstance(inputs, (list, tuple)):
            elems = list(inputs)
            if elems and all(isinstance(t, torch.Tensor) for t in elems):
                try:
                    inputs = torch.stack(elems, dim=0)
                except RuntimeError:
                    # Fallback: concat unsqueezed tensors
                    inputs = torch.cat([t.unsqueeze(0) for t in elems], dim=0)

        # Similarly, allow targets to be list/tuple of tensors if needed
        if isinstance(targets, (list, tuple)):
            elems = list(targets)
            if elems and all(isinstance(t, torch.Tensor) for t in elems):
                try:
                    targets = torch.stack(elems, dim=0)
                except RuntimeError:
                    targets = torch.cat([t.unsqueeze(0) for t in elems], dim=0)

        if not isinstance(inputs, torch.Tensor):
            raise EvaluationError(
                f"Expected inputs to be a Tensor after device move, got {type(inputs)!r}"
            )
        if targets is not None and not isinstance(targets, torch.Tensor):
            raise EvaluationError(
                f"Expected targets to be a Tensor after device move, got {type(targets)!r}"
            )

        return inputs, targets

    def _move_to_device(self, obj: Any, device: torch.device) -> Any:
        """Recursively move tensors in a structure to the given device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True)

        if isinstance(obj, (list, tuple)):
            return type(obj)(self._move_to_device(o, device) for o in obj)

        if isinstance(obj, dict):
            return {k: self._move_to_device(v, device) for k, v in obj.items()}

        return obj
