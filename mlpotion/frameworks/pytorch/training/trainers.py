"""PyTorch model training."""
import time
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mlpotion.core.exceptions import TrainingError
from mlpotion.core.results import TrainingResult
from mlpotion.frameworks.pytorch.config import ModelTrainingConfig
from mlpotion.core.protocols import ModelTrainerProtocol
from mlpotion.utils import trycatch
from mlpotion.core.exceptions import ModelTrainerError
from loguru import logger


class PyTorchModelTrainer(ModelTrainerProtocol[nn.Module]):
    """Generic trainer for PyTorch models.

    This trainer makes minimal assumptions about the model:

    - If a batch is `(inputs, targets)`, it uses `loss_fn(outputs, targets)`.
    - If a batch is just `inputs`, it uses `loss_fn(outputs, inputs)`
      (useful for autoencoders / unsupervised setups).
    - Any `nn.Module` is supported as long as its outputs/targets are compatible
      with the configured loss function.

    Optional config fields (if present on ModelTrainingConfig):
        - max_batches_per_epoch: Optional[int]
          Limit the number of batches processed in each epoch.
        - max_batches: Optional[int]
          Fallback name; if max_batches_per_epoch is not set, this is used.
    """

    @trycatch(
        error=ModelTrainerError,
        success_msg="✅ Successfully trained PyTorch model",
    )
    def train(
        self,
        model: nn.Module,
        dataloader: DataLoader[Any],
        config: ModelTrainingConfig,
        validation_dataloader: DataLoader[Any] | None = None,
    ) -> TrainingResult[nn.Module]:
        """Train a PyTorch model.

        Args:
            model: PyTorch model (nn.Module).
            dataloader: Training DataLoader.
            config: Training configuration.
            validation_dataloader: Optional validation DataLoader.

        Returns:
            TrainingResult: trained model, history, metrics, config, etc.

        Raises:
            TrainingError: If training fails.
        """
        try:
            logger.info("Starting PyTorch model training...")
            logger.info(
                "Config: epochs={epochs}, lr={lr}, optimizer={opt}, "
                "loss_fn={loss_fn}, device={device}",
                epochs=config.epochs,
                lr=config.learning_rate,
                opt=config.optimizer,
                loss_fn=config.loss_fn,
                device=config.device,
            )

            # Setup device
            device = torch.device(config.device)
            model = model.to(device)

            # Setup optimizer and loss
            optimizer = self._create_optimizer(model, config)
            criterion = self._create_loss_fn(config)

            # Optional limit on batches per epoch
            max_batches_per_epoch = getattr(config, "max_batches_per_epoch", None)
            if max_batches_per_epoch is None:
                max_batches_per_epoch = getattr(config, "max_batches", None)

            history: dict[str, list[float]] = {"loss": []}
            if validation_dataloader is not None:
                history["val_loss"] = []

            start_time = time.time()

            for epoch in range(config.epochs):
                model.train()
                epoch_loss = 0.0
                num_batches = 0

                for batch in dataloader:
                    inputs, targets = self._prepare_batch(batch, device=device)

                    optimizer.zero_grad()
                    outputs = model(inputs)

                    # Supervised vs unsupervised / autoencoder
                    if targets is not None:
                        loss = criterion(outputs, targets)
                    else:
                        loss = criterion(outputs, inputs)

                    loss.backward()
                    optimizer.step()

                    epoch_loss += float(loss.item())
                    num_batches += 1

                    if (
                        max_batches_per_epoch is not None
                        and num_batches >= max_batches_per_epoch
                    ):
                        logger.info(
                            "Reached max_batches_per_epoch={mb}; "
                            "stopping epoch {epoch} early.",
                            mb=max_batches_per_epoch,
                            epoch=epoch + 1,
                        )
                        break

                if num_batches == 0:
                    raise TrainingError("Training dataloader yielded no batches.")

                avg_loss = epoch_loss / num_batches
                history["loss"].append(avg_loss)

                # Validation phase
                if validation_dataloader is not None:
                    val_loss = self._validate(
                        model=model,
                        dataloader=validation_dataloader,
                        criterion=criterion,
                        device=device,
                    )
                    history["val_loss"].append(val_loss)
                else:
                    val_loss = None

                # Logging
                if getattr(config, "verbose", True):
                    msg = f"Epoch {epoch + 1}/{config.epochs} - loss: {avg_loss:.4f}"
                    if val_loss is not None:
                        msg += f" - val_loss: {val_loss:.4f}"
                    logger.info(msg)

            training_time = time.time() - start_time

            # Final metrics
            metrics: dict[str, float] = {"loss": float(history["loss"][-1])}
            if "val_loss" in history and history["val_loss"]:
                metrics["val_loss"] = float(history["val_loss"][-1])

            best_epoch = self._find_best_epoch(history)

            logger.info("Training completed in {t:.2f}s", t=training_time)
            logger.info("Final metrics: {metrics}", metrics=metrics)

            return TrainingResult(
                model=model,
                history=history,
                metrics=metrics,
                config=config,
                training_time=training_time,
                best_epoch=best_epoch,
            )

        except TrainingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TrainingError(f"Training failed: {exc!s}") from exc

    # ------------------------------------------------------------------ #
    # Optimizer / loss helpers
    # ------------------------------------------------------------------ #
    def _create_optimizer(
        self,
        model: nn.Module,
        config: ModelTrainingConfig,
    ) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        name = (config.optimizer or "adam").lower()
        lr = config.learning_rate

        if name == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr)
        if name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr)
        if name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr)
        if name == "rmsprop":
            return torch.optim.RMSprop(model.parameters(), lr=lr)

        raise TrainingError(f"Unknown optimizer: {config.optimizer!r}")

    def _create_loss_fn(
        self,
        config: ModelTrainingConfig,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Create loss function from config.

        Supports:
            - string key (e.g. 'mse', 'cross_entropy')
            - nn.Module instance
            - callable taking (outputs, targets)
        """
        loss_cfg = config.loss_fn

        # If user already gave a loss module or callable, use it directly.
        if isinstance(loss_cfg, nn.Module):
            return loss_cfg
        if callable(loss_cfg):
            return loss_cfg  # type: ignore[return-value]

        # Otherwise, treat as string key.
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
                "Unknown loss_fn '{name}', falling back to MSELoss(). "
                "Available: {keys}",
                name=name,
                keys=list(loss_map.keys()),
            )
        return loss_map.get(name, nn.MSELoss())

    # ------------------------------------------------------------------ #
    # Validation / metrics helpers
    # ------------------------------------------------------------------ #
    def _validate(
        self,
        model: nn.Module,
        dataloader: DataLoader[Any],
        criterion: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device,
    ) -> float:
        """Run validation."""
        model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = self._prepare_batch(batch, device=device)
                outputs = model(inputs)

                if targets is not None:
                    loss = criterion(outputs, targets)
                else:
                    loss = criterion(outputs, inputs)

                val_loss += float(loss.item())
                num_batches += 1

        if num_batches == 0:
            logger.warning("Validation dataloader yielded no batches; val_loss is NaN.")
            return float("nan")

        return val_loss / num_batches

    def _find_best_epoch(self, history: dict[str, list[float]]) -> int | None:
        """Find epoch with best validation loss (smallest val_loss)."""
        if "val_loss" in history and history["val_loss"]:
            val_losses = history["val_loss"]
            best_idx = min(enumerate(val_losses), key=lambda x: x[1])[0]
            return best_idx + 1  # 1-based
        return None

    # ------------------------------------------------------------------ #
    # Batch / device helpers
    # ------------------------------------------------------------------ #
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
        # Split
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            inputs, targets = batch
        else:
            inputs = batch
            targets = None

        # Move to device
        inputs = self._move_to_device(inputs, device)
        if targets is not None:
            targets = self._move_to_device(targets, device)

        # If inputs is list/tuple of tensors → stack/concat
        if isinstance(inputs, (list, tuple)):
            elems = list(inputs)
            if elems and all(isinstance(t, torch.Tensor) for t in elems):
                try:
                    inputs = torch.stack(elems, dim=0)
                except RuntimeError:
                    inputs = torch.cat([t.unsqueeze(0) for t in elems], dim=0)

        # If targets is list/tuple of tensors → stack/concat
        if isinstance(targets, (list, tuple)):
            elems = list(targets)
            if elems and all(isinstance(t, torch.Tensor) for t in elems):
                try:
                    targets = torch.stack(elems, dim=0)
                except RuntimeError:
                    targets = torch.cat([t.unsqueeze(0) for t in elems], dim=0)

        # Final type checks
        if not isinstance(inputs, torch.Tensor):
            raise TrainingError(
                f"Expected inputs to be a Tensor after device move, got {type(inputs)!r}"
            )

        if targets is not None and not isinstance(targets, torch.Tensor):
            raise TrainingError(
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

        # Non-tensor: leave as-is
        return obj
