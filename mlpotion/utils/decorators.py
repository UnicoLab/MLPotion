from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Type

from loguru import logger

from mlpotion.core.exceptions import MLPotionError


class trycatch:
    """Decorator for wrapping methods with unified exception handling and logging.

    Args:
        error: Exception type to raise when unexpected errors occur.
        success_msg: Optional success message to log on method completion.

    Example:
        ```python
        @trycatch(error=DataLoadingError, success_msg="Dataset loaded")
        def load(self):
            ...
        ```
    """

    def __init__(
        self,
        error: Type[Exception],
        success_msg: str | None = None,
    ) -> None:
        self.error = error
        self.success_msg = success_msg

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = func(*args, **kwargs)
                if self.success_msg:
                    logger.info(self.success_msg)
                return result

            except self.error:
                # Let known custom errors propagate unchanged
                raise

            except MLPotionError:
                # Let all MLPotionError subclasses propagate unchanged
                # This includes ExportError, EvaluationError, TrainingError, etc.
                raise

            except (RuntimeError, TypeError, ValueError, AttributeError, KeyError, IndexError):
                # Let standard Python exceptions propagate unchanged
                # These are often used for validation and should not be wrapped
                raise

            except Exception as exc:
                logger.error(
                    f"❌ Unexpected error in {func.__name__}: {exc!s}"
                )
                raise self.error(f"Unexpected error: {exc!s}") from exc

        return wrapper
