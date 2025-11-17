"""ZenML integration (optional).

Only available if zenml is installed.
"""

try:
    import zenml

    _zenml_available = True
except ImportError:
    _zenml_available = False
    raise ImportError(
        "ZenML is not installed. Install with: pip install mlpotion[zenml]"
    )

from mlpotion.integrations.zenml.adapters import ZenMLAdapter

__all__ = ["ZenMLAdapter"]