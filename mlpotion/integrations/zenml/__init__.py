"""ZenML integration (optional).

Only available if zenml is installed.
"""

try:
    _zenml_available = True
    from mlpotion.integrations.zenml.adapters import ZenMLAdapter

    __all__ = ["ZenMLAdapter"]
except ImportError:
    _zenml_available = False
    __all__ = []
