"""Connectivity estimation (Stage 2): correlation, c-GC/c-GC*, PCMCI+, JPCMCI+.

Importing this package registers all shipped estimators in the registry.
"""

# Import side-effect: register estimators.
from . import correlation as _correlation  # noqa: E402,F401
from . import granger as _granger  # noqa: E402,F401
from . import pcmci as _pcmci  # noqa: E402,F401
from . import time_varying as _time_varying  # noqa: E402,F401
from .base import ConnectivityConfig, ConnectivityEstimator
from .registry import CONNECTIVITY_REGISTRY, ConnectivityFactory, register_connectivity

__all__ = [
    "ConnectivityConfig",
    "ConnectivityEstimator",
    "ConnectivityFactory",
    "register_connectivity",
    "CONNECTIVITY_REGISTRY",
]
