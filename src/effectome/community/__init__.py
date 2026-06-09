"""Community detection (Stage 4): static, multiscale, and temporal/evolving.

Importing this package registers all shipped detectors.
"""

from . import multiscale as _multiscale  # noqa: E402,F401

# Import side-effect: register detectors.
from . import static as _static  # noqa: E402,F401
from . import temporal as _temporal  # noqa: E402,F401
from .base import (
    COMMUNITY_REGISTRY,
    CommunityConfig,
    CommunityDetector,
    CommunityFactory,
    register_community,
)

__all__ = [
    "CommunityConfig",
    "CommunityDetector",
    "CommunityFactory",
    "register_community",
    "COMMUNITY_REGISTRY",
]
