"""Shared utilities: reproducibility, IO, and synthetic ground-truth data."""

from .seed import set_seed
from .synthetic import SyntheticConfig, make_synthetic_recording

__all__ = ["set_seed", "SyntheticConfig", "make_synthetic_recording"]
