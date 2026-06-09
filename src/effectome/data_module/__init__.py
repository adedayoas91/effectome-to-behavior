"""Data module: schema, loaders, preprocessing, and windowing (Stages 0-1)."""

from .loaders import get_loader, register_loader
from .preprocess import PreprocessConfig, preprocess
from .schema import (
    CommunitySeries,
    ConnectivitySeries,
    NeuralRecording,
    Window,
    WindowedSegments,
)
from .windowing import WindowConfig, make_windows

__all__ = [
    "NeuralRecording",
    "Window",
    "WindowedSegments",
    "ConnectivitySeries",
    "CommunitySeries",
    "get_loader",
    "register_loader",
    "PreprocessConfig",
    "preprocess",
    "WindowConfig",
    "make_windows",
]
