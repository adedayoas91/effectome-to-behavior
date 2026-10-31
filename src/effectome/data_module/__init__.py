"""Data module: schema, loaders, preprocessing, and windowing (Stages 0-1)."""

from .loaders import get_loader, register_loader
from .preprocess import (
    FittedPreprocessor,
    PreprocessConfig,
    bundle_net_bandpass,
    exclude_named_neurons,
    preprocess,
)
from .schema import (
    ArtifactProvenance,
    CommunitySeries,
    ConnectivitySeries,
    EndpointMetadata,
    ExampleSupport,
    FoldSpec,
    NeuralRecording,
    RawInterval,
    RecordingIdentity,
    TemporalAnchor,
    Window,
    WindowedSegments,
)
from .segmentation import (
    CalciumSegmentationConfig,
    make_overlapping_calcium_windows,
    make_taper,
    segment_calcium_traces,
)
from .windowing import WindowConfig, make_windows

__all__ = [
    "NeuralRecording",
    "RecordingIdentity",
    "ArtifactProvenance",
    "RawInterval",
    "ExampleSupport",
    "FoldSpec",
    "EndpointMetadata",
    "TemporalAnchor",
    "Window",
    "WindowedSegments",
    "ConnectivitySeries",
    "CommunitySeries",
    "get_loader",
    "register_loader",
    "PreprocessConfig",
    "FittedPreprocessor",
    "bundle_net_bandpass",
    "exclude_named_neurons",
    "preprocess",
    "CalciumSegmentationConfig",
    "make_overlapping_calcium_windows",
    "make_taper",
    "segment_calcium_traces",
    "WindowConfig",
    "make_windows",
]
