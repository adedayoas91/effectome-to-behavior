"""Chunk-resumable connectivity estimation for method-specific notebooks."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from effectome.connectivity.base import ConnectivityEstimator
from effectome.data_module.schema import ConnectivitySeries, WindowedSegments

from .resume import ResumableRun, content_fingerprint


def _slice_segments(segments: WindowedSegments, start: int, stop: int) -> WindowedSegments:
    return WindowedSegments(
        segments=np.asarray(segments.segments[start:stop]),
        windows=list(segments.windows[start:stop]),
        behavior_per_window={
            key: np.asarray(values[start:stop]) for key, values in segments.behavior_per_window.items()
        },
        n_neurons=segments.n_neurons,
        fps=segments.fps,
        metadata={**segments.metadata, "chunk_start": start, "chunk_stop": stop},
        anchors=list(segments.anchors[start:stop]) if segments.anchors else [],
        provenance=segments.provenance,
    )


def _combine_series(chunks: list[ConnectivitySeries]) -> ConnectivitySeries:
    if not chunks:
        raise ValueError("at least one connectivity chunk is required")
    first = chunks[0]
    if any(
        chunk.method != first.method
        or chunk.directed != first.directed
        or chunk.weight_semantics != first.weight_semantics
        for chunk in chunks[1:]
    ):
        raise ValueError("connectivity chunks have incompatible estimator semantics")
    runtime_seconds = sum(
        float(chunk.diagnostics.get("runtime", {}).get("elapsed_seconds", 0.0))
        for chunk in chunks
    )
    matrices = np.concatenate([chunk.matrices for chunk in chunks], axis=0)
    lagged_matrices = None
    if all(chunk.lagged_matrices is not None for chunk in chunks):
        lagged_matrices = np.concatenate(
            [np.asarray(chunk.lagged_matrices) for chunk in chunks], axis=0
        )
    lag_resolved = [
        item
        for chunk in chunks
        for item in chunk.diagnostics.get("lag_resolved", [])
    ]
    return ConnectivitySeries(
        matrices=matrices,
        window_starts=np.concatenate([chunk.window_starts for chunk in chunks]),
        method=first.method,
        directed=first.directed,
        behavior_per_window={
            key: np.concatenate([chunk.behavior_per_window[key] for chunk in chunks])
            for key in first.behavior_per_window
        },
        anchors=[anchor for chunk in chunks for anchor in chunk.anchors],
        signed=first.signed,
        weighted=first.weighted,
        storage=first.storage,
        weight_semantics=first.weight_semantics,
        diagnostics={
            **first.diagnostics,
            **({"lag_resolved": lag_resolved} if lag_resolved else {}),
            "resume": {
                "chunk_count": len(chunks),
                "chunk_sizes": [chunk.n_windows for chunk in chunks],
                "estimator_runtime_seconds": runtime_seconds,
            },
            "n_windows": int(matrices.shape[0]),
        },
        lagged_matrices=lagged_matrices,
        provenance=first.provenance,
    )


def estimate_connectivity_resumable(
    run: ResumableRun,
    estimator: ConnectivityEstimator,
    segments: WindowedSegments,
    *,
    chunk_size: int = 16,
    input_artifacts: dict[str, str | Path] | None = None,
    force: bool = False,
) -> ConnectivitySeries:
    """Estimate independent windows in atomic chunks and resume completed chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunks = [
        (start, min(start + chunk_size, segments.n_windows))
        for start in range(0, segments.n_windows, chunk_size)
    ]
    config = {
        "estimator": asdict(estimator.cfg),
        "chunk_size": chunk_size,
        "n_windows": segments.n_windows,
        "n_neurons": segments.n_neurons,
        "window_mode": segments.metadata.get("window_mode"),
        "input_content": content_fingerprint(
            {
                "segments": segments.segments,
                "windows": segments.windows,
                "anchors": segments.anchors,
                "behavior": segments.behavior_per_window,
                "provenance": segments.provenance,
            }
        ),
    }
    return run.execute_chunks(
        "connectivity",
        chunks,
        lambda bounds: estimator.run(_slice_segments(segments, bounds[0], bounds[1])),
        _combine_series,
        config=config,
        dependencies=input_artifacts,
        force=force,
    )
