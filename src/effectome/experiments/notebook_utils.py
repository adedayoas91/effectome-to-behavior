"""Shared helpers for notebook-based effectome analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_project_root() -> Path:
    """Locate the repository root from a notebook working directory."""
    project_root = next(
        (
            path
            for path in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
            if (path / "src" / "effectome").exists()
        ),
        None,
    )
    if project_root is None:
        raise FileNotFoundError("Could not locate effectome project root")
    return project_root


def summarize_stack(matrices: np.ndarray) -> dict[str, float]:
    """Summarize a stack of connectivity matrices."""
    nonzero = matrices[np.abs(matrices) > 0]
    total = matrices.shape[0] * matrices.shape[1] * matrices.shape[2]
    return {
        "n_windows": int(matrices.shape[0]),
        "n_neurons": int(matrices.shape[1]),
        "density": float(nonzero.size / total) if total else 0.0,
        "mean_abs": float(np.mean(np.abs(matrices))),
        "mean_signed": float(np.mean(matrices)),
        "positive_fraction": float(np.mean(nonzero > 0)) if nonzero.size else 0.0,
        "negative_fraction": float(np.mean(nonzero < 0)) if nonzero.size else 0.0,
    }


def plot_depth_summary(summary_df: pd.DataFrame, x_key: str, title: str):
    """Plot a compact four-panel summary against depth or tau."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ["density", "mean_abs", "positive_fraction", "negative_fraction"]
    titles = ["Edge density", "Mean |weight|", "Positive edge fraction", "Negative edge fraction"]
    for axis, metric, metric_title in zip(axes.flatten(), metrics, titles, strict=True):
        axis.plot(summary_df[x_key], summary_df[metric], marker="o")
        axis.set_xlabel(x_key)
        axis.set_ylabel(metric)
        axis.set_title(metric_title)
        axis.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    return fig


def collapse_tigramite_results(results: dict[str, Any], alpha: float, tau_min: int) -> np.ndarray:
    """Collapse Tigramite lagged outputs into one signed source->target matrix."""
    val = np.asarray(results["val_matrix"], dtype=np.float64)
    pmat = np.asarray(results["p_matrix"], dtype=np.float64)
    lag_slice = slice(max(tau_min, 0), val.shape[2])
    sig = np.where(pmat[:, :, lag_slice] <= alpha, val[:, :, lag_slice], 0.0)
    strongest = np.argmax(np.abs(sig), axis=2)
    rows = np.arange(sig.shape[0])[:, None]
    cols = np.arange(sig.shape[1])[None, :]
    influence = sig[rows, cols, strongest]
    np.fill_diagonal(influence, 0.0)
    return influence


def normalize_multiple_recordings(
    recordings: dict[Any, np.ndarray] | list[np.ndarray] | tuple[np.ndarray, ...],
) -> dict[Any, np.ndarray]:
    """Normalize multiple context recordings into Tigramite's dict format."""
    if isinstance(recordings, dict):
        data_dict = {key: np.asarray(value, dtype=np.float64) for key, value in recordings.items()}
    else:
        data_dict = {idx: np.asarray(value, dtype=np.float64) for idx, value in enumerate(recordings)}
    if not data_dict:
        raise ValueError("recordings must contain at least one dataset.")
    n_vars = next(iter(data_dict.values())).shape[1]
    for key, value in data_dict.items():
        if value.ndim != 2:
            raise ValueError(f"Recording {key!r} must have shape (T, N), got {value.shape}.")
        if value.shape[1] != n_vars:
            raise ValueError("All recordings must share the same number of variables.")
    return data_dict


__all__ = [
    "collapse_tigramite_results",
    "find_project_root",
    "normalize_multiple_recordings",
    "plot_depth_summary",
    "summarize_stack",
]
