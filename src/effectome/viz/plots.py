"""Matplotlib plotting helpers (headless-safe)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_connectivity_matrix(matrix: np.ndarray, title: str, path: str | Path) -> Path:
    """Heatmap of an N x N connectivity matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-np.abs(matrix).max(), vmax=np.abs(matrix).max())
    ax.set_title(title)
    ax.set_xlabel("target neuron")
    ax.set_ylabel("source neuron")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_transition_matrix(p_matrix: np.ndarray, title: str, path: str | Path) -> Path:
    """Heatmap of a connectivity-state transition matrix."""
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(p_matrix, cmap="viridis", vmin=0, vmax=1)
    for i in range(p_matrix.shape[0]):
        for j in range(p_matrix.shape[1]):
            ax.text(j, i, f"{p_matrix[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("to state")
    ax.set_ylabel("from state")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_manifold(embedding: np.ndarray, color: np.ndarray, title: str, path: str | Path) -> Path:
    """Scatter of the first two manifold dimensions, colored by behavior."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="plasma", s=6, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.colorbar(sc, ax=ax, fraction=0.046, label="behavior")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
