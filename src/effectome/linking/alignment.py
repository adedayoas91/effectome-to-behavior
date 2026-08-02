"""Lead-lag alignment between connectivity dynamics, manifold dynamics, and behavior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats import discretize


def change_signal(labels: np.ndarray) -> np.ndarray:
    """Binary signal marking where a (discrete) label sequence changes value."""
    lab = np.asarray(labels)
    return np.concatenate([[0], (np.diff(lab) != 0).astype(float)])


def manifold_velocity(embedding: np.ndarray) -> np.ndarray:
    """Per-window latent displacement vectors."""
    emb = np.asarray(embedding, dtype=float)
    vel = np.zeros_like(emb)
    vel[1:] = np.diff(emb, axis=0)
    return vel


def manifold_speed(embedding: np.ndarray) -> np.ndarray:
    """Euclidean speed of the window-aligned manifold trajectory."""
    vel = manifold_velocity(embedding)
    return np.linalg.norm(vel, axis=1)


def _normalized_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    a = (a - a.mean()) / (a.std() or 1e-12)
    b = (b - b.mean()) / (b.std() or 1e-12)
    lags = np.arange(-max_lag, max_lag + 1)
    out = np.empty(len(lags))
    n = len(a)
    for i, lag in enumerate(lags):
        if lag < 0:
            out[i] = np.mean(a[-lag:] * b[: n + lag])
        elif lag > 0:
            out[i] = np.mean(a[: n - lag] * b[lag:])
        else:
            out[i] = np.mean(a * b)
    return lags, out


@dataclass
class LeadLagResult:
    """Result of a connectivity-leads-target cross-correlation analysis."""

    lags: np.ndarray
    xcorr: np.ndarray
    best_lag: int
    best_value: float
    connectivity_leads: bool
    target_name: str = "behavior"


def lead_lag(
    state_labels: np.ndarray,
    behavior: np.ndarray,
    max_lag: int = 10,
    n_bins: int = 5,
    target_name: str = "behavior",
) -> LeadLagResult:
    """Cross-correlate connectivity-state changes against behavior changes over lags."""
    conn_change = change_signal(state_labels)
    beh_change = change_signal(discretize(behavior, n_bins))
    lags, xc = _normalized_xcorr(conn_change, beh_change, max_lag)
    best_idx = int(np.argmax(np.abs(xc)))
    best_lag = int(lags[best_idx])
    return LeadLagResult(
        lags=lags,
        xcorr=xc,
        best_lag=best_lag,
        best_value=float(xc[best_idx]),
        connectivity_leads=best_lag > 0,
        target_name=target_name,
    )
