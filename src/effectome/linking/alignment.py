"""Lead-lag alignment between connectivity dynamics and behavior (tests H4).

Tests whether connectivity-state transitions *precede* behavioral transitions by computing the
cross-correlation between the connectivity-state-change signal and the behavior-change signal as
a function of lag, and reporting the lag of peak association. A positive optimal lag means
connectivity changes lead behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats import discretize


def change_signal(labels: np.ndarray) -> np.ndarray:
    """Binary signal marking where a (discrete) label sequence changes value."""
    lab = np.asarray(labels)
    return np.concatenate([[0], (np.diff(lab) != 0).astype(float)])


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
    """Result of a connectivity-leads-behavior cross-correlation analysis.

    Attributes:
        lags: Lag axis in windows (positive = connectivity leads behavior).
        xcorr: Cross-correlation value per lag.
        best_lag: Lag of maximum |cross-correlation|.
        best_value: Cross-correlation at `best_lag`.
        connectivity_leads: True if best_lag > 0 (connectivity change precedes behavior change).
    """

    lags: np.ndarray
    xcorr: np.ndarray
    best_lag: int
    best_value: float
    connectivity_leads: bool


def lead_lag(
    state_labels: np.ndarray, behavior: np.ndarray, max_lag: int = 10, n_bins: int = 5
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
    )
