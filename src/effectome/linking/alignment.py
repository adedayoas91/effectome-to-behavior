"""Lead-lag alignment between connectivity dynamics, manifold dynamics, and behavior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from effectome.data_module.schema import TemporalAnchor

from .stats import discretize


def combined_continuity_groups(
    groups: np.ndarray | None,
    strata: np.ndarray | None,
    n_samples: int,
) -> np.ndarray | None:
    """Create 1D group keys that break derivatives at either group or stratum changes."""
    if groups is None and strata is None:
        return None
    group_arr = np.zeros(n_samples, dtype=int) if groups is None else np.asarray(groups, dtype=object)
    strata_arr = (
        np.zeros(n_samples, dtype=int) if strata is None else np.asarray(strata, dtype=object)
    )
    if group_arr.shape[0] != n_samples or strata_arr.shape[0] != n_samples:
        raise ValueError("groups and strata must align with the sequence")
    combined = np.empty(n_samples, dtype=object)
    for index in range(n_samples):
        combined[index] = (group_arr[index], strata_arr[index])
    return combined


def change_signal(labels: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """Binary signal marking where a (discrete) label sequence changes value."""
    lab = np.asarray(labels)
    changed = (np.diff(lab) != 0).astype(float)
    if groups is not None:
        group_arr = np.asarray(groups)
        if group_arr.shape[0] != lab.shape[0]:
            raise ValueError("groups must align with labels")
        changed[group_arr[1:] != group_arr[:-1]] = 0.0
    return np.concatenate([[0], changed])


def manifold_velocity(embedding: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """Per-window latent displacement vectors."""
    emb = np.asarray(embedding, dtype=float)
    vel = np.zeros_like(emb)
    vel[1:] = np.diff(emb, axis=0)
    if groups is not None:
        group_arr = np.asarray(groups)
        if group_arr.shape[0] != emb.shape[0]:
            raise ValueError("groups must align with embedding")
        vel[1:][group_arr[1:] != group_arr[:-1]] = 0.0
    return vel


def manifold_speed(embedding: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """Euclidean speed of the window-aligned manifold trajectory."""
    vel = manifold_velocity(embedding, groups=groups)
    return np.linalg.norm(vel, axis=1)


def future_manifold_displacement(
    embedding: np.ndarray,
    origins: np.ndarray,
    *,
    lag: int = 1,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return future latent displacements and their chart-invariant magnitudes.

    Each row is ``z[k + lag] - z[k]`` for a supplied effectome origin ``k``.
    When fold/recording groups are provided, crossings between coordinate charts or
    recording segments are rejected rather than silently differenced.
    """

    emb = np.asarray(embedding, dtype=float)
    origin_arr = np.asarray(origins, dtype=int)
    if emb.ndim != 2:
        raise ValueError("embedding must have shape (anchor, latent_dimension)")
    if lag <= 0:
        raise ValueError("lag must be strictly positive")
    if origin_arr.ndim != 1:
        raise ValueError("origins must be a 1D integer array")
    if origin_arr.size and (origin_arr.min() < 0 or origin_arr.max() + lag >= emb.shape[0]):
        raise ValueError("origins and lag must select complete future embedding rows")
    if groups is not None:
        group_arr = np.asarray(groups, dtype=object)
        if group_arr.shape[0] != emb.shape[0]:
            raise ValueError("groups must align with embedding")
        if np.any(group_arr[origin_arr] != group_arr[origin_arr + lag]):
            raise ValueError("future manifold displacement cannot cross a continuity group")
    displacement = emb[origin_arr + lag] - emb[origin_arr]
    return displacement, np.linalg.norm(displacement, axis=1)


def activity_magnitude_features(
    time_by_neuron: np.ndarray,
    anchors: list[TemporalAnchor] | tuple[TemporalAnchor, ...],
) -> np.ndarray:
    """Summarize population activity over each anchor's target interval.

    Columns are mean absolute activity, root-mean-square activity, and the
    absolute population mean.  These nuisance covariates help distinguish
    connectivity-linked effects from changes in overall activity magnitude.
    """
    traces = np.asarray(time_by_neuron, dtype=float)
    if traces.ndim != 2:
        raise ValueError("time_by_neuron must have shape (time, neuron)")
    rows: list[list[float]] = []
    for anchor in anchors:
        target = traces[anchor.target_start : anchor.target_stop]
        if target.shape[0] != anchor.target_length:
            raise ValueError("anchor target interval falls outside activity samples")
        if not np.any(np.isfinite(target)):
            raise ValueError("anchor target interval has no finite activity samples")
        finite_per_frame = np.isfinite(target).sum(axis=1)
        population_mean = np.divide(
            np.nansum(target, axis=1),
            finite_per_frame,
            out=np.full(target.shape[0], np.nan, dtype=float),
            where=finite_per_frame > 0,
        )
        rows.append(
            [
                float(np.nanmean(np.abs(target))),
                float(np.sqrt(np.nanmean(target**2))),
                float(np.nanmean(np.abs(population_mean))),
            ]
        )
    return np.asarray(rows, dtype=float)


def _residualize(signal: np.ndarray, controls: np.ndarray) -> np.ndarray:
    control_arr = np.asarray(controls, dtype=float)
    if control_arr.ndim == 1:
        control_arr = control_arr[:, None]
    if control_arr.shape[0] != signal.shape[0]:
        raise ValueError("controls must align with the signals")
    design = np.concatenate([np.ones((len(signal), 1)), control_arr], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, signal, rcond=None)
    return signal - design @ coefficients


def _normalized_xcorr(
    a: np.ndarray,
    b: np.ndarray,
    max_lag: int,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    a = (a - a.mean()) / (a.std() or 1e-12)
    b = (b - b.mean()) / (b.std() or 1e-12)
    lags = np.arange(-max_lag, max_lag + 1)
    out = np.empty(len(lags))
    n = len(a)
    group_arr = None if groups is None else np.asarray(groups)
    if group_arr is not None and group_arr.shape[0] != n:
        raise ValueError("groups must align with signals")
    for i, lag in enumerate(lags):
        if lag < 0:
            left, right = a[-lag:], b[: n + lag]
            valid = None if group_arr is None else group_arr[-lag:] == group_arr[: n + lag]
        elif lag > 0:
            left, right = a[: n - lag], b[lag:]
            valid = None if group_arr is None else group_arr[: n - lag] == group_arr[lag:]
        else:
            left, right = a, b
            valid = None
        products = left * right
        selected = products if valid is None else products[valid]
        out[i] = float(np.mean(selected)) if selected.size else 0.0
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
    groups: np.ndarray | None = None,
    controls: np.ndarray | None = None,
) -> LeadLagResult:
    """Cross-correlate state and target changes, optionally adjusting nuisance activity."""
    conn_change = change_signal(state_labels, groups=groups)
    beh_change = change_signal(discretize(behavior, n_bins), groups=groups)
    if controls is not None:
        conn_change = _residualize(conn_change, controls)
        beh_change = _residualize(beh_change, controls)
    lags, xc = _normalized_xcorr(conn_change, beh_change, max_lag, groups=groups)
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
