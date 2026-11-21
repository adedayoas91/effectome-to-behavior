"""Select local refits from matured forecast errors within each trial."""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def refit_decisions(
    origins: np.ndarray,
    pooled: np.ndarray,
    local: np.ndarray,
    observed_targets: np.ndarray,
    *,
    horizon: int,
    lookback: int,
    margin: float,
) -> np.ndarray:
    """Return a trial-by-event mask where true selects the local forecast.

    A forecast issued at sample ``s`` can first inform the choice at
    ``s + horizon``. Each choice uses the most recent ``lookback`` matured
    forecasts from its own trial. Burn-in and ties select the pooled forecast.
    """
    gains, matured = _matured_gains(origins, pooled, local, observed_targets, horizon=horizon)
    return _rolling_choice(gains, matured, lookback, margin)


def fast_refit_decisions(
    origins: np.ndarray,
    pooled: np.ndarray,
    local: np.ndarray,
    observed_targets: np.ndarray,
    *,
    horizon: int,
    lookback: int = 16,
    margin: float = 0.0,
    consecutive_losses: int = 2,
    recovery_forecasts: int = 4,
) -> np.ndarray:
    """Suspend the trailing-average rule on recent losses and require new evidence to resume."""
    for name, count in (
        ("consecutive_losses", consecutive_losses),
        ("recovery_forecasts", recovery_forecasts),
    ):
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise ValueError(f"{name} must be a positive forecast count")
    gains, matured = _matured_gains(origins, pooled, local, observed_targets, horizon=horizon)
    choose = _rolling_choice(gains, matured, lookback, margin)
    for trial, (history, counts) in enumerate(zip(gains, matured, strict=True)):
        blocked = False
        loss_run = recovered = previous = 0
        for event, count in enumerate(counts):
            for k in range(previous, count):
                loss_run = loss_run + 1 if history[k] < 0 else 0
                if loss_run >= consecutive_losses:
                    blocked, recovered = True, 0
                elif blocked:
                    recovered += 1
                    if (
                        recovered >= recovery_forecasts
                        and history[k + 1 - recovery_forecasts : k + 1].mean() > margin
                    ):
                        blocked = False
            previous = count
            choose[trial, event] &= not blocked
    return choose


def _rolling_choice(gains: np.ndarray, matured: np.ndarray, lookback: int, margin: float) -> np.ndarray:
    if isinstance(lookback, bool) or not isinstance(lookback, Integral) or lookback < 1:
        raise ValueError("lookback must be a positive forecast count")
    if isinstance(margin, bool) or not isinstance(margin, Real) or not isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    if lookback > gains.shape[1]:
        return np.zeros(gains.shape, dtype=bool)
    with np.errstate(over="raise", invalid="raise"):
        try:
            recent_gains = sliding_window_view(gains, lookback, axis=1).mean(axis=2)
        except FloatingPointError as error:
            raise ValueError("forecast loss calculation must be finite") from error
    starts = np.maximum(matured - int(lookback), 0)
    return (matured >= lookback) & (recent_gains[np.arange(len(gains))[:, None], starts] > margin)


def _matured_gains(
    origins: np.ndarray,
    pooled: np.ndarray,
    local: np.ndarray,
    observed_targets: np.ndarray,
    *,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    max_sample = np.iinfo(np.int64).max
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or not 0 < horizon <= max_sample:
        raise ValueError("horizon must be a positive sample count")

    times = np.asarray(origins)
    if (
        times.ndim != 2
        or not times.size
        or times.dtype.kind not in "iu"
        or np.any(times < 0)
        or np.any(times > max_sample)
    ):
        raise ValueError("origins must be nonempty trial-by-event nonnegative integer sample indices")
    times = times.astype(np.int64, copy=False)
    if np.any(times[:, 1:] <= times[:, :-1]):
        raise ValueError("origins must increase strictly within every trial")

    values = []
    for name, array in (("pooled", pooled), ("local", local), ("observed_targets", observed_targets)):
        data = np.asarray(array)
        if data.shape != (*times.shape, 2) or data.dtype.kind not in "iuf" or not np.isfinite(data).all():
            raise ValueError(f"{name} must be finite real values with shape {(*times.shape, 2)}")
        values.append(data.astype(float, copy=False))
    pooled_values, local_values, targets = values
    with np.errstate(over="raise", invalid="raise"):
        try:
            gains = np.abs(pooled_values - targets).mean(axis=2) - np.abs(local_values - targets).mean(axis=2)
        except FloatingPointError as error:
            raise ValueError("forecast loss calculation must be finite") from error

    # Searching for the cutoff avoids adding the horizon to a large origin.
    matured = np.stack([np.searchsorted(row, row - int(horizon), side="right") for row in times])
    return gains, matured
