"""Compact probabilistic state models for connectivity trajectories.

The transparent baseline remains cluster-then-Markov.  This module adds a
lightweight generative counterpart: a diagonal-emission HMM on low-dimensional
features, plus an explicit duration diagnostic that can motivate a later HSMM.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import blake2b

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from effectome.data_module.schema import ConnectivitySeries

from .graph_states import boundary_indices_for_series
from .metrics import vectorize


@dataclass(frozen=True)
class ProbabilisticStateConfig:
    n_states: int = 3
    n_components: int = 8
    standardize: bool = True
    covariance_floor: float = 1e-4
    n_iter: int = 50
    tol: float = 1e-4
    seed: int = 42


@dataclass
class DurationDiagnostics:
    empirical_mean: np.ndarray
    empirical_variance: np.ndarray
    geometric_mean: np.ndarray
    total_variation: np.ndarray
    overdispersion: np.ndarray


@dataclass
class HSMMCandidate:
    status: str
    diagnostics: DurationDiagnostics
    non_geometric_states: np.ndarray


@dataclass
class ProbabilisticStateProjection:
    filtered_probs: np.ndarray
    smoothed_probs: np.ndarray
    filtered_labels: np.ndarray
    smoothed_labels: np.ndarray
    filtered_entropy: np.ndarray
    smoothed_entropy: np.ndarray
    labels: np.ndarray
    uncertainty: np.ndarray
    log_likelihood: float


@dataclass(frozen=True)
class ProbabilisticFilterState:
    model_hash: str
    filtered_prob: np.ndarray
    samples_seen: int


@dataclass(frozen=True)
class ProbabilisticFilterResult:
    filtered_probs: np.ndarray
    filtered_labels: np.ndarray
    filtered_entropy: np.ndarray
    state_snapshots: tuple[ProbabilisticFilterState, ...]
    final_state: ProbabilisticFilterState


@dataclass
class ProbabilisticStateModel:
    n_states: int
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    projection: np.ndarray
    transition_matrix: np.ndarray
    initial_probs: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    filtered_probs: np.ndarray
    smoothed_probs: np.ndarray
    filtered_labels: np.ndarray
    smoothed_labels: np.ndarray
    filtered_entropy: np.ndarray
    smoothed_entropy: np.ndarray
    labels: np.ndarray
    uncertainty: np.ndarray
    log_likelihood: float
    heldout_log_likelihood: float | None
    heldout_iid_log_likelihood: float | None
    test_indices: np.ndarray
    boundary_indices: np.ndarray
    duration_diagnostics: DurationDiagnostics
    hsmm_candidate: HSMMCandidate

    @property
    def model_hash(self) -> str:
        digest = blake2b(digest_size=16)
        for value in (
            self.feature_mean,
            self.feature_scale,
            self.projection,
            self.transition_matrix,
            self.initial_probs,
            self.means,
            self.variances,
        ):
            array = np.ascontiguousarray(value)
            digest.update(array.view(np.uint8).tobytes())
        return digest.hexdigest()

    def transform(
        self,
        series: ConnectivitySeries | np.ndarray,
        *,
        boundary_indices: np.ndarray | None = None,
    ) -> ProbabilisticStateProjection:
        matrices = series.matrices if isinstance(series, ConnectivitySeries) else np.asarray(series)
        if boundary_indices is None and isinstance(series, ConnectivitySeries):
            boundary_indices = boundary_indices_for_series(series)
        features = _project_features(
            np.asarray(matrices, dtype=np.float64),
            self.feature_mean,
            self.feature_scale,
            self.projection,
        )
        filtered, smoothed, loglik = _forward_backward(
            features,
            self.initial_probs,
            self.transition_matrix,
            self.means,
            self.variances,
            _segment_slices(features.shape[0], boundary_indices),
        )
        filtered_labels = np.argmax(filtered, axis=1).astype(np.int64)
        smoothed_labels = np.argmax(smoothed, axis=1).astype(np.int64)
        filtered_entropy = -np.sum(filtered * np.log(filtered + 1e-12), axis=1)
        smoothed_entropy = -np.sum(smoothed * np.log(smoothed + 1e-12), axis=1)
        return ProbabilisticStateProjection(
            filtered_probs=filtered,
            smoothed_probs=smoothed,
            filtered_labels=filtered_labels,
            smoothed_labels=smoothed_labels,
            filtered_entropy=filtered_entropy,
            smoothed_entropy=smoothed_entropy,
            labels=filtered_labels,
            uncertainty=filtered_entropy,
            log_likelihood=loglik,
        )

    def filter(
        self,
        series: ConnectivitySeries | np.ndarray,
        *,
        state: ProbabilisticFilterState | None = None,
    ) -> ProbabilisticFilterResult:
        matrices = series.matrices if isinstance(series, ConnectivitySeries) else np.asarray(series)
        features = _project_features(
            np.asarray(matrices, dtype=np.float64),
            self.feature_mean,
            self.feature_scale,
            self.projection,
        )
        if state is not None and state.model_hash != self.model_hash:
            raise ValueError("filter state belongs to a different probabilistic model")
        previous = None if state is None else np.asarray(state.filtered_prob, dtype=np.float64)
        samples_seen = 0 if state is None else state.samples_seen
        emissions = _gaussian_log_emissions(features, self.means, self.variances)
        probabilities = np.empty((features.shape[0], self.n_states), dtype=np.float64)
        snapshots: list[ProbabilisticFilterState] = []
        for row, log_emission in enumerate(emissions):
            prior = self.initial_probs if previous is None else previous @ self.transition_matrix
            log_prob = np.log(prior + 1e-12) + log_emission
            log_prob -= logsumexp(log_prob)
            previous = np.exp(log_prob)
            probabilities[row] = previous
            snapshot = ProbabilisticFilterState(
                model_hash=self.model_hash,
                filtered_prob=previous.copy(),
                samples_seen=samples_seen + row + 1,
            )
            snapshots.append(snapshot)
        final = (
            snapshots[-1]
            if snapshots
            else ProbabilisticFilterState(
                model_hash=self.model_hash,
                filtered_prob=(
                    self.initial_probs.copy() if previous is None else previous.copy()
                ),
                samples_seen=samples_seen,
            )
        )
        labels = np.argmax(probabilities, axis=1).astype(np.int64)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
        return ProbabilisticFilterResult(
            filtered_probs=probabilities,
            filtered_labels=labels,
            filtered_entropy=entropy,
            state_snapshots=tuple(snapshots),
            final_state=final,
        )


def _segment_slices(n_items: int, boundary_indices: np.ndarray | None) -> list[slice]:
    if boundary_indices is None or len(boundary_indices) == 0:
        return [slice(0, n_items)]
    boundaries = sorted(set(int(x) for x in boundary_indices if 0 <= int(x) < n_items))
    if not boundaries or boundaries[0] != 0:
        boundaries = [0] + boundaries
    boundaries.append(n_items)
    return [slice(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def _subset_boundaries(
    indices: np.ndarray,
    original_boundaries: np.ndarray,
) -> np.ndarray:
    """Map recording boundaries and gaps onto an ordered subset of window indices."""
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("fit/test indices must be a non-empty 1D array")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("fit/test indices must be strictly increasing")
    boundary_set = {int(value) for value in original_boundaries}
    starts = [0]
    for position in range(1, len(indices)):
        if indices[position] != indices[position - 1] + 1 or int(indices[position]) in boundary_set:
            starts.append(position)
    return np.asarray(starts, dtype=int)


def _fit_projection(
    matrices: np.ndarray,
    fit_idx: np.ndarray,
    cfg: ProbabilisticStateConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = vectorize(matrices)
    train = raw[fit_idx]
    if cfg.standardize:
        mean = train.mean(axis=0, keepdims=True)
        scale = train.std(axis=0, keepdims=True)
        scale[scale < 1e-12] = 1.0
    else:
        mean = np.zeros((1, raw.shape[1]), dtype=np.float64)
        scale = np.ones((1, raw.shape[1]), dtype=np.float64)
    centered_train = (train - mean) / scale
    max_rank = min(centered_train.shape[0], centered_train.shape[1])
    n_components = max(1, min(cfg.n_components, max_rank))
    _, _, vt = np.linalg.svd(centered_train, full_matrices=False)
    projection = vt[:n_components].T
    features = ((raw - mean) / scale) @ projection
    return features, mean, scale, projection


def _gaussian_log_emissions(
    features: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    diff = features[:, None, :] - means[None, :, :]
    return -0.5 * (
        np.sum(np.log(2.0 * np.pi * variances[None, :, :]), axis=2)
        + np.sum((diff * diff) / variances[None, :, :], axis=2)
    )


def _initialize_hmm(
    features: np.ndarray,
    n_states: int,
    boundary_indices: np.ndarray,
    covariance_floor: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = KMeans(n_clusters=n_states, random_state=seed, n_init=10).fit(features).labels_
    means = np.stack(
        [
            features[labels == state].mean(axis=0) if np.any(labels == state) else features.mean(axis=0)
            for state in range(n_states)
        ]
    )
    variances = np.stack(
        [
            features[labels == state].var(axis=0) + covariance_floor
            if np.any(labels == state)
            else features.var(axis=0) + covariance_floor
            for state in range(n_states)
        ]
    )
    initial = np.ones(n_states, dtype=np.float64)
    transition = np.ones((n_states, n_states), dtype=np.float64)
    for slc in _segment_slices(features.shape[0], boundary_indices):
        seg = labels[slc]
        if len(seg) == 0:
            continue
        initial[seg[0]] += 1.0
        for a, b in zip(seg[:-1], seg[1:], strict=False):
            transition[a, b] += 1.0
    initial /= initial.sum()
    transition /= transition.sum(axis=1, keepdims=True)
    return initial, transition, means, variances


def _forward_backward(
    features: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    segments: list[slice],
) -> tuple[np.ndarray, np.ndarray, float]:
    n_samples, n_states = features.shape[0], initial.shape[0]
    log_initial = np.log(initial + 1e-12)
    log_transition = np.log(transition + 1e-12)
    log_emission = _gaussian_log_emissions(features, means, variances)
    filtered = np.zeros((n_samples, n_states), dtype=np.float64)
    smoothed = np.zeros((n_samples, n_states), dtype=np.float64)
    total_loglik = 0.0

    for slc in segments:
        seg_emission = log_emission[slc]
        seg_len = seg_emission.shape[0]
        log_alpha = np.zeros((seg_len, n_states), dtype=np.float64)
        log_beta = np.zeros((seg_len, n_states), dtype=np.float64)

        log_alpha[0] = log_initial + seg_emission[0]
        norm = logsumexp(log_alpha[0])
        log_alpha[0] -= norm
        total_loglik += norm
        for t in range(1, seg_len):
            log_alpha[t] = seg_emission[t] + logsumexp(log_alpha[t - 1][:, None] + log_transition, axis=0)
            norm = logsumexp(log_alpha[t])
            log_alpha[t] -= norm
            total_loglik += norm

        for t in range(seg_len - 2, -1, -1):
            log_beta[t] = logsumexp(
                log_transition + seg_emission[t + 1][None, :] + log_beta[t + 1][None, :],
                axis=1,
            )
            log_beta[t] -= logsumexp(log_beta[t])

        gamma = log_alpha + log_beta
        gamma -= logsumexp(gamma, axis=1, keepdims=True)
        filtered[slc] = np.exp(log_alpha)
        smoothed[slc] = np.exp(gamma)

    return filtered, smoothed, float(total_loglik)


def _em_update(
    features: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    segments: list[slice],
    covariance_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    n_states = initial.shape[0]
    log_transition = np.log(transition + 1e-12)
    log_emission = _gaussian_log_emissions(features, means, variances)
    gamma = np.zeros((features.shape[0], n_states), dtype=np.float64)
    xi_sum = np.zeros((n_states, n_states), dtype=np.float64)
    initial_sum = np.zeros(n_states, dtype=np.float64)
    total_loglik = 0.0

    for slc in segments:
        seg_emission = log_emission[slc]
        seg_len = seg_emission.shape[0]
        log_alpha = np.zeros((seg_len, n_states), dtype=np.float64)
        log_beta = np.zeros((seg_len, n_states), dtype=np.float64)
        scales = np.zeros(seg_len, dtype=np.float64)

        log_alpha[0] = np.log(initial + 1e-12) + seg_emission[0]
        scales[0] = logsumexp(log_alpha[0])
        log_alpha[0] -= scales[0]
        for t in range(1, seg_len):
            log_alpha[t] = seg_emission[t] + logsumexp(log_alpha[t - 1][:, None] + log_transition, axis=0)
            scales[t] = logsumexp(log_alpha[t])
            log_alpha[t] -= scales[t]
        total_loglik += float(scales.sum())

        for t in range(seg_len - 2, -1, -1):
            log_beta[t] = logsumexp(
                log_transition + seg_emission[t + 1][None, :] + log_beta[t + 1][None, :],
                axis=1,
            )
            log_beta[t] -= logsumexp(log_beta[t])

        seg_gamma = log_alpha + log_beta
        seg_gamma -= logsumexp(seg_gamma, axis=1, keepdims=True)
        seg_gamma = np.exp(seg_gamma)
        gamma[slc] = seg_gamma
        initial_sum += seg_gamma[0]

        for t in range(seg_len - 1):
            log_xi = (
                log_alpha[t][:, None]
                + log_transition
                + seg_emission[t + 1][None, :]
                + log_beta[t + 1][None, :]
            )
            log_xi -= logsumexp(log_xi)
            xi_sum += np.exp(log_xi)

    initial_new = initial_sum / max(initial_sum.sum(), 1e-12)
    transition_new = xi_sum + 1e-6
    transition_new /= transition_new.sum(axis=1, keepdims=True)

    state_mass = gamma.sum(axis=0) + 1e-12
    means_new = (gamma.T @ features) / state_mass[:, None]
    centered = features[:, None, :] - means_new[None, :, :]
    variances_new = np.sum(gamma[:, :, None] * centered * centered, axis=0) / state_mass[:, None]
    variances_new = np.maximum(variances_new, covariance_floor)

    return initial_new, transition_new, means_new, variances_new, gamma, xi_sum, total_loglik


def _fit_hmm(
    features: np.ndarray,
    boundary_indices: np.ndarray,
    cfg: ProbabilisticStateConfig,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    initial, transition, means, variances = _initialize_hmm(
        features,
        cfg.n_states,
        boundary_indices,
        cfg.covariance_floor,
        cfg.seed,
    )
    segments = _segment_slices(features.shape[0], boundary_indices)
    prev_loglik = -np.inf
    gamma = np.zeros((features.shape[0], cfg.n_states), dtype=np.float64)
    for iteration in range(cfg.n_iter):
        initial, transition, means, variances, gamma, _, loglik = _em_update(
            features,
            initial,
            transition,
            means,
            variances,
            segments,
            cfg.covariance_floor,
        )
        if progress_callback is not None:
            progress_callback(iteration + 1, cfg.n_iter)
        if loglik - prev_loglik < cfg.tol:
            break
        prev_loglik = loglik
    filtered, smoothed, loglik = _forward_backward(
        features, initial, transition, means, variances, segments
    )
    return initial, transition, means, variances, filtered, smoothed, loglik


def _iid_log_likelihood(
    features: np.ndarray,
    mixture: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> float:
    log_emission = _gaussian_log_emissions(features, means, variances)
    return float(np.sum(logsumexp(np.log(mixture + 1e-12)[None, :] + log_emission, axis=1)))


def _duration_diagnostics(
    labels: np.ndarray,
    transition: np.ndarray,
    n_states: int,
    boundary_indices: np.ndarray | None = None,
) -> DurationDiagnostics:
    runs: dict[int, list[int]] = {state: [] for state in range(n_states)}
    for slc in _segment_slices(len(labels), boundary_indices):
        segment = labels[slc]
        if segment.size == 0:
            continue
        cur = int(segment[0])
        length = 1
        for value in segment[1:]:
            value = int(value)
            if value == cur:
                length += 1
            else:
                runs[cur].append(length)
                cur = value
                length = 1
        runs[cur].append(length)

    empirical_mean = np.zeros(n_states, dtype=np.float64)
    empirical_variance = np.zeros(n_states, dtype=np.float64)
    geometric_mean = np.zeros(n_states, dtype=np.float64)
    total_variation = np.zeros(n_states, dtype=np.float64)
    overdispersion = np.zeros(n_states, dtype=np.float64)

    for state in range(n_states):
        dwell = np.asarray(runs[state], dtype=np.float64)
        if dwell.size == 0:
            continue
        empirical_mean[state] = float(dwell.mean())
        empirical_variance[state] = float(dwell.var()) if dwell.size > 1 else 0.0
        p_stay = float(np.clip(transition[state, state], 1e-6, 1.0 - 1e-6))
        geom_p = 1.0 - p_stay
        geometric_mean[state] = 1.0 / geom_p
        support = np.arange(1, int(dwell.max()) + 1, dtype=np.float64)
        empirical_hist = np.array([(dwell == s).mean() for s in support], dtype=np.float64)
        geometric_hist = np.power(p_stay, support - 1.0) * geom_p
        geometric_hist /= geometric_hist.sum()
        total_variation[state] = 0.5 * float(np.abs(empirical_hist - geometric_hist).sum())
        overdispersion[state] = empirical_variance[state] / max(empirical_mean[state], 1e-12)

    return DurationDiagnostics(
        empirical_mean=empirical_mean,
        empirical_variance=empirical_variance,
        geometric_mean=geometric_mean,
        total_variation=total_variation,
        overdispersion=overdispersion,
    )


def _project_features(
    matrices: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    return ((vectorize(matrices) - mean) / scale) @ projection


def fit_probabilistic_states(
    series: ConnectivitySeries,
    cfg: ProbabilisticStateConfig,
    *,
    fit_indices: np.ndarray | None = None,
    test_indices: np.ndarray | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ProbabilisticStateModel:
    matrices = np.asarray(series.matrices, dtype=np.float64)
    n_windows = matrices.shape[0]
    fit_idx = np.arange(n_windows, dtype=int) if fit_indices is None else np.asarray(fit_indices, dtype=int)
    boundary_indices = boundary_indices_for_series(series)
    if fit_idx.ndim != 1 or fit_idx.size < cfg.n_states:
        raise ValueError("fit_indices must contain at least n_states ordered windows")
    if np.any(fit_idx < 0) or np.any(fit_idx >= n_windows):
        raise ValueError("fit_indices are outside the connectivity series")
    test_idx = np.array([], dtype=int) if test_indices is None else np.asarray(test_indices, dtype=int)
    if test_idx.ndim != 1 or np.any(test_idx < 0) or np.any(test_idx >= n_windows):
        raise ValueError("test_indices are outside the connectivity series")
    if test_idx.size and np.any(np.diff(test_idx) <= 0):
        raise ValueError("test_indices must be strictly increasing")
    if np.intersect1d(fit_idx, test_idx).size:
        raise ValueError("fit_indices and test_indices must be disjoint")
    fit_boundaries = _subset_boundaries(fit_idx, boundary_indices)
    features, mean, scale, projection = _fit_projection(matrices, fit_idx, cfg)
    fit_features = features[fit_idx]
    initial, transition, hmm_means, variances, filtered_fit, smoothed_fit, fit_loglik = _fit_hmm(
        fit_features,
        boundary_indices=fit_boundaries,
        cfg=cfg,
        progress_callback=progress_callback,
    )

    projection_all = _forward_backward(
        features,
        initial,
        transition,
        hmm_means,
        variances,
        _segment_slices(n_windows, boundary_indices),
    )
    filtered_all, smoothed_all, full_loglik = projection_all
    filtered_labels = np.argmax(filtered_all, axis=1).astype(np.int64)
    smoothed_labels = np.argmax(smoothed_all, axis=1).astype(np.int64)
    filtered_entropy = -np.sum(filtered_all * np.log(filtered_all + 1e-12), axis=1)
    smoothed_entropy = -np.sum(smoothed_all * np.log(smoothed_all + 1e-12), axis=1)
    labels = filtered_labels
    uncertainty = filtered_entropy
    duration = _duration_diagnostics(labels, transition, cfg.n_states, boundary_indices)
    hsmm_candidate = HSMMCandidate(
        status="duration_diagnostic_only",
        diagnostics=duration,
        non_geometric_states=np.flatnonzero(duration.total_variation > 0.15),
    )

    heldout_loglik = None
    heldout_iid = None
    if test_idx.size:
        test_features = features[test_idx]
        if test_features.size > 0:
            test_boundaries = _subset_boundaries(test_idx, boundary_indices)
            heldout_loglik = _forward_backward(
                test_features,
                initial,
                transition,
                hmm_means,
                variances,
                _segment_slices(test_features.shape[0], test_boundaries),
            )[2]
            mixture = smoothed_fit.mean(axis=0)
            heldout_iid = _iid_log_likelihood(test_features, mixture, hmm_means, variances)

    return ProbabilisticStateModel(
        n_states=cfg.n_states,
        feature_mean=np.asarray(mean, dtype=np.float64),
        feature_scale=np.asarray(scale, dtype=np.float64),
        projection=np.asarray(projection, dtype=np.float64),
        transition_matrix=transition,
        initial_probs=initial,
        means=hmm_means,
        variances=variances,
        filtered_probs=filtered_all,
        smoothed_probs=smoothed_all,
        filtered_labels=filtered_labels,
        smoothed_labels=smoothed_labels,
        filtered_entropy=filtered_entropy,
        smoothed_entropy=smoothed_entropy,
        labels=labels,
        uncertainty=uncertainty,
        log_likelihood=float(full_loglik),
        heldout_log_likelihood=None if heldout_loglik is None else float(heldout_loglik),
        heldout_iid_log_likelihood=None if heldout_iid is None else float(heldout_iid),
        test_indices=test_idx,
        boundary_indices=boundary_indices,
        duration_diagnostics=duration,
        hsmm_candidate=hsmm_candidate,
    )


__all__ = [
    "DurationDiagnostics",
    "HSMMCandidate",
    "ProbabilisticStateConfig",
    "ProbabilisticFilterResult",
    "ProbabilisticFilterState",
    "ProbabilisticStateModel",
    "ProbabilisticStateProjection",
    "fit_probabilistic_states",
]
