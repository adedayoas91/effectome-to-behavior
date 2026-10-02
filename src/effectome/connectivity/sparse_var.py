"""Constant and independent-window sparse VAR connectivity baselines."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Any

import numpy as np

from effectome.data_module.schema import TemporalAnchor, WindowedSegments

from .base import ConnectivityConfig, ConnectivityEstimator
from .predictive import (
    FilterState,
    GraphEstimate,
    NeuralForecast,
    PredictiveCapabilities,
    UnsupportedForecastError,
)
from .registry import register_connectivity
from .var_utils import LagDesign, VARFit, build_lag_design, fit_sparse_var


def _continuous_anchor_key(anchor: TemporalAnchor) -> tuple[str, str, str | None, str | None, str | None]:
    return (
        anchor.dataset_id,
        anchor.recording_id,
        anchor.animal_id,
        anchor.session_id,
        anchor.segment_id,
    )


def _hash_arrays(*arrays: np.ndarray, extras: tuple[Any, ...] = ()) -> str:
    digest = blake2b(digest_size=16)
    for array in arrays:
        arr = np.ascontiguousarray(array)
        digest.update(str(arr.shape).encode())
        digest.update(str(arr.dtype).encode())
        digest.update(arr.view(np.uint8).tobytes())
    for item in extras:
        digest.update(repr(item).encode())
    return digest.hexdigest()


def _aggregate_lags(lagged: np.ndarray, mode: str) -> np.ndarray:
    if mode == "sum":
        return lagged.sum(axis=0)
    if mode == "mean":
        return lagged.mean(axis=0)
    if mode == "first":
        return lagged[0]
    raise ValueError("lag_aggregation must be one of: sum, mean, first")


def _fit_options(cfg: ConnectivityConfig) -> dict[str, Any]:
    extra = dict(cfg.extra)
    return {
        "n_lags": int(extra.get("n_lags", cfg.max_lag)),
        "lambda1": float(extra.get("lambda1", 0.0)),
        "lambda2": float(extra.get("lambda2", cfg.ridge)),
        "standardize": bool(extra.get("standardize", True)),
        "max_iter": int(extra.get("max_iter", 10_000)),
        "tol": float(extra.get("tol", 1e-6)),
        "lag_aggregation": str(extra.get("lag_aggregation", "sum")),
    }


@dataclass(frozen=True)
class SparseVARModel:
    fit: VARFit
    capabilities: PredictiveCapabilities
    lag_aggregation: str = "sum"

    @property
    def model_hash(self) -> str:
        return _hash_arrays(
            self.fit.coefficients,
            self.fit.intercept,
            extras=(
                self.fit.diagnostics.get("lambda1"),
                self.fit.diagnostics.get("lambda2"),
                self.fit.diagnostics.get("n_lags"),
            ),
        )

    @property
    def n_lags(self) -> int:
        return int(self.fit.diagnostics["n_lags"])

    @property
    def n_neurons(self) -> int:
        return int(self.fit.diagnostics["n_neurons"])

    @property
    def lagged_coefficients(self) -> np.ndarray:
        return self.fit.lagged_coefficients

    def initial_state(self, *, recording_identity: Any = None) -> FilterState:
        return FilterState(
            model_hash=self.model_hash,
            history=np.empty((0, self.n_neurons), dtype=np.float64),
            recording_identity=recording_identity,
        )

    def _validate_state(self, state: FilterState) -> None:
        if state.model_hash != self.model_hash:
            raise ValueError("filter state belongs to a different fitted model")
        if state.history.ndim != 2 or state.history.shape[1] != self.n_neurons:
            raise ValueError("filter state history has an incompatible shape")

    def _state_hash(self, state: FilterState) -> str:
        return _hash_arrays(
            state.history,
            extras=(state.model_hash, state.sample_index, state.recording_identity),
        )

    def observe(
        self,
        state: FilterState,
        samples: np.ndarray,
        *,
        sample_indices: np.ndarray | None = None,
        masks: np.ndarray | None = None,
        available_inputs: np.ndarray | None = None,
        recording_identity: Any = None,
    ) -> FilterState:
        self._validate_state(state)
        if available_inputs is not None:
            raise UnsupportedForecastError("sparse VAR does not support exogenous inputs")
        if recording_identity is not None and state.recording_identity not in (None, recording_identity):
            raise ValueError("filter state belongs to a different recording")
        identity = state.recording_identity if recording_identity is None else recording_identity
        observed = np.asarray(samples, dtype=np.float64)
        if observed.ndim != 2 or observed.shape[1] != self.n_neurons:
            raise ValueError("samples must have shape (time, n_neurons)")
        if sample_indices is not None:
            raw_indices = np.asarray(sample_indices)
            if raw_indices.shape != (observed.shape[0],):
                raise ValueError("sample_indices must align with samples")
            if not np.issubdtype(raw_indices.dtype, np.integer):
                raise ValueError("sample_indices must be integers")
            indices = raw_indices.astype(np.int64, copy=False)
            if indices.size and np.any(np.diff(indices) <= 0):
                raise ValueError("sample_indices must be strictly increasing")
            if indices.size and state.sample_index is not None and indices[0] < state.sample_index:
                raise ValueError("sample_indices replay previously observed samples")
        else:
            indices = None
        valid = np.ones(observed.shape[0], dtype=bool)
        if masks is not None:
            mask = np.asarray(masks, dtype=bool)
            if mask.shape != observed.shape:
                raise ValueError("masks must align with samples")
            valid &= mask.all(axis=1)
        valid &= np.isfinite(observed).all(axis=1)
        history = np.asarray(state.history, dtype=np.float64).copy()
        previous = None if state.sample_index is None else state.sample_index - 1
        for row, row_valid, index in zip(
            observed,
            valid,
            range(observed.shape[0]) if indices is None else indices,
            strict=True,
        ):
            if indices is not None and previous is not None and int(index) != previous + 1:
                history = np.empty((0, self.n_neurons), dtype=np.float64)
            if row_valid:
                history = np.vstack([history, row])[-self.n_lags :]
            else:
                history = np.empty((0, self.n_neurons), dtype=np.float64)
            previous = int(index) if indices is not None else None
        origin = (
            state.sample_index
            if observed.shape[0] == 0
            else (None if indices is None else int(indices[-1]) + 1)
        )
        return FilterState(
            model_hash=self.model_hash,
            history=history,
            sample_index=origin,
            recording_identity=identity,
            diagnostics={"observed_rows": int(valid.sum())},
        )

    def forecast(
        self,
        state: FilterState,
        *,
        horizon_samples: int,
        known_future_inputs: np.ndarray | None = None,
    ) -> NeuralForecast:
        self._validate_state(state)
        if known_future_inputs is not None:
            raise UnsupportedForecastError("sparse VAR does not support exogenous inputs")
        if horizon_samples < 1:
            raise ValueError("horizon_samples must be positive")
        state_hash = self._state_hash(state)
        if not self.fit.converged:
            return NeuralForecast(
                mean=np.empty((0, self.n_neurons), dtype=np.float64),
                horizon_samples=horizon_samples,
                origin_sample=state.sample_index,
                failure_reason="nonconverged_fit",
                model_hash=self.model_hash,
                state_hash=state_hash,
            )
        if state.history.shape[0] < self.n_lags:
            return NeuralForecast(
                mean=np.empty((0, self.n_neurons), dtype=np.float64),
                horizon_samples=horizon_samples,
                origin_sample=state.sample_index,
                failure_reason="insufficient_history",
                model_hash=self.model_hash,
                state_hash=state_hash,
            )
        history = np.asarray(state.history[-self.n_lags :], dtype=np.float64).copy()
        means: list[np.ndarray] = []
        for _step in range(horizon_samples):
            row = np.concatenate([history[-lag] for lag in range(1, self.n_lags + 1)])
            pred = self.fit.intercept + row @ self.fit.coefficients
            means.append(pred)
            history = np.vstack([history[1:], pred])
        return NeuralForecast(
            mean=np.asarray(means, dtype=np.float64),
            horizon_samples=horizon_samples,
            origin_sample=state.sample_index,
            model_hash=self.model_hash,
            state_hash=state_hash,
            raw_support={
                "history_interval": (
                    None
                    if state.sample_index is None
                    else (state.sample_index - self.n_lags, state.sample_index)
                ),
                "recording_identity": state.recording_identity,
                "residual_covariance": self.fit.residual_covariance,
                "density": "unsupported",
            },
        )

    def graph_at(self, state: FilterState | None = None) -> GraphEstimate:
        if state is not None:
            self._validate_state(state)
        if not self.fit.converged:
            raise ValueError("cannot export a graph from a nonconverged fit")
        lagged = np.asarray(self.lagged_coefficients, dtype=np.float64)
        return GraphEstimate(
            matrix=_aggregate_lags(lagged, self.lag_aggregation),
            lagged_coefficients=lagged,
            coefficient_semantics="signed_sparse_var_coefficient_source_to_target",
            model_hash=self.model_hash,
            diagnostics=dict(self.fit.diagnostics),
        )


def fit_sparse_var_model(
    samples: np.ndarray,
    *,
    cfg: ConnectivityConfig,
    sample_indices: np.ndarray | None = None,
    segment_ids: np.ndarray | None = None,
    masks: np.ndarray | None = None,
    available_inputs: np.ndarray | None = None,
) -> SparseVARModel:
    if available_inputs is not None:
        raise UnsupportedForecastError("sparse VAR does not support exogenous inputs")
    options = _fit_options(cfg.resolve(1.0))
    design = build_lag_design(
        np.asarray(samples, dtype=np.float64),
        n_lags=options["n_lags"],
        sample_indices=sample_indices,
        segment_ids=segment_ids,
        masks=masks,
    )
    if design.design.shape[0] and design.segment_ids is not None:
        keep: list[int] = []
        seen: dict[tuple[Any, int], int] = {}
        for row, (segment, response_index) in enumerate(
            zip(design.segment_ids, design.response_indices, strict=True)
        ):
            key = (segment.item() if hasattr(segment, "item") else segment, int(response_index))
            if key in seen:
                first = seen[key]
                if not (
                    np.array_equal(design.design[first], design.design[row])
                    and np.array_equal(design.response[first], design.response[row])
                ):
                    raise ValueError("duplicate owned lag rows disagree")
                continue
            seen[key] = row
            keep.append(row)
        selected = np.asarray(keep, dtype=int)
        design = LagDesign(
            design.design[selected],
            design.response[selected],
            design.response_indices[selected],
            design.n_lags,
            design.n_neurons,
            design.segment_ids[selected],
        )
    fit = fit_sparse_var(
        design,
        lambda1=options["lambda1"],
        lambda2=options["lambda2"],
        standardize=options["standardize"],
        max_iter=options["max_iter"],
        tol=options["tol"],
    )
    return SparseVARModel(
        fit=fit,
        capabilities=PredictiveCapabilities(neural_forecast=True),
        lag_aggregation=options["lag_aggregation"],
    )


def _fit_from_design(design: LagDesign, cfg: ConnectivityConfig) -> SparseVARModel:
    options = _fit_options(cfg)
    fit = fit_sparse_var(
        design,
        lambda1=options["lambda1"],
        lambda2=options["lambda2"],
        standardize=options["standardize"],
        max_iter=options["max_iter"],
        tol=options["tol"],
    )
    return SparseVARModel(
        fit=fit,
        capabilities=PredictiveCapabilities(neural_forecast=True),
        lag_aggregation=options["lag_aggregation"],
    )


def _reconstruct_samples(segments: WindowedSegments) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not segments.anchors:
        return None
    n_neurons = segments.n_neurons
    start = min(anchor.context_start for anchor in segments.anchors)
    stop = max(anchor.context_stop for anchor in segments.anchors)
    samples = np.full((stop - start, n_neurons), np.nan, dtype=np.float64)
    sample_indices = np.arange(start, stop, dtype=np.int64)
    segment_ids = np.full(stop - start, "", dtype=object)
    for window, segment, anchor in zip(segments.windows, segments.segments, segments.anchors, strict=True):
        if segment.shape[0] != anchor.history_length:
            return None
        key = repr(_continuous_anchor_key(anchor))
        local_start = anchor.context_start - start
        local_stop = anchor.context_stop - start
        existing = samples[local_start:local_stop]
        filled = np.isfinite(existing).all(axis=1)
        if np.any(filled) and not np.allclose(existing[filled], segment[filled], equal_nan=True):
            return None
        samples[local_start:local_stop] = np.where(np.isfinite(existing), existing, segment)
        segment_ids[local_start:local_stop] = key
        if anchor.gap_before and local_start > 0:
            segment_ids[local_start - 1] = f"gap-before-{window.start}"
        if anchor.gap_after and local_stop < len(segment_ids):
            segment_ids[local_stop] = f"gap-after-{window.stop}"
    return samples, sample_indices, segment_ids


class _SparseVARBase(ConnectivityEstimator):
    directed = True
    weighted = True
    weight_semantics = "signed_sparse_var_coefficient"

    def __init__(self, cfg: ConnectivityConfig) -> None:
        super().__init__(cfg)
        self._window_lagged: list[np.ndarray] = []
        self._diagnostic_update: dict[str, Any] = {}

    @property
    def capabilities(self) -> PredictiveCapabilities:
        return PredictiveCapabilities(neural_forecast=True, independent_window=False)

    def fit(self, samples: np.ndarray) -> SparseVARModel:
        return fit_sparse_var_model(samples, cfg=self.cfg)

    def estimate(self, segment: np.ndarray) -> np.ndarray:
        model = self.fit(np.asarray(segment, dtype=np.float64))
        graph = model.graph_at()
        self._window_lagged = [graph.lagged_coefficients]
        return graph.matrix

    def _lagged_output(self) -> np.ndarray | None:
        return np.stack(self._window_lagged) if self._window_lagged else None

    def _diagnostics(self, segments: WindowedSegments, mats: np.ndarray) -> dict[str, Any]:
        diagnostics = super()._diagnostics(segments, mats)
        diagnostics.update(
            {
                "predictive_capabilities": self.capabilities.__dict__,
                "lag_design": "same_contiguous_segment_required",
                "input_axis_convention": "design_matrix_times_input_coefficients",
                **self._diagnostic_update,
            }
        )
        return diagnostics


@register_connectivity("sparse_var")
class SparseVARConnectivity(_SparseVARBase):
    """Constant sparse VAR fit over deduplicated raw rows reconstructed from anchors."""

    estimation_mode = "constant_sparse_var"

    def estimate_sequence(self, segments: WindowedSegments) -> np.ndarray:
        reconstructed = _reconstruct_samples(segments)
        if reconstructed is None:
            matrices = []
            self._window_lagged = []
            fit_statuses = []
            for segment in segments.segments:
                model = self.fit(segment)
                graph = model.graph_at()
                matrices.append(graph.matrix)
                self._window_lagged.append(graph.lagged_coefficients)
                fit_statuses.append(graph.diagnostics.get("status"))
            self._diagnostic_update = {
                "adaptation_policy": "fallback_independent_window_no_anchor_reconstruction",
                "fit_statuses": fit_statuses,
            }
            return np.asarray(matrices, dtype=np.float64)
        samples, sample_indices, segment_ids = reconstructed
        options = _fit_options(self.cfg)
        design = build_lag_design(
            samples,
            n_lags=options["n_lags"],
            sample_indices=sample_indices,
            segment_ids=segment_ids,
        )
        model = _fit_from_design(design, self.cfg)
        graph = model.graph_at()
        self._window_lagged = [graph.lagged_coefficients for _ in range(segments.n_windows)]
        self._diagnostic_update = {
            "adaptation_policy": "constant_fit_repeated_for_windows",
            "valid_design_rows": int(design.design.shape[0]),
            "model_hash": model.model_hash,
            **graph.diagnostics,
        }
        return np.repeat(graph.matrix[None, :, :], segments.n_windows, axis=0)


@register_connectivity("windowed_sparse_var")
class WindowedSparseVARConnectivity(_SparseVARBase):
    """Independent sparse VAR refit per window."""

    estimation_mode = "independent_window_sparse_var"

    @property
    def capabilities(self) -> PredictiveCapabilities:
        return PredictiveCapabilities(neural_forecast=True, independent_window=True)

    def estimate_sequence(self, segments: WindowedSegments) -> np.ndarray:
        self._window_lagged = []
        matrices = []
        fit_statuses = []
        for segment in segments.segments:
            model = self.fit(segment)
            graph = model.graph_at()
            matrices.append(graph.matrix)
            self._window_lagged.append(graph.lagged_coefficients)
            fit_statuses.append(graph.diagnostics.get("status"))
        self._diagnostic_update = {
            "adaptation_policy": "independent_window_refit",
            "fit_statuses": fit_statuses,
        }
        return np.asarray(matrices, dtype=np.float64)
