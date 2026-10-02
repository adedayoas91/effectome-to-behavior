"""Outer-fold ownership skeleton for confirmatory effectome evaluation.

This module owns split construction before any learned preprocessing, graph fitting,
state fitting, manifold fitting, or decoder training.  The first implementation is a
small auditable runner that can execute spy/dummy fitters in tests and persist a
machine-readable fold manifest for later WP4/WP5 expansion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from effectome.connectivity import ConnectivityConfig
from effectome.connectivity.sparse_var import fit_sparse_var_model
from effectome.data_module.schema import TemporalAnchor, WindowedSegments
from effectome.linking import (
    DependencySupport,
    anchor_group_labels,
    purged_blocked_splits,
    valid_positive_lag_origins,
)


@dataclass(frozen=True)
class EvaluationConfig:
    profile: str = "revised_primary"
    mode: str = "development"
    protocol: str = "purged_blocked"
    n_outer_folds: int = 5
    embargo: int = 0
    group_by: str = "recording"
    min_train_examples: int = 2
    min_test_examples: int = 1
    update_policy: str = "frozen"
    states_enabled: bool = False
    groups_enabled: bool = False
    endpoint_name: str = "motif"
    endpoint_kind: str = "unknown"
    endpoint_type: str = "categorical"
    positive_lag: int = 1
    class_vocabulary: tuple[Any, ...] = ()
    probability_floor: float = 1e-12
    practical_gain_nats: float | None = None
    neural_lambda1_grid: tuple[float, ...] = (0.0, 0.01, 0.05)
    neural_lambda2: float = 1e-3
    neural_max_lag: int = 1
    decoder_c_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    decoder_alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    selection_seed: int = 260926
    training_calibration_policy: str = "outer_training_only"
    neural_selection_metric: str = "standardized_one_step_mse"
    decoder_selection_metric: str = "log_loss"
    max_missing_fraction: float = 0.05
    failure_policy: str = "record_and_continue"
    artifact_schema_version: str = "2.0"
    reject_legacy_for_confirmation: bool = True
    endpoint_verified: bool = False
    panel_mapping_id: str | None = None
    calibration_status: str = "unresolved"
    resource_budget_id: str | None = None
    source_manifest_hash: str | None = None
    claim_to_test_map: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.profile not in {"revised_primary", "legacy_descriptive"}:
            raise ValueError("unsupported evaluation profile")
        if self.mode not in {"development", "confirmatory", "legacy_descriptive"}:
            raise ValueError("unsupported evaluation mode")
        if self.update_policy not in {"frozen", "declared_past_only"}:
            raise ValueError("unsupported evaluation update policy")
        if self.positive_lag < 1 or self.neural_max_lag < 1:
            raise ValueError("positive_lag and neural_max_lag must be positive")
        if self.endpoint_type not in {"categorical", "continuous"}:
            raise ValueError("endpoint_type must be categorical or continuous")
        if not 0 < self.probability_floor < 1:
            raise ValueError("probability_floor must lie in (0, 1)")
        if not self.neural_lambda1_grid or min(self.neural_lambda1_grid) < 0:
            raise ValueError("neural lambda grid must be non-empty and nonnegative")
        if not self.decoder_c_grid or min(self.decoder_c_grid) <= 0:
            raise ValueError("decoder C grid must be non-empty and positive")
        if not self.decoder_alpha_grid or min(self.decoder_alpha_grid) < 0:
            raise ValueError("decoder alpha grid must be non-empty and nonnegative")
        if self.training_calibration_policy not in {
            "outer_training_only",
            "registered_unscored_prefix",
        }:
            raise ValueError("unsupported training calibration policy")
        if self.neural_selection_metric != "standardized_one_step_mse":
            raise ValueError("unsupported neural selection metric")
        if self.decoder_selection_metric != "log_loss":
            raise ValueError("unsupported decoder selection metric")
        if not 0 <= self.max_missing_fraction < 1:
            raise ValueError("max_missing_fraction must lie in [0, 1)")
        if self.failure_policy not in {"record_and_continue", "fail_fast"}:
            raise ValueError("unsupported evaluation failure policy")
        if self.artifact_schema_version != "2.0":
            raise ValueError("the revised evaluator requires artifact schema 2.0")
        if self.profile == "legacy_descriptive" and self.mode != "legacy_descriptive":
            raise ValueError("legacy_descriptive profile requires legacy_descriptive mode")
        if self.mode == "confirmatory":
            if self.endpoint_kind == "unknown":
                raise ValueError("confirmatory evaluation requires endpoint lineage")
            if self.endpoint_type == "categorical" and not self.class_vocabulary:
                raise ValueError("confirmatory categorical evaluation requires a vocabulary")
            if self.practical_gain_nats is None:
                raise ValueError("confirmatory evaluation requires a frozen practical gain")
            unresolved = []
            if not self.endpoint_verified:
                unresolved.append("endpoint verification")
            if self.panel_mapping_id is None:
                unresolved.append("panel mapping")
            if self.calibration_status != "passed":
                unresolved.append("calibration")
            if self.resource_budget_id is None:
                unresolved.append("resource budget")
            if self.source_manifest_hash is None:
                unresolved.append("source manifest")
            if not self.claim_to_test_map:
                unresolved.append("claim-to-test map")
            if unresolved:
                raise ValueError(
                    "confirmatory evaluation has unresolved requirements: "
                    + ", ".join(unresolved)
                )


@dataclass(frozen=True)
class EvaluationFold:
    fold_id: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    protocol: str
    update_policy: str
    status: str = "available"
    failure_reason: str | None = None


@dataclass(frozen=True)
class SplitManifest:
    outer_folds: tuple[EvaluationFold, ...]
    n_examples: int
    protocol: str
    group_by: str
    embargo: int
    states_enabled: bool
    groups_enabled: bool


@dataclass(frozen=True)
class PredictionRecord:
    origin_index: int
    fold_id: str
    model_name: str
    status: str
    endpoint_name: str | None = None
    endpoint_kind: str | None = None
    endpoint_type: str = "categorical"
    class_order: tuple[Any, ...] = ()
    probabilities: tuple[float, ...] = ()
    issue_sample: int | None = None
    target_start: int | None = None
    target_stop: int | None = None
    animal_id: str | None = None
    recording_id: str | None = None
    prediction: Any = None
    predictive_scale: float | None = None
    observed: Any = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    manifest: SplitManifest
    predictions: tuple[PredictionRecord, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelScore:
    model_name: str
    endpoint_name: str
    log_loss_nats: float
    n_predictions: int
    n_failed: int
    accuracy: float | None = None


@dataclass(frozen=True)
class RegressionScore:
    model_name: str
    endpoint_name: str
    gaussian_log_loss_nats: float
    rmse: float
    n_predictions: int
    n_failed: int


@dataclass(frozen=True)
class PairedModelDelta:
    baseline_model: str
    candidate_model: str
    endpoint_name: str
    equal_animal_delta_log_loss: float
    per_animal_delta: dict[str, float]
    n_matched_predictions: int
    n_baseline_only: int = 0
    n_candidate_only: int = 0
    n_failed_baseline: int = 0
    n_failed_candidate: int = 0


class FoldFitter(Protocol):
    """Minimal interface used by the skeleton runner and spy tests."""

    name: str

    def fit(self, train_indices: np.ndarray, fold: EvaluationFold) -> Any:
        """Fit using only the fold-owned training indices."""

    def predict(
        self,
        fitted: Any,
        test_indices: np.ndarray,
        fold: EvaluationFold,
    ) -> Sequence[PredictionRecord]:
        """Issue immutable prediction records for the fold-owned test indices."""


def _fold(
    index: int,
    train: np.ndarray,
    test: np.ndarray,
    cfg: EvaluationConfig,
    *,
    status: str = "available",
    failure_reason: str | None = None,
) -> EvaluationFold:
    return EvaluationFold(
        fold_id=f"outer-{index:03d}",
        train_indices=tuple(int(item) for item in train),
        test_indices=tuple(int(item) for item in test),
        protocol=cfg.protocol,
        update_policy=cfg.update_policy,
        status=status,
        failure_reason=failure_reason,
    )


def _blocked_candidates(n_examples: int, n_folds: int) -> list[np.ndarray]:
    effective = max(2, min(int(n_folds), max(2, n_examples // 2)))
    return [block.astype(int) for block in np.array_split(np.arange(n_examples), effective)]


def _mark_fold_availability(
    index: int,
    train: np.ndarray,
    test: np.ndarray,
    cfg: EvaluationConfig,
) -> EvaluationFold:
    if train.size < cfg.min_train_examples or test.size < cfg.min_test_examples:
        return _fold(
            index,
            train,
            test,
            cfg,
            status="unavailable",
            failure_reason="fold_size_below_minimum",
        )
    return _fold(index, train, test, cfg)


def _chronological_splits(
    n_examples: int,
    cfg: EvaluationConfig,
    anchors: Sequence[TemporalAnchor],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build forward-only folds without borrowing observations after a test block."""
    blocks = _blocked_candidates(n_examples, cfg.n_outer_folds + 1)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for test in blocks[1:]:
        first_test = int(test[0])
        train = np.arange(first_test, dtype=int)
        if train.size:
            protected_start = min(anchors[int(index)].context_start for index in test)
            train = np.asarray(
                [
                    int(index)
                    for index in train
                    if anchors[int(index)].context_stop + cfg.embargo <= protected_start
                ],
                dtype=int,
            )
        splits.append((train, test))
    return splits


def build_split_manifest(
    n_examples: int,
    cfg: EvaluationConfig,
    *,
    anchors: Sequence[TemporalAnchor] | None = None,
    dependency_support: DependencySupport | None = None,
) -> SplitManifest:
    """Build outer folds before any learned model fitting.

    Unavailable folds are retained in the manifest with a reason instead of shrinking
    the purge/embargo until a convenient score is calculable.
    """
    if cfg.protocol not in {"purged_blocked", "chronological_online", "leave_animal_out"}:
        raise ValueError(
            "protocol must be one of: purged_blocked, chronological_online, leave_animal_out"
        )
    if n_examples < 1:
        raise ValueError("n_examples must be positive")
    if anchors is not None and len(anchors) != n_examples:
        raise ValueError("anchors must align with examples")

    if cfg.protocol in {"chronological_online", "leave_animal_out"} and anchors is None:
        raise ValueError(f"protocol={cfg.protocol!r} requires physical temporal anchors")

    group_by = "animal" if cfg.protocol == "leave_animal_out" else cfg.group_by
    groups = anchor_group_labels(anchors, group_by=group_by).tolist() if anchors is not None else None
    candidate_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        if cfg.protocol == "chronological_online":
            assert anchors is not None
            candidate_pairs = _chronological_splits(n_examples, cfg, anchors)
        else:
            candidate_pairs = list(
                purged_blocked_splits(
                    n_examples,
                    cfg.n_outer_folds,
                    embargo=cfg.embargo,
                    anchors=anchors,
                    groups=groups,
                    dependency_support=dependency_support or DependencySupport(),
                )
            )
    except ValueError as exc:
        failure_reason = (
            "purging_left_insufficient_training_examples"
            if cfg.protocol == "purged_blocked"
            else f"split_construction_failed:{exc}"
        )
        fallback_tests = _blocked_candidates(n_examples, cfg.n_outer_folds)
        return SplitManifest(
            outer_folds=tuple(
                _fold(
                    index,
                    np.empty(0, dtype=int),
                    test,
                    cfg,
                    status="unavailable",
                    failure_reason=failure_reason,
                )
                for index, test in enumerate(fallback_tests)
            ),
            n_examples=int(n_examples),
            protocol=cfg.protocol,
            group_by=group_by,
            embargo=int(cfg.embargo),
            states_enabled=bool(cfg.states_enabled),
            groups_enabled=bool(cfg.groups_enabled),
        )

    folds = [
        _mark_fold_availability(index, train, test, cfg)
        for index, (train, test) in enumerate(candidate_pairs)
    ]

    return SplitManifest(
        outer_folds=tuple(folds),
        n_examples=int(n_examples),
        protocol=cfg.protocol,
        group_by=group_by,
        embargo=int(cfg.embargo),
        states_enabled=bool(cfg.states_enabled),
        groups_enabled=bool(cfg.groups_enabled),
    )


def run_evaluation_skeleton(
    windows: WindowedSegments,
    cfg: EvaluationConfig,
    *,
    fitters: Sequence[FoldFitter] = (),
    dependency_support: DependencySupport | None = None,
) -> EvaluationResult:
    manifest = build_split_manifest(
        windows.n_windows,
        cfg,
        anchors=windows.anchors or None,
        dependency_support=dependency_support,
    )
    predictions: list[PredictionRecord] = []
    fit_calls = 0
    unavailable = 0
    for fold in manifest.outer_folds:
        if fold.status != "available":
            unavailable += 1
            continue
        train = np.asarray(fold.train_indices, dtype=int)
        test = np.asarray(fold.test_indices, dtype=int)
        for fitter in fitters:
            fitted = fitter.fit(train, fold)
            fit_calls += 1
            predictions.extend(fitter.predict(fitted, test, fold))
    return EvaluationResult(
        manifest=manifest,
        predictions=tuple(predictions),
        diagnostics={
            "fit_calls": fit_calls,
            "available_folds": sum(fold.status == "available" for fold in manifest.outer_folds),
            "unavailable_folds": unavailable,
            "prediction_count": len(predictions),
            "skeleton_status": "split_owner_only",
        },
    )


@dataclass(frozen=True)
class _ProbabilityDecoder:
    scaler: StandardScaler
    model: LogisticRegression | None
    class_order: tuple[Any, ...]
    prior: np.ndarray
    probability_floor: float

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            probabilities = np.repeat(self.prior[None, :], features.shape[0], axis=0)
        else:
            raw = self.model.predict_proba(self.scaler.transform(features))
            probabilities = np.full(
                (features.shape[0], len(self.class_order)), self.probability_floor, dtype=float
            )
            for column, label in enumerate(self.model.classes_):
                probabilities[:, self.class_order.index(label)] = raw[:, column]
        probabilities = np.maximum(probabilities, self.probability_floor)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class _RegressionDecoder:
    scaler: StandardScaler
    model: Ridge
    predictive_scale: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(self.scaler.transform(features)), dtype=float)


def _fit_regression_decoder(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float,
) -> _RegressionDecoder:
    scaler = StandardScaler().fit(features)
    model = Ridge(alpha=alpha).fit(scaler.transform(features), labels.astype(float))
    residuals = labels.astype(float) - model.predict(scaler.transform(features))
    scale = max(float(np.sqrt(np.mean(residuals**2))), 1e-8)
    return _RegressionDecoder(scaler, model, scale)


def _select_decoder_alpha(
    features: np.ndarray,
    labels: np.ndarray,
    cfg: EvaluationConfig,
) -> float:
    if features.shape[0] < 4:
        return float(cfg.decoder_alpha_grid[0])
    try:
        splits = purged_blocked_splits(features.shape[0], min(3, cfg.n_outer_folds))
    except ValueError:
        return float(cfg.decoder_alpha_grid[0])
    losses: dict[float, list[float]] = {float(value): [] for value in cfg.decoder_alpha_grid}
    for train, test in splits:
        for alpha in losses:
            decoder = _fit_regression_decoder(features[train], labels[train], alpha=alpha)
            error = decoder.predict(features[test]) - labels[test].astype(float)
            losses[alpha].append(float(np.mean(error**2)))
    return min(losses, key=lambda value: (float(np.mean(losses[value])), value))


def _fit_probability_decoder(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    class_order: tuple[Any, ...],
    c_value: float,
    probability_floor: float,
    seed: int,
) -> _ProbabilityDecoder:
    scaler = StandardScaler().fit(features)
    counts = np.asarray([np.sum(labels == label) for label in class_order], dtype=float) + 1.0
    prior = counts / counts.sum()
    observed_classes = np.unique(labels)
    model = None
    if observed_classes.size > 1:
        model = LogisticRegression(
            C=c_value,
            max_iter=2_000,
            random_state=seed,
        ).fit(scaler.transform(features), labels)
    return _ProbabilityDecoder(scaler, model, class_order, prior, probability_floor)


def _mean_log_loss(
    decoder: _ProbabilityDecoder,
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    probabilities = decoder.predict_proba(features)
    indices = np.asarray([decoder.class_order.index(label) for label in labels], dtype=int)
    return float(-np.mean(np.log(probabilities[np.arange(labels.size), indices])))


def _select_decoder_c(
    features: np.ndarray,
    labels: np.ndarray,
    cfg: EvaluationConfig,
    class_order: tuple[Any, ...],
) -> float:
    if features.shape[0] < 4:
        return float(cfg.decoder_c_grid[0])
    try:
        splits = purged_blocked_splits(features.shape[0], min(3, cfg.n_outer_folds))
    except ValueError:
        return float(cfg.decoder_c_grid[0])
    losses: dict[float, list[float]] = {float(value): [] for value in cfg.decoder_c_grid}
    for train, test in splits:
        for c_value in losses:
            decoder = _fit_probability_decoder(
                features[train],
                labels[train],
                class_order=class_order,
                c_value=c_value,
                probability_floor=cfg.probability_floor,
                seed=cfg.selection_seed,
            )
            losses[c_value].append(_mean_log_loss(decoder, features[test], labels[test]))
    return min(losses, key=lambda value: (float(np.mean(losses[value])), value))


def _neural_holdout_loss(segment: np.ndarray, cfg: EvaluationConfig, lambda1: float) -> float:
    cut = max(cfg.neural_max_lag + 2, int(np.floor(0.8 * segment.shape[0])))
    if cut >= segment.shape[0]:
        return float("inf")
    model = fit_sparse_var_model(
        segment[:cut],
        cfg=ConnectivityConfig(
            name="sparse_var",
            max_lag=cfg.neural_max_lag,
            ridge=cfg.neural_lambda2,
            extra={
                "lambda1": lambda1,
                "lambda2": cfg.neural_lambda2,
                "standardize": True,
            },
        ),
    )
    state = model.initial_state()
    history_start = cut - cfg.neural_max_lag
    state = model.observe(
        state,
        segment[history_start:cut],
        sample_indices=np.arange(history_start, cut),
    )
    errors: list[float] = []
    for sample_index in range(cut, segment.shape[0]):
        forecast = model.forecast(state, horizon_samples=1)
        if forecast.failure_reason is not None:
            return float("inf")
        errors.append(float(np.mean((forecast.mean[0] - segment[sample_index]) ** 2)))
        state = model.observe(
            state,
            segment[sample_index : sample_index + 1],
            sample_indices=np.asarray([sample_index]),
        )
    return float(np.mean(errors)) if errors else float("inf")


def _select_neural_lambda(
    windows: WindowedSegments,
    origin_indices: np.ndarray,
    cfg: EvaluationConfig,
) -> float:
    losses: dict[float, list[float]] = {
        float(value): [] for value in cfg.neural_lambda1_grid
    }
    for origin in origin_indices:
        segment = np.asarray(windows.segments[int(origin)], dtype=np.float64)
        for value in losses:
            try:
                losses[value].append(_neural_holdout_loss(segment, cfg, value))
            except (ValueError, np.linalg.LinAlgError):
                losses[value].append(float("inf"))
    return min(losses, key=lambda value: (float(np.mean(losses[value])), value))


def _baseline_feature(segment: np.ndarray) -> np.ndarray:
    values = np.asarray(segment, dtype=np.float64)
    return np.concatenate([values[-1], values.mean(axis=0), values.std(axis=0)])


def _effectome_feature(
    segment: np.ndarray,
    cfg: EvaluationConfig,
    lambda1: float,
) -> tuple[np.ndarray | None, str | None]:
    try:
        model = fit_sparse_var_model(
            np.asarray(segment, dtype=np.float64),
            cfg=ConnectivityConfig(
                name="sparse_var",
                max_lag=cfg.neural_max_lag,
                ridge=cfg.neural_lambda2,
                extra={
                    "lambda1": lambda1,
                    "lambda2": cfg.neural_lambda2,
                    "standardize": True,
                },
            ),
        )
        graph = model.graph_at()
        return graph.lagged_coefficients.ravel(), None
    except (ValueError, np.linalg.LinAlgError) as exc:
        return None, f"{type(exc).__name__}:{exc}"


def run_primary_evaluation(
    windows: WindowedSegments,
    cfg: EvaluationConfig,
    *,
    dependency_support: DependencySupport | None = None,
    selected_fold_ids: frozenset[str] | None = None,
) -> EvaluationResult:
    """Run the no-state/no-group probability-level M0/M1 comparison."""
    if cfg.states_enabled or cfg.groups_enabled:
        raise ValueError("primary M0/M1 requires optional states and groups to be disabled")
    if cfg.endpoint_name not in windows.behavior_per_window:
        raise KeyError(f"endpoint {cfg.endpoint_name!r} is absent from windowed behavior")
    origins = valid_positive_lag_origins(
        windows.n_windows,
        cfg.positive_lag,
        windows.anchors or None,
        strict=True,
    )
    if origins.size < 4:
        raise ValueError("too few strictly future endpoint origins for primary evaluation")
    origin_anchors = [windows.anchors[int(index)] for index in origins]
    manifest = build_split_manifest(
        len(origins), cfg, anchors=origin_anchors, dependency_support=dependency_support
    )
    endpoint = np.asarray(windows.behavior_per_window[cfg.endpoint_name])
    class_order = ()
    if cfg.endpoint_type == "categorical":
        class_order = (
            tuple(cfg.class_vocabulary)
            if cfg.class_vocabulary
            else tuple(np.unique(endpoint[origins + cfg.positive_lag]).tolist())
        )
    baseline_features = np.stack(
        [_baseline_feature(windows.segments[int(origin)]) for origin in origins]
    )
    labels = endpoint[origins + cfg.positive_lag]
    predictions: list[PredictionRecord] = []
    fold_diagnostics: dict[str, Any] = {}

    for fold in manifest.outer_folds:
        if fold.status != "available":
            continue
        if selected_fold_ids is not None and fold.fold_id not in selected_fold_ids:
            continue
        train = np.asarray(fold.train_indices, dtype=int)
        test = np.asarray(fold.test_indices, dtype=int)
        lambda1 = _select_neural_lambda(windows, origins[train], cfg)
        graph_features: dict[int, np.ndarray] = {}
        graph_failures: dict[int, str] = {}
        for local_index in np.concatenate([train, test]):
            feature, failure = _effectome_feature(
                windows.segments[int(origins[local_index])], cfg, lambda1
            )
            if feature is None:
                graph_failures[int(local_index)] = failure or "effectome_fit_failed"
            else:
                graph_features[int(local_index)] = feature
        m1_training_available = all(int(index) in graph_features for index in train)
        m0_tuning: float
        m0_probabilities = np.empty((0, len(class_order)))
        m0_predictions = np.empty(0, dtype=float)
        m0_scale: float | None = None
        if cfg.endpoint_type == "categorical":
            m0_tuning = _select_decoder_c(
                baseline_features[train], labels[train], cfg, class_order
            )
            m0 = _fit_probability_decoder(
                baseline_features[train],
                labels[train],
                class_order=class_order,
                c_value=m0_tuning,
                probability_floor=cfg.probability_floor,
                seed=cfg.selection_seed,
            )
            m0_probabilities = m0.predict_proba(baseline_features[test])
        else:
            m0_tuning = _select_decoder_alpha(
                baseline_features[train], labels[train], cfg
            )
            m0_regression = _fit_regression_decoder(
                baseline_features[train], labels[train], alpha=m0_tuning
            )
            m0_predictions = m0_regression.predict(baseline_features[test])
            m0_scale = m0_regression.predictive_scale
        m1_tuning: float | None = None
        m1_probabilities = np.empty((0, len(class_order)))
        m1_predictions = np.empty(0, dtype=float)
        m1_scale: float | None = None
        available_test = np.zeros(test.size, dtype=bool)
        if m1_training_available:
            graph_width = next(iter(graph_features.values())).size
            m1_features = np.stack(
                [
                    np.concatenate([baseline_features[index], graph_features[int(index)]])
                    if int(index) in graph_features
                    else np.full(baseline_features.shape[1] + graph_width, np.nan)
                    for index in range(len(origins))
                ]
            )
            available_test = np.asarray([int(index) in graph_features for index in test])
            if cfg.endpoint_type == "categorical":
                m1_tuning = _select_decoder_c(
                    m1_features[train], labels[train], cfg, class_order
                )
                m1 = _fit_probability_decoder(
                    m1_features[train],
                    labels[train],
                    class_order=class_order,
                    c_value=m1_tuning,
                    probability_floor=cfg.probability_floor,
                    seed=cfg.selection_seed,
                )
                if np.any(available_test):
                    m1_probabilities = m1.predict_proba(m1_features[test[available_test]])
            else:
                m1_tuning = _select_decoder_alpha(m1_features[train], labels[train], cfg)
                m1_regression = _fit_regression_decoder(
                    m1_features[train], labels[train], alpha=m1_tuning
                )
                m1_scale = m1_regression.predictive_scale
                if np.any(available_test):
                    m1_predictions = m1_regression.predict(m1_features[test[available_test]])
        m1_row = 0
        for row, local_index in enumerate(test):
            origin = int(origins[local_index])
            source_anchor = windows.anchors[origin]
            target_anchor = windows.anchors[origin + cfg.positive_lag]
            common = {
                "origin_index": origin,
                "fold_id": fold.fold_id,
                "endpoint_name": cfg.endpoint_name,
                "endpoint_kind": cfg.endpoint_kind,
                "endpoint_type": cfg.endpoint_type,
                "class_order": class_order,
                "issue_sample": source_anchor.context_stop,
                "target_start": target_anchor.target_start,
                "target_stop": target_anchor.target_stop,
                "animal_id": source_anchor.animal_id,
                "recording_id": source_anchor.recording_id,
                "observed": labels[local_index],
            }
            if cfg.endpoint_type == "categorical":
                predictions.append(
                    PredictionRecord(
                        model_name="M0_activity",
                        status="issued",
                        probabilities=tuple(float(value) for value in m0_probabilities[row]),
                        **common,
                    )
                )
            else:
                predictions.append(
                    PredictionRecord(
                        model_name="M0_activity",
                        status="issued",
                        prediction=float(m0_predictions[row]),
                        predictive_scale=m0_scale,
                        **common,
                    )
                )
            if available_test[row]:
                if cfg.endpoint_type == "categorical":
                    predictions.append(
                        PredictionRecord(
                            model_name="M1_effectome",
                            status="issued",
                            probabilities=tuple(
                                float(value) for value in m1_probabilities[m1_row]
                            ),
                            **common,
                        )
                    )
                else:
                    predictions.append(
                        PredictionRecord(
                            model_name="M1_effectome",
                            status="issued",
                            prediction=float(m1_predictions[m1_row]),
                            predictive_scale=m1_scale,
                            **common,
                        )
                    )
                m1_row += 1
            else:
                predictions.append(
                    PredictionRecord(
                        model_name="M1_effectome",
                        status="failed",
                        failure_reason=(
                            graph_failures.get(int(local_index), "effectome_fit_failed")
                            if m1_training_available
                            else "outer_training_effectome_feature_failure"
                        ),
                        **common,
                    )
                )
        fold_diagnostics[fold.fold_id] = {
            "lambda1": lambda1,
            "decoder_parameter": "C" if cfg.endpoint_type == "categorical" else "alpha",
            "m0_decoder_value": m0_tuning,
            "m1_decoder_value": m1_tuning,
            "m0_decoder_c": m0_tuning if cfg.endpoint_type == "categorical" else None,
            "m1_decoder_c": m1_tuning if cfg.endpoint_type == "categorical" else None,
            "training_examples": int(train.size),
            "test_examples": int(test.size),
        }
    return EvaluationResult(
        manifest=manifest,
        predictions=tuple(predictions),
        diagnostics={
            "status": "primary_probability_comparison",
            "profile": cfg.profile,
            "mode": cfg.mode,
            "endpoint": cfg.endpoint_name,
            "origin_count": int(origins.size),
            "available_folds": sum(
                fold.status == "available" for fold in manifest.outer_folds
            ),
            "unavailable_folds": sum(
                fold.status != "available" for fold in manifest.outer_folds
            ),
            "claim_to_test_map": dict(cfg.claim_to_test_map)
            or {"incremental_predictive_value": "paired_M0_M1_log_loss"},
            "folds": fold_diagnostics,
        },
    )


def primary_evaluation_fold_ids(
    windows: WindowedSegments,
    cfg: EvaluationConfig,
    *,
    dependency_support: DependencySupport | None = None,
) -> tuple[str, ...]:
    """Return the stable available outer-fold IDs without fitting any model."""
    origins = valid_positive_lag_origins(
        windows.n_windows,
        cfg.positive_lag,
        windows.anchors or None,
        strict=True,
    )
    if origins.size < 4:
        raise ValueError("too few strictly future endpoint origins for primary evaluation")
    manifest = build_split_manifest(
        len(origins),
        cfg,
        anchors=[windows.anchors[int(index)] for index in origins],
        dependency_support=dependency_support,
    )
    return tuple(
        fold.fold_id for fold in manifest.outer_folds if fold.status == "available"
    )


def combine_primary_evaluation_results(
    results: Sequence[EvaluationResult],
    *,
    cfg: EvaluationConfig,
) -> EvaluationResult:
    """Combine independently checkpointed fold results into one immutable artifact."""
    if not results:
        raise ValueError("at least one available primary-evaluation fold is required")
    manifest = results[0].manifest
    if any(result.manifest != manifest for result in results[1:]):
        raise ValueError("primary evaluation fold manifests do not match")
    predictions = tuple(
        prediction for result in results for prediction in result.predictions
    )
    fold_diagnostics: dict[str, Any] = {}
    for result in results:
        overlap = set(fold_diagnostics) & set(result.diagnostics.get("folds", {}))
        if overlap:
            raise ValueError(f"duplicate primary evaluation folds: {sorted(overlap)}")
        fold_diagnostics.update(result.diagnostics.get("folds", {}))
    expected = {
        fold.fold_id for fold in manifest.outer_folds if fold.status == "available"
    }
    if set(fold_diagnostics) != expected:
        missing = sorted(expected - set(fold_diagnostics))
        raise ValueError(f"primary evaluation fold results are incomplete: {missing}")
    return EvaluationResult(
        manifest=manifest,
        predictions=predictions,
        diagnostics={
            "status": "primary_probability_comparison",
            "profile": cfg.profile,
            "mode": cfg.mode,
            "endpoint": cfg.endpoint_name,
            "origin_count": manifest.n_examples,
            "available_folds": len(expected),
            "unavailable_folds": sum(
                fold.status != "available" for fold in manifest.outer_folds
            ),
            "claim_to_test_map": dict(cfg.claim_to_test_map)
            or {"incremental_predictive_value": "paired_M0_M1_log_loss"},
            "folds": fold_diagnostics,
            "checkpoint_mode": "outer_fold",
        },
    )
def _probability_for_observed(record: PredictionRecord) -> float:
    if not record.class_order or not record.probabilities:
        raise ValueError("classification prediction records require class_order and probabilities")
    if len(record.class_order) != len(record.probabilities):
        raise ValueError("class_order and probabilities must have the same length")
    probs = np.asarray(record.probabilities, dtype=float)
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("probabilities must have positive total mass")
    probs = probs / total
    try:
        index = tuple(record.class_order).index(record.observed)
    except ValueError as exc:
        raise ValueError("observed class is absent from the registered vocabulary") from exc
    return float(np.clip(probs[index], 1e-12, 1.0))


def score_classification_records(
    records: Sequence[PredictionRecord],
    *,
    model_name: str,
    endpoint_name: str,
) -> ModelScore:
    selected = [
        record
        for record in records
        if record.model_name == model_name and record.endpoint_name == endpoint_name
    ]
    issued = [record for record in selected if record.status == "issued"]
    failed = len(selected) - len(issued)
    if not issued:
        return ModelScore(
            model_name=model_name,
            endpoint_name=endpoint_name,
            log_loss_nats=float("nan"),
            n_predictions=0,
            n_failed=failed,
            accuracy=None,
        )
    losses = [-np.log(_probability_for_observed(record)) for record in issued]
    predictions = [
        record.class_order[int(np.argmax(np.asarray(record.probabilities, dtype=float)))]
        for record in issued
    ]
    accuracy = float(
        np.mean(
            [pred == record.observed for pred, record in zip(predictions, issued, strict=True)]
        )
    )
    return ModelScore(
        model_name=model_name,
        endpoint_name=endpoint_name,
        log_loss_nats=float(np.mean(losses)),
        n_predictions=len(issued),
        n_failed=failed,
        accuracy=accuracy,
    )


def _gaussian_loss(record: PredictionRecord) -> float:
    if record.prediction is None or record.predictive_scale is None:
        raise ValueError("continuous prediction records require a mean and predictive scale")
    mean = float(record.prediction)
    observed = float(record.observed)
    scale = float(record.predictive_scale)
    if not np.isfinite([mean, observed, scale]).all() or scale <= 0:
        raise ValueError("continuous prediction parameters must be finite with positive scale")
    residual = (observed - mean) / scale
    return float(0.5 * np.log(2.0 * np.pi * scale**2) + 0.5 * residual**2)


def score_regression_records(
    records: Sequence[PredictionRecord],
    *,
    model_name: str,
    endpoint_name: str,
) -> RegressionScore:
    selected = [
        record
        for record in records
        if record.model_name == model_name and record.endpoint_name == endpoint_name
    ]
    issued = [record for record in selected if record.status == "issued"]
    failed = len(selected) - len(issued)
    if not issued:
        return RegressionScore(model_name, endpoint_name, float("nan"), float("nan"), 0, failed)
    losses = [_gaussian_loss(record) for record in issued]
    squared_errors = [
        (float(record.observed) - float(record.prediction)) ** 2 for record in issued
    ]
    return RegressionScore(
        model_name=model_name,
        endpoint_name=endpoint_name,
        gaussian_log_loss_nats=float(np.mean(losses)),
        rmse=float(np.sqrt(np.mean(squared_errors))),
        n_predictions=len(issued),
        n_failed=failed,
    )


def _predictive_loss(record: PredictionRecord) -> float:
    if record.endpoint_type == "continuous":
        return _gaussian_loss(record)
    return float(-np.log(_probability_for_observed(record)))


def paired_log_loss_delta(
    records: Sequence[PredictionRecord],
    *,
    baseline_model: str,
    candidate_model: str,
    endpoint_name: str,
) -> PairedModelDelta:
    by_key: dict[
        tuple[str, str, str, int, int | None, int | None, int | None],
        dict[str, PredictionRecord],
    ] = {}
    failed = {baseline_model: 0, candidate_model: 0}
    for record in records:
        if record.endpoint_name != endpoint_name:
            continue
        if record.model_name not in {baseline_model, candidate_model}:
            continue
        if record.status != "issued":
            failed[record.model_name] += 1
            continue
        animal = record.animal_id or record.recording_id or "unknown"
        recording = record.recording_id or "unknown"
        key = (
            animal,
            recording,
            record.fold_id,
            int(record.origin_index),
            record.issue_sample,
            record.target_start,
            record.target_stop,
        )
        models = by_key.setdefault(key, {})
        if record.model_name in models:
            raise ValueError(
                "duplicate prediction record for one model and physical origin: "
                f"{record.model_name} {key}"
            )
        models[record.model_name] = record

    animal_deltas: dict[str, list[float]] = {}
    matched = 0
    baseline_only = 0
    candidate_only = 0
    for (animal, _recording, _fold, _origin, _issue, _start, _stop), pair in by_key.items():
        if baseline_model not in pair or candidate_model not in pair:
            baseline_only += int(baseline_model in pair)
            candidate_only += int(candidate_model in pair)
            continue
        if pair[baseline_model].observed != pair[candidate_model].observed:
            raise ValueError("paired models disagree on the observed endpoint")
        if pair[baseline_model].class_order != pair[candidate_model].class_order:
            raise ValueError("paired models use different endpoint class vocabularies")
        baseline_loss = _predictive_loss(pair[baseline_model])
        candidate_loss = _predictive_loss(pair[candidate_model])
        animal_deltas.setdefault(animal, []).append(float(baseline_loss - candidate_loss))
        matched += 1

    per_animal = {animal: float(np.mean(values)) for animal, values in animal_deltas.items()}
    equal_animal = float(np.mean(list(per_animal.values()))) if per_animal else float("nan")
    return PairedModelDelta(
        baseline_model=baseline_model,
        candidate_model=candidate_model,
        endpoint_name=endpoint_name,
        equal_animal_delta_log_loss=equal_animal,
        per_animal_delta=per_animal,
        n_matched_predictions=matched,
        n_baseline_only=baseline_only,
        n_candidate_only=candidate_only,
        n_failed_baseline=failed[baseline_model],
        n_failed_candidate=failed[candidate_model],
    )


__all__ = [
    "EvaluationConfig",
    "EvaluationFold",
    "EvaluationResult",
    "FoldFitter",
    "PredictionRecord",
    "RegressionScore",
    "SplitManifest",
    "build_split_manifest",
    "combine_primary_evaluation_results",
    "paired_log_loss_delta",
    "primary_evaluation_fold_ids",
    "run_evaluation_skeleton",
    "run_primary_evaluation",
    "score_classification_records",
    "score_regression_records",
]
