"""Estimator-specific scaling and sign/rank agreement for effectome sensitivity analyses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from effectome.data_module.schema import ConnectivitySeries


@dataclass(frozen=True)
class SignedEffectomeScaler:
    """A train-fitted positive scale that preserves zero and coefficient sign."""

    scale: float

    @classmethod
    def fit(
        cls,
        matrices: np.ndarray,
        fit_indices: np.ndarray | list[int] | None = None,
    ) -> SignedEffectomeScaler:
        values = np.asarray(matrices, dtype=float)
        if values.ndim != 3 or values.shape[1] != values.shape[2]:
            raise ValueError("matrices must have shape (anchor, node, node)")
        fitted = values if fit_indices is None else values[np.asarray(fit_indices, dtype=int)]
        nonzero = np.abs(fitted[np.abs(fitted) > 1e-12])
        scale = float(np.median(nonzero)) if nonzero.size else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        return cls(scale=scale)

    def transform(self, matrices: np.ndarray) -> np.ndarray:
        return np.asarray(matrices, dtype=float) / self.scale


@dataclass(frozen=True)
class EffectomeAgreement:
    """Scale-free agreement between two aligned estimator outputs."""

    edge_rank_correlation: float
    sign_agreement: float
    support_jaccard: float
    outgoing_role_rank_correlation: float
    incoming_role_rank_correlation: float
    n_mutually_present_edges: int


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_arr = np.asarray(left, dtype=float).ravel()
    right_arr = np.asarray(right, dtype=float).ravel()
    if left_arr.size < 2 or np.all(left_arr == left_arr[0]) or np.all(right_arr == right_arr[0]):
        return 0.0
    statistic = float(spearmanr(left_arr, right_arr).statistic)
    return statistic if np.isfinite(statistic) else 0.0


def signed_effectome_agreement(
    reference: ConnectivitySeries,
    candidate: ConnectivitySeries,
    zero_tolerance: float = 1e-12,
) -> EffectomeAgreement:
    """Compare aligned estimators without pooling their incomparable raw magnitudes."""
    left = np.asarray(reference.matrices, dtype=float)
    right = np.asarray(candidate.matrices, dtype=float)
    if left.shape != right.shape:
        raise ValueError("effectome series must have identical aligned shapes")
    if reference.window_starts.shape != candidate.window_starts.shape or not np.array_equal(
        reference.window_starts, candidate.window_starts
    ):
        raise ValueError("effectome series must use identical temporal anchors")

    n_nodes = left.shape[1]
    off_diagonal = ~np.eye(n_nodes, dtype=bool)
    left_edges = left[:, off_diagonal]
    right_edges = right[:, off_diagonal]
    left_present = np.abs(left_edges) > zero_tolerance
    right_present = np.abs(right_edges) > zero_tolerance
    mutual = left_present & right_present
    union = left_present | right_present

    sign_agreement = (
        float(np.mean(np.sign(left_edges[mutual]) == np.sign(right_edges[mutual])))
        if np.any(mutual)
        else 0.0
    )
    support_jaccard = float(np.sum(mutual) / np.sum(union)) if np.any(union) else 1.0

    left_scaler = SignedEffectomeScaler.fit(left)
    right_scaler = SignedEffectomeScaler.fit(right)
    left_scaled = left_scaler.transform(left)
    right_scaled = right_scaler.transform(right)
    left_outgoing = left_scaled.sum(axis=(0, 2))
    right_outgoing = right_scaled.sum(axis=(0, 2))
    left_incoming = left_scaled.sum(axis=(0, 1))
    right_incoming = right_scaled.sum(axis=(0, 1))

    return EffectomeAgreement(
        edge_rank_correlation=_rank_correlation(left_edges, right_edges),
        sign_agreement=sign_agreement,
        support_jaccard=support_jaccard,
        outgoing_role_rank_correlation=_rank_correlation(left_outgoing, right_outgoing),
        incoming_role_rank_correlation=_rank_correlation(left_incoming, right_incoming),
        n_mutually_present_edges=int(np.sum(mutual)),
    )
