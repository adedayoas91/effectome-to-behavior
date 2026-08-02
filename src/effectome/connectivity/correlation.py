"""Correlation-based functional connectivity (non-causal baseline)."""

from __future__ import annotations

import numpy as np

from .base import ConnectivityEstimator
from .registry import register_connectivity


@register_connectivity("correlation")
class CorrelationConnectivity(ConnectivityEstimator):
    """Pearson or partial-correlation functional connectivity.

    Symmetric (undirected). Partial correlation conditions out all other neurons via the
    precision matrix, giving a sparser, more specific functional graph.
    """

    directed = False
    weight_semantics = "functional_association"

    def estimate(self, segment: np.ndarray) -> np.ndarray:
        # segment: (L, N) -> N x N
        x = segment - segment.mean(axis=0, keepdims=True)
        if not self.cfg.partial:
            c = np.corrcoef(x, rowvar=False)
            return np.nan_to_num(c)

        # Partial correlation from the (regularized) precision matrix.
        cov = np.cov(x, rowvar=False)
        cov += self.cfg.ridge * 1e-3 * np.eye(cov.shape[0])
        precision = np.linalg.pinv(cov)
        d = np.sqrt(np.diag(precision))
        denom = np.outer(d, d)
        pcorr = -precision / np.where(denom == 0, 1.0, denom)
        np.fill_diagonal(pcorr, 1.0)
        return np.nan_to_num(pcorr)
