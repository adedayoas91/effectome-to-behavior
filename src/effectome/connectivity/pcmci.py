"""PCMCI+ causal discovery (constraint-based) via tigramite.

PCMCI+ is well-suited to neural time series: it controls for autocorrelation and conditions on
parents to remove spurious links, returning lagged *and* contemporaneous causal graphs. This
wraps tigramite so it plugs into the same `ConnectivityEstimator` interface. The heavy `csl`
extra (tigramite) is imported lazily so the package installs without it.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import ConnectivityEstimator
from .registry import register_connectivity

logger = logging.getLogger(__name__)


@register_connectivity("pcmci")
class PCMCIConnectivity(ConnectivityEstimator):
    """Directed connectivity from PCMCI+ (partial-correlation conditional-independence test).

    Returns an N x N matrix whose entry [i, j] is the strongest |MCI| test statistic for a
    causal link i -> j across lags 1..max_lag (plus contemporaneous if enabled), with
    non-significant links (p > alpha) set to zero.
    """

    directed = True

    def estimate(self, segment: np.ndarray) -> np.ndarray:
        try:
            from tigramite import data_processing as pp
            from tigramite.independence_tests.parcorr import ParCorr
            from tigramite.pcmci import PCMCI
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "PCMCI requires the 'csl' extra: `uv pip install -e '.[csl]'` (installs tigramite)."
            ) from exc

        x = np.asarray(segment, dtype=np.float64)  # (L, N)
        n = x.shape[1]
        dataframe = pp.DataFrame(x, var_names=[str(i) for i in range(n)])
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)

        tau_max = max(1, self.cfg.max_lag)
        results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=self.cfg.alpha)
        val = np.abs(results["val_matrix"])  # (N, N, tau_max+1), [i, j, lag] = i -> j
        pmat = results["p_matrix"]

        sig = val * (pmat <= self.cfg.alpha)
        influence = sig.max(axis=2)  # collapse lags -> N x N (source i -> target j)
        return influence
