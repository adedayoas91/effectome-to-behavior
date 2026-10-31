"""Registry wrappers for Tigramite causal discovery estimators."""

from __future__ import annotations

import numpy as np

from .base import ConnectivityEstimator
from .registry import register_connectivity


def _collapse_tigramite_results(
    val_matrix: np.ndarray,
    p_matrix: np.ndarray,
    *,
    alpha: float,
    tau_min: int,
) -> np.ndarray:
    val = np.asarray(val_matrix, dtype=np.float64)
    pmat = np.asarray(p_matrix, dtype=np.float64)
    if not 1 <= tau_min < val.shape[2]:
        raise ValueError("directed PCMCI adapters require positive lags within tau_max")
    lag_slice = slice(tau_min, val.shape[2])
    sig = np.where(pmat[:, :, lag_slice] <= alpha, val[:, :, lag_slice], 0.0)
    strongest = np.argmax(np.abs(sig), axis=2)
    rows = np.arange(sig.shape[0])[:, None]
    cols = np.arange(sig.shape[1])[None, :]
    influence = sig[rows, cols, strongest]
    np.fill_diagonal(influence, 0.0)
    return influence


def _jpcmci_node_classification(
    n_variables: int, node_classification: dict[int, str] | None
) -> dict[int, str]:
    """Treat unspecified variables as system nodes; contexts require explicit labels."""
    if node_classification is None:
        return {i: "system" for i in range(n_variables)}
    valid_types = {"system", "time_context", "space_context", "time_dummy", "space_dummy"}
    if set(node_classification) != set(range(n_variables)):
        raise ValueError("node_classification must classify every variable by integer index")
    if not set(node_classification.values()) <= valid_types:
        raise ValueError(f"unknown JPCMCI+ node classification; expected one of {sorted(valid_types)}")
    return dict(node_classification)


@register_connectivity("pcmci")
class PCMCIConnectivity(ConnectivityEstimator):
    """Directed connectivity from PCMCI+."""

    directed = True
    weight_semantics = "signed_mci_dependence_statistic"

    def estimate(self, segment: np.ndarray) -> np.ndarray:
        try:
            from tigramite import data_processing as pp
            from tigramite.independence_tests.parcorr import ParCorr
            from tigramite.pcmci import PCMCI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PCMCI+ requires Tigramite in the active venv. Install it in `.venv` first."
            ) from exc

        x = np.asarray(segment, dtype=np.float64)
        tau_min = int(self.cfg.extra.get("tau_min", 1))
        pcmci = PCMCI(
            dataframe=pp.DataFrame(x, var_names=[str(i) for i in range(x.shape[1])]),
            cond_ind_test=ParCorr(),
            verbosity=0,
        )
        results = pcmci.run_pcmciplus(
            tau_min=tau_min,
            tau_max=max(1, self.cfg.max_lag),
            pc_alpha=self.cfg.extra.get("pc_alpha", self.cfg.alpha),
            **self.cfg.extra.get("run_kwargs", {}),
        )
        return _collapse_tigramite_results(
            results["val_matrix"],
            results["p_matrix"],
            alpha=self.cfg.alpha,
            tau_min=tau_min,
        )


@register_connectivity("jpcmci")
class JPCMCIConnectivity(ConnectivityEstimator):
    """JPCMCI+ wrapper for pooled recordings in notebook-style workflows."""

    directed = True
    weight_semantics = "signed_mci_dependence_statistic"

    def estimate(self, segment: np.ndarray) -> np.ndarray:
        try:
            from tigramite import data_processing as pp
            from tigramite.independence_tests.parcorr_mult import ParCorrMult
            from tigramite.jpcmciplus import JPCMCIplus
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "JPCMCI+ requires Tigramite with `tigramite.jpcmciplus` in the active venv."
            ) from exc

        tau_min = int(self.cfg.extra.get("tau_min", 1))
        recordings = self.cfg.extra.get("recordings")
        if recordings is None:
            recordings = {0: np.asarray(segment, dtype=np.float64)}
        elif not isinstance(recordings, dict):
            recordings = {idx: np.asarray(value, dtype=np.float64) for idx, value in enumerate(recordings)}
        first = next(iter(recordings.values()))
        estimator = JPCMCIplus(
            dataframe=pp.DataFrame(
                data=recordings,
                analysis_mode="multiple",
                vector_vars=self.cfg.extra.get("vector_vars"),
                var_names=self.cfg.extra.get("var_names", [str(i) for i in range(first.shape[1])]),
            ),
            cond_ind_test=ParCorrMult(significance="analytic"),
            node_classification=_jpcmci_node_classification(
                len(self.cfg.extra["vector_vars"]) if self.cfg.extra.get("vector_vars") else first.shape[1],
                self.cfg.extra.get("node_classification"),
            ),
            verbosity=0,
        )
        results = estimator.run_jpcmciplus(
            tau_min=tau_min,
            tau_max=max(1, self.cfg.max_lag),
            pc_alpha=self.cfg.extra.get("pc_alpha", self.cfg.alpha),
            **self.cfg.extra.get("run_kwargs", {}),
        )
        return _collapse_tigramite_results(
            results["val_matrix"],
            results["p_matrix"],
            alpha=self.cfg.alpha,
            tau_min=tau_min,
        )
