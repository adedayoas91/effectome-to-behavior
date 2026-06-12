"""Notebook-facing analyzers modeled on markovianity_diagnostic."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from effectome.core import CausalisedGC


def _collapse_tigramite_results(
    val_matrix: np.ndarray,
    p_matrix: np.ndarray,
    *,
    alpha: float,
    tau_min: int,
) -> np.ndarray:
    val = np.asarray(val_matrix, dtype=np.float64)
    pmat = np.asarray(p_matrix, dtype=np.float64)
    lag_slice = slice(max(tau_min, 0), val.shape[2])
    sig = np.where(pmat[:, :, lag_slice] <= alpha, val[:, :, lag_slice], 0.0)
    strongest = np.argmax(np.abs(sig), axis=2)
    rows = np.arange(sig.shape[0])[:, None]
    cols = np.arange(sig.shape[1])[None, :]
    influence = sig[rows, cols, strongest]
    np.fill_diagonal(influence, 0.0)
    return influence


def _load_tigramite_pcmci():
    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PCMCI+ requires Tigramite in the active venv. Install it in `.venv` first."
        ) from exc
    return pp, ParCorr, PCMCI


def _load_tigramite_jpcmci():
    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr_mult import ParCorrMult
        from tigramite.jpcmciplus import JPCMCIplus
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "JPCMCI+ requires Tigramite with `tigramite.jpcmciplus` in the active venv."
        ) from exc
    return pp, ParCorrMult, JPCMCIplus


def _normalize_multiple_recordings(
    recordings: dict[Any, np.ndarray] | list[np.ndarray] | tuple[np.ndarray, ...],
) -> dict[Any, np.ndarray]:
    if isinstance(recordings, dict):
        data_dict = {key: np.asarray(value, dtype=np.float64) for key, value in recordings.items()}
    else:
        data_dict = {idx: np.asarray(value, dtype=np.float64) for idx, value in enumerate(recordings)}
    if not data_dict:
        raise ValueError("recordings must contain at least one dataset.")
    n_vars = next(iter(data_dict.values())).shape[1]
    for key, value in data_dict.items():
        if value.ndim != 2:
            raise ValueError(f"Recording {key!r} must have shape (T, N), got {value.shape}.")
        if value.shape[1] != n_vars:
            raise ValueError("All recordings must share the same number of variables.")
    return data_dict


def _run_pcmciplus_matrix(
    x: np.ndarray,
    *,
    tau_max: int,
    alpha: float,
    tau_min: int,
    pc_alpha: float | None,
    run_kwargs: dict[str, Any],
) -> np.ndarray:
    pp, ParCorr, PCMCI = _load_tigramite_pcmci()
    x = np.asarray(x, dtype=np.float64)
    pcmci = PCMCI(
        dataframe=pp.DataFrame(x, var_names=[str(i) for i in range(x.shape[1])]),
        cond_ind_test=ParCorr(),
        verbosity=0,
    )
    results = pcmci.run_pcmciplus(
        tau_min=tau_min,
        tau_max=tau_max,
        pc_alpha=pc_alpha if pc_alpha is not None else alpha,
        **run_kwargs,
    )
    return _collapse_tigramite_results(
        results["val_matrix"],
        results["p_matrix"],
        alpha=alpha,
        tau_min=tau_min,
    )


def _run_jpcmciplus_matrix(
    recordings: dict[Any, np.ndarray] | list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    tau_max: int,
    alpha: float,
    tau_min: int,
    pc_alpha: float | None,
    node_classification: dict[int, str] | None,
    vector_vars: dict[int, list[tuple[int, int]]] | None,
    var_names: list[str] | None,
    run_kwargs: dict[str, Any],
) -> np.ndarray:
    pp, ParCorrMult, JPCMCIplus = _load_tigramite_jpcmci()
    data_dict = _normalize_multiple_recordings(recordings)
    first = next(iter(data_dict.values()))
    estimator = JPCMCIplus(
        dataframe=pp.DataFrame(
            data=data_dict,
            analysis_mode="multiple",
            vector_vars=vector_vars,
            var_names=var_names if var_names is not None else [str(i) for i in range(first.shape[1])],
        ),
        cond_ind_test=ParCorrMult(significance="analytic"),
        node_classification=node_classification,
        verbosity=0,
    )
    results = estimator.run_jpcmciplus(
        tau_min=tau_min,
        tau_max=tau_max,
        pc_alpha=pc_alpha if pc_alpha is not None else alpha,
        **run_kwargs,
    )
    return _collapse_tigramite_results(
        results["val_matrix"],
        results["p_matrix"],
        alpha=alpha,
        tau_min=tau_min,
    )


def _run_causalised_gc_single_depth(
    x: np.ndarray,
    p: int,
    *,
    method: str,
    alpha: float,
    beta: float,
    n_perm: int,
    n_lags: int,
    temporal: bool,
    verbose: int,
    simulation: bool,
    signed: bool,
) -> np.ndarray:
    estimator = CausalisedGC(
        n_perm=n_perm,
        n_pasts=int(p),
        n_lags=n_lags,
        temporal=temporal,
        method=method,
        signed=signed,
    )
    estimator.fit(np.asarray(x, dtype=np.float64).T, verbose=verbose)
    return estimator.get_connectivity_matrix(
        simulation=simulation,
        alpha=alpha,
        beta=beta,
    )


def make_causalised_gc_analyzer(
    method: str,
    *,
    alpha: float = 0.01,
    beta: float = 0.001,
    n_perm: int = 200,
    n_lags: int = 1,
    temporal: bool = True,
    verbose: int = 0,
    simulation: bool = True,
    signed: bool = True,
) -> Callable[[np.ndarray, list[int]], dict[int, np.ndarray]]:
    if method not in {"cgc", "fcgc"}:
        raise ValueError("method must be 'cgc' or 'fcgc'.")

    def analyze(x: np.ndarray, p_values: list[int]) -> dict[int, np.ndarray]:
        return {
            int(p_value): _run_causalised_gc_single_depth(
                x,
                int(p_value),
                method=method,
                alpha=alpha,
                beta=beta,
                n_perm=n_perm,
                n_lags=n_lags,
                temporal=temporal,
                verbose=verbose,
                simulation=simulation,
                signed=signed,
            )
            for p_value in p_values
        }

    return analyze


def analyze_with_cgc(x: np.ndarray, p_values: list[int]) -> dict[int, np.ndarray]:
    return make_causalised_gc_analyzer("cgc")(x, p_values)


def analyze_with_cgc_star(x: np.ndarray, p_values: list[int]) -> dict[int, np.ndarray]:
    return make_causalised_gc_analyzer("fcgc")(x, p_values)


def make_pcmciplus_analyzer(
    *,
    alpha: float = 0.05,
    tau_min: int = 1,
    pc_alpha: float | None = None,
    **kwargs: Any,
) -> Callable[[np.ndarray, list[int]], dict[int, np.ndarray]]:
    def analyze(x: np.ndarray, p_values: list[int]) -> dict[int, np.ndarray]:
        return {
            int(p_value): _run_pcmciplus_matrix(
                x,
                tau_max=int(p_value),
                alpha=alpha,
                tau_min=tau_min,
                pc_alpha=pc_alpha,
                run_kwargs=kwargs,
            )
            for p_value in p_values
        }

    return analyze


def make_jpcmciplus_analyzer(
    *,
    alpha: float = 0.05,
    tau_min: int = 1,
    pc_alpha: float | None = None,
    node_classification: dict[int, str] | None = None,
    vector_vars: dict[int, list[tuple[int, int]]] | None = None,
    var_names: list[str] | None = None,
    **kwargs: Any,
) -> Callable[
    [dict[Any, np.ndarray] | list[np.ndarray] | tuple[np.ndarray, ...], list[int]],
    dict[int, np.ndarray],
]:
    def analyze(
        recordings: dict[Any, np.ndarray] | list[np.ndarray] | tuple[np.ndarray, ...],
        p_values: list[int],
    ) -> dict[int, np.ndarray]:
        return {
            int(p_value): _run_jpcmciplus_matrix(
                recordings,
                tau_max=int(p_value),
                alpha=alpha,
                tau_min=tau_min,
                pc_alpha=pc_alpha,
                node_classification=node_classification,
                vector_vars=vector_vars,
                var_names=var_names,
                run_kwargs=kwargs,
            )
            for p_value in p_values
        }

    return analyze


def analyze_with_pcmciplus(x: np.ndarray, p_values: list[int]) -> dict[int, np.ndarray]:
    return make_pcmciplus_analyzer()(x, p_values)


def analyze_with_jpcmciplus(
    recordings: dict[Any, np.ndarray] | list[np.ndarray] | tuple[np.ndarray, ...],
    p_values: list[int],
) -> dict[int, np.ndarray]:
    return make_jpcmciplus_analyzer()(recordings, p_values)


METHODS = {
    "cgc": analyze_with_cgc,
    "cgc_star": analyze_with_cgc_star,
    "pcmciplus": analyze_with_pcmciplus,
    "jpcmciplus": analyze_with_jpcmciplus,
}

__all__ = [
    "METHODS",
    "analyze_with_cgc",
    "analyze_with_cgc_star",
    "analyze_with_jpcmciplus",
    "analyze_with_pcmciplus",
    "make_causalised_gc_analyzer",
    "make_jpcmciplus_analyzer",
    "make_pcmciplus_analyzer",
]
