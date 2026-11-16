"""Thin adapters for the four imported causalized-GC support estimators.

These adapters export upstream binary, positive-lag decisions. They neither fit
VAR coefficients nor provide a neural forecasting model. The conventional CPU
package runs in its own supported Python environment, once per window sequence.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from ._native_gc_runtime import conventional_gc_runtime, current_conventional_runtime
from .base import ConnectivityEstimator
from .predictive import PredictiveCapabilities
from .registry import register_connectivity

_EVIDENCE_ATTRIBUTES = (
    "marginal_statistic_", "marginal_effect_", "marginal_pvalue_",
    "conditional_statistic_", "conditional_effect_", "conditional_pvalue_",
    "conditional_evaluated_", "corr_", "pVal_corr_", "inv_corr_", "pVal_inv_corr_",
    "lag_score_", "lag_adjacency_", "decision_mask_",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _package_provenance(module: Any, distribution: str) -> dict[str, Any]:
    result: dict[str, Any] = {"module": module.__name__, "module_file": module.__file__}
    try:
        installed = metadata.distribution(distribution)
        result["distribution"] = distribution
        result["version"] = installed.version
        direct_url = installed.read_text("direct_url.json")
        if direct_url:
            result["direct_url"] = json.loads(direct_url)
    except metadata.PackageNotFoundError:
        result["version"] = "unavailable"
    return result


class _NativeGCConnectivity(ConnectivityEstimator):
    directed = True
    weighted = False
    weight_semantics = "binary_upstream_causal_support"
    conventional = False
    upstream_method = "cgc"

    @property
    def capabilities(self) -> PredictiveCapabilities:
        return PredictiveCapabilities(independent_window=True, neural_forecast=False)

    def _options(self) -> dict[str, Any]:
        if self.cfg.max_lag != 1:
            raise ValueError("native four-method GC comparison supports max_lag=1 only")
        if self.cfg.max_lag_seconds is not None and "lag_contract" not in self.cfg.extra:
            raise ValueError("resolve duration-based native GC lags with the recording fps first")
        if self.cfg.absolute:
            raise ValueError("native GC support requires absolute=False; its decisions are unsigned")
        if not 0.0 <= self.cfg.threshold <= 1.0:
            raise ValueError("a binary support threshold must lie in [0, 1]")
        options: dict[str, Any] = {
            "alpha": float(self.cfg.alpha),
            "beta": float(self.cfg.extra.get("beta", 0.005)),
            "n_perm": int(self.cfg.extra.get("n_perm", 1000 if self.conventional else 0)),
            "seed": int(self.cfg.extra.get("seed", 0)),
            "verbose": int(self.cfg.extra.get("verbose", 0)),
            "method": self.upstream_method,
        }
        if not 0.0 <= options["alpha"] <= 1.0 or not 0.0 <= options["beta"] <= 1.0:
            raise ValueError("native GC alpha and beta must lie in [0, 1]")
        if options["n_perm"] < 0:
            raise ValueError("native GC n_perm must be nonnegative")
        if self.conventional and options["n_perm"] == 0:
            raise ValueError("conventional GC requires n_perm>=1 for its upstream permutation test")
        return options

    def _estimate_windows(self, values: np.ndarray) -> np.ndarray:
        options = self._options()
        segments = np.asarray(values, dtype=np.float64)
        if segments.ndim != 3 or segments.shape[0] < 1 or segments.shape[1] < 3:
            raise ValueError("native GC requires nonempty windows shaped (window, time>=3, neuron)")
        if not np.isfinite(segments).all():
            raise ValueError("native GC requires finite input; no local imputation is performed")
        if not self.conventional:
            # Upstream full circular-shift tests have minimum p=1/(T-n_pasts).
            # Reject impossible discovery thresholds instead of reporting a silent empty graph.
            minimum_pvalue = 1.0 / (segments.shape[1] - 1)
            if min(options["alpha"], options["beta"]) < minimum_pvalue:
                raise ValueError(
                    f"native fast GC thresholds are below the attainable complete-shift p-value "
                    f"floor {minimum_pvalue:.6g} for {segments.shape[1]} samples; "
                    "increase context length or use declared attainable thresholds"
                )
        if self.conventional:
            matrices = self._conventional_windows(segments, options)
        else:
            matrices = self._fast_windows(segments, options)
        if matrices.shape != (len(segments), segments.shape[2], segments.shape[2]):
            raise RuntimeError("upstream GC support does not align with window/neuron axes")
        if not np.isin(matrices, [0, 1]).all():
            raise RuntimeError("upstream GC adapter returned nonbinary support")
        self._window_lagged = matrices[:, None, :, :].copy()
        return matrices

    def estimate(self, segment: np.ndarray) -> np.ndarray:
        return self._estimate_windows(np.asarray(segment)[None, :, :])[0]

    def estimate_sequence(self, segments) -> np.ndarray:
        return self._estimate_windows(segments.segments)

    def _fast_windows(self, segments: np.ndarray, options: dict[str, Any]) -> np.ndarray:
        try:
            module = importlib.import_module("fastCGC")
        except ImportError as error:
            raise ImportError("install the project's pinned fastcgc dependency for native fast GC") from error
        self._upstream_provenance = _package_provenance(module, "fastcgc")
        self._window_diagnostics = []
        matrices = []
        for index, segment in enumerate(segments):
            seed = options["seed"] + index
            estimator = module.FastGcStar(
                n_perm=options["n_perm"], n_pasts=1, n_lags=1, temporal=True,
                method=options["method"], parallel=False, random_state=seed,
                dependence_profile="linear", evaluation_policy="full",
            )
            estimator.fit(
                segment.T, verbose=options["verbose"],
                alpha=options["alpha"], beta=options["beta"],
            )
            native_matrix = estimator.get_connectivity_matrix(
                simulation=True, alpha=options["alpha"], beta=options["beta"],
            )
            # The public native lag mask avoids conflating edge scores with support.
            lagged = np.asarray(estimator.lag_adjacency_, dtype=bool)
            if lagged.shape != (2, segment.shape[1], segment.shape[1]):
                raise RuntimeError("FastGcStar did not expose the expected tau0/tau1 native decisions")
            support = lagged[1].astype(np.float64)
            np.fill_diagonal(support, 0.0)
            matrices.append(support)
            self._window_diagnostics.append({
                "native_matrix": _jsonable(np.asarray(native_matrix)),
                "native_evidence": {
                    name: _jsonable(getattr(estimator, name))
                    for name in _EVIDENCE_ATTRIBUTES if hasattr(estimator, name)
                },
                "backend_metadata": _jsonable(getattr(estimator, "backend_metadata_", {})),
                "requested_seed": seed,
                "seed_policy": "upstream_FastGcStar_random_state",
                "minimum_attainable_shift_pvalue": 1.0 / (len(segment) - 1),
            })
        return np.stack(matrices)

    def _conventional_windows(self, segments: np.ndarray, options: dict[str, Any]) -> np.ndarray:
        project_root = Path(__file__).resolve().parents[3]
        runtime = Path(self.cfg.extra.get(
            "runtime_python", project_root / "tools/conventional-cgc/.venv/bin/python"
        )).expanduser()
        if not runtime.is_file():
            raise FileNotFoundError(
                f"conventional GC runtime is missing: {runtime}. "
                "Install tools/conventional-cgc with its pinned upstream package; "
                "no fast-method fallback is used."
            )
        options["upstream_module"] = str(self.cfg.extra.get("upstream_module", "causalized_gc"))
        worker = Path(__file__).with_name("_native_gc_worker.py")
        with tempfile.TemporaryDirectory(prefix="effectome-native-cgc-") as directory:
            root = Path(directory)
            input_path, output_path, metadata_path = (root / name for name in (
                "input.npz", "output.npz", "metadata.json"
            ))
            np.savez(input_path, segments=segments, options=np.asarray(json.dumps(options)))
            environment = {
                **os.environ,
                "MPLCONFIGDIR": str(root / "matplotlib"),
                "XDG_CACHE_HOME": str(root / "cache"),
            }
            session = current_conventional_runtime()
            if session is None:
                result = subprocess.run(
                    [str(runtime), str(worker), str(input_path), str(output_path), str(metadata_path)],
                    capture_output=True, text=True, check=False, env=environment,
                )
                if result.returncode:
                    raise RuntimeError(
                        "official conventional GC runtime failed; no estimator fallback was used:\n"
                        + (result.stderr or result.stdout)[-4000:]
                    )
            else:
                session.run(runtime, worker, input_path, output_path, metadata_path, environment)
            payload = json.loads(metadata_path.read_text())
            self._upstream_provenance = payload["upstream_provenance"]
            self._window_diagnostics = payload["windows"]
            with np.load(output_path, allow_pickle=False) as output:
                matrices = np.asarray(output["matrices"], dtype=np.float64)
                for window in self._window_diagnostics:
                    window["native_evidence"] = {
                        name: output[key].tolist()
                        for name, key in window.pop("evidence_keys").items()
                    }
        return matrices

    def _lagged_output(self) -> np.ndarray | None:
        return getattr(self, "_window_lagged", None)

    def _diagnostics(self, segments, mats):
        diagnostics = super()._diagnostics(segments, mats)
        diagnostics.update({
            "signed": False,
            "backend": "conventional_cgc" if self.conventional else "fastcgc",
            "support_variant": self.upstream_method,
            "support_kind": "native_upstream_decision",
            "primary_tau_policy": "lagged_only_tau_1",
            "native_lag_selection": "tau1; tau0 excluded; self-edges excluded",
            "neural_forecast": False,
            "directed_drive_available": False,
            "upstream_provenance": self._upstream_provenance,
            "lag_resolved": self._window_diagnostics,
            "evidence_axis_convention": "rows=lag_block*neurons+source; columns=target",
        })
        identity = segments.provenance.identity
        if ("node_ids" in segments.metadata
                and identity.dataset_id not in {"", "unknown"}
                and identity.recording_id not in {"", "unknown"}):
            identifiers = list(segments.metadata["node_ids"])
            diagnostics.update({
                "node_ids": [json.dumps([
                    identity.dataset_id, identity.recording_id, identity.animal_id,
                    identity.session_id, identity.segment_id, str(identifier),
                ], separators=(",", ":")) for identifier in identifiers],
                "recording_local_neuron_ids": identifiers,
                "node_registration_scope": "recording_local_ROIs_not_cross_animal_anatomical_alignment",
                "physical_lags": [1.0 / segments.fps],  # Native adapter exports tau=1 only.
                "representation_units": self.weight_semantics,
            })
        return diagnostics

    def run(self, segments):
        series = super().run(segments)
        series.signed = False
        return series


__all__ = ["conventional_gc_runtime"]


@register_connectivity("fast_cgc")
class FastCGCConnectivity(_NativeGCConnectivity):
    """Imported FastGcStar marginal/conditional c-GC decisions."""


@register_connectivity("fast_cgc_star")
class FastCGCStarConnectivity(_NativeGCConnectivity):
    """Imported FastGcStar full-conditioning c-GC* decisions."""

    upstream_method = "fcgc"


@register_connectivity("conventional_cgc")
class ConventionalCGCConnectivity(_NativeGCConnectivity):
    """Official conventional CPU c-GC, imported in its supported environment."""

    conventional = True


@register_connectivity("conventional_cgc_star")
class ConventionalCGCStarConnectivity(_NativeGCConnectivity):
    """Official conventional CPU c-GC*, imported in its supported environment."""

    conventional = True
    upstream_method = "cgc_star"
