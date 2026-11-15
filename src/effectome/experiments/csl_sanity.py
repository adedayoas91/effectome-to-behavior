"""Native CSL graph recovery on paired planted-chain and independent null data.

This is a graph-discovery sanity check, separate from manifold-transition and
interventional driver recovery. No discovery algorithm is implemented here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from effectome.connectivity import ConnectivityConfig, ConnectivityFactory
from effectome.data_module.schema import Window, WindowedSegments
from effectome.workflows.cross_method import _json_safe, load_cross_method_profile
from effectome.workflows.resume import content_fingerprint

CSL_METHODS = ("conventional_cgc", "conventional_cgc_star", "fast_cgc", "fast_cgc_star")
CSL_SCENARIOS = ("chain", "null")


@dataclass(frozen=True)
class CSLSanityConfig:
    """Freeze the comparison before inspecting recovery or pairwise agreement."""

    seeds: tuple[int, ...] = (0, 1, 2)
    n_samples: int = 1024
    burn_in: int = 256
    coupling: float = 0.8

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if any(not isinstance(seed, int) or seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be nonnegative integers")
        if self.n_samples < 256 or self.burn_in < 2:
            raise ValueError("sanity data require >=256 samples and >=2 burn-in samples")
        if not np.isfinite(self.coupling) or not 0 < self.coupling < 1:
            raise ValueError("coupling must lie strictly between zero and one")


def make_csl_sanity_data(config: CSLSanityConfig, scenario: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate four observed neurons with independent Gaussian innovations.

    In the chain, x1[t] = c*x0[t-1]+e1[t], x2[t] = -c*x1[t-1]+e2[t].
    Neuron3 is an isolated negative control. The null has no cross-neuron links.
    The same seed pairs innovations across conditions. A linear lagged generator
    is the known truth, never a fitted VAR estimator or an inference input.
    """
    if scenario not in CSL_SCENARIOS:
        raise ValueError(f"unknown sanity scenario: {scenario}")
    coefficients = np.zeros((4, 4))
    if scenario == "chain":
        coefficients[0, 1] = config.coupling
        coefficients[1, 2] = -config.coupling
    innovations = np.random.default_rng(seed).normal(size=(config.n_samples + config.burn_in, 4))
    values = innovations.copy()
    for sample in range(1, len(values)):
        values[sample] += values[sample - 1] @ coefficients
    return values[config.burn_in :], coefficients != 0


def graph_recovery_metrics(inferred: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """Evaluate directed off-diagonal support; undefined precision stays absent."""
    predicted, target = np.asarray(inferred, dtype=bool), np.asarray(truth, dtype=bool)
    if predicted.ndim != 2 or predicted.shape != target.shape or predicted.shape[0] != predicted.shape[1]:
        raise ValueError("inferred and truth must be matching square matrices")
    mask = ~np.eye(len(predicted), dtype=bool)
    positive, actual = predicted[mask], target[mask]
    tp = int((positive & actual).sum())
    fp = int((positive & ~actual).sum())
    fn = int((~positive & actual).sum())
    tn = int((~positive & ~actual).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_selected": tp + fp, "n_true": tp + fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "exact_support": not (fp or fn),
    }


def support_agreement(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    """Jaccard is undefined for two empty graphs, even when they match exactly."""
    a, b = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ValueError("support matrices must be matching and square")
    mask = ~np.eye(len(a), dtype=bool)
    intersection = int((a & b & mask).sum())
    union = int(((a | b) & mask).sum())
    return {
        "intersection": intersection, "union": union,
        "jaccard": intersection / union if union else None,
        "both_empty": union == 0,
        "exact_support": bool(np.array_equal(a[mask], b[mask])),
    }


def run_csl_sanity(
    config: CSLSanityConfig,
    output_dir: str | Path,
    *,
    methods: dict[str, ConnectivityConfig] | None = None,
) -> dict[str, Any]:
    """Batch the same independent recordings through each imported implementation."""
    if methods is None:
        _, methods = load_cross_method_profile("csl_sanity")
    if set(methods) != set(CSL_METHODS):
        raise ValueError(f"sanity panel must contain exactly {CSL_METHODS}")
    cells = [(scenario, seed) for scenario in CSL_SCENARIOS for seed in config.seeds]
    examples = [make_csl_sanity_data(config, scenario, seed) for scenario, seed in cells]
    segments = np.stack([values for values, _ in examples])
    windows = WindowedSegments(
        segments=segments,
        windows=[Window(0, config.n_samples) for _ in cells],
        behavior_per_window={}, n_neurons=4, fps=1.0,
        metadata={"independent_recordings": [f"{s}-seed-{i}" for s, i in cells]},
    )
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    graphs: dict[str, np.ndarray] = {}
    for method in CSL_METHODS:
        try:
            series = ConnectivityFactory(methods[method]).run(windows)
            graphs[method] = series.matrices != 0
            diagnostics[method] = {"status": "available", "evidence": series.diagnostics}
            for index, (scenario, seed) in enumerate(cells):
                rows.append({
                    "method": method, "scenario": scenario, "seed": seed, "status": "available",
                    "support": graphs[method][index],
                    **graph_recovery_metrics(graphs[method][index], examples[index][1]),
                })
        except (ValueError, RuntimeError, ImportError, OSError, np.linalg.LinAlgError) as error:
            diagnostics[method] = {"status": "unavailable", "reason": str(error)}
            rows.extend({"method": method, "scenario": s, "seed": seed, "status": "unavailable",
                         "reason": str(error)} for s, seed in cells)
    pairs = []
    for first, second in combinations(CSL_METHODS, 2):
        for index, (scenario, seed) in enumerate(cells):
            available = first in graphs and second in graphs
            pairs.append({
                "first": first, "second": second, "scenario": scenario, "seed": seed,
                "status": "available" if available else "unavailable",
                **(support_agreement(graphs[first][index], graphs[second][index]) if available else {}),
            })
    report = {
        "schema_version": "four_csl_graph_sanity_v1", "config": asdict(config),
        "method_configs": {method: asdict(cfg) for method, cfg in methods.items()},
        "input_fingerprint": content_fingerprint(segments),
        "comparison_fingerprint": content_fingerprint((asdict(config), methods, segments)),
        "truth": {scenario: make_csl_sanity_data(config, scenario, config.seeds[0])[1]
                  for scenario in CSL_SCENARIOS},
        "seed_rows": rows, "pairwise_rows": pairs, "diagnostics": diagnostics,
        "attempted_runs": len(cells) * len(methods),
        "available_runs": sum(row["status"] == "available" for row in rows),
        "claim_scope": "descriptive_native_graph_recovery_sanity_only",
        "transition_driver_recovery_validated": False,
        "independent_estimator_families": 1,
        "conventional_rng_caveat": "Upstream CPU threaded Numba permutations are not guaranteed seeded.",
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "sanity.json").write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n")
    return report
