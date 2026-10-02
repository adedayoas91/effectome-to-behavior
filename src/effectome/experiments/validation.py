"""Validation manifest helpers for revised synthetic benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

SuiteName = Literal["core", "stress", "nonlinear", "grouping", "independent_calibration"]


@dataclass(frozen=True)
class ValidationSuiteSpec:
    name: SuiteName
    cells: int
    replicates_per_cell: int
    recordings: int
    optional: bool = False


@dataclass(frozen=True)
class ValidationCell:
    """One persisted dataset attempt in the finite validation manifest."""

    suite: SuiteName
    scenario: str
    graph: int
    replicate: int
    condition: str = "baseline"

    @property
    def cell_id(self) -> str:
        return (
            f"{self.suite}:{self.scenario}:{self.condition}:"
            f"g{self.graph:03d}:r{self.replicate:03d}"
        )


@dataclass(frozen=True)
class ValidationBudget:
    """Per-cell resource policy recorded before benchmark execution."""

    timeout_seconds: float = 3_600.0
    memory_limit_mb: float = 16_384.0
    failure_policy: str = "record_and_continue"

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.memory_limit_mb <= 0:
            raise ValueError("validation resource limits must be positive")
        if self.failure_policy not in {"record_and_continue", "fail_fast"}:
            raise ValueError("unsupported validation failure policy")


@dataclass(frozen=True)
class ValidationAttempt:
    """Machine-readable outcome for one finite registered benchmark cell."""

    cell_id: str
    status: str
    seed: int
    config_hash: str
    elapsed_seconds: float
    peak_memory_mb: float
    budget: ValidationBudget
    result: dict[str, Any]
    failure_reason: str | None = None
    resumed: bool = False


def _stable_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "sha256": hashlib.sha256(array.view(np.uint8).tobytes()).hexdigest(),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _stable_json(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_stable_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _atomic_attempt(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def run_validation_cells(
    cells: Sequence[ValidationCell],
    *,
    output_dir: str | Path,
    master_seed: int,
    config_factory: Callable[[ValidationCell, int], Mapping[str, Any]],
    evaluator: Callable[[ValidationCell, Mapping[str, Any]], Mapping[str, Any]],
    budget: ValidationBudget | None = None,
    resume: bool = True,
    progress_callback: Callable[[int, int, ValidationAttempt], None] | None = None,
) -> tuple[ValidationAttempt, ...]:
    """Execute and atomically persist registered cells with resumable attempt records.

    The caller supplies the raw-data/evaluation adapter so benchmark cells use the same
    evaluator as real recordings. This function deliberately does not select thresholds or
    silently retry failures; every registered attempt remains represented in the output.
    """
    destination = Path(output_dir)
    budget = ValidationBudget() if budget is None else budget
    attempts: list[ValidationAttempt] = []
    total = len(cells)
    for ordinal, cell in enumerate(cells, start=1):
        path = destination / cell.suite / f"{cell.cell_id.replace(':', '__')}.json"
        if resume and path.is_file():
            with path.open(encoding="utf-8") as handle:
                stored = json.load(handle)
            attempt = ValidationAttempt(
                cell_id=str(stored["cell_id"]),
                status=str(stored["status"]),
                seed=int(stored["seed"]),
                config_hash=str(stored["config_hash"]),
                elapsed_seconds=float(stored["elapsed_seconds"]),
                peak_memory_mb=float(stored["peak_memory_mb"]),
                budget=ValidationBudget(**stored["budget"]),
                result=dict(stored.get("result", {})),
                failure_reason=stored.get("failure_reason"),
                resumed=True,
            )
        else:
            seed = deterministic_seed(
                master_seed,
                suite=cell.suite,
                scenario=cell.scenario,
                graph=cell.graph,
                replicate=cell.replicate,
                purpose="recording",
            )
            config = dict(config_factory(cell, seed))
            encoded = json.dumps(_stable_json(config), sort_keys=True, separators=(",", ":"))
            config_hash = hashlib.sha256(encoded.encode()).hexdigest()
            started = time.perf_counter()
            tracemalloc.start()
            status = "completed"
            failure_reason = None
            result: dict[str, Any] = {}
            try:
                result = dict(_stable_json(dict(evaluator(cell, config))))
            except Exception as exc:  # every failed registered attempt must remain visible
                status = "failed"
                failure_reason = f"{type(exc).__name__}:{exc}"
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            elapsed = time.perf_counter() - started
            peak_mb = float(peak / (1024**2))
            if status == "completed" and elapsed > budget.timeout_seconds:
                status = "timeout"
                failure_reason = "elapsed time exceeded the registered cell budget"
            if status == "completed" and peak_mb > budget.memory_limit_mb:
                status = "memory_exceeded"
                failure_reason = "traced Python memory exceeded the registered cell budget"
            attempt = ValidationAttempt(
                cell_id=cell.cell_id,
                status=status,
                seed=seed,
                config_hash=config_hash,
                elapsed_seconds=float(elapsed),
                peak_memory_mb=peak_mb,
                budget=budget,
                result=result,
                failure_reason=failure_reason,
            )
            _atomic_attempt(path, _stable_json(asdict(attempt)))
            if status != "completed" and budget.failure_policy == "fail_fast":
                attempts.append(attempt)
                if progress_callback is not None:
                    progress_callback(ordinal, total, attempt)
                raise RuntimeError(f"validation cell {cell.cell_id} {status}: {failure_reason}")
        attempts.append(attempt)
        if progress_callback is not None:
            progress_callback(ordinal, total, attempt)
    return tuple(attempts)


REVISED_VALIDATION_SUITES: tuple[ValidationSuiteSpec, ...] = (
    ValidationSuiteSpec("core", cells=100, replicates_per_cell=5, recordings=500),
    ValidationSuiteSpec("stress", cells=90, replicates_per_cell=3, recordings=270),
    ValidationSuiteSpec("nonlinear", cells=10, replicates_per_cell=3, recordings=30),
    ValidationSuiteSpec("grouping", cells=40, replicates_per_cell=3, recordings=120, optional=True),
    ValidationSuiteSpec(
        "independent_calibration",
        cells=600,
        replicates_per_cell=1,
        recordings=600,
    ),
)


def benchmark_cells(*, include_grouping: bool = False) -> tuple[ValidationCell, ...]:
    """Materialize the finite Appendix-C benchmark without running any cell."""
    cells: list[ValidationCell] = []
    temporal = (
        "constant",
        "smooth_drift",
        "nonrecurring_epochs",
        "recurrent_geometric",
        "recurrent_non_geometric",
    )
    for scenario in temporal:
        for size in (20, 109):
            for graph in range(10):
                for replicate in range(5):
                    cells.append(
                        ValidationCell(
                            "core",
                            scenario,
                            graph,
                            replicate,
                            condition=f"n{size}",
                        )
                    )

    stress_conditions = (
        "snr_low",
        "snr_high",
        "dwell_short",
        "dwell_long",
        "kernel_heterogeneity",
        "burst_missingness",
        "latent_input",
        "instantaneous",
        "correlated_innovations",
    )
    for condition in stress_conditions:
        for graph in range(10):
            for replicate in range(3):
                cells.append(ValidationCell("stress", "recurrent_geometric", graph, replicate, condition))

    for graph in range(10):
        for replicate in range(3):
            cells.append(ValidationCell("nonlinear", "nonlinear_saturation", graph, replicate))

    if include_grouping:
        for family in ("assortative", "feedforward", "core_periphery", "no_group"):
            for graph in range(10):
                for replicate in range(3):
                    cells.append(ValidationCell("grouping", family, graph, replicate))

    for family in ("variance", "measurement", "endpoint"):
        for graph in range(200):
            cells.append(ValidationCell("independent_calibration", family, graph, 0))

    ids = [cell.cell_id for cell in cells]
    if len(ids) != len(set(ids)):  # defensive guard for future manifest edits
        raise RuntimeError("validation cell IDs must be unique")
    return tuple(cells)


def validation_totals(*, include_grouping: bool = False) -> dict[str, int]:
    selected = [
        suite for suite in REVISED_VALIDATION_SUITES if include_grouping or not suite.optional
    ]
    core_like = [
        suite
        for suite in selected
        if suite.name in {"core", "stress", "nonlinear", "grouping"}
    ]
    return {
        "core_stress_nonlinear": sum(
            suite.recordings
            for suite in REVISED_VALIDATION_SUITES
            if suite.name in {"core", "stress", "nonlinear"}
        ),
        "with_grouping": sum(suite.recordings for suite in core_like),
        "independent_calibration": next(
            suite.recordings
            for suite in REVISED_VALIDATION_SUITES
            if suite.name == "independent_calibration"
        ),
        "selected_total": sum(suite.recordings for suite in selected),
    }


def deterministic_seed(
    master_seed: int,
    *,
    suite: str,
    scenario: str,
    graph: int,
    replicate: int,
    purpose: str,
) -> int:
    """Derive stable integer streams without Python's randomized hash()."""
    tokens = [suite, scenario, str(graph), str(replicate), purpose]
    entropy = [int(master_seed)]
    for token in tokens:
        encoded = token.encode("utf-8")
        entropy.extend(encoded)
        entropy.append(0)
    sequence = np.random.SeedSequence(entropy)
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


__all__ = [
    "REVISED_VALIDATION_SUITES",
    "ValidationCell",
    "ValidationAttempt",
    "ValidationBudget",
    "ValidationSuiteSpec",
    "benchmark_cells",
    "deterministic_seed",
    "run_validation_cells",
    "validation_totals",
]
