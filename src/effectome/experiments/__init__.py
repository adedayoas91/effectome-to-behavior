"""Notebook-facing experiment adapters."""

from .adapters import (
    METHODS,
    analyze_with_cgc,
    analyze_with_cgc_star,
    analyze_with_jpcmciplus,
    analyze_with_pcmciplus,
    make_causalised_gc_analyzer,
    make_jpcmciplus_analyzer,
    make_pcmciplus_analyzer,
)
from .notebook_utils import (
    collapse_tigramite_results,
    find_project_root,
    normalize_multiple_recordings,
    plot_depth_summary,
    summarize_stack,
)
from .validation import (
    REVISED_VALIDATION_SUITES,
    ValidationAttempt,
    ValidationBudget,
    ValidationCell,
    ValidationSuiteSpec,
    benchmark_cells,
    deterministic_seed,
    run_validation_cells,
    validation_totals,
)

__all__ = [
    "METHODS",
    "analyze_with_cgc",
    "analyze_with_cgc_star",
    "analyze_with_jpcmciplus",
    "analyze_with_pcmciplus",
    "collapse_tigramite_results",
    "find_project_root",
    "make_causalised_gc_analyzer",
    "make_jpcmciplus_analyzer",
    "make_pcmciplus_analyzer",
    "normalize_multiple_recordings",
    "plot_depth_summary",
    "REVISED_VALIDATION_SUITES",
    "ValidationAttempt",
    "ValidationBudget",
    "ValidationCell",
    "summarize_stack",
    "ValidationSuiteSpec",
    "benchmark_cells",
    "deterministic_seed",
    "run_validation_cells",
    "validation_totals",
]
