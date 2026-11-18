"""Inventory, execute, or inspect the fresh-seed calibrated mechanism study."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from effectome.experiments.csl_sanity import CSL_METHODS
from effectome.workflows.cross_method import load_cross_method_profile

if TYPE_CHECKING:
    from effectome.connectivity import ConnectivityConfig
    from effectome.experiments.mechanism_replication_config import MechanismReplicationConfig

logger = logging.getLogger(__name__)


def replication_inventory(
    config: MechanismReplicationConfig, methods: dict[str, ConnectivityConfig]
) -> dict[str, Any]:
    """Count declared work without creating output directories or fitting models."""
    mechanism = config.mechanism
    training_trials = int(mechanism.generator.n_trials * (1 - mechanism.evaluation_fraction))
    validation_trials = max(1, math.ceil(mechanism.fit.validation_fraction * training_trials))
    discovery_trials = training_trials - validation_trials
    primary_cells = len(config.calibration_seeds) + len(config.positive_seeds) + len(config.null_seeds)
    comparison_cells = len(config.positive_seeds)
    total_cells = primary_cells + comparison_cells
    lanes = ["known_support", "empty_support", *methods]
    return {
        "profile": config.profile,
        "primary_history": config.primary_history,
        "comparison_history": config.comparison_history,
        "calibration_null_seeds": list(config.calibration_seeds),
        "positive_test_seeds": list(config.positive_seeds),
        "null_test_seeds": list(config.null_seeds),
        "primary_cells": primary_cells,
        "comparison_positive_cells": comparison_cells,
        "planned_cells": total_cells,
        "planned_method_runs": total_cells * len(lanes),
        "lanes": lanes,
        "all_four_native_methods_requested": set(methods) == set(CSL_METHODS),
        "training_trials_per_recording": training_trials,
        "support_discovery_inner_training_trials": discovery_trials,
        "native_window_fits": total_cells * len(methods) * discovery_trials,
        "calibration_alpha": config.calibration_alpha,
        "minimum_nomination_effect_cutoff": config.nomination_floor,
        "calibration_scope": "primary_history_null_seed_maximum_across_units_and_directions",
        "comparison_scope": "paired_positive_seeds_only_not_independently_calibrated_nomination",
        "claim_scope": "bounded_fully_observed_model_matched_replication_not_biological_necessity",
        "fitting_enabled": False,
    }


def inspect_replication_status(output_root: Path) -> dict[str, Any]:
    """Read the latest atomic report without loading a fitting implementation."""
    path = output_root / "replication-report.json"
    if not path.is_file():
        return {"status": "not_started", "report": str(path), "fitting_enabled": False}
    report = json.loads(path.read_text())
    return {"status": "saved_report", "report": str(path), "fitting_enabled": False, "saved": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("pilot", "replication"), default="replication")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true", help="Fit the declared study; default only inventories")
    action.add_argument("--status", action="store_true", help="Read an existing report; never fit")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--method", action="append", choices=CSL_METHODS)
    parser.add_argument("--output-root", "--output-dir", dest="output_root", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.workers < 1:
        parser.error("--workers must be a positive integer")
    output = args.output_root or (
        Path("outputs/driver-mechanism-replication") / args.profile / "mechanism-replication-01"
    )
    if args.status:
        status = inspect_replication_status(output)
        saved = status.pop("saved", {})
        status.update({key: saved[key] for key in (
            "profile", "execution_complete", "stage", "progress", "planned_cells", "completed_cells",
            "attempted_method_runs", "available_method_runs", "calibration_complete", "claim_scope",
        ) if key in saved})
        logger.info("%s", json.dumps(status, indent=2))
        return

    from effectome.experiments.mechanism_replication_config import load_mechanism_replication_profile

    config = load_mechanism_replication_profile(args.profile)
    _, methods = load_cross_method_profile("csl_sanity")
    if args.method:
        methods = {name: methods[name] for name in dict.fromkeys(args.method)}
    inventory = replication_inventory(config, methods)
    inventory["output_root"] = str(output)
    inventory["workers"] = args.workers
    inventory["fitting_enabled"] = args.run
    logger.info("Mechanism replication inventory: %s", json.dumps(inventory, indent=2))
    if not args.run:
        return

    from effectome.experiments.mechanism_replication import (
        render_mechanism_replication_report,
        run_mechanism_replication,
    )

    from notebooks._shared import require_conventional_runtime

    require_conventional_runtime(methods)
    report = run_mechanism_replication(config, output, workers=args.workers, methods=methods)
    analysis = render_mechanism_replication_report(report, output / "analysis")
    logger.info("Saved replication report: %s; analysis: %s", output / "replication-report.json", analysis)


if __name__ == "__main__":
    main()
