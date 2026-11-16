"""Inventory or execute the development-only structural driver diagnostic."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

from effectome.experiments.csl_sanity import CSL_METHODS
from effectome.experiments.mechanism_reporting import render_mechanism_report
from effectome.experiments.mechanism_validation import ORACLE_LANES, run_mechanism_validation
from effectome.experiments.mechanism_validation_config import load_mechanism_validation_profile
from effectome.utils.seed import set_seed
from effectome.workflows.cross_method import load_cross_method_profile

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "development"), default="smoke")
    parser.add_argument("--run", action="store_true", help="Default only inventories the development study")
    parser.add_argument(
        "--oracle-only", action="store_true", help="Known/empty support, without native discovery"
    )
    parser.add_argument("--method", action="append", choices=CSL_METHODS,
                        help="Explicit subset of estimated-support lanes; oracle controls always retained")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_mechanism_validation_profile(args.profile)
    set_seed(config.seeds[0])
    _, methods = load_cross_method_profile("csl_sanity")
    if args.method:
        methods = {name: methods[name] for name in dict.fromkeys(args.method)}
    if args.oracle_only and args.method:
        parser.error("--oracle-only cannot be combined with --method")
    lanes = [*ORACLE_LANES, *([] if args.oracle_only else methods)]
    fit_trials = int(config.generator.n_trials * (1 - config.evaluation_fraction))
    discovery_trials = fit_trials - max(1, math.ceil(config.fit.validation_fraction * fit_trials))
    inventory = {
        "profile": config.profile, "planned_cells": 2 * len(config.seeds),
        "planned_method_runs": 2 * len(config.seeds) * len(lanes), "lanes": lanes,
        "training_trials_per_recording": fit_trials,
        "training_one_step_pairs_per_recording": fit_trials * (config.generator.trial_length - 1),
        "heldout_trials_per_recording": config.generator.n_trials - fit_trials,
        "support_discovery_inner_training_trials": discovery_trials,
        "native_window_fits": (
            0 if args.oracle_only else 2 * len(config.seeds) * len(methods) * discovery_trials
        ),
        "scope": "model_matched_fully_observed_development_only",
    }
    logger.info("Mechanism study inventory: %s", json.dumps(inventory))
    if not args.run:
        return
    output = args.output_dir or Path("outputs/driver-mechanism-validation") / args.profile
    report = run_mechanism_validation(config, output, methods=methods, oracle_only=args.oracle_only)
    analysis = render_mechanism_report(report, output / "analysis")
    logger.info("Available runs: %s/%s; descriptive report: %s", report["available_method_runs"],
                report["attempted_method_runs"], analysis)


if __name__ == "__main__":
    main()
