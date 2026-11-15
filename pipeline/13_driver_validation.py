"""Inventory or execute simulator-grounded transition-driver recovery."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from effectome.experiments.driver_study import run_driver_study
from effectome.experiments.driver_validation import (
    driver_validation_cells,
    load_driver_validation_profile,
    run_driver_validation,
)
from effectome.experiments.driver_validation_reporting import render_driver_validation_report
from effectome.utils.seed import set_seed
from effectome.workflows.cross_method import load_cross_method_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "development", "confirmatory"), default="smoke")
    parser.add_argument("--run", action="store_true", help="Default only inventories the planned protocol")
    parser.add_argument("--max-cells", type=int, help="Record explicitly partial execution")
    parser.add_argument("--method", action="append", help="Explicitly restrict the requested estimator lanes")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--workers", type=int, default=1, help="Independent persistent native runtime workers"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_driver_validation_profile(args.profile)
    set_seed(config.positive_seeds[0])
    _, methods = load_cross_method_profile(config.estimator_profile)
    if args.method:
        if set(args.method) - methods.keys():
            parser.error(f"unknown method; choose from {list(methods)}")
        methods = {name: methods[name] for name in args.method}
    cells = driver_validation_cells(config)
    print(
        json.dumps(
            {
                "profile": args.profile,
                "planned_cells": len(cells),
                "planned_method_runs": len(cells) * len(methods),
                "scenarios": config.scenarios,
                "positive_seeds": config.positive_seeds,
                "null_seeds": config.null_seeds,
                "planned_window_fits": len(cells) * len(methods) * config.generator.n_trials
                if config.trial_origin_offset is not None else None,
                "protocol_status": "proposed_not_preregistered",
            }
        )
    )
    if args.run:
        output = args.output_dir or (
            Path("outputs/driver-validation") / config.estimator_profile
            / ("scheduled-v2" if config.trial_origin_offset is not None else "dense-v1") / args.profile
        )
        if args.workers > 1:
            if args.method:
                parser.error("parallel study retains all four methods; omit --method")
            report = run_driver_study(config, output, workers=args.workers, max_cells=args.max_cells)
        else:
            from effectome.connectivity.native_gc import conventional_gc_runtime

            with conventional_gc_runtime():
                report = run_driver_validation(config, output, methods=methods, max_cells=args.max_cells)
        render_driver_validation_report(report, output / "analysis")
        print(f"Driver validation: {output / 'driver-validation.json'}")
        print(f"Available method runs: {report['available_method_runs']}/{report['attempted_method_runs']}")
        print(f"Benchmark acceptance: {report['acceptance']['overall_status']}")


if __name__ == "__main__":
    main()
