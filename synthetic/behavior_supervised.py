"""Inventory, run or inspect the distributed continuous-behavior experiment."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from effectome.experiments.behavior_supervised import (
    REPORT,
    behavior_supervised_inventory,
    run_behavior_supervised,
)
from effectome.experiments.behavior_supervised_config import load_behavior_supervised_profile
from effectome.experiments.csl_sanity import CSL_METHODS
from effectome.workflows.cross_method import load_cross_method_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "development", "confirmation"), default="development")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true")
    action.add_argument("--status", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--method", action="append", choices=CSL_METHODS)
    parser.add_argument("--oracle-only", action="store_true", help="Run simulator, known and empty support")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--development-report", type=Path)
    parser.add_argument("--encoder-python", type=Path, help="Use an explicit compatible encoder runtime")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(process)d %(message)s")
    if args.workers < 1 or (args.oracle_only and args.method):
        parser.error("workers must be positive; choose either --oracle-only or --method")
    root = args.output_root or Path("outputs/behavior-supervised") / args.profile / "distributed-behavior-01"
    if args.status:
        path = root / REPORT
        saved = json.loads(path.read_text()) if path.is_file() else {}
        summary = {
            key: saved[key]
            for key in (
                "profile",
                "stage",
                "completed_cells",
                "planned_cells",
                "available_method_runs",
                "attempted_method_runs",
                "execution_complete",
                "nomination_calibrated",
                "scientific_validation_passed",
            )
            if key in saved
        }
        print(
            json.dumps(
                {"report": str(path), "status": "saved" if saved else "not_started", **summary}, indent=2
            )
        )
        return
    config = load_behavior_supervised_profile(args.profile)
    if args.encoder_python:
        config = replace(config, learning=replace(config.learning, runtime_python=str(args.encoder_python)))
    _, methods = load_cross_method_profile("csl_sanity")
    if args.oracle_only:
        methods = {}
    elif args.method:
        methods = {name: methods[name] for name in dict.fromkeys(args.method)}
    print(
        json.dumps(
            {
                **behavior_supervised_inventory(config, methods),
                "output": str(root),
                "fitting_enabled": args.run,
            },
            indent=2,
        ),
        flush=True,
    )
    if not args.run:
        return
    from notebooks._shared import require_conventional_runtime

    require_conventional_runtime(methods)
    report = run_behavior_supervised(
        config, root, workers=args.workers, methods=methods, development_report=args.development_report
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "execution_complete",
                    "all_requested_lanes_available",
                    "completed_cells",
                    "available_method_runs",
                    "native_window_fits",
                    "confirmatory_executed",
                    "scientific_validation_passed",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
