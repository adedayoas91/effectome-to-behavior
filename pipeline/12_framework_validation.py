"""Execute or inventory the reproducible shared-transition validation matrix."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from effectome.experiments.framework_validation import (
    framework_validation_cells,
    load_framework_validation_profile,
    run_framework_validation,
)
from effectome.experiments.framework_validation_analysis import (
    analyze_framework_validation,
    export_manuscript_validation_bundle,
)
from effectome.workflows.cross_method import load_cross_method_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "development", "manuscript"), default="smoke")
    parser.add_argument(
        "--run", action="store_true", help="Execute; default only inventories the declared matrix"
    )
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--manuscript-dir", type=Path, help="Export provenance-linked generated artifacts")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_framework_validation_profile(args.profile)
    output_dir = args.output_dir or (
        Path("outputs/framework-validation") / config.estimator_profile / args.profile
    )
    _, methods = load_cross_method_profile(config.estimator_profile)
    cells = framework_validation_cells(config)
    print(
        json.dumps(
            {
                "profile": args.profile,
                "planned_cells": len(cells),
                "planned_method_runs": len(cells) * len(methods),
                "methods": list(methods),
                "seeds": config.seeds,
            }
        )
    )
    if args.run:
        report = run_framework_validation(config, output_dir, max_cells=args.max_cells)
        analyze_framework_validation(report, output_dir / "analysis")
        if args.manuscript_dir is not None:
            export_manuscript_validation_bundle(output_dir / "analysis", args.manuscript_dir)
        print(f"Validation manifest: {output_dir / 'validation.json'}")
        print(f"Available method runs: {report['available_methods']}/{report['attempted_methods']}")


if __name__ == "__main__":
    main()
