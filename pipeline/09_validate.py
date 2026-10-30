"""Preflight or explicitly execute the registered raw-process benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from effectome.experiments.benchmark import (
    benchmark_preflight,
    load_benchmark_config,
    print_benchmark_summary,
    run_benchmark,
    write_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="conf/validation/revised.yaml")
    parser.add_argument("--profile", choices=("smoke", "publication"), default="smoke")
    parser.add_argument(
        "--run", action="store_true", help="Execute fits; default only validates/lists the grid"
    )
    parser.add_argument("--suite", action="append", dest="suites", help="Repeat to select registered suites")
    parser.add_argument("--max-cells", type=int, help="Bound attempted raw recordings")
    parser.add_argument("--offset", type=int, default=0, help="Stable offset in the selected manifest")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--list-cells", action="store_true", help="Include every cell in stdout preflight")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--calibration-tolerance", type=float, help="Predeclared error tolerance; freeze before execution"
    )
    args = parser.parse_args()
    overrides = {
        "profile": args.profile,
        "master_seed": 260925 if args.profile == "smoke" else 260926,
        "cell_offset": args.offset,
        "resume": not args.no_resume,
        "output_dir": args.output_dir or Path("outputs/validation") / args.profile,
    }
    if args.profile == "smoke":
        overrides.update(max_cells=1, max_origins=2, calibration_draws=19)
    if args.suites:
        overrides["suites"] = tuple(args.suites)
    if args.max_cells is not None:
        overrides["max_cells"] = args.max_cells
    if args.calibration_tolerance is not None:
        overrides["calibration_tolerance"] = args.calibration_tolerance
    config = load_benchmark_config(args.config, **overrides)
    if args.preflight_output is not None:
        write_preflight(args.preflight_output, config)
    if args.run:
        print(print_benchmark_summary(run_benchmark(config)))
    else:
        preflight = benchmark_preflight(config)
        if not args.list_cells:
            preflight.pop("cells")
        print(json.dumps(preflight, indent=2))


if __name__ == "__main__":
    main()
