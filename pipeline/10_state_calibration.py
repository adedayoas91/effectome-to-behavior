"""Execute finite raw-data checks of optional graph-state summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from effectome.connectivity import ConnectivityConfig
from effectome.experiments.state_calibration import (
    SCENARIOS,
    run_state_calibration,
    state_calibration_profile,
    write_state_calibration_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "development"), default="smoke")
    parser.add_argument("--scenario", choices=SCENARIOS, action="append", dest="scenarios")
    parser.add_argument("--history", type=int, action="append", dest="histories")
    parser.add_argument("--dwell", type=int, action="append", dest="dwells")
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--max-draws", type=int)
    parser.add_argument("--neurons", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/state-calibration/smoke"))
    args = parser.parse_args()
    overrides: dict[str, Any] = {}
    for name, value in (
        ("scenarios", args.scenarios), ("history_lengths", args.histories), ("dwell_lengths", args.dwells),
    ):
        if value is not None:
            overrides[name] = tuple(value)
    if args.replicates is not None:
        overrides["replicates"] = args.replicates
    if args.max_draws is not None:
        overrides["max_draws"] = args.max_draws
    for name, value in (("n_neurons", args.neurons), ("n_samples", args.samples),
                        ("stride", args.stride), ("fps", args.fps), ("seed", args.seed)):
        if value is not None:
            overrides[name] = value
    overrides["connectivity"] = ConnectivityConfig(
        name="cgc", max_lag=args.lag, alpha=0.05,
        extra={"support_test": "fastcgc", "beta": args.beta, "support_ridge": 1e-6},
    )
    report = run_state_calibration(state_calibration_profile(args.profile, **overrides))
    paths = write_state_calibration_report(report, args.output_dir)
    print(f"State calibration: {report['summary']}")
    print(f"Manifest: {paths[0]}\nScorecard: {paths[1]}")


if __name__ == "__main__":
    main()
