"""Run the bounded four-upstream-method native graph sanity comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from effectome.experiments.csl_sanity import CSLSanityConfig, run_csl_sanity
from effectome.experiments.csl_sanity_reporting import render_csl_sanity_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/csl-sanity/smoke"))
    args = parser.parse_args()
    config = CSLSanityConfig(seeds=tuple(args.seeds), n_samples=args.samples)
    print(f"Declared independent seeds: {config.seeds}; samples per recording: {config.n_samples}")
    print(f"Planned fits: {2 * 4 * len(config.seeds)} (chain and null, four upstream methods)")
    if args.run:
        report = run_csl_sanity(config, args.output_dir)
        render_csl_sanity_report(report, args.output_dir / "analysis")
        print(f"Available fits: {report['available_runs']}/{report['attempted_runs']}")
        print(f"Report: {args.output_dir / 'analysis' / 'analysis-report.md'}")


if __name__ == "__main__":
    main()
