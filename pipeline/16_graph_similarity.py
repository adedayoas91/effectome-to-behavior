"""Inventory or run a bounded graph-identity/topology grouping diagnostic."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from effectome.experiments.graph_similarity_validation import (
    graph_similarity_inventory,
    load_graph_similarity_profile,
    render_graph_similarity_report,
    run_graph_similarity_validation,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "development"), default="smoke")
    parser.add_argument(
        "--run", action="store_true", help="Default only inventories the structural diagnostic"
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_graph_similarity_profile(args.profile)
    logger.info("Graph similarity inventory: %s", json.dumps(graph_similarity_inventory(config)))
    if not args.run:
        return
    output = args.output_dir or Path("outputs/graph-similarity") / args.profile
    report = run_graph_similarity_validation(config, output)
    analysis = render_graph_similarity_report(report, output / "analysis")
    logger.info("Available fits: %s/%s; descriptive report: %s", report["available_fits"],
                report["attempted_fits"], analysis)


if __name__ == "__main__":
    main()
