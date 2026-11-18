"""Inventory, run, or inspect the training-frozen learned behavioral endpoint bridge."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from effectome.experiments.csl_sanity import CSL_METHODS
from effectome.experiments.manifold_validation_config import load_manifold_validation_profile
from effectome.workflows.cross_method import load_cross_method_profile

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "development"), default="development")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true")
    action.add_argument("--status", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--method", action="append", choices=CSL_METHODS)
    parser.add_argument("--oracle-only", action="store_true", help="Run oracle/known/empty controls only")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.oracle_only and args.method:
        parser.error("--oracle-only and --method cannot be combined")
    root = args.output_root or (
        Path("outputs/learned-manifold-validation") / args.profile / "endpoint-bridge-03"
    )
    if args.status:
        path = root / "manifold-validation.json"
        saved = json.loads(path.read_text()) if path.is_file() else {}
        status = {key: saved[key] for key in ("profile", "completed_cells", "planned_cells",
                  "attempted_method_runs", "available_method_runs", "execution_complete",
                  "all_requested_lanes_available", "claim_scope") if key in saved}
        status.update({"report": str(path), "status": "saved" if saved else "not_started",
                       "cached_cell_files": len(list(root.glob("cells/*/cell.json")))})
        logger.info("%s", json.dumps(status, indent=2))
        return
    from effectome.experiments.manifold_validation import (
        manifold_validation_inventory,
        run_manifold_validation,
    )

    config = load_manifold_validation_profile(args.profile)
    _, methods = load_cross_method_profile("csl_sanity")
    if args.oracle_only:
        methods = {}
    elif args.method:
        methods = {name: methods[name] for name in dict.fromkeys(args.method)}
    inventory = manifold_validation_inventory(config, methods)
    inventory.update({"fitting_enabled": args.run, "output_root": str(root), "workers": args.workers})
    logger.info("%s", json.dumps(inventory, indent=2))
    if args.run:
        from notebooks._shared import require_conventional_runtime

        require_conventional_runtime(methods)
        report = run_manifold_validation(config, root, workers=args.workers, methods=methods)
        logger.info("Execution complete=%s; all requested lanes available=%s; report=%s",
                    report["execution_complete"], report["all_requested_lanes_available"],
                    root / "manifold-validation.json")


if __name__ == "__main__":
    main()
