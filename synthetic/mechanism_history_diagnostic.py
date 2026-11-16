"""Compare discovery histories on identical development recordings and endpoints.

This is a power diagnostic, not additional independent-seed confirmation. All
settings and short-history failures are retained; alpha, beta and support quorum
are unchanged. A longer context is compatible here because the mechanism is fixed.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from effectome.experiments.csl_sanity import CSL_METHODS
from effectome.experiments.mechanism_reporting import render_mechanism_report
from effectome.experiments.mechanism_validation import run_mechanism_validation
from effectome.experiments.mechanism_validation_config import load_mechanism_validation_profile
from effectome.workflows.cross_method import load_cross_method_profile
from effectome.workflows.resume import content_fingerprint

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Default only inventories the paired diagnostic")
    parser.add_argument("--method", action="append", choices=CSL_METHODS)
    parser.add_argument("--output-root", type=Path,
                        default=Path("outputs/driver-mechanism-validation/history-comparison"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base = load_mechanism_validation_profile("smoke")
    _, methods = load_cross_method_profile("csl_sanity")
    if args.method:
        methods = {name: methods[name] for name in dict.fromkeys(args.method)}
    histories = (256, 1024)
    # Same full trial arrays and learned-mechanism fitting data in both cases.
    generator = replace(base.generator, trial_length=1152)
    logger.info("Histories=%s; fixed trial length=1152; burn-in=128; native lanes=%s; development seed=%s",
                histories, list(methods), base.seeds)
    if not args.run:
        return
    comparison: dict = {
        "scope": "paired_history_power_diagnostic_same_raw_seed_not_independent_confirmation",
        "histories": list(histories), "nomination_calibrated": False,
        "scientific_validation_passed": False, "runs": [],
    }
    identities = []
    for history in histories:
        config = replace(base, generator=generator, discovery_burn_in=128, discovery_history_length=history)
        output = args.output_root / f"history-{history}"
        report = run_mechanism_validation(config, output, methods=methods)
        render_mechanism_report(report, output / "analysis")
        identities.append(content_fingerprint([
            (cell["scenario"], cell["seed"], cell["training_input_fingerprint"],
             cell["evaluation_initial_state_fingerprint"], cell["oracle"])
            for cell in report["cells"]
        ]))
        comparison["runs"].append({
            "history_length": history,
            "report": str(output / "mechanism-validation.json"),
            "summary": report["summary"],
            "execution_complete": report["execution_complete"],
            "available_method_runs": report["available_method_runs"],
            "attempted_method_runs": report["attempted_method_runs"],
        })
    if len(set(identities)) != 1:
        raise ValueError(
            "history comparison did not preserve identical training data, origins and oracle events"
        )
    comparison["paired_inputs_verified"] = True
    comparison["paired_input_fingerprint"] = identities[0]
    path = args.output_root / "history-comparison.json"
    path.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
    logger.info("Paired diagnostic complete: %s", path)


if __name__ == "__main__":
    main()
