"""Compare eight effectomes against shared, strictly future manifold transitions."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from effectome.data_module.schema import WindowedSegments
from effectome.experiments.transition_examples import (
    TransitionExampleConfig,
    make_transition_example,
    make_transition_example_windows,
)
from effectome.utils.io import load_artifact
from effectome.workflows.cross_method import (
    load_cross_method_profile,
    run_cross_method_comparison,
    write_cross_method_report,
)
from effectome.workflows.evaluation import build_primary_inputs_from_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "development"), default="smoke")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--windows", type=Path, help="Trusted local identity-preprocessed windows artifact")
    source.add_argument("--raw-manifest", type=Path, help="Existing raw-evaluation YAML manifest")
    parser.add_argument(
        "--scenario",
        choices=("one_to_one", "matched_marginals", "fixed_circuit"),
        default="matched_marginals",
    )
    parser.add_argument("--samples", type=int, default=1536)
    parser.add_argument("--neurons", type=int, default=4)
    parser.add_argument("--method", action="append", dest="methods", help="Subset of the eight lane names")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cross-method/smoke"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config, methods = load_cross_method_profile(args.profile)
    neuron_ids = None
    if args.raw_manifest is not None:
        from omegaconf import OmegaConf

        manifest = OmegaConf.to_container(OmegaConf.load(args.raw_manifest), resolve=True)
        if not isinstance(manifest, dict):
            parser.error("raw manifest must be a YAML mapping")
        windows, _ = build_primary_inputs_from_manifest(manifest)
    elif args.windows is not None:
        windows = load_artifact(args.windows)
        if not isinstance(windows, WindowedSegments):
            parser.error("--windows must load a WindowedSegments artifact")
    else:
        recording = make_transition_example(
            TransitionExampleConfig(
                scenario=args.scenario,
                n_samples=args.samples,
                n_neurons=args.neurons,
                seed=config.seed,
            )
        )
        windows = make_transition_example_windows(recording)
        neuron_ids = [str(value) for value in recording.neuron_ids]
    if args.methods:
        unknown = set(args.methods) - set(methods)
        if unknown:
            parser.error(f"unknown estimator lanes: {sorted(unknown)}")
        methods = {name: methods[name] for name in args.methods}
    report = run_cross_method_comparison(windows, config, method_configs=methods, neuron_ids=neuron_ids)
    paths = write_cross_method_report(report, args.output_dir)
    available = sum(result["status"] == "available" for result in report["methods"].values())
    logger = logging.getLogger(__name__)
    logger.info("Available methods: %d/%d. Report: %s", available, len(methods), paths[0])
    if available != len(methods):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
