"""A sanity report must survive missing runtimes and preserve negative results."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

from effectome.connectivity import ConnectivityConfig
from effectome.experiments import csl_sanity
from effectome.experiments.csl_sanity_reporting import render_csl_sanity_report

matplotlib.use("Agg")


@pytest.mark.parametrize("available_methods", [("fast_cgc", "fast_cgc_star"), ()])
def test_rendering_retains_unavailable_methods_empty_agreement_and_missed_truth(
    available_methods, tmp_path, monkeypatch,
):
    # Exercise the producer's actual report schema. Available test backends
    # deliberately miss both planted links, rather than manufacture recovery.
    def factory(config):
        if config.name not in available_methods:
            raise FileNotFoundError(f"missing official runtime for {config.name}")

        def run(windows):
            return SimpleNamespace(
                matrices=np.zeros((windows.n_windows, windows.n_neurons, windows.n_neurons)),
                diagnostics={"backend": "test_transport_only"},
            )

        return SimpleNamespace(run=run)

    monkeypatch.setattr(csl_sanity, "ConnectivityFactory", factory)
    methods = {name: ConnectivityConfig(name=name) for name in csl_sanity.CSL_METHODS}
    config = csl_sanity.CSLSanityConfig(seeds=(2,), n_samples=256)
    csl_sanity.run_csl_sanity(config, tmp_path, methods=methods)
    report = json.loads((tmp_path / "sanity.json").read_text())
    render_csl_sanity_report(report, tmp_path)

    with (tmp_path / "seed-metrics.csv").open(newline="") as handle:
        seed_rows = list(csv.DictReader(handle))
    assert len(seed_rows) == 8
    missing = [row for row in seed_rows if row["status"] == "unavailable"]
    assert len(missing) == 8 - 2 * len(available_methods)
    assert all("missing official runtime" in row["reason"] for row in missing)
    for row in seed_rows:
        if row["status"] == "available" and row["scenario"] == "chain":
            assert row["tp"] == "0" and row["fn"] == "2"
            assert row["recall"] == "0.0" and row["exact_support"] == "False"
            assert row["precision"] == ""

    with (tmp_path / "pairwise-agreement.csv").open(newline="") as handle:
        pairs = list(csv.DictReader(handle))
    assert len(pairs) == 12
    assert all(row["jaccard"] == "" for row in pairs)
    available_pairs = [row for row in pairs if row["status"] == "available"]
    assert len(available_pairs) == (2 if available_methods else 0)
    assert all(row["both_empty"] == "True" for row in available_pairs)
    assert pairs[0]["status"] == "unavailable"

    summary = (tmp_path / "analysis-report.md").read_text()
    assert f"Available fits: {2 * len(available_methods)}/8" in summary
    assert "undefined Jaccard" in summary
    assert "does not establish manifold transitions" in summary
    for figure in ("figure-01-native-support", "figure-02-seed-recovery"):
        assert (tmp_path / "figures" / f"{figure}.png").stat().st_size > 1000
        assert (tmp_path / "figures" / f"{figure}.pdf").read_bytes().startswith(b"%PDF")
