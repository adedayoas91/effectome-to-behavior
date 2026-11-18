"""Fresh-seed entrypoints expose workload and read evidence without fitting."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

from effectome.connectivity import ConnectivityConfig
from effectome.experiments.csl_sanity import CSL_METHODS
from synthetic.mechanism_replication import inspect_replication_status, replication_inventory

ROOT = Path(__file__).resolve().parents[1]


def test_inventory_counts_primary_and_positive_only_comparison_without_fitting():
    config = SimpleNamespace(
        profile="replication", calibration_seeds=tuple(range(19)), positive_seeds=tuple(range(20)),
        null_seeds=tuple(range(60)), primary_history=1024, comparison_history=256,
        calibration_alpha=0.05, nomination_floor=0.1,
        mechanism=SimpleNamespace(
            generator=SimpleNamespace(n_trials=24), evaluation_fraction=0.5,
            fit=SimpleNamespace(validation_fraction=0.25),
        ),
    )
    methods = {name: ConnectivityConfig(name=name) for name in CSL_METHODS}
    inventory = replication_inventory(config, methods)
    assert inventory["planned_cells"] == 119
    assert inventory["primary_cells"] == 99
    assert inventory["comparison_positive_cells"] == 20
    assert inventory["planned_method_runs"] == 714
    assert inventory["native_window_fits"] == 4284
    assert not inventory["fitting_enabled"]
    assert inventory["all_four_native_methods_requested"]
    subset = replication_inventory(config, {"fast_cgc": methods["fast_cgc"]})
    assert subset["planned_method_runs"] == 357
    assert not subset["all_four_native_methods_requested"]


def test_status_reads_only_the_requested_report(tmp_path):
    output = tmp_path / "not-created"
    missing = inspect_replication_status(output)
    assert missing["status"] == "not_started"
    assert not output.exists()
    output.mkdir()
    report = {"execution_complete": False, "progress": {"completed_cells": 3}}
    (output / "replication-report.json").write_text(json.dumps(report))
    status = inspect_replication_status(output)
    assert status["saved"] == report
    assert not status["fitting_enabled"]


def test_replication_notebook_is_runnable_and_keeps_seed_pool_ownership_visible():
    path = ROOT / "synthetic/07_mechanism_replication.ipynb"
    payload = json.loads(path.read_text())
    code = []
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            ast.parse(source, filename=f"{path}:cell-{index}")
            code.append(source)
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
    combined = "\n".join(code)
    assert 'PROFILE = "replication"' in combined
    assert "RUN = True" in combined
    assert "WORKERS = 4" in combined
    assert "require_conventional_runtime(selected_methods" in combined
    assert '"--run"' in combined
    assert '"outputs/driver-mechanism-replication"' in combined
    assert "calibration_seeds" not in combined or "replication_inventory" in combined
    markdown = "\n".join("".join(c["source"]) for c in payload["cells"] if c["cell_type"] == "markdown")
    assert "19 separate calibration-null seeds" in markdown
    assert "60 untouched null-test seeds" in markdown
    assert "256-sample history" in markdown
    assert "not receive an independent calibrated-null claim" in markdown
