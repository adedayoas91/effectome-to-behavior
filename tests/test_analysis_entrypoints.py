"""Executable raw-manifest readiness and reporting contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("preflight", [True, False])
def test_evaluation_cli_raw_manifest_reports_without_global_artifacts(tmp_path, preflight):
    output = tmp_path / "artifacts"
    command = [
        sys.executable,
        "pipeline/08_evaluate.py",
        "paths.evaluation_manifest=conf/evaluation-manifests/synthetic-smoke.yaml",
        f"paths.artifacts={output}",
        f"hydra.run.dir={tmp_path / 'hydra'}",
    ]
    if preflight:
        command.append("+preflight=true")
    subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True,
        timeout=120, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    report = json.loads(
        (output / ("preflight.json" if preflight else "evaluation-report.json")).read_text()
    )
    if preflight:
        assert report["status"] == "ready_for_model_fitting"
        assert report["models_fitted"] is False
        assert report["available_folds"] > 0
        assert not (output / "evaluation.pkl").exists()
    else:
        assert report["prediction_count"] > 0
        assert report["diagnostics"]["available_folds"] > 0
        assert (output / "evaluation.pkl").is_file()
        assert (output / "evaluation-manifest.yaml").is_file()
        assert report["provenance"]["claim_eligibility"] == "development_only"
        repeated = subprocess.run(
            command, cwd=ROOT, check=True, capture_output=True, text=True,
            timeout=120, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )
        assert "reused checkpoint" in repeated.stdout
