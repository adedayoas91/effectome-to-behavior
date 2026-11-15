"""Visual exports retain unsupported results and frozen evidence provenance."""

import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from synthetic._validation_plots import _matrix  # noqa: E402
from synthetic.validation_figures import _tables, _worked_case, render_validation_visuals  # noqa: E402


def test_tables_keep_negative_gains_and_incomplete_seed_denominators(tmp_path):
    report = {
        "planned_cells": [{"scenario": "stationary", "spec_id": "first"}],
        "planned_methods": ["cgc", "cgc_star"],
        "config": {"seeds": [0, 1, 2]},
    }
    statistics = {
        "summaries": [
            {
                "spec_id": "first",
                "method": "cgc",
                "mean_gain_nats": -0.25,
                "sd_gain_nats": 0.5,
                "n_available_seeds": 2,
                "n_planned_seeds": 3,
            },
            {
                "spec_id": "first",
                "method": "cgc_star",
                "mean_gain_nats": None,
                "sd_gain_nats": None,
                "n_available_seeds": 0,
                "n_planned_seeds": 3,
            },
        ]
    }
    _tables(report, statistics, tmp_path)
    markdown = (tmp_path / "table-01-predictive-summary.md").read_text()
    latex = (tmp_path / "table-01-predictive-summary.tex").read_text()
    assert "-0.250 ± 0.500 (2/3)" in markdown
    assert "Unavailable (0/3)" in markdown
    assert r"$-0.250\pm 0.500$" in latex
    assert "2/3" in latex and "0/3" in latex
    assert "NA" in latex


def test_matrix_does_not_display_unsupported_zero_rows_as_probabilities():
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    matrix = _matrix(axis, [[0, 0], [0.2, 0.8]], [0], "Observed")
    assert np.asarray(matrix.get_array().mask).tolist() == [[True, True], [False, False]]
    assert [text.get_text() for text in axis.texts] == ["NA", "NA", "0.20", "0.80"]
    plt.close(figure)


def test_visuals_reject_mismatched_statistics_and_do_not_select_a_successful_case(tmp_path):
    source = tmp_path / "source"
    (source / "analysis").mkdir(parents=True)
    (source / "validation.json").write_text(json.dumps({"manifest_fingerprint": "simulation"}))
    (source / "analysis/statistics.json").write_text(json.dumps({"manifest_fingerprint": "other"}))
    with pytest.raises(ValueError, match="provenance disagree"):
        render_validation_visuals(source, tmp_path / "visuals")
    report = {
        "planned_cells": [
            {"scenario": "matched_marginals", "cell_id": "first"},
            {"scenario": "matched_marginals", "cell_id": "second"},
        ],
        "rows": [{"cell_id": "second", "method": "cgc", "status": "available"}],
    }
    comparison, _, status = _worked_case(report, tmp_path)
    assert comparison is None
    assert status["reason"] == "first declared worked case was not executed"
