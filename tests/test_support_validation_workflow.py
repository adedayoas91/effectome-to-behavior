"""Check the shipped run order and the completed-result boundary."""

import ast
import json
from pathlib import Path

import pytest

from effectome.workflows.resume import content_fingerprint
from synthetic.support_validation import validate_and_visualize


def test_active_synthetic_notebooks_are_ordered_and_confirmation_is_enabled():
    directory = Path(__file__).resolve().parents[1] / "synthetic"
    paths = sorted(directory.glob("*.ipynb"))
    assert [path.name for path in paths] == [
        "00_simulator_and_sanity.ipynb",
        "01_run_support_benchmark.ipynb",
        "02_validate_and_visualize.ipynb",
    ]
    codes = []
    for path in paths:
        notebook = json.loads(path.read_text())
        sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        for source in sources:
            ast.parse(source)
        assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
        codes.append("\n".join(sources))
    assert "RUN_SANITY = True" in codes[0]
    assert "RUN = True" in codes[1]
    assert 'PROFILE = "confirmation"' in codes[1] and 'PROFILE = "confirmation"' in codes[2]
    assert "synthetic.behavior_rollout_support" in codes[1]
    assert "validate_and_visualize(RUN_ROOT)" in codes[2]


def test_validation_requires_a_complete_matched_run(tmp_path):
    protocol = {
        "profile": "smoke",
        "settings": {"seed_start": 9, "seed_count": 1, "shuffled_masks": 2, "rollouts": 32},
        "scenarios": ["driver_switch", "constant", "nuisance_switch"],
    }
    signature = content_fingerprint(protocol)
    (tmp_path / "support-protocol.json").write_text(json.dumps({**protocol, "signature": signature}))
    (tmp_path / "support-report.json").write_text(json.dumps({
        "profile": "smoke", "signature": signature, "execution_complete": False,
        "completed_cases": 1, "planned_cases": 3,
    }))
    with pytest.raises(ValueError, match="incomplete"):
        validate_and_visualize(tmp_path)
    assert not (tmp_path / "validation").exists()
