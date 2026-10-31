"""Notebook entry points identify and validate the installed backend environment."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from notebooks._build_notebooks import (
    BACKEND_ENVIRONMENT_CHECK,
    ENVIRONMENT_NOTE,
    code_cell,
    markdown_cell,
    write_notebook,
)

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1] / "notebooks"


def test_all_notebooks_validate_the_migrated_environment_and_compile():
    notebooks = sorted(NOTEBOOK_ROOT.rglob("*.ipynb"))
    assert notebooks
    for path in notebooks:
        payload = json.loads(path.read_text())
        assert payload["metadata"]["language_info"]["version"] == "3.14", path
        first_code = next(cell for cell in payload["cells"] if cell["cell_type"] == "code")
        assert BACKEND_ENVIRONMENT_CHECK.strip() in "".join(first_code["source"]), path
        markdown = "\n".join(
            "".join(cell["source"]) for cell in payload["cells"] if cell["cell_type"] == "markdown"
        )
        assert ENVIRONMENT_NOTE in markdown, path
        assert not re.search(r"^#.*(?<![A-Za-z])c-GC", markdown, re.MULTILINE), path
        for cell_index, cell in enumerate(payload["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                ast.parse(source, filename=f"{path}:cell-{cell_index}")
                assert "git clone" not in source, path
                assert not re.search(r"support_test['\"]?\s*:\s*['\"]analytic['\"]", source), path


def test_generated_notebooks_include_the_same_environment_contract(tmp_path):
    path = tmp_path / "generated.ipynb"
    cells = [markdown_cell("# Example"), code_cell("print('analysis')")]
    write_notebook(path, cells)
    payload = json.loads(path.read_text())
    assert payload["metadata"]["language_info"]["version"] == "3.14"
    assert BACKEND_ENVIRONMENT_CHECK.strip() in "".join(payload["cells"][1]["source"])
    assert ENVIRONMENT_NOTE in "".join(payload["cells"][0]["source"])
    assert cells[0]["source"] == ["# Example\n"]
