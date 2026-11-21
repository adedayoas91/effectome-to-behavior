"""Export a browser view of saved simulator and fitted disconnection effects."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

TEMPLATE = Path(__file__).with_suffix(".html")


def write_intervention_view(path: Path, cases: list[dict[str, Any]], *, profile: str) -> Path:
    """Write a standalone view of matched effects without running the simulator."""
    payload = {
        "profile": profile,
        # Array columns follow the saved dictionary order, including numeric keys.
        "cases": [{**case, "unit_order": list(case["units"])} for case in cases],
    }
    embedded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    embedded = embedded.replace("<", "\\u003c")
    page = TEMPLATE.read_text(encoding="utf-8").replace("__INTERVENTION_DATA__", embedded, 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(page)
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path
