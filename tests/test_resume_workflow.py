"""Resumable workflow checkpoints must be atomic and reject stale inputs."""

from __future__ import annotations

import numpy as np

from effectome.workflows import ResumableRun, content_fingerprint


def test_content_fingerprint_rejects_same_shape_changed_arrays() -> None:
    first = np.zeros((4, 3), dtype=np.float32)
    changed = first.copy()
    changed[0, 0] = 1.0
    assert content_fingerprint(first) == content_fingerprint(first.copy())
    assert content_fingerprint(first) != content_fingerprint(changed)


def test_completed_stage_is_reused_only_for_matching_signature(tmp_path) -> None:
    dependency = tmp_path / "input.bin"
    dependency.write_bytes(b"version-one")
    run = ResumableRun(tmp_path / "runs", "run-01", "cgc")
    calls = 0

    def compute() -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"value": calls}

    first = run.execute(
        "connectivity",
        compute,
        config={"alpha": 0.05},
        dependencies={"windows": dependency},
    )
    second = run.execute(
        "connectivity",
        compute,
        config={"alpha": 0.05},
        dependencies={"windows": dependency},
    )
    assert first == second == {"value": 1}
    assert calls == 1
    assert run.last_record is not None and run.last_record.reused

    dependency.write_bytes(b"version-two")
    refreshed = run.execute(
        "connectivity",
        compute,
        config={"alpha": 0.05},
        dependencies={"windows": dependency},
    )
    assert refreshed == {"value": 2}
    assert calls == 2
    assert run.last_record is not None and not run.last_record.reused


def test_chunks_and_combination_resume_independently(tmp_path) -> None:
    run = ResumableRun(tmp_path / "runs", "run-02", "cgc-star")
    calls: list[int] = []

    def compute_chunk(value: int) -> int:
        calls.append(value)
        return value * value

    first = run.execute_chunks(
        "connectivity",
        [1, 2, 3],
        compute_chunk,
        sum,
        config={"chunk_size": 1},
    )
    second = run.execute_chunks(
        "connectivity",
        [1, 2, 3],
        compute_chunk,
        sum,
        config={"chunk_size": 1},
    )
    assert first == second == 14
    assert calls == [1, 2, 3]
    assert run.completed_stages() == [
        "connectivity",
        "connectivity.chunk-00000",
        "connectivity.chunk-00001",
        "connectivity.chunk-00002",
    ]


def test_method_namespaces_do_not_share_checkpoints(tmp_path) -> None:
    root = tmp_path / "runs"
    cgc = ResumableRun(root, "same-run", "cgc")
    cgc_star = ResumableRun(root, "same-run", "cgc-star")
    assert cgc.execute("fit", lambda: "primary", config={}) == "primary"
    assert cgc_star.execute("fit", lambda: "robustness", config={}) == "robustness"
    assert cgc.run_dir != cgc_star.run_dir
