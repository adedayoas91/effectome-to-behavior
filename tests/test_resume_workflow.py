"""Resumable workflow checkpoints must be atomic and reject stale inputs."""

from __future__ import annotations

import numpy as np
import pytest

from effectome.workflows import ResumableRun, content_fingerprint, implementation_fingerprint


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
    progress_events: list[tuple[int, int, bool]] = []

    def compute_chunk(value: int) -> int:
        calls.append(value)
        return value * value

    first = run.execute_chunks(
        "connectivity",
        [1, 2, 3],
        compute_chunk,
        sum,
        config={"chunk_size": 1},
        progress_callback=lambda done, total, reused: progress_events.append((done, total, reused)),
    )
    second = run.execute_chunks(
        "connectivity",
        [1, 2, 3],
        compute_chunk,
        sum,
        config={"chunk_size": 1},
        progress_callback=lambda done, total, reused: progress_events.append((done, total, reused)),
    )
    assert first == second == 14
    assert calls == [1, 2, 3]
    assert run.completed_stages() == [
        "connectivity",
        "connectivity.chunk-00000",
        "connectivity.chunk-00001",
        "connectivity.chunk-00002",
    ]
    assert progress_events[:3] == [(1, 3, False), (2, 3, False), (3, 3, False)]
    assert progress_events[3:] == [(1, 3, True), (2, 3, True), (3, 3, True)]


def test_method_namespaces_do_not_share_checkpoints(tmp_path) -> None:
    root = tmp_path / "runs"
    cgc = ResumableRun(root, "same-run", "cgc")
    cgc_star = ResumableRun(root, "same-run", "cgc-star")
    assert cgc.execute("fit", lambda: "primary", config={}) == "primary"
    assert cgc_star.execute("fit", lambda: "robustness", config={}) == "robustness"
    assert cgc.run_dir != cgc_star.run_dir


def test_implementation_version_invalidates_completed_stage(tmp_path) -> None:
    root = tmp_path / "runs"
    old = ResumableRun(root, "same-run", "cgc", implementation_version="old")
    new = ResumableRun(root, "same-run", "cgc", implementation_version="new")
    assert old.execute("fit", lambda: "old", config={}) == "old"
    assert new.execute("fit", lambda: "new", config={}) == "new"
    assert new.last_record is not None and not new.last_record.reused


def test_load_completed_rejects_a_stale_implementation(tmp_path) -> None:
    root = tmp_path / "runs"
    old = ResumableRun(root, "same-run", "cgc", implementation_version="old")
    old.execute("fit", lambda: "old", config={})
    new = ResumableRun(root, "same-run", "cgc", implementation_version="new")

    assert new.completed_stages() == []
    with pytest.raises(RuntimeError, match="implementation"):
        new.load_completed("fit")


def test_identical_relocated_dependency_keeps_semantic_signature(tmp_path) -> None:
    first = tmp_path / "first" / "input.bin"
    second = tmp_path / "second" / "input.bin"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"same-content")
    second.write_bytes(b"same-content")
    run = ResumableRun(tmp_path / "runs", "portable", "cgc", implementation_version="v1")
    assert run.execute("fit", lambda: 1, config={}, dependencies={"input": first}) == 1
    assert run.execute("fit", lambda: 2, config={}, dependencies={"input": second}) == 1
    assert run.last_record is not None and run.last_record.reused


def test_failed_recomputation_cannot_serve_stale_artifact(tmp_path) -> None:
    run = ResumableRun(tmp_path / "runs", "failed", "cgc", implementation_version="v1")
    run.execute("fit", lambda: "old", config={"version": 1})
    with pytest.raises(RuntimeError, match="boom"):
        run.execute(
            "fit",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            config={"version": 2},
        )
    with pytest.raises(RuntimeError, match="not completed"):
        run.load_completed("fit")


def test_implementation_fingerprint_covers_pipeline_and_notebook_orchestration(tmp_path) -> None:
    root = tmp_path / "repo"
    (root / "src" / "effectome").mkdir(parents=True)
    (root / "pipeline").mkdir()
    (root / "notebooks").mkdir()
    (root / "src" / "effectome" / "core.py").write_text("VALUE = 1\n")
    (root / "pipeline" / "08_evaluate.py").write_text("VALUE = 1\n")
    (root / "notebooks" / "_shared.py").write_text("VALUE = 1\n")
    (root / "notebooks" / "_build_notebooks.py").write_text("VALUE = 1\n")
    (root / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")

    initial = implementation_fingerprint(root)
    (root / "pipeline" / "08_evaluate.py").write_text("VALUE = 2\n")
    after_pipeline = implementation_fingerprint(root)
    (root / "notebooks" / "_shared.py").write_text("VALUE = 2\n")
    after_helper = implementation_fingerprint(root)
    (root / "notebooks" / "_build_notebooks.py").write_text("VALUE = 2\n")
    after_builder = implementation_fingerprint(root)

    assert len({initial, after_pipeline, after_helper, after_builder}) == 4
