"""Smoke tests for method-split resumable notebook helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import notebooks._shared as shared
from effectome.community import CommunityConfig
from effectome.connectivity import ConnectivityConfig
from effectome.data_module import PreprocessConfig, WindowConfig
from effectome.data_module.schema import WindowedSegments
from effectome.dynamics import GraphStateConfig, ProbabilisticStateConfig, TransitionConfig
from effectome.utils.io import save_artifact
from notebooks._shared import (
    make_run,
    run_bundle_net_reference_preprocessing,
    run_community,
    run_connectivity,
    run_graph_states_and_transitions,
    run_preprocessing,
    run_probabilistic_states,
)


def _subset_windows(windows, n_windows: int = 6) -> WindowedSegments:
    return WindowedSegments(
        segments=np.asarray(windows.segments[:n_windows]),
        windows=list(windows.windows[:n_windows]),
        behavior_per_window={
            key: np.asarray(values[:n_windows]) for key, values in windows.behavior_per_window.items()
        },
        n_neurons=windows.n_neurons,
        fps=windows.fps,
        metadata=dict(windows.metadata),
        anchors=list(windows.anchors[:n_windows]),
        provenance=windows.provenance,
    )


def test_method_split_notebook_files_exist():
    root = Path(__file__).resolve().parents[1] / "notebooks"
    expected_analysis = [
        Path("effectomes/c-GC/01_connectivity.ipynb"),
        Path("linking/c-GC/06_linking.ipynb"),
        Path("effectomes/c-GC-star/01_connectivity.ipynb"),
        Path("counterfactuals/c-GC-star/07_attribution_counterfactual.ipynb"),
        Path("effectomes/partial-correlation/01_connectivity.ipynb"),
        Path("linking/partial-correlation/06_linking.ipynb"),
        Path("clustering/c-GC/02_cluster_markov.ipynb"),
        Path("probabilistic_states/c-GC/03_hmm.ipynb"),
        Path("communities/c-GC/04_temporal_community.ipynb"),
        Path("manifolds/c-GC/05_manifold.ipynb"),
    ]
    dataset_directories = [
        root / "c_elegans",
        root / "v2a_rsns" / "220119_F2_run11",
        root / "v2a_rsns" / "220127_F4_run2",
    ]
    for directory in dataset_directories:
        assert (directory / "preprocessing" / "00_preprocess.ipynb").is_file()
        for relative_path in expected_analysis:
            assert (directory / relative_path).is_file(), (
                f"{directory.relative_to(root)}/{relative_path}"
            )


def test_all_notebooks_bootstrap_repository_imports():
    root = Path(__file__).resolve().parents[1] / "notebooks"

    for path in sorted(root.rglob("*.ipynb")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        first_code_cell = next(
            cell for cell in payload["cells"] if cell["cell_type"] == "code"
        )
        source = "".join(first_code_cell["source"])

        assert "_REPOSITORY_ROOT = next(" in source, path
        assert 'str(_REPOSITORY_ROOT / "src")' in source, path
        assert "str(_REPOSITORY_ROOT)" in source, path


def test_all_notebooks_explain_live_and_resumed_progress():
    root = Path(__file__).resolve().parents[1] / "notebooks"

    for path in sorted(root.rglob("*.ipynb")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        markdown = "\n".join(
            "".join(cell["source"])
            for cell in payload["cells"]
            if cell["cell_type"] == "markdown"
        )
        assert "iterative stages also display live progress bars" in markdown, path
        assert "Reused checkpoints are reported explicitly" in markdown, path


def test_notebook_setup_imports_from_nested_directories():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "notebooks" / "c_elegans" / "preprocessing" / "00_preprocess.ipynb",
        root / "notebooks" / "c_elegans" / "effectomes" / "c-GC" / "01_connectivity.ipynb",
    ]

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
        setup_source = "\n".join("".join(cell["source"]) for cell in code_cells[:2])

        completed = subprocess.run(
            [sys.executable, "-I", "-c", setup_source],
            cwd=path.parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert completed.returncode == 0, f"{path}:\n{completed.stderr}"


def test_preprocess_notebooks_use_configured_output_ids():
    root = Path(__file__).resolve().parents[1] / "notebooks"
    expectations = {
        root / "c_elegans" / "preprocessing" / "00_preprocess.ipynb": (
            'DATASET_OUTPUT_ID = "bundle-net-c-elegans"',
            'RECORDING_ID = "bundle-net-worm-0"',
            'EXPECTED_BEHAVIOR_KEYS = ["motif"]',
        ),
        root / "v2a_rsns" / "220119_F2_run11" / "preprocessing" / "00_preprocess.ipynb": (
            'DATASET_OUTPUT_ID = "v2a-rsns"',
            'RECORDING_ID = "220119_F2_run11"',
            'EXPECTED_BEHAVIOR_KEYS = ["continuous"]',
        ),
        root / "v2a_rsns" / "220127_F4_run2" / "preprocessing" / "00_preprocess.ipynb": (
            'DATASET_OUTPUT_ID = "v2a-rsns"',
            'RECORDING_ID = "220127_F4_run2"',
            'EXPECTED_BEHAVIOR_KEYS = ["continuous"]',
        ),
    }

    for path, snippets in expectations.items():
        notebook = json.loads(path.read_text(encoding="utf-8"))
        text = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"
        )
        for snippet in snippets:
            assert snippet in text, f"{path.name} missing {snippet}"
        assert (
            'WINDOW_CONFIG_PATH = PROJECT_ROOT / "conf" / "windowing" '
            '/ "primary_physical.yaml"'
        ) in text
        assert "WINDOW_CFG = WindowConfig(**WINDOWING_CFG)" in text

    c_elegans_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(
            (root / "c_elegans" / "preprocessing" / "00_preprocess.ipynb").read_text(
                encoding="utf-8"
            )
        )["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "run_bundle_net_reference_preprocessing" in c_elegans_text
    assert "bundle_net_reference_recording" in c_elegans_text
    assert "bundle_net_training_pairs" in c_elegans_text
    assert "BUNDLE_NET_EXCLUDE_NEURONS" in c_elegans_text
    assert "BUNDLE_NET_EXCLUDE_NAME_SOURCE" in c_elegans_text


def test_c_elegans_counterfactual_targets_manifold_not_motif_codes():
    root = Path(__file__).resolve().parents[1] / "notebooks" / "c_elegans"
    for method_directory in ("c-GC", "c-GC-star"):
        path = (
            root
            / "counterfactuals"
            / method_directory
            / "07_attribution_counterfactual.ipynb"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        text = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload["cells"]
            if cell.get("cell_type") == "code"
        )
        assert 'BEHAVIOR_KEY = "motif"' in text
        assert 'endpoint="manifold"' in text


def test_c_elegans_manifold_notebooks_load_canonical_bunddle_profile():
    root = Path(__file__).resolve().parents[1] / "notebooks" / "c_elegans" / "manifolds"
    for method_directory in ("c-GC", "c-GC-star", "partial-correlation"):
        payload = json.loads(
            (root / method_directory / "05_manifold.ipynb").read_text(encoding="utf-8")
        )
        text = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload["cells"]
            if cell.get("cell_type") == "code"
        )
        assert 'MANIFOLD_CONFIG_NAME = "bunddle"' in text
        assert 'MANIFOLD_BEHAVIOR_KEY = "motif"' in text
        assert "OmegaConf.load(MANIFOLD_CONFIG_PATH)" in text


def test_effectome_notebooks_load_canonical_connectivity_profiles():
    root = Path(__file__).resolve().parents[1] / "notebooks" / "c_elegans" / "effectomes"
    expected = {
        "c-GC": "cgc.yaml",
        "c-GC-star": "cgc_star.yaml",
        "partial-correlation": "correlation.yaml",
    }
    for method_directory, config_name in expected.items():
        payload = json.loads(
            (root / method_directory / "01_connectivity.ipynb").read_text(encoding="utf-8")
        )
        text = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload["cells"]
            if cell.get("cell_type") == "code"
        )
        assert f'/ "{config_name}"' in text
        assert "ConnectivityConfig(**CONNECTIVITY_YAML)" in text


def test_attribution_helper_uses_window_aligned_behavior(monkeypatch, tmp_path):
    window_behavior = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    connectivity = SimpleNamespace(
        directed=True,
        behavior_per_window={"motif": window_behavior},
    )
    artifacts = {
        "connectivity": connectivity,
        "community": object(),
        "manifold": object(),
        "linking": object(),
    }
    paths = {name: tmp_path / f"{name}.pkl" for name in artifacts}
    observed: dict[str, np.ndarray] = {}

    class ImmediateRun:
        run_id = "reference"
        method = "cgc"
        run_dir = tmp_path

        def execute(self, _stage, compute, **_kwargs):
            return compute()

    monkeypatch.setattr(
        shared,
        "require_stage",
        lambda _run, stage: (artifacts[stage], paths[stage]),
    )
    monkeypatch.setattr(
        shared,
        "_checkpoint_input",
        lambda *_args, **_kwargs: SimpleNamespace(
            behavior={"motif": np.arange(100, dtype=np.int64)}
        ),
    )

    def capture(_series, _community, _manifold, behavior, _behavior_key, **_kwargs):
        observed["behavior"] = np.asarray(behavior)
        return "candidate-result"

    monkeypatch.setattr(shared, "qualify_candidate_drivers", capture)
    result = shared.run_candidate_drivers(
        ImmediateRun(),
        recording_path=tmp_path / "recording.pkl",
        behavior_key="motif",
    )

    assert result == "candidate-result"
    assert np.array_equal(observed["behavior"], window_behavior)


def test_method_notebooks_reference_shared_artifacts_under_dataset_output_root():
    root = Path(__file__).resolve().parents[1] / "notebooks"
    notebook = root / "c_elegans" / "effectomes" / "c-GC" / "01_connectivity.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"] if cell.get("cell_type") == "code"
    )

    assert 'OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "analysis" / DATASET_OUTPUT_ID / RECORDING_ID' in text
    assert 'WINDOWS_PATH = OUTPUT_ROOT / RUN_ID / "shared" / "stages" / "windows" / "artifact.pkl"' in text
    assert "ARTIFACT_ROOT" not in text


def test_preprocessing_notebook_helper_checkpoints_shared_inputs(tmp_path):
    run = make_run(tmp_path / "analysis" / "synthetic" / "recording-0", "reference", "shared")
    recording, prepared = run_preprocessing(
        run,
        data_cfg={
            "name": "synthetic",
            "dataset_id": "synthetic-test",
            "recording_id": "recording-0",
            "n_neurons": 5,
            "n_timepoints": 90,
            "n_states": 2,
            "regime_dwell": 30,
            "behavior_drivers": 2,
            "seed": 0,
        },
        preprocess_cfg=PreprocessConfig(),
        window_cfg=WindowConfig(
            length=30,
            history_length=30,
            target_length=5,
            stride=5,
            mode="temporal",
        ),
    )

    assert recording.identity.recording_id == "recording-0"
    assert prepared.n_windows == 13
    assert run.completed_stages() == ["raw_recording", "recording", "windows"]

    reference, pairs = run_bundle_net_reference_preprocessing(
        run,
        preprocess_cfg=PreprocessConfig(bundle_net_bandpass=True),
        target_length=15,
    )
    assert reference.metadata["preprocess_dependency"]["kind"] == "global_noncausal_iir"
    assert pairs.x_t.shape == (75, 15, 5)
    assert run.completed_stages() == [
        "bundle_net_reference_recording",
        "bundle_net_training_pairs",
        "raw_recording",
        "recording",
        "windows",
    ]


def test_notebook_helpers_run_resumable_method_lane(tmp_path, windows):
    small = _subset_windows(windows, n_windows=6)
    windows_path = save_artifact(small, tmp_path / "artifacts" / "windows.pkl")
    run = make_run(tmp_path / "runs", "demo", "cgc")

    connectivity = run_connectivity(
        run,
        windows_path=windows_path,
        connectivity_cfg=ConnectivityConfig(
            name="cgc",
            max_lag=1,
            extra={"support_test": "analytic", "n_perm": 0, "n_lags": 1, "seed": 0},
        ),
        chunk_size=3,
    )
    assert connectivity.n_windows == 6
    assert run.completed_stages() == [
        "connectivity",
        "connectivity.chunk-00000",
        "connectivity.chunk-00001",
        "windows_input",
    ]

    states, transitions = run_graph_states_and_transitions(
        run,
        graph_cfg=GraphStateConfig(n_states=2, metric="causal_kernel", select_k=False, seed=0),
        transition_cfg=TransitionConfig(n_null=8, seed=0),
    )
    probabilistic = run_probabilistic_states(
        run,
        probabilistic_cfg=ProbabilisticStateConfig(n_states=2, n_components=2, n_iter=10, seed=0),
    )
    community = run_community(
        run,
        community_cfg=CommunityConfig(
            name="temporal",
            resolution=1.0,
            symmetrize=False,
            use_absolute=False,
            seed=0,
            extra={"n_communities": 2, "temporal_penalty": 1.0, "n_runs": 2, "max_iter": 10},
        ),
    )

    assert states.labels.shape[0] == connectivity.n_windows
    assert transitions.transition_matrix.shape == (2, 2)
    assert probabilistic.labels.shape[0] == connectivity.n_windows
    assert community.labels.shape[0] == connectivity.n_windows
