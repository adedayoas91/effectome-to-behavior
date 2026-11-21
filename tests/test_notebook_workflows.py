"""Smoke tests for flattened, CSL-first resumable notebook helpers."""

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
from effectome.manifold import ManifoldConfig
from effectome.utils.io import save_artifact
from notebooks._build_notebooks import (
    DatasetSpec,
    datasetize_cells,
    notebook_attribution,
    notebook_preprocessing,
)
from notebooks._shared import (
    make_run,
    run_bundle_net_reference_preprocessing,
    run_community,
    run_connectivity,
    run_graph_states_and_transitions,
    run_preprocessing,
    run_probabilistic_states,
)

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
V2A_REFERENCE = NOTEBOOKS / "v2a_rsns" / "220119_F2_run11"
WORM_ROLLOUT = NOTEBOOKS / "c_elegans"
LEGACY_WORM_SPEC = DatasetSpec(
    slug="c_elegans",
    label="C. elegans",
    config_name="c_elegans",
    relative_directory=Path("c_elegans"),
    dataset_id="bundle-net-c-elegans",
    recording_id="bundle-net-worm-0",
    behavior_keys=("motif",),
    primary_behavior="motif",
    endpoint_type="categorical",
    feature_sensitivity_endpoint="manifold",
    bundle_net_reference=True,
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


def test_method_notebook_files_are_flattened():
    root = Path(__file__).resolve().parents[1] / "notebooks"
    methods = {
        "cgc",
        "cgc_star",
        "correlation_partial",
        "sparse_var",
        "windowed_sparse_var",
        "switching_var",
        "tvvar_smooth",
        "tvvar_fused",
    }
    expected_by_stage = {
        "01_effectomes": {f"{method}.ipynb" for method in methods},
        "02_clustering": {f"{method}.ipynb" for method in methods},
        "03_probabilistic_states": {f"{method}.ipynb" for method in methods},
        "04_communities": {f"{method}.ipynb" for method in methods},
        "05_manifolds": {f"{method}.ipynb" for method in methods},
        "06_linking": {f"{method}.ipynb" for method in methods},
        "07_counterfactuals": {
            f"{method}.ipynb" for method in methods - {"correlation_partial"}
        },
    }
    dataset_directories = [
        root / "v2a_rsns" / "220119_F2_run11",
        root / "v2a_rsns" / "220127_F4_run2",
    ]
    for directory in dataset_directories:
        assert (directory / "00_preprocessing" / "preprocess.ipynb").is_file()
        assert (directory / "08_evaluation" / "primary_m0_m1.ipynb").is_file()
        for stage, expected_names in expected_by_stage.items():
            stage_directory = directory / stage
            assert {path.name for path in stage_directory.glob("*.ipynb")} == expected_names
            assert not any(path.is_dir() for path in stage_directory.iterdir())


def test_worm_rollout_has_three_ordered_notebooks_using_one_run():
    expected_workflow = {
        "00_data_and_timing.ipynb": (
            "preflight(REPO, OUTPUT_ROOT, worm_indices=WORM_INDICES, data_root=DATA_ROOT)"
        ),
        "01_fit_and_forecast.ipynb": "run_forecasts(REPO, OUTPUT_ROOT)",
        "02_score_and_disconnections.ipynb": "summarize(REPO, OUTPUT_ROOT, disconnect_rois=DISCONNECT_ROIS)",
    }
    paths = sorted(WORM_ROLLOUT.rglob("*.ipynb"))
    assert paths == [WORM_ROLLOUT / name for name in expected_workflow]
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
        assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in cells)
        code = "\n".join("".join(cell["source"]) for cell in cells)
        assert 'RUN_ID = "motif-rollout-01"' in code
        assert 'OUTPUT_ROOT = REPO / "outputs" / "celegans-rollout" / RUN_ID' in code
        assert expected_workflow[path.name] in code
        assert "from effectome.workflows.celegans_rollout import" in code
    assert "WORM_INDICES = (0, 1, 2, 3, 4)" in paths[0].read_text(encoding="utf-8")


def test_legacy_notebooks_bootstrap_repository_imports():
    root = Path(__file__).resolve().parents[1] / "notebooks"

    for path in sorted(root.rglob("*.ipynb")):
        if path.parent == WORM_ROLLOUT:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        first_code_cell = next(
            cell for cell in payload["cells"] if cell["cell_type"] == "code"
        )
        source = "".join(first_code_cell["source"])

        assert "_REPOSITORY_ROOT = next(" in source, path
        assert 'str(_REPOSITORY_ROOT / "src")' in source, path
        assert "str(_REPOSITORY_ROOT)" in source, path


def test_legacy_notebooks_explain_live_and_resumed_progress():
    root = Path(__file__).resolve().parents[1] / "notebooks"

    for path in sorted(root.rglob("*.ipynb")):
        if path.parent == WORM_ROLLOUT:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        markdown = "\n".join(
            "".join(cell["source"])
            for cell in payload["cells"]
            if cell["cell_type"] == "markdown"
        )
        assert "iterative stages also display live progress bars" in markdown, path
        assert "Reused, failed, running, and missing checkpoints are reported explicitly" in markdown, path


def test_legacy_notebooks_print_resume_header():
    root = Path(__file__).resolve().parents[1] / "notebooks"

    for path in sorted(root.rglob("*.ipynb")):
        if path.parent == WORM_ROLLOUT:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload["cells"]
            if cell.get("cell_type") == "code"
        )
        assert "print_notebook_header" in code, path
        assert "expected_stages=" in code, path
        assert "force=" in code, path


def test_notebook_setup_imports_from_nested_directories():
    paths = [
        V2A_REFERENCE / "00_preprocessing" / "preprocess.ipynb",
        V2A_REFERENCE / "01_effectomes" / "cgc.ipynb",
        *sorted(WORM_ROLLOUT.glob("*.ipynb")),
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
        root / "v2a_rsns" / "220119_F2_run11" / "00_preprocessing" / "preprocess.ipynb": (
            'DATASET_OUTPUT_ID = "v2a-rsns"',
            'RECORDING_ID = "220119_F2_run11"',
            'EXPECTED_BEHAVIOR_KEYS = ["continuous"]',
        ),
        root / "v2a_rsns" / "220127_F4_run2" / "00_preprocessing" / "preprocess.ipynb": (
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


def test_legacy_worm_preprocessing_template_keeps_retrospective_reference():
    c_elegans_text = "\n".join(
        "".join(cell["source"])
        for cell in notebook_preprocessing(LEGACY_WORM_SPEC)
        if cell["cell_type"] == "code"
    )
    assert "run_bundle_net_reference_preprocessing" in c_elegans_text
    assert "bundle_net_reference_recording" in c_elegans_text
    assert "bundle_net_training_pairs" in c_elegans_text
    assert "BUNDLE_NET_EXCLUDE_NEURONS" in c_elegans_text
    assert "BUNDLE_NET_EXCLUDE_NAME_SOURCE" in c_elegans_text
    assert 'DATA_CFG["exclude_neuron_names"]' in c_elegans_text
    assert 'DATA_CFG["exclude_neuron_name_source"]' in c_elegans_text
    assert 'DATA_CFG["bundle_net_reference"]["exclude_neuron_names"]' not in c_elegans_text
    assert "source -> {recording.n_neurons} analysis" in c_elegans_text


def test_legacy_worm_counterfactual_template_targets_manifold():
    cells = datasetize_cells(notebook_attribution("cgc", "fastC-GC"), LEGACY_WORM_SPEC)
    text = "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")
    assert 'BEHAVIOR_KEY = "motif"' in text
    assert 'endpoint="manifold"' in text


def test_counterfactual_notebooks_label_feature_sensitivity_claim_boundary():
    root = Path(__file__).resolve().parents[1] / "notebooks"
    paths = sorted(root.rglob("07_counterfactuals/*.ipynb"))
    assert paths

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload["cells"]
            if cell.get("cell_type") == "markdown"
        )
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload["cells"]
            if cell.get("cell_type") == "code"
        )
        assert "predictor-feature sensitivity only" in markdown, path
        assert "does not identify a biological intervention" in markdown, path
        assert "run_feature_sensitivity" in code, path
        assert "run_counterfactual_perturbation" not in code, path
        assert "biological_intervention_identified" in code, path


def test_v2a_manifold_notebooks_load_supported_pca_profile():
    root = V2A_REFERENCE / "05_manifolds"
    for method in ("cgc", "cgc_star", "correlation_partial"):
        payload = json.loads((root / f"{method}.ipynb").read_text(encoding="utf-8"))
        text = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload["cells"]
            if cell.get("cell_type") == "code"
        )
        assert 'MANIFOLD_CONFIG_NAME = "classical"' in text
        assert 'MANIFOLD_BEHAVIOR_KEY = "continuous"' in text
        assert "OmegaConf.load(MANIFOLD_CONFIG_PATH)" in text


def test_bunddle_notebook_manifold_uses_target_window_indexing(
    monkeypatch, tmp_path, synthetic_recording, windows
):
    target_length = 15
    connectivity = SimpleNamespace(
        anchors=[],
        n_windows=2,
        window_starts=np.array(
            [0, synthetic_recording.n_timepoints - 80],
            dtype=np.int64,
        ),
        provenance=windows.provenance,
    )

    class FakeBunddleEmbedder:
        def fit(self, neural, _behavior, *, progress_callback=None):
            del neural, progress_callback
            return self

        def transform(self, neural):
            return np.zeros(
                (neural.shape[0] - target_length + 1, 3),
                dtype=np.float32,
            )

        def transform_targets(self, _neural, target_slices, _behavior):
            return np.zeros((len(target_slices), 3), dtype=np.float32)

        def save(self, path):
            return path

    class ImmediateRun:
        run_id = "reference"
        run_dir = tmp_path

        def __init__(self, method):
            self.method = method

        def execute(self, _stage, compute, **_kwargs):
            return compute()

    monkeypatch.setattr(shared, "ManifoldFactory", lambda _cfg: FakeBunddleEmbedder())
    monkeypatch.setattr(shared, "_checkpoint_input", lambda *_args, **_kwargs: synthetic_recording)
    monkeypatch.setattr(
        shared,
        "require_stage",
        lambda _run, _stage: (connectivity, tmp_path / "connectivity.pkl"),
    )

    for method in ("cgc", "cgc_star", "correlation_partial"):
        artifact = shared.run_manifold(
            ImmediateRun(method),
            recording_path=tmp_path / "recording.pkl",
            manifold_cfg=ManifoldConfig(
                name="bunddle",
                n_dims=3,
                behavior_key="motif",
                target_length=target_length,
                target_seconds=None,
                cross_fit=False,
            ),
            linking_cfg={},
            history_length=80,
        )

        assert artifact.full_embedding.shape[0] == synthetic_recording.n_timepoints - 14
        assert artifact.metadata["full_embedding_indexing"] == "target_window_end"
        assert artifact.metadata["full_embedding_sample_offset"] == 14


def test_effectome_notebooks_load_canonical_connectivity_profiles():
    root = V2A_REFERENCE / "01_effectomes"
    expected = {
        "cgc": "cgc.yaml",
        "cgc_star": "cgc_star.yaml",
        "correlation_partial": "correlation.yaml",
        "sparse_var": "sparse_var.yaml",
        "windowed_sparse_var": "windowed_sparse_var.yaml",
        "switching_var": "switching_var.yaml",
        "tvvar_smooth": "tvvar_smooth.yaml",
        "tvvar_fused": "tvvar_fused.yaml",
    }
    for method, config_name in expected.items():
        payload = json.loads((root / f"{method}.ipynb").read_text(encoding="utf-8"))
        text = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload["cells"]
            if cell.get("cell_type") == "code"
        )
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload["cells"]
            if cell.get("cell_type") == "markdown"
        )
        assert f'/ "{config_name}"' in text
        assert "ConnectivityConfig(**CONNECTIVITY_YAML)" in text
        if method in {"cgc", "cgc_star"}:
            assert "CHUNK_SIZE = 16" in text
            assert "chunk_size=CHUNK_SIZE" in text
            assert "`shared/windows` checkpoint must exist" in markdown


def test_attribution_helper_uses_window_aligned_behavior(monkeypatch, tmp_path):
    window_behavior = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    connectivity = SimpleNamespace(
        directed=True,
        behavior_per_window={"motif": window_behavior},
    )
    artifacts = {
        "outer_owned_features": SimpleNamespace(
            connectivity=connectivity, community=object(), manifold=object(),
            fit_indices=np.arange(3), evaluation_indices=np.arange(3, 5),
            fit_scope_id="outer-0",
        ),
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
        fit_indices=np.arange(3),
        evaluation_indices=np.arange(3, 5),
        fit_scope_id="outer-0",
    )

    assert result == "candidate-result"
    assert np.array_equal(observed["behavior"], window_behavior)


def test_method_notebooks_reference_shared_artifacts_under_dataset_output_root():
    notebook = V2A_REFERENCE / "01_effectomes" / "cgc.ipynb"
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


def test_notebook_helpers_run_resumable_method_lane(monkeypatch, tmp_path, windows):
    small = _subset_windows(windows, n_windows=6)
    windows_path = save_artifact(small, tmp_path / "artifacts" / "windows.pkl")
    run = make_run(tmp_path / "runs", "demo", "cgc")
    progress_events: list[tuple[int, int, bool]] = []

    class ProgressRecorder:
        def __init__(self, _description, _unit):
            pass

        def __call__(self, completed, total, reused=False):
            progress_events.append((completed, total, reused))

        def close(self):
            pass

    monkeypatch.setattr(shared, "_ProgressReporter", ProgressRecorder)

    connectivity_cfg = ConnectivityConfig(
        name="cgc",
        max_lag=1,
        extra={"support_test": "analytic", "n_perm": 0, "n_lags": 1, "seed": 0},
    )
    connectivity = run_connectivity(
        run,
        windows_path=windows_path,
        connectivity_cfg=connectivity_cfg,
        chunk_size=3,
    )
    resumed = run_connectivity(
        run,
        windows_path=windows_path,
        connectivity_cfg=connectivity_cfg,
        chunk_size=3,
    )
    assert connectivity.n_windows == 6
    assert resumed.n_windows == 6
    assert progress_events == [
        (3, 6, False),
        (6, 6, False),
        (3, 6, True),
        (6, 6, True),
    ]
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
