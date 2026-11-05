"""Generate method-split resumable analysis notebooks."""

# ruff: noqa: E501, UP031 -- template strings are kept verbatim for readable notebook cells.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent

METHOD_DIRECTORIES = {
    "cgc": "c-GC",
    "cgc_star": "c-GC-star",
    "correlation_partial": "partial-correlation",
    "sparse_var": "sparse-VAR",
    "windowed_sparse_var": "windowed-sparse-VAR",
    "switching_var": "switching-VAR",
    "tvvar_smooth": "TVVAR-smooth",
    "tvvar_fused": "TVVAR-fused",
}

ENVIRONMENT_NOTE = (
    "Environment: select the project Python 3.14 `.venv` kernel. Install the locked packages "
    "with `uv sync --python 3.14 --locked --extra dev --extra csl --extra community --extra notebooks` "
    "from the repository root. The setup cell checks installed package versions; no upstream "
    "repository clone is required. Stored outputs may predate the backend migration; rerun "
    "the notebook to produce results with the current backend."
)

PROGRESS_NOTE = (
    "Progress: long stages print checkpoint-aware timing; iterative stages also display live "
    "progress bars. Reused, failed, running, and missing checkpoints are reported explicitly."
)

BACKEND_ENVIRONMENT_CHECK = '''
if sys.version_info[:2] != (3, 14):
    raise RuntimeError(
        "Select the project Python 3.14 .venv kernel before running this notebook. "
        f"Current interpreter: {sys.executable}"
    )
_BACKEND_DISTRIBUTIONS = ("fastcgc", "tigramite", "scikit-learn", "kmedoids", "igraph", "leidenalg")
try:
    _BACKEND_VERSIONS = {name: version(name) for name in _BACKEND_DISTRIBUTIONS}
except PackageNotFoundError as error:
    raise RuntimeError(
        "The notebook kernel is missing a required installed backend. Run "
        "uv sync --python 3.14 --locked --extra dev --extra csl --extra community --extra notebooks "
        "from the repository root, then restart the project .venv kernel."
    ) from error
print("Notebook interpreter:", sys.executable)
print("Installed backends:", _BACKEND_VERSIONS)
'''

REPOSITORY_BOOTSTRAP = '''
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    _REPOSITORY_ROOT = next(
        path
        for path in (Path.cwd(), *Path.cwd().parents)
        if (path / "notebooks" / "_shared.py").is_file()
    )
except StopIteration as error:
    raise RuntimeError(
        "Could not locate the effectome repository from the notebook working directory"
    ) from error
sys.path[:0] = [
    path
    for path in (str(_REPOSITORY_ROOT / "src"), str(_REPOSITORY_ROOT))
    if path not in sys.path
]
''' + BACKEND_ENVIRONMENT_CHECK


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset-specific notebook namespace and behavioral contract."""

    slug: str
    label: str
    config_name: str
    relative_directory: Path
    dataset_id: str
    recording_id: str
    behavior_keys: tuple[str, ...]
    primary_behavior: str = "continuous"
    endpoint_type: str = "continuous"
    feature_sensitivity_endpoint: str = "behavior"
    bundle_net_reference: bool = False


def markdown_cell(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line if line.endswith("\n") else f"{line}\n" for line in lines],
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line if line.endswith("\n") else f"{line}\n" for line in source.strip("\n").splitlines()],
    }


def with_repository_bootstrap(cells: list[dict]) -> list[dict]:
    """Insert path initialization immediately before a notebook's first code cell."""
    first_code = next(
        (index for index, cell in enumerate(cells) if cell["cell_type"] == "code"),
        len(cells),
    )
    cells = json.loads(json.dumps(cells))
    first_markdown = next((cell for cell in cells if cell["cell_type"] == "markdown"), None)
    if first_markdown is not None:
        first_markdown["source"].extend(["\n", f"{ENVIRONMENT_NOTE}\n"])
    return [*cells[:first_code], code_cell(REPOSITORY_BOOTSTRAP), *cells[first_code:]]


def write_notebook(path: Path, cells: list[dict]) -> None:
    payload = {
        "cells": with_repository_bootstrap(cells),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3.14 (effectome .venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.14",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def notebook_preprocessing(dataset: DatasetSpec) -> list[dict]:
    behavior_keys = json.dumps(list(dataset.behavior_keys))
    expected_stages = ["raw_recording", "recording", "windows"]
    if dataset.bundle_net_reference:
        expected_stages.extend(["bundle_net_reference_recording", "bundle_net_training_pairs"])
    expected_stage_literal = "[\n        " + ",\n        ".join(
        json.dumps(stage) for stage in expected_stages
    ) + "\n    ]"
    reference_import = (
        "    run_bundle_net_reference_preprocessing,\n" if dataset.bundle_net_reference else ""
    )
    reference_setup = (
        '''
BUNDLE_NET_REFERENCE_CFG = PreprocessConfig(
    **DATA_CFG["bundle_net_reference"]["preprocessing"]
)
BUNDLE_NET_TARGET_LENGTH = int(
    DATA_CFG["bundle_net_reference"]["paired_window_length"]
)
BUNDLE_NET_EXCLUDE_NEURONS = list(
    DATA_CFG["exclude_neuron_names"]
)
BUNDLE_NET_EXCLUDE_NAME_SOURCE = str(
    DATA_CFG["exclude_neuron_name_source"]
)
'''
        if dataset.bundle_net_reference
        else ""
    )
    cells = [
        markdown_cell(
            f"# {dataset.label}: Load, Preprocess, and Window",
            "",
            "This dataset-level notebook creates the immutable recording and temporal-window checkpoints shared by every method lane.",
            "Confirm the raw-data path, array keys, sampling rate, behavior mapping, valid ranges, and gaps in the named dataset config before running.",
            PROGRESS_NOTE,
        ),
        code_cell(
            f'''
from omegaconf import OmegaConf

from effectome.data_module import PreprocessConfig, WindowConfig
from notebooks._shared import (
    PROJECT_ROOT,
    make_run,
    print_notebook_header,
    print_stage_status,
{reference_import}    run_preprocessing,
    stage_artifact_path,
)

DATASET_ID = "{dataset.slug}"
DATASET_OUTPUT_ID = "{dataset.dataset_id}"
RECORDING_ID = "{dataset.recording_id}"
RUN_ID = "reference"
METHOD = "shared"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "analysis" / DATASET_OUTPUT_ID / RECORDING_ID
DATA_CONFIG_PATH = PROJECT_ROOT / "conf" / "data" / "{dataset.config_name}.yaml"
WINDOW_CONFIG_PATH = PROJECT_ROOT / "conf" / "windowing" / "primary_physical.yaml"
FORCE = False

DATA_CFG = OmegaConf.to_container(OmegaConf.load(DATA_CONFIG_PATH), resolve=True)
if not isinstance(DATA_CFG, dict):
    raise TypeError(f"Expected a mapping in {{DATA_CONFIG_PATH}}")
DATA_CFG["path"] = str((PROJECT_ROOT / str(DATA_CFG["path"])).resolve())
WINDOWING_CFG = OmegaConf.to_container(OmegaConf.load(WINDOW_CONFIG_PATH), resolve=True)
if not isinstance(WINDOWING_CFG, dict):
    raise TypeError(f"Expected a mapping in {{WINDOW_CONFIG_PATH}}")
if str(DATA_CFG.get("dataset_id", "")) != DATASET_OUTPUT_ID:
    raise ValueError(
        f"Notebook output dataset id {{DATASET_OUTPUT_ID}} does not match config dataset id "
        f"{{DATA_CFG.get('dataset_id')}}"
    )
if str(DATA_CFG.get("recording_id", "")) != RECORDING_ID:
    raise ValueError(
        f"Notebook output recording id {{RECORDING_ID}} does not match config recording id "
        f"{{DATA_CFG.get('recording_id')}}"
    )

PREPROCESS_CFG = PreprocessConfig(
    detrend=False,
    zscore=False,
    deconvolve=False,
    smooth_window=0,
    drop_low_variance=0.0,
)
{reference_setup}
WINDOW_CFG = WindowConfig(**WINDOWING_CFG)
EXPECTED_BEHAVIOR_KEYS = {behavior_keys}

run = make_run(OUTPUT_ROOT, RUN_ID, METHOD)
print_notebook_header(
    run,
    title=f"{{DATASET_ID}} preprocessing",
    expected_stages={expected_stage_literal},
    force=FORCE,
)
'''
        ),
        code_cell(
            '''
recording, windows = run_preprocessing(
    run,
    data_cfg=DATA_CFG,
    preprocess_cfg=PREPROCESS_CFG,
    window_cfg=WINDOW_CFG,
    force=FORCE,
)
missing_behavior = sorted(set(EXPECTED_BEHAVIOR_KEYS) - set(recording.behavior))
if missing_behavior:
    raise KeyError(f"Dataset is missing configured behavior keys: {missing_behavior}")

print(f"recording: {recording.n_neurons} neurons x {recording.n_timepoints} samples")
source_neuron_count = recording.metadata.get("source_neuron_count")
if source_neuron_count is not None:
    print(
        "neuron cohort: "
        f"{source_neuron_count} source -> {recording.n_neurons} analysis; "
        f"excluded={recording.metadata.get('excluded_neuron_names', [])}"
    )
print(f"windows: {windows.n_windows}; shape={windows.segments.shape}")
timing = windows.metadata["temporal_contract"]
print(
    "duration-first timing: "
    f"history={timing['history_length']} frames/{timing['history']['resolved_seconds']:.3f}s, "
    f"target={timing['target_length']} frames/{timing['target']['resolved_seconds']:.3f}s, "
    f"stride={timing['stride_length']} frames/{timing['stride']['resolved_seconds']:.3f}s, "
    f"adjacent-context overlap={1.0 - timing['stride_length'] / timing['history_length']:.1%}"
)
print(f"recording artifact: {stage_artifact_path(run, 'recording')}")
print(f"windows artifact: {stage_artifact_path(run, 'windows')}")
print_stage_status(run)
'''
        ),
    ]
    if dataset.bundle_net_reference:
        cells.append(
            code_cell(
                '''
bundle_reference, bundle_pairs = run_bundle_net_reference_preprocessing(
    run,
    preprocess_cfg=BUNDLE_NET_REFERENCE_CFG,
    exclude_neuron_names=BUNDLE_NET_EXCLUDE_NEURONS,
    exclude_neuron_name_source=BUNDLE_NET_EXCLUDE_NAME_SOURCE,
    behavior_key="motif",
    target_length=BUNDLE_NET_TARGET_LENGTH,
    force=FORCE,
)
print(
    "Bundle-Net reference: "
    f"{bundle_reference.n_neurons} neurons x {bundle_reference.n_timepoints} samples; "
    f"paired windows={bundle_pairs.x_t.shape[0]}"
)
print(
    "reference recording artifact: "
    f"{stage_artifact_path(run, 'bundle_net_reference_recording')}"
)
print(f"training-pair artifact: {stage_artifact_path(run, 'bundle_net_training_pairs')}")
print_stage_status(run)
'''
            )
        )
    return cells


def notebook_primary_evaluation(dataset: DatasetSpec) -> list[dict]:
    return [
        markdown_cell(
            f"# {dataset.label}: Primary M0/M1 Evaluation",
            "",
            "This notebook runs the outer-owned probability comparison directly from shared temporal windows. Optional states, communities, and manifolds are not required.",
            PROGRESS_NOTE,
        ),
        code_cell(
            f'''
from omegaconf import OmegaConf

from effectome.workflows import (
    EvaluationConfig,
    paired_log_loss_delta,
    score_classification_records,
    score_regression_records,
)
from notebooks._shared import (
    PROJECT_ROOT,
    make_run,
    print_notebook_header,
    print_stage_status,
    run_primary_probability_evaluation,
    stage_artifact_path,
)

DATASET_OUTPUT_ID = "{dataset.dataset_id}"
RECORDING_ID = "{dataset.recording_id}"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "analysis" / DATASET_OUTPUT_ID / RECORDING_ID
RUN_ID = "primary"
METHOD = "primary-evaluation"
FORCE = False

shared_run = make_run(OUTPUT_ROOT, "reference", "shared")
WINDOWS_PATH = stage_artifact_path(shared_run, "windows")
EVALUATION_CONFIG_PATH = PROJECT_ROOT / "conf" / "evaluation" / "default.yaml"
evaluation_data = OmegaConf.to_container(OmegaConf.load(EVALUATION_CONFIG_PATH), resolve=True)
if not isinstance(evaluation_data, dict):
    raise TypeError(f"Expected a mapping in {{EVALUATION_CONFIG_PATH}}")
evaluation_data["endpoint_name"] = "{dataset.primary_behavior}"
evaluation_data["endpoint_type"] = "{dataset.endpoint_type}"
evaluation_cfg = EvaluationConfig(**evaluation_data)

run = make_run(OUTPUT_ROOT, RUN_ID, METHOD)
print_notebook_header(
    run,
    title="{dataset.label} primary probability evaluation",
    expected_stages=["windows_input", "primary_evaluation"],
    force=FORCE,
)
'''
        ),
        code_cell(
            '''
result = run_primary_probability_evaluation(
    run,
    windows_path=WINDOWS_PATH,
    evaluation_cfg=evaluation_cfg,
    force=FORCE,
)
records = result.predictions
score_records = (
    score_classification_records
    if evaluation_cfg.endpoint_type == "categorical"
    else score_regression_records
)
for model_name in ("M0_activity", "M1_effectome"):
    print(score_records(
        records,
        model_name=model_name,
        endpoint_name=evaluation_cfg.endpoint_name,
    ))
print(paired_log_loss_delta(
    records,
    baseline_model="M0_activity",
    candidate_model="M1_effectome",
    endpoint_name=evaluation_cfg.endpoint_name,
))
print_stage_status(run)
'''
        ),
    ]


def datasetize_cells(cells: list[dict], dataset: DatasetSpec) -> list[dict]:
    """Bind a generic method notebook to one dataset/recording output namespace."""
    payload = json.loads(json.dumps(cells))
    behavior_keys = json.dumps(list(dataset.behavior_keys))
    replacements = [
        (
            'RUN_ID = "demo"',
            (
                f'DATASET_ID = "{dataset.slug}"\n'
                f'DATASET_OUTPUT_ID = "{dataset.dataset_id}"\n'
                f'RECORDING_ID = "{dataset.recording_id}"\n'
                'RUN_ID = "reference"'
            ),
        ),
        (
            'ARTIFACT_ROOT = PROJECT_ROOT / "outputs" / "notebook_runs"',
            'OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "analysis" / DATASET_OUTPUT_ID / RECORDING_ID',
        ),
        ('make_run(ARTIFACT_ROOT, RUN_ID, METHOD)', 'make_run(OUTPUT_ROOT, RUN_ID, METHOD)'),
        (
            'WINDOWS_PATH = PROJECT_ROOT / "outputs" / "artifacts" / "windows.pkl"',
            'WINDOWS_PATH = OUTPUT_ROOT / RUN_ID / "shared" / "stages" / "windows" / "artifact.pkl"',
        ),
        (
            'RECORDING_PATH = PROJECT_ROOT / "outputs" / "artifacts" / "recording.pkl"',
            'RECORDING_PATH = OUTPUT_ROOT / RUN_ID / "shared" / "stages" / "recording" / "artifact.pkl"',
        ),
        ('["continuous", "motif"]', behavior_keys),
        (
            'behavior_key="continuous"',
            f'behavior_key="{dataset.primary_behavior}"',
        ),
        (
            'BEHAVIOR_KEY = "continuous"',
            f'BEHAVIOR_KEY = "{dataset.primary_behavior}"',
        ),
        (
            'endpoint="behavior"',
            f'endpoint="{dataset.feature_sensitivity_endpoint}"',
        ),
        (
            'outputs/notebook_runs/<RUN_ID>/<METHOD>/',
            'outputs/analysis/<DATASET_ID>/<RECORDING_ID>/<RUN_ID>/<METHOD>/',
        ),
        ('`outputs/artifacts/windows.pkl`', 'the dataset `shared/windows` checkpoint'),
        ('`outputs/artifacts/recording.pkl`', 'the dataset `shared/recording` checkpoint'),
    ]
    for cell in payload:
        source = cell.get("source", [])
        for index, line in enumerate(source):
            for old, new in replacements:
                line = line.replace(old, new)
            source[index] = line
    title_source = payload[0].get("source", [])
    if title_source:
        title_source[0] = title_source[0].replace("# ", f"# {dataset.label} — ", 1)
    return payload


def notebook_connectivity(title: str, method: str, config_name: str) -> list[dict]:
    return [
        markdown_cell(
            f"# {title}",
            "",
            "Prerequisite: `outputs/artifacts/windows.pkl` must exist.",
            "Parallel lane: yes. This notebook owns only its method namespace under `outputs/notebook_runs/<RUN_ID>/<METHOD>/`.",
            "Execution: one model is fit per window. `CHECKPOINT_CHUNK_SIZE` controls how many fits share an atomic resume checkpoint; it is not the total fit count.",
            PROGRESS_NOTE,
        ),
        code_cell(
            """
from omegaconf import OmegaConf

from effectome.connectivity import ConnectivityConfig
from notebooks._shared import (
    PROJECT_ROOT,
    make_run,
    print_notebook_header,
    print_stage_status,
    run_connectivity,
    stage_artifact_path,
)

RUN_ID = "demo"
METHOD = "%s"
ARTIFACT_ROOT = PROJECT_ROOT / "outputs" / "notebook_runs"
WINDOWS_PATH = PROJECT_ROOT / "outputs" / "artifacts" / "windows.pkl"
CONNECTIVITY_CONFIG_PATH = PROJECT_ROOT / "conf" / "connectivity" / "%s.yaml"
FORCE = False
CHECKPOINT_CHUNK_SIZE = 16

CONNECTIVITY_YAML = OmegaConf.to_container(
    OmegaConf.load(CONNECTIVITY_CONFIG_PATH), resolve=True
)
if not isinstance(CONNECTIVITY_YAML, dict):
    raise TypeError(f"Expected a mapping in {CONNECTIVITY_CONFIG_PATH}")
CONNECTIVITY_CFG = ConnectivityConfig(**CONNECTIVITY_YAML)

run = make_run(ARTIFACT_ROOT, RUN_ID, METHOD)
print_notebook_header(
    run,
    title=f"{METHOD} connectivity",
    expected_stages=["windows_input", "connectivity"],
    force=FORCE,
)
"""
            % (method, config_name)
        ),
        code_cell(
            """
connectivity = run_connectivity(
    run,
    windows_path=WINDOWS_PATH,
    connectivity_cfg=CONNECTIVITY_CFG,
    chunk_size=CHECKPOINT_CHUNK_SIZE,
    force=FORCE,
)
print(connectivity.matrices.shape)
print(connectivity.diagnostics["runtime"])
print(stage_artifact_path(run, "connectivity"))
print_stage_status(run)
"""
        ),
    ]


def notebook_states(method: str, method_label: str) -> list[dict]:
    return [
        markdown_cell(
            f"# Prospective Graph-State Comparison: {method_label}",
            "",
            f"Prerequisite: the `{method}` connectivity stage must already exist for the selected `RUN_ID`.",
            "Compare one-state and continuous baselines with K-means and regularized mixtures using inner selection and a frozen held-out evaluation.",
            "Raw-history purging separates training and future graph targets. Regimes remain unknown until raw-process calibration; Label-permutation tests do not establish neural regimes.",
            PROGRESS_NOTE,
        ),
        code_cell(
            """
from effectome.dynamics import GraphStateConfig, TransitionConfig
from effectome.dynamics.state_evaluation import StateEvaluationConfig
from notebooks._shared import (
    PROJECT_ROOT,
    make_run,
    print_notebook_header,
    print_stage_status,
    run_graph_states_and_transitions,
)

RUN_ID = "demo"
METHOD = "%s"
ARTIFACT_ROOT = PROJECT_ROOT / "outputs" / "notebook_runs"
FORCE = False

GRAPH_CFG = GraphStateConfig(
    n_states=1, feature_representation="signed_lag_offdiag", assignment="kmeans",
    metric="frobenius", pca_components=8, analysis_scope="prospective", seed=42,
)
TRANSITION_CFG = TransitionConfig(laplace=1.0, n_null=200, heldout_fraction=0.3, seed=42)

run = make_run(ARTIFACT_ROOT, RUN_ID, METHOD)
print_notebook_header(
    run,
    title=f"{METHOD} graph states and transitions",
    expected_stages=["state_evaluation", "graph_states", "transitions"],
    force=FORCE,
)
"""
            % method
        ),
        code_cell(
            """
states, transitions = run_graph_states_and_transitions(
    run,
    graph_cfg=GRAPH_CFG,
    transition_cfg=TRANSITION_CFG,
    selection_cfg=StateEvaluationConfig(graph=GRAPH_CFG),
    analysis_scope="prospective",
    on_unavailable="return",
    force=FORCE,
)
if states is None:
    print("State evaluation unavailable:", run.load_completed("state_evaluation").summary())
else:
    print(states.labels.shape, states.metric, states.silhouette)
    print(transitions.transition_matrix)
    print("Conditional label-sequence diagnostic only:", transitions.p_value)
print("Neural regimes remain unvalidated without independent raw-process calibration.")
print_stage_status(run)
"""
        ),
    ]


def notebook_probabilistic(method: str, method_label: str) -> list[dict]:
    return [
        markdown_cell(
            f"# Probabilistic State Notebook: {method_label}",
            "",
            f"Prerequisite: the `{method}` connectivity stage must already exist for the selected `RUN_ID`.",
            "Optional descriptive HMM fit, starting from one state. Duration diagnostics can motivate later models; an HMM fit does not establish neural regimes and is separate from prospective state selection.",
            PROGRESS_NOTE,
        ),
        code_cell(
            """
from effectome.dynamics import ProbabilisticStateConfig
from notebooks._shared import (
    PROJECT_ROOT,
    make_run,
    print_notebook_header,
    print_stage_status,
    run_probabilistic_states,
)

RUN_ID = "demo"
METHOD = "%s"
ARTIFACT_ROOT = PROJECT_ROOT / "outputs" / "notebook_runs"
FORCE = False

PROBABILISTIC_CFG = ProbabilisticStateConfig(
    n_states=1,
    n_components=8,
    standardize=True,
    covariance_floor=1.0e-4,
    n_iter=50,
    tol=1.0e-4,
    seed=42,
)

run = make_run(ARTIFACT_ROOT, RUN_ID, METHOD)
print_notebook_header(
    run,
    title=f"{METHOD} probabilistic states",
    expected_stages=["probabilistic_states"],
    force=FORCE,
)
"""
            % method
        ),
        code_cell(
            """
probabilistic = run_probabilistic_states(run, probabilistic_cfg=PROBABILISTIC_CFG, force=FORCE)
print(probabilistic.labels.shape)
print(probabilistic.log_likelihood)
print(probabilistic.hsmm_candidate.status)
print(probabilistic.hsmm_candidate.non_geometric_states)
print_stage_status(run)
"""
        ),
    ]


def notebook_community(method: str, method_label: str) -> list[dict]:
    return [
        markdown_cell(
            f"# Temporal Neuron Roles: {method_label}",
            "",
            f"Prerequisite: the `{method}` connectivity stage must already exist for the selected `RUN_ID`.",
            "This optional analysis groups neurons by signed incoming/outgoing roles. Structural community detection is retained separately through the installed Leiden backend; roles are not evidence of densely coupled modules.",
            "One role is the null summary. Increase the role count only with held-out evidence or an independently declared structural hypothesis.",
            PROGRESS_NOTE,
        ),
        code_cell(
            """
from effectome.community import CommunityConfig
from notebooks._shared import PROJECT_ROOT, make_run, print_notebook_header, print_stage_status, run_community

RUN_ID = "demo"
METHOD = "%s"
ARTIFACT_ROOT = PROJECT_ROOT / "outputs" / "notebook_runs"
FORCE = False

COMMUNITY_CFG = CommunityConfig(
    name="temporal",
    resolution=1.0,
    symmetrize=False,
    use_absolute=False,
    seed=42,
    extra={
        "n_communities": 1,
        "feature_representation": "signed_in_out",
        "mode": "prospective",
        "temporal_penalty": 1.5,
        "n_runs": 5,
        "max_iter": 20,
        "init_window_count": 4,
        "online_learning_rate": 0.25,
    },
)

run = make_run(ARTIFACT_ROOT, RUN_ID, METHOD)
print_notebook_header(
    run,
    title=f"{METHOD} temporal neuron roles",
    expected_stages=["community"],
    force=FORCE,
)
"""
            % method
        ),
        code_cell(
            """
community = run_community(run, community_cfg=COMMUNITY_CFG, force=FORCE)
print(community.labels.shape)
print(community.method, getattr(community, "mode", None))
print_stage_status(run)
"""
        ),
    ]


def notebook_manifold(method: str, method_label: str) -> list[dict]:
    return [
        markdown_cell(
            f"# Manifold Notebook: {method_label}",
            "",
            f"Prerequisites: `outputs/artifacts/recording.pkl` and the `{method}` connectivity stage for the selected run.",
            "Parallel lane: yes, once the method-specific connectivity stage exists.",
            PROGRESS_NOTE,
        ),
        code_cell(
            """
from omegaconf import OmegaConf

from effectome.manifold import ManifoldConfig
from notebooks._shared import PROJECT_ROOT, make_run, print_notebook_header, print_stage_status, run_manifold

RUN_ID = "demo"
METHOD = "%s"
ARTIFACT_ROOT = PROJECT_ROOT / "outputs" / "notebook_runs"
RECORDING_PATH = PROJECT_ROOT / "outputs" / "artifacts" / "recording.pkl"
MANIFOLD_CONFIG_NAME = "classical"
MANIFOLD_CONFIG_PATH = PROJECT_ROOT / "conf" / "manifold" / f"{MANIFOLD_CONFIG_NAME}.yaml"
MANIFOLD_BEHAVIOR_KEY = "continuous"
FORCE = False

MANIFOLD_YAML = OmegaConf.to_container(OmegaConf.load(MANIFOLD_CONFIG_PATH), resolve=True)
if not isinstance(MANIFOLD_YAML, dict):
    raise TypeError(f"Expected a mapping in {MANIFOLD_CONFIG_PATH}")
MANIFOLD_CFG = ManifoldConfig(
    **{**MANIFOLD_YAML, "behavior_key": MANIFOLD_BEHAVIOR_KEY}
)
LINKING_CFG = {
    "embargo": 4,
    "group_by": "recording",
    "lag_extension": 0,
    "preprocessing_past_support": 0,
    "preprocessing_future_support": 0,
}

run = make_run(ARTIFACT_ROOT, RUN_ID, METHOD)
print_notebook_header(
    run,
    title=f"{METHOD} manifold",
    expected_stages=["recording_input", "manifold"],
    force=FORCE,
)
"""
            % method
        ),
        code_cell(
            """
manifold = run_manifold(
    run,
    recording_path=RECORDING_PATH,
    manifold_cfg=MANIFOLD_CFG,
    linking_cfg=LINKING_CFG,
    force=FORCE,
)
print(manifold.window_embedding.shape)
print(manifold.metadata["window_embedding_mode"])
print_stage_status(run)
"""
        ),
    ]


def notebook_linking(method: str, method_label: str, *, directed: bool) -> list[dict]:
    baseline_note = (
        "This lane is directed, so the reported source/target readouts keep their signed directional semantics."
        if directed
        else "This lane is an undirected baseline. Use it for comparison, not for directional driver claims."
    )
    return [
        markdown_cell(
            f"# Linking Notebook: {method_label}",
            "",
            f"Prerequisites: `graph_states`, `community`, `manifold`, and `{method}` `connectivity` stages for the selected run, plus `outputs/artifacts/recording.pkl`.",
            "This notebook preserves the anchor-aware embargo and activity-covariate controls in the resumed stage artifact.",
            baseline_note,
            PROGRESS_NOTE,
        ),
        code_cell(
            """
from notebooks._shared import PROJECT_ROOT, make_run, print_notebook_header, print_stage_status, run_linking

RUN_ID = "demo"
METHOD = "%s"
ARTIFACT_ROOT = PROJECT_ROOT / "outputs" / "notebook_runs"
RECORDING_PATH = PROJECT_ROOT / "outputs" / "artifacts" / "recording.pkl"
FORCE = False

LINKING_CFG = {
    "behavior_keys": ["continuous", "motif"],
    "n_null": 1000,
    "n_bins": 5,
    "max_lag": 10,
    "positive_lag": 1,
    "n_folds": 5,
    "embargo": 4,
    "group_by": "recording",
    "lag_extension": 0,
    "preprocessing_past_support": 0,
    "preprocessing_future_support": 0,
    "allow_global_preprocessing": False,
    "null_kind": "circular_shift",
    "block_length": 8,
    "seed": 42,
}

run = make_run(ARTIFACT_ROOT, RUN_ID, METHOD)
print_notebook_header(
    run,
    title=f"{METHOD} linking",
    expected_stages=["recording_input", "linking"],
    force=FORCE,
)
"""
            % method
        ),
        code_cell(
            """
report = run_linking(
    run,
    recording_path=RECORDING_PATH,
    linking_cfg=LINKING_CFG,
    force=FORCE,
)
if report.get("status") == "unavailable":
    print("Linking unavailable:", report["reason"])
else:
    print(len(report["decoding"]))
    print(report["splitter"])
    print(report["positive_lag_incremental"].keys())
    print(report["effectome_to_future_manifold"])
print_stage_status(run)
"""
        ),
    ]


def notebook_attribution(method: str, method_label: str) -> list[dict]:
    return [
        markdown_cell(
            f"# Attribution and Feature Sensitivity Notebook: {method_label}",
            "",
            "Prerequisites: identity-preprocessed raw windows and `outputs/artifacts/recording.pkl` from preprocessing. This notebook fits a separate purged chronological outer-owned feature bundle.",
            "This notebook reports predictor-feature sensitivity only; it does not identify a biological intervention.",
            PROGRESS_NOTE,
        ),
        code_cell(
            """
from omegaconf import OmegaConf

from effectome.connectivity import ConnectivityConfig
from notebooks._shared import (
    PROJECT_ROOT,
    make_run,
    print_notebook_header,
    print_stage_status,
    run_candidate_drivers,
    run_feature_sensitivity,
    run_owned_feature_artifacts,
)

RUN_ID = "demo"
METHOD = "%s"
ARTIFACT_ROOT = PROJECT_ROOT / "outputs" / "notebook_runs"
RECORDING_PATH = PROJECT_ROOT / "outputs" / "artifacts" / "recording.pkl"
WINDOWS_PATH = PROJECT_ROOT / "outputs" / "artifacts" / "windows.pkl"
BEHAVIOR_KEY = "continuous"
FIT_SCOPE_ID = f"{RUN_ID}/{METHOD}/sensitivity-outer-0"
MULTIPLICITY_FAMILY = (f"{BEHAVIOR_KEY}:feature-edit",)
connectivity_data = OmegaConf.to_container(
    OmegaConf.load(PROJECT_ROOT / "conf" / "connectivity" / f"{METHOD}.yaml"), resolve=True
)
CONNECTIVITY_CFG = ConnectivityConfig(**connectivity_data)
FORCE = False

run = make_run(ARTIFACT_ROOT, RUN_ID, METHOD)
print_notebook_header(
    run,
    title=f"{METHOD} attribution and feature sensitivity",
    expected_stages=[
        "recording_input",
        "outer_owned_features",
        "candidate_drivers",
        "surrogate_model",
        "feature_sensitivity",
    ],
    force=FORCE,
)
"""
            % method
        ),
        code_cell(
            """
owned = run_owned_feature_artifacts(
    run, windows_path=WINDOWS_PATH, connectivity_cfg=CONNECTIVITY_CFG,
    fit_scope_id=FIT_SCOPE_ID, evaluation_fraction=0.2, force=FORCE,
)
print(f"Training windows: {len(owned.fit_indices)}, held-out windows: {len(owned.evaluation_indices)}")

candidates = run_candidate_drivers(
    run,
    recording_path=RECORDING_PATH,
    behavior_key=BEHAVIOR_KEY,
    lag=1,
    n_folds=5,
    embargo=4,
    seed=42,
    matched_control_percentile=95.0,
    fit_indices=owned.fit_indices,
    evaluation_indices=owned.evaluation_indices,
    fit_scope_id=owned.fit_scope_id,
    force=FORCE,
)
top = candidates.scores[0]
print(top)
print_stage_status(run)
"""
        ),
        code_cell(
            """
sensitivity = run_feature_sensitivity(
    run,
    recording_path=RECORDING_PATH,
    behavior_key=BEHAVIOR_KEY,
    lag=1,
    ridge_alpha=1.0,
    n_folds=5,
    embargo=4,
    min_skill=0.0,
    scale=0.0,
    dose_scales=[1.0, 0.5, 0.0],
    top_k=1,
    endpoint="behavior",
    seed=42,
    fit_indices=owned.fit_indices,
    evaluation_indices=owned.evaluation_indices,
    fit_scope_id=owned.fit_scope_id,
    multiplicity_family=MULTIPLICITY_FAMILY,
    hypothesis_id=f"{BEHAVIOR_KEY}:feature-edit",
    min_absolute_skill=0.0,
    max_distribution_shift=5.0,
    n_controls=64,
    control_resolution_policy="unavailable",
    force=FORCE,
)
print(sensitivity)
print(sensitivity.analysis_kind)
print(sensitivity.biological_intervention_identified)
print_stage_status(run)
"""
        ),
    ]


def notebook_state_calibration() -> list[dict]:
    return [
        markdown_cell(
            "# Raw State Calibration: Development Checks",
            "",
            "Regenerate raw recordings and rerun connectivity and nested state selection for fixed-law, measurement-change, drift, recurrent, nonrecurring and nonlinear scenarios.",
            "This smoke grid is a software/development check. Multi-state selection is not a validated neural-regime claim; mixed histories are excluded from instantaneous recovery scores.",
            PROGRESS_NOTE,
            "The complete bounded calibration grid is checkpointed. Individual graph fits do not display their own progress bars.",
        ),
        code_cell(
            """
from dataclasses import asdict

from effectome.experiments.state_calibration import (
    run_state_calibration,
    state_calibration_profile,
    write_state_calibration_report,
)
from notebooks._shared import PROJECT_ROOT, make_run, print_notebook_header, print_stage_status

CALIBRATION_CFG = state_calibration_profile("smoke")
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "state-calibration" / "smoke"
FORCE = False
run = make_run(PROJECT_ROOT / "outputs" / "analysis" / "validation", "smoke", "raw-state-calibration")
print_notebook_header(
    run, title="Raw state calibration", expected_stages=["state_calibration"], force=FORCE,
)
print(CALIBRATION_CFG)
"""
        ),
        code_cell(
            """
report = run.execute(
    "state_calibration", lambda: run_state_calibration(CALIBRATION_CFG),
    config=asdict(CALIBRATION_CFG), force=FORCE,
)
manifest_path, scorecard_path = write_state_calibration_report(report, OUTPUT_DIRECTORY)
print(report["summary"])
print("Manifest:", manifest_path)
print("Scorecard:", scorecard_path)
print_stage_status(run)
"""
        ),
    ]


def notebook_manifold_trajectory_visualization() -> list[dict]:
    return [
        markdown_cell(
            "# Interactive 3D Manifold Trajectory and Effectomes",
            "",
            "Explore a consecutive manifold window with rounded temporal paths and synchronized active/mean effectome panels. Start-step and segment-count sliders choose the visible range; Previous and Next move by the selected range size.",
            "The rounded paths use endpoint-preserving centripetal Catmull--Rom interpolation inspired by the BunDLe trajectory presentation. Every observed manifold point is retained. Curves stop at recording gaps, bad-frame boundaries, and cross-fit fold boundaries, so smoothing never bridges a declared discontinuity.",
            "The visualization is descriptive. A smooth display curve does not add observations, establish continuous-time dynamics, or support causal interpretation.",
            PROGRESS_NOTE,
        ),
        code_cell(
            '''
import importlib
import sys

from IPython import get_ipython

shell = get_ipython()
if shell is None:
    raise RuntimeError("Run this file as a Jupyter notebook")

# Prefer an embedded widget when ipympl is available. The project adds no new
# dependency for this visualization, so a local native window is the default
# fallback (macOS first, then Tk, then a noninteractive inline figure).
backend_candidates = ["widget"]
if sys.platform == "darwin":
    backend_candidates.append("osx")
else:
    backend_candidates.append("tk")
backend_candidates.append("inline")
backend_errors = {}
for backend_name in backend_candidates:
    try:
        shell.run_line_magic("matplotlib", backend_name)
        INTERACTIVE_BACKEND = backend_name
        break
    except (ImportError, ModuleNotFoundError, RuntimeError) as error:
        backend_errors[backend_name] = str(error)
else:
    raise RuntimeError(f"No Matplotlib notebook backend is available: {backend_errors}")

shared = importlib.import_module("notebooks._shared")
visualization = importlib.import_module("scripts.plot_manifold_trajectory")
PROJECT_ROOT = shared.PROJECT_ROOT
make_run = shared.make_run
print_notebook_header = shared.print_notebook_header
print_stage_status = shared.print_stage_status
TrajectoryExplorer = visualization.TrajectoryExplorer
load_trajectory_data = visualization.load_trajectory_data

OUTPUT_BASE = PROJECT_ROOT / "outputs"
DATASET_OUTPUT_ID = "bundle-net-c-elegans"
RECORDING_ID = "bundle-net-worm-0"
RUN_ID = "reference"
METHOD = "cgc"
COLOR_BY = "motif"  # "motif", "graph", or "hmm"
BEHAVIOR_KEY = "motif"

# Initial control values. The sliders can change all three after this cell runs.
START_STEP = 0
SEGMENTS_PER_VIEW = 25
ACTIVE_EFFECTOME_STEP = START_STEP
CURVE_SAMPLES_PER_SEGMENT = 20
ARROW_STRIDE = 1

OUTPUT_ROOT = OUTPUT_BASE / "analysis" / DATASET_OUTPUT_ID / RECORDING_ID
RUN_ROOT = OUTPUT_ROOT / RUN_ID
run = make_run(OUTPUT_ROOT, RUN_ID, METHOD)
expected = ["connectivity", "manifold"]
if COLOR_BY == "graph":
    expected.append("graph_states")
elif COLOR_BY == "hmm":
    expected.append("probabilistic_states")
print_notebook_header(
    run,
    title="Interactive 3D manifold trajectory",
    expected_stages=expected,
    force=False,
)
print("Matplotlib backend:", INTERACTIVE_BACKEND)
'''
        ),
        code_cell(
            '''
trajectory = load_trajectory_data(
    RUN_ROOT,
    method=METHOD,
    color_by=COLOR_BY,
    behavior_key=BEHAVIOR_KEY,
)
print(
    f"Loaded {trajectory.embedding.shape[0]} manifold steps, "
    f"{trajectory.effectomes.shape[1]} effectome rows, "
    f"and {trajectory.valid_steps.sum()} valid temporal joins"
)
print_stage_status(run)
'''
        ),
        markdown_cell(
            "## Explore consecutive segments",
            "",
            "Use **Start step** and **Segments** to select the consecutive trajectory window. **Previous** and **Next** move by the current segment count. **Effectome step** selects the black-diamond manifold point and its matching matrix; the lower matrix is the mean across all points in the visible window.",
            "",
            "Drag the 3D axis to rotate it and scroll to zoom. If the backend falls back to `inline`, edit the initial values above and rerun this cell; sliders need an interactive Matplotlib backend.",
        ),
        code_cell(
            '''
explorer = TrajectoryExplorer(
    trajectory,
    start_step=START_STEP,
    segment_count=SEGMENTS_PER_VIEW,
    active_step=ACTIVE_EFFECTOME_STEP,
    curve_samples=CURVE_SAMPLES_PER_SEGMENT,
    arrow_stride=ARROW_STRIDE,
)
explorer.figure
'''
        ),
        markdown_cell(
            "## Export the current selection",
            "",
            "Move the controls to the desired window before running this cell. The PNG and GIF preserve the active and mean effectome panels. Adjust frame count, rate, and DPI as needed.",
        ),
        code_cell(
            '''
EXPORT_DIRECTORY = PROJECT_ROOT / "outputs" / "analysis-output" / "manifold-3d"
start, count, active = explorer.selection
stem = f"manifold-3d-{METHOD}-{COLOR_BY}-steps-{start}-{start + count}-active-{active}"
png_path = explorer.save_png(EXPORT_DIRECTORY / f"{stem}.png", dpi=160)
gif_path = explorer.save_gif(
    EXPORT_DIRECTORY / f"{stem}.gif",
    frames=72,
    fps=12,
    dpi=110,
)
print("Saved:", png_path)
print("Saved:", gif_path)
'''
        ),
    ]


def main() -> None:
    write_notebook(ROOT / "validation" / "10_state_calibration.ipynb", notebook_state_calibration())
    write_notebook(
        ROOT / "visualization" / "11_manifold_trajectory_3d.ipynb",
        notebook_manifold_trajectory_visualization(),
    )
    datasets = [
        DatasetSpec(
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
        ),
        DatasetSpec(
            slug="220119_F2_run11",
            label="V2a-RSN 220119_F2_run11",
            config_name="220119_F2_run11",
            relative_directory=Path("v2a_rsns") / "220119_F2_run11",
            dataset_id="v2a-rsns",
            recording_id="220119_F2_run11",
            behavior_keys=("continuous",),
        ),
        DatasetSpec(
            slug="220127_F4_run2",
            label="V2a-RSN 220127_F4_run2",
            config_name="220127_F4_run2",
            relative_directory=Path("v2a_rsns") / "220127_F4_run2",
            dataset_id="v2a-rsns",
            recording_id="220127_F4_run2",
            behavior_keys=("continuous",),
        ),
    ]
    lanes = [
        (
            "cgc",
            "fastC-GC",
            True,
            "cgc",
        ),
        (
            "cgc_star",
            "fastC-GC*",
            True,
            "cgc_star",
        ),
        (
            "correlation_partial",
            "Partial Correlation Baseline",
            False,
            "correlation",
        ),
        (
            "sparse_var",
            "Sparse VAR Baseline",
            True,
            "sparse_var",
        ),
        (
            "windowed_sparse_var",
            "Windowed Sparse VAR",
            True,
            "windowed_sparse_var",
        ),
        (
            "switching_var",
            "Switching VAR",
            True,
            "switching_var",
        ),
        (
            "tvvar_smooth",
            "Smooth TVVAR",
            True,
            "tvvar_smooth",
        ),
        (
            "tvvar_fused",
            "Group-fused TVVAR",
            True,
            "tvvar_fused",
        ),
    ]
    for dataset in datasets:
        dataset_root = ROOT / dataset.relative_directory
        preprocessing_path = Path("preprocessing") / "00_preprocess.ipynb"
        write_notebook(dataset_root / preprocessing_path, notebook_preprocessing(dataset))
        print(f"Wrote {dataset.relative_directory / preprocessing_path}")
        evaluation_path = Path("evaluation") / "08_primary_m0_m1.ipynb"
        write_notebook(
            dataset_root / evaluation_path,
            notebook_primary_evaluation(dataset),
        )
        print(f"Wrote {dataset.relative_directory / evaluation_path}")
        for method, label, directed, connectivity_config_name in lanes:
            method_directory = METHOD_DIRECTORIES[method]
            notebooks = {
                Path("effectomes") / method_directory / "01_connectivity.ipynb": notebook_connectivity(
                    f"Connectivity Notebook: {label}", method, connectivity_config_name
                ),
                Path("clustering")
                / method_directory
                / "02_cluster_markov.ipynb": notebook_states(method, label),
                Path("probabilistic_states")
                / method_directory
                / "03_hmm.ipynb": notebook_probabilistic(method, label),
                Path("communities")
                / method_directory
                / "04_temporal_community.ipynb": notebook_community(method, label),
                Path("manifolds")
                / method_directory
                / "05_manifold.ipynb": notebook_manifold(method, label),
                Path("linking")
                / method_directory
                / "06_linking.ipynb": notebook_linking(method, label, directed=directed),
            }
            if directed:
                notebooks[
                    Path("counterfactuals")
                    / method_directory
                    / "07_attribution_counterfactual.ipynb"
                ] = notebook_attribution(method, label)
            for relative_path, cells in notebooks.items():
                write_notebook(dataset_root / relative_path, datasetize_cells(cells, dataset))
                print(f"Wrote {dataset.relative_directory / relative_path}")


if __name__ == "__main__":
    main()
