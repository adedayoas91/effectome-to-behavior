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
}


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
    counterfactual_endpoint: str = "behavior"
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


def write_notebook(path: Path, cells: list[dict]) -> None:
    payload = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def notebook_preprocessing(dataset: DatasetSpec) -> list[dict]:
    behavior_keys = json.dumps(list(dataset.behavior_keys))
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
    DATA_CFG["bundle_net_reference"]["exclude_neuron_names"]
)
BUNDLE_NET_EXCLUDE_NAME_SOURCE = str(
    DATA_CFG["bundle_net_reference"]["exclude_neuron_name_source"]
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
        ),
        code_cell(
            f'''
from omegaconf import OmegaConf

from effectome.data_module import PreprocessConfig, WindowConfig
from notebooks._shared import (
    PROJECT_ROOT,
    make_run,
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
print_stage_status(run)
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
            f'endpoint="{dataset.counterfactual_endpoint}"',
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
    is_manifold_notebook = any(
        "MANIFOLD_CFG" in line for cell in payload for line in cell.get("source", [])
    )
    if dataset.bundle_net_reference and is_manifold_notebook:
        bundle_replacements = [
            (
                'RECORDING_PATH = OUTPUT_ROOT / RUN_ID / "shared" / "stages" / "recording" / "artifact.pkl"',
                'RECORDING_PATH = (\n'
                '    OUTPUT_ROOT / RUN_ID / "shared" / "stages"\n'
                '    / "bundle_net_reference_recording" / "artifact.pkl"\n'
                ')',
            ),
            ('MANIFOLD_CONFIG_NAME = "classical"', 'MANIFOLD_CONFIG_NAME = "bunddle"'),
        ]
        for cell in payload:
            source = cell.get("source", [])
            for index, line in enumerate(source):
                for old, new in bundle_replacements:
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
        ),
        code_cell(
            """
from omegaconf import OmegaConf

from effectome.connectivity import ConnectivityConfig
from notebooks._shared import (
    PROJECT_ROOT,
    make_run,
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
CHUNK_SIZE = 16

CONNECTIVITY_YAML = OmegaConf.to_container(
    OmegaConf.load(CONNECTIVITY_CONFIG_PATH), resolve=True
)
if not isinstance(CONNECTIVITY_YAML, dict):
    raise TypeError(f"Expected a mapping in {CONNECTIVITY_CONFIG_PATH}")
CONNECTIVITY_CFG = ConnectivityConfig(**CONNECTIVITY_YAML)

run = make_run(ARTIFACT_ROOT, RUN_ID, METHOD)
print_stage_status(run)
"""
            % (method, config_name)
        ),
        code_cell(
            """
connectivity = run_connectivity(
    run,
    windows_path=WINDOWS_PATH,
    connectivity_cfg=CONNECTIVITY_CFG,
    chunk_size=CHUNK_SIZE,
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
            f"# Cluster-Markov State Notebook: {method_label}",
            "",
            f"Prerequisite: the `{method}` connectivity stage must already exist for the selected `RUN_ID`.",
            "Parallel lane: yes, independent across method namespaces.",
        ),
        code_cell(
            """
from effectome.dynamics import GraphStateConfig, TransitionConfig
from notebooks._shared import PROJECT_ROOT, make_run, print_stage_status, run_graph_states_and_transitions

RUN_ID = "demo"
METHOD = "%s"
ARTIFACT_ROOT = PROJECT_ROOT / "outputs" / "notebook_runs"
FORCE = False

GRAPH_CFG = GraphStateConfig(n_states=3, metric="causal_kernel", select_k=False, seed=42)
TRANSITION_CFG = TransitionConfig(laplace=1.0, n_null=200, heldout_fraction=0.3, seed=42)

run = make_run(ARTIFACT_ROOT, RUN_ID, METHOD)
print_stage_status(run)
"""
            % method
        ),
        code_cell(
            """
states, transitions = run_graph_states_and_transitions(
    run,
    graph_cfg=GRAPH_CFG,
    transition_cfg=TRANSITION_CFG,
    force=FORCE,
)
print(states.labels.shape, states.metric, states.silhouette)
print(transitions.transition_matrix)
print(transitions.p_value)
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
            "This notebook fits the HMM and records duration diagnostics that indicate whether an HSMM is warranted later.",
        ),
        code_cell(
            """
from effectome.dynamics import ProbabilisticStateConfig
from notebooks._shared import PROJECT_ROOT, make_run, print_stage_status, run_probabilistic_states

RUN_ID = "demo"
METHOD = "%s"
ARTIFACT_ROOT = PROJECT_ROOT / "outputs" / "notebook_runs"
FORCE = False

PROBABILISTIC_CFG = ProbabilisticStateConfig(
    n_states=3,
    n_components=8,
    standardize=True,
    covariance_floor=1.0e-4,
    n_iter=50,
    tol=1.0e-4,
    seed=42,
)

run = make_run(ARTIFACT_ROOT, RUN_ID, METHOD)
print_stage_status(run)
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
            f"# Temporal Community Notebook: {method_label}",
            "",
            f"Prerequisite: the `{method}` connectivity stage must already exist for the selected `RUN_ID`.",
            "Use `mode='prospective'` if you want positive-lag community prediction later in the linking notebook.",
        ),
        code_cell(
            """
from effectome.community import CommunityConfig
from notebooks._shared import PROJECT_ROOT, make_run, print_stage_status, run_community

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
        "n_communities": 3,
        "temporal_penalty": 1.5,
        "n_runs": 4,
        "max_iter": 20,
        "mode": "prospective",
    },
)

run = make_run(ARTIFACT_ROOT, RUN_ID, METHOD)
print_stage_status(run)
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
        ),
        code_cell(
            """
from omegaconf import OmegaConf

from effectome.manifold import ManifoldConfig
from notebooks._shared import PROJECT_ROOT, make_run, print_stage_status, run_manifold

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
print_stage_status(run)
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
        ),
        code_cell(
            """
from notebooks._shared import PROJECT_ROOT, make_run, print_stage_status, run_linking

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
print_stage_status(run)
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
            f"# Attribution and Counterfactual Notebook: {method_label}",
            "",
            f"Prerequisites: `community`, `manifold`, `{method}` `connectivity`, and `linking` stages, plus `outputs/artifacts/recording.pkl`.",
            "This notebook stops at validated in-silico counterfactual perturbations and stores the surrogate and perturbation checkpoints separately.",
        ),
        code_cell(
            """
from notebooks._shared import (
    PROJECT_ROOT,
    make_run,
    print_stage_status,
    run_candidate_drivers,
    run_counterfactual_perturbation,
)

RUN_ID = "demo"
METHOD = "%s"
ARTIFACT_ROOT = PROJECT_ROOT / "outputs" / "notebook_runs"
RECORDING_PATH = PROJECT_ROOT / "outputs" / "artifacts" / "recording.pkl"
BEHAVIOR_KEY = "continuous"
FORCE = False

run = make_run(ARTIFACT_ROOT, RUN_ID, METHOD)
print_stage_status(run)
"""
            % method
        ),
        code_cell(
            """
candidates = run_candidate_drivers(
    run,
    recording_path=RECORDING_PATH,
    behavior_key=BEHAVIOR_KEY,
    lag=1,
    n_folds=5,
    embargo=4,
    seed=42,
    matched_control_percentile=95.0,
    force=FORCE,
)
top = candidates.scores[0]
print(top)
print_stage_status(run)
"""
        ),
        code_cell(
            """
perturbation = run_counterfactual_perturbation(
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
    force=FORCE,
)
print(perturbation)
print_stage_status(run)
"""
        ),
    ]


def main() -> None:
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
            counterfactual_endpoint="manifold",
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
            "c-GC",
            True,
            "cgc",
        ),
        (
            "cgc_star",
            "c-GC*",
            True,
            "cgc_star",
        ),
        (
            "correlation_partial",
            "Partial Correlation Baseline",
            False,
            "correlation",
        ),
    ]
    for dataset in datasets:
        dataset_root = ROOT / dataset.relative_directory
        preprocessing_path = Path("preprocessing") / "00_preprocess.ipynb"
        write_notebook(dataset_root / preprocessing_path, notebook_preprocessing(dataset))
        print(f"Wrote {dataset.relative_directory / preprocessing_path}")
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
