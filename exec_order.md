# Experiment execution order

Run the current synthetic benchmark before the C. elegans motif rollout.
See the [synthetic development archive](synthetic/archive/README.md) for
the original four-CSL checks, manuscript simulation panels and driver studies.

## 1. Set up the application and encoder runtimes

From the application repository on `dev`:

```bash
uv sync --python 3.14 --locked --extra notebooks
uv sync --project tools/behavior-encoder --python 3.12 --locked
```

Choose the project Python 3.14 `.venv` notebook kernel.

## 2. Run the synthetic support comparison

Follow [00 simulator and sanity, 01 benchmark, and 02 validation](synthetic/README.md).
Notebook 01 defaults to 20 seeds and 60 cases. Use the same profile and run ID
in notebooks 01 and 02. The validation stage regenerates the paired decisions
and exports forecast plots, signed simulation-effect calibration and an
interactive view of saved interventions.

## 3. Run the C. elegans motif rollout

Use the [C. elegans notebooks](notebooks/README.md#c-elegans-motif-rollout)
in their numbered order. Notebook 00 verifies the NoStim files and freezes a
chronological plan for each selected worm. Notebook 01 fits and forecasts
motif probabilities. Notebook 02 scores held-out forecasts and can compute
paired model-based disconnections. The default processes all five worms.
See the [adaptation and provenance notes](docs/celegans_rollout.md).

## 4. Prepare each V2a recording independently

The recording roots are `notebooks/v2a_rsns/220119_F2_run11/` and
`notebooks/v2a_rsns/220127_F4_run2/`.

Run `00_preprocessing/preprocess.ipynb` in each recording. Retain its
`raw_recording` checkpoint under the shared run namespace; the canonical
comparison rebuilds identity-preprocessed windows from that raw artifact and
disables descriptive per-window scaling. Preserve acquisition gaps, invalid
intervals and the original neuron IDs.

## 5. Run the V2a shared comparison

Run `09_cross_method/shared_transitions.ipynb` for the same recording.
Its only biological prerequisite is the raw preprocessing checkpoint; individual
effectome, state, community, manifold and linking notebooks are not prerequisites.

The notebook fits one training-owned manifold and frozen region vocabulary,
builds one strictly future event table, and currently retains the historical
eight-estimator recording configuration on
identical held-out origins. It reports native graph-state transitions separately
from observed and predicted shared manifold transitions. It also supplies signed
features, neuron/group omission scores, matched contrasts/controls, and
estimator-family sensitivity.

Set the raw preprocessing run ID, comparison settings and optional neuron groups
before execution. Keep `FORCE=False` for content-checked resume. Nonfinite raw
histories or insufficient purged support produce explicit errors; do not weaken
purging or invent missing-data handling to obtain a result.

## 6. Optional V2a analyses

| Notebook/stage | Purpose and prerequisites |
|---|---|
| `08_evaluation/primary_m0_m1.ipynb` | Complementary behavior-endpoint comparison after preprocessing |
| `01_effectomes/<method>.ipynb` | Individual-estimator inspection after preprocessing |
| `02_clustering/<method>.ipynb` | Native graph codes after that method's `01` |
| `03_probabilistic_states/<method>.ipynb` | Optional descriptive HMM after `01` |
| `04_communities/<method>.ipynb` | Optional neuronal roles/structure after `01` |
| `05_manifolds/<method>.ipynb` | Optional method-specific charts after `01`; not the shared reference |
| `06_linking/<method>.ipynb` | Historical stitched analysis after `02`, `04`, `05`; not the canonical comparison |
| `07_counterfactuals/<method>.ipynb` | Separate feature sensitivity after preprocessing; no biological intervention claim |

Partial correlation remains undirected and has no directional `07`. Exact BunDLe
reproduction is optional and needs a supported PyTorch platform. Structural
Leiden and temporal roles answer different questions. Endpoint-directed graph
selection remains a separately declared sensitivity in copied raw manifests.

## 7. Candidate replication and manuscript reporting

Inspect calibration, effect sizes, controls, graph/state uncertainty, time-window
sensitivity and availability before selecting candidate neurons or groups.
Simulation seeds support simulation replication; methods and overlapping windows
do not supply independent biological samples. Independent animals, verified
behavior endpoints and intervention validation remain separate scientific gates.

Different recordings may run independently. Never run two writers against the
same recording/run/stage namespace. CLI and notebook artifact layouts differ.
Historical stored outputs retain their original scope; restart and rerun before
interpreting them under the revised framework.
