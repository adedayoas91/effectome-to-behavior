# Experiment execution order — shared-transition framework

Updated 16 November 2026. Run from the application repository on `dev` with
the project Python 3.14 kernel and the pinned Python 3.13 conventional c-GC runtime.
The canonical scientific path is four-CSL planted-graph sanity check → synthetic
framework validation → shared raw preprocessing → one shared cross-method
transition comparison per recording → candidate replication and reporting.

## 1. Environment and protocol inventory

```bash
uv sync --python 3.14 --locked --extra dev --extra csl --extra community --extra notebooks
UV_CACHE_DIR=/tmp/effectome-cgc-uv-cache uv sync \
  --project tools/conventional-cgc --python 3.13.5 --locked
.venv/bin/python pipeline/14_csl_sanity.py
.venv/bin/python pipeline/12_framework_validation.py --profile smoke
.venv/bin/python pipeline/12_framework_validation.py --profile manuscript
.venv/bin/python pipeline/13_driver_validation.py --profile smoke
.venv/bin/python pipeline/13_driver_validation.py --profile confirmatory
.venv/bin/python pipeline/15_mechanism_validation.py --profile smoke
```

Without `--run`, these commands only print the declared workload. The manuscript
matrix is large: inspect its complete cell/method counts and resources before
execution. Profiles are proposed settings, not retrospective preregistration.
The current four-method framework manuscript matrix contains 7,680 raw-process
cells and 30,720 method fits. The driver confirmation contains 140 cells and 560
method fits. The larger framework matrix remains unexecuted; the driver-study
execution status is recorded below and in its live artifact.

The scheduled-origin driver confirmation completed at
`outputs/driver-validation/csl_sanity/scheduled-v2/confirmatory/`.
It uses fresh seeds300–319/400–459 and one fixed origin per trial: sample303
to335, history256,target8,N8,PCA1 and half-trial holdout. The560 method/recording
entries contain71,680 native history fits. Check `driver-validation.json` for
actual completion and acceptance: all560 runs are numerically available and all
four methods failed the frozen gates. The finalizer archived complete evidence
under `synthetic/results/driver_confirmation_v2/` and compiled the manuscript.
The larger framework sensitivity matrix remains separate.

The subsequent development path is `pipeline/15_mechanism_validation.py` and
`synthetic/06_mechanism_driver_validation.ipynb`; see the
[cited rationale](docs/driver_methodology_revision.md). Its fully observed,
model-matched control compares known support with imported native CSL support
through learned nonlinear mechanisms and paired outgoing disconnections. It uses
`outputs/driver-mechanism-validation/`, new development seeds and separate
probability-effect metrics. It is not revised independent confirmation or a
replacement for the completed failure. Start with the smoke inventory above,
then add `--run` for all lanes or `--oracle-only --run` for the readout diagnostic.

## 2. Synthetic framework validation before biological interpretation

Read [the validation guide](synthetic/README.md). Begin with the planted controls
and native graph sanity check before fitting the larger manifold comparison:

1. `synthetic/00_ground_truth.ipynb`: designed circuit and observation controls.
2. `synthetic/01_four_csl_sanity.ipynb`: conventional c-GC/c-GC* and
   fastC-GC/fastC-GC* on identical planted-chain and independent-null recordings;
   inspect edge recovery, false links and nonempty-graph agreement.
3. `synthetic/02_shared_transition_comparison.ipynb`: inspect one shared comparison.
4. `synthetic/03_correspondence_and_controls.ipynb`: inspect graph/manifold relationships.
5. `synthetic/04_framework_validation.ipynb`: repeated-seed experiment matrix,
   uncertainty, failure denominators, figures and manuscript exports.
6. `synthetic/05_driver_recovery.ipynb`: intervention-defined driver truth,
   transition-specific candidate recovery and proposed acceptance criteria.
7. `synthetic/06_mechanism_driver_validation.ipynb`: inspect or rerun the separate
   mechanism revision, three-seed oracle control and paired 256/1024 histories.

The filenames follow this order. Notebook 06 runs independently; follow the
[fresh-machine setup and run switches](synthetic/README.md#evaluate-the-latest-changes-on-the-faster-machine)
to evaluate the latest revision without repeating the old confirmation.

Run the small native graph check first, then the current four-method repeated
framework check when its workload and interpretation are appropriate:

```bash
.venv/bin/python pipeline/14_csl_sanity.py --run
.venv/bin/python pipeline/12_framework_validation.py --profile smoke --run \
  --output-dir outputs/framework-validation/csl_sanity/smoke
```

The first command writes to `outputs/csl-sanity/smoke/`. The four variants import
upstream code and retain unsigned positive-lag support; no VAR estimator or local
coefficient readout participates. All four belong to one GC family. Agreement
must be considered alongside recovery and false links; empty agreement is not
positive-control success. A graph-recovery result does not establish
transition-driver identification.

The smoke profile exercises implementation; it does not calibrate publication
claims. The development profile expands history, horizon and observation stresses
with engineering estimator settings. Before scientific interpretation, freeze
the manuscript profile and run its independent seeds and complete sensitivity
matrix with frozen upstream estimator configurations. `--max-cells` records partial
execution, including all unexecuted contrasts; it cannot complete the study.
Current histories are at least 256 samples for the fast beta=0.005 resolution.
Driver trials contain512 samples with cue onset288 and offset480; scheduled
origins end303 and future targets end335, a3.2-second horizon. These
settings were developed before fresh confirmation. Opposite rotations in the
matched-marginal control have identical support, so unsigned graphs cannot
identify their sign changes. A support-changing matched-marginal control is
needed before claiming this panel discovers circuit/manifold degeneracy.

Outputs include `validation.json`, seed-level `run_metrics.csv`, a strict
`analysis/` bundle, vector figures and generated LaTeX. Use
`--manuscript-dir ../manuscript-260926` to copy provenance-linked artifacts into
the active manuscript. Negative, failed and unavailable results remain visible.

The older `pipeline/09_validate.py` and `pipeline/10_state_calibration.py` remain
complementary estimator/state qualification tools.

Before claiming reliable transition-driver identification, follow
[the driver-ground-truth protocol](docs/driver_ground_truth_protocol.md).
Inventory `pipeline/13_driver_validation.py` first, then execute a bounded smoke
pilot with `--run`. Compare actual simulator outgoing-disconnection effects
against predictive omission scores. The earlier circuit controls alone do not
provide this causal recovery target.

The proposed confirmation uses independent seeds disjoint from the pilot,
requires at least 20 positive-control and 60 null seeds per method, and retains
failed fits and insufficient event/semantic-alignment support. Do not weaken
the gates after seeing confirmation outcomes. `--max-cells` and method subsets
remain explicit partial evaluations; a small pilot stays inconclusive. The
hidden-confounder challenge reports a failure mode rather than proving causal
identifiability. Inspect runtime and support before starting the larger study.
Current driver output defaults to `outputs/driver-validation/csl_sanity/scheduled-v2/<profile>/`;
framework output defaults to `outputs/framework-validation/csl_sanity/<profile>/`.
VAR-estimator simulation results and their manuscript figures/tables have been
removed from the current presentation. Raw output archives remain available for
audit; current synthetic notebooks neither load nor export them. VAR backends
and real-data notebooks remain unchanged.

## 3. Prepare each biological recording independently

Recording roots are `notebooks/c_elegans/`,
`notebooks/v2a_rsns/220119_F2_run11/`, and
`notebooks/v2a_rsns/220127_F4_run2/`.

Run `00_preprocessing/preprocess.ipynb` in each recording. Retain its
`raw_recording` checkpoint under the shared run namespace; the canonical
comparison rebuilds identity-preprocessed windows from that raw artifact and
disables descriptive per-window scaling. Preserve acquisition gaps, invalid
intervals and the original neuron IDs.

## 4. Run the canonical shared comparison

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

## 5. Optional complementary analyses

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

## 6. Candidate replication and manuscript reporting

Inspect calibration, effect sizes, controls, graph/state uncertainty, time-window
sensitivity and availability before selecting candidate neurons or groups.
Simulation seeds support simulation replication; methods and overlapping windows
do not supply independent biological samples. Independent animals, verified
behavior endpoints and intervention validation remain separate scientific gates.

Different recordings may run independently. Never run two writers against the
same recording/run/stage namespace. CLI and notebook artifact layouts differ.
Historical stored outputs retain their original scope; restart and rerun before
interpreting them under the revised framework.
