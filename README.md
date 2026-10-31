# effectome

**From dynamic effectomes to behavior.** Infer time-varying signed, directed, weighted effective
interactions from C. elegans and larval-zebrafish calcium imaging, model recurrent whole-effectome
states, detect evolving neural communities, and test whether their reconfiguration precedes and
predicts later manifold or behavioral change. The implemented perturbation-stage output is
predictor-feature sensitivity, not a biological intervention or a generative counterfactual.

The revised implementation contract is [Appendix C of the review](../report.md), with current
scientific status in [revision notes](../manuscript-260926/REVISION_NOTES.md). The parent
[`goal.md`](../goal.md) preserves the earlier research protocol. See
[`ANALYSIS_RUNS.md`](ANALYSIS_RUNS.md) for executable preflight, benchmark and analysis commands.

## Analysis Profiles

The revised primary analysis is the outer-fold-owned M0 activity versus M1 effectome comparison.
The CLI starts from a complete raw-data manifest; each dataset notebook starts from the
provenance-validated, identity-preprocessed window checkpoint created by its paired preprocessing
notebook. Neither path requires graph states, communities, or a manifold.
The historical stitched workflow remains available under the explicit `legacy_descriptive`
evaluation profile and must not be interpreted as confirmatory prospective evidence.

Connectivity notebook lanes cover fastC-GC, fastC-GC* (configuration names `cgc`, `cgc_star`), partial correlation, constant and windowed sparse
VAR, switching VAR, and smooth/fused TVVAR. Notebook stages are checkpointed; sequence estimators
checkpoint the complete sequence, while the primary evaluator resumes at outer-fold boundaries.

## Pipeline

Primary: raw recordings → training-owned graph features → M0/M1 prediction →
held-out evaluation. This comparison does not require states, groups, or a manifold.

Optional analyses include nested graph-summary selection and frozen transitions;
neuronal roles or structural communities; descriptive HMM/manifold analyses; and
training-owned feature sensitivity. Follow [exec_order.md](exec_order.md) for
notebook dependencies and run order.

Each stage is a registry-backed module under `src/effectome/`, configured via a Hydra config
group under `conf/`, and exposed as a numbered entry point under `pipeline/`.

## Install

```bash
uv sync --python 3.14 --locked --extra dev --extra csl --extra community --extra notebooks
source .venv/bin/activate
# Optional heavier backends (requires a supported PyTorch platform):
uv sync --python 3.14 --locked --extra full --extra dev --extra notebooks
```

The fastC-GC package is installed into `.venv` from an immutable source archive
at commit `985de54bf3d2474a5e692ad1d236c74f8452e197` (upstream package version
`0.1.0`). It requires Python 3.14. The archive and dependency hashes are recorded
in `uv.lock`; no repository clone, editable upstream checkout, or `sys.path`
injection is used. Upstream currently has no published release or PyPI package,
so this is a reproducible source pin, not a stable-release designation.

The default linear backend uses standardized ridge residuals and complete
FFT circular-shift enumeration. `support_ridge: 1e-6` is separate from the
signed VAR readout ridge. Production configs use `alpha: 0.05`, `beta: 0.005`;
positive thresholds below `1/(retained lag-design rows)` are rejected. Low-level
`ConnectivityConfig` calls without `extra.beta` use `alpha` for both thresholds
in fast mode; explicit legacy modes retain their original threshold defaults.
Explicit `support_test: analytic` and `support_test: circular_shift` retain the
legacy implementation for sensitivity comparisons. Existing `cgc` / `cgc_star`
configuration keys and notebook directories remain supported. Outputs identify
the actual backend and installed version; previous checkpoints are invalidated
by the source and lock fingerprints.

### Installed analysis backends

Tigramite supplies PCMCI+ (`connectivity=pcmci`) and JPCMCI+
(`connectivity=jpcmci`). Both call the installed upstream algorithms, with
positive-lag MCI statistics as graph weights. JPCMCI+ defaults to system-node
labels; explicitly classify context/dummy variables and scope pooled recordings
when using its context-aware analysis.

Leiden uses installed `igraph` and `leidenalg` and reports an error if those
packages are missing. Directed input retains both matrix triangles; RB modularity
requires nonnegative weights (the default uses absolute weights). K-means,
spectral clustering and PCA use scikit-learn. Distance-based k-medoids uses the
published `kmedoids` alternating solver with seeded initial medoids.

All 173 notebooks identify Python 3.14 and check the installed packages. Their
configuration keys and directory names remain compatible; displayed method names
identify fastC-GC/fastC-GC*. Existing stored outputs were preserved and need a new
run to reflect the upgraded backend. See [the backend audit](BACKEND_PACKAGE_AUDIT.md)
and [notebook readiness](NOTEBOOK_READINESS.md) for the PCA default and the
supplied V2a recordings' insufficient-data state results.

The temporal signed-role model, boundary handling, and fold-specific assignment
remain project-specific procedures built around these packages. The optional
Markov-stability entry remains explicitly labeled as a NetworkX proxy; the
optional graph distances and diagonal HMM retain their existing implementations.

## Quickstart (synthetic data — no real dataset needed)

```bash
# Run the revised outer-owned M0/M1 evaluation from a complete raw-data manifest.
python pipeline/08_evaluate.py \
  paths.evaluation_manifest=conf/evaluation-manifests/synthetic-smoke.yaml

# Exercise the five predictive adapters and benchmark report on one small raw recording.
python pipeline/09_validate.py --profile smoke --run --output-dir outputs/validation/smoke

# Optional descriptive feature stages. These do not establish prospective ownership.
python pipeline/00_preprocess.py data=synthetic
python pipeline/01_connectivity.py connectivity=cgc
python pipeline/02_dynamics.py
python pipeline/03_community.py    community=leiden
python pipeline/04_manifold.py     manifold=classical
python pipeline/05_linking.py
```

Each script writes typed artifacts under `outputs/` (gitignored) that the next stage consumes.

The default temporal profile is duration-first: 120 seconds of history, a
5.164-second manifold target/reporting cadence, and a 0.7-second c-GC lag
horizon. Each recording's `fps` resolves these to samples with rounding
provenance. Use `windowing=reference_500_15_15` only for the explicit sample
reference; named 90/180-second and fast/slow-cadence sensitivity profiles live
under `conf/windowing/`.

Stages 6/7 and the attribution notebooks construct a separate chronological feature bundle
from identity-preprocessed raw windows. They purge overlapping training histories, fit scale,
PCA and role parameters on training data, and freeze candidate/control selection under one
`fit_scope_id`. No passing candidate yields an explicit unavailable result. Their outputs
remain development feature sensitivity. The optional coupling-aware generative intervention
model is not implemented.

Primary evaluation supports development folds and a frozen-development confirmatory cohort
with independent evaluation animals, explicitly aligned neuron panels, registered sign-flip
inference and complete BY multiplicity accounting. Scientific registration gates remain
required. The raw-process benchmark CLI defaults to preflight; `--run` explicitly starts fits:

```bash
python pipeline/09_validate.py --profile publication
python pipeline/09_validate.py --profile smoke --run --output-dir outputs/validation/smoke
```

## Tests

```bash
pytest                 # unit + synthetic ground-truth recovery tests
ruff check . && mypy src/
```

## Design conventions

- Hydra + frozen-dataclass configs; `set_seed` everywhere for reproducibility.
- Factory + Registry per module — add a method by registering it, never by branching on strings.
- Files kept to 200–400 lines; type hints and docstrings throughout; logging not `print`.

## Layout

```
src/effectome/
├── data_module/     # loaders, preprocessing, windowing (shifted segments)
├── connectivity/    # c-GC, correlation, sparse/switching/time-varying VAR methods
├── dynamics/        # graph-state clustering + Markov/HMM transitions
├── community/       # static / multiscale / temporal community detection
├── manifold/        # CEBRA / BundDLe-Net / classical behavioral manifolds
├── linking/         # alignment, decoding, lead-lag, behavior↔community stats
├── attribution/     # signed neuron/group roles and preliminary candidate screening
├── intervention/    # validated-surrogate gates and predictor-feature sensitivity
├── viz/             # plotting helpers
└── utils/           # seed, io, synthetic ground-truth generator
```

## Citing the connectivity methods

The fast connectivity backend builds on S. A. Adedayo's
[c-GC/c-GC* methodology](https://arxiv.org/abs/2501.02672), *Re-examining Granger
Causality with Causal Bayesian Networks and Reichenbach's Principles* (2025;
revised 2026). Cite that work for the RCCP marginal/conditional decision rule
and conditioning sets, together with [fastC-GC and fastC-GC*: Efficient
RCCP-Guided Graph Estimation for Time Series](https://arxiv.org/abs/2610.37883)
(Adedayo, 2026) for the standardized ridge-residual estimator and complete-shift
FFT implementation. The fast estimators can select different graphs. This
pipeline's signed VAR readout, which retains target self-history, is a separate
step from fastC-GC's dependence scores. Bibliographic records for both papers are
included in [`../manuscript-260926/references.bib`](../manuscript-260926/references.bib).

For preprocessing and state-adequacy analyses, see Adedayo's
[BSS trace-preservation study](https://arxiv.org/abs/2608.00655) and
[conditioning-depth diagnostics](https://arxiv.org/abs/2606.01214).
The latter holds reported lags fixed while varying conditioning history; its
published evaluation uses c-GC/c-GC*, and extension to fastC-GC requires separate
calibration.

## Review-driven method changes

Community detection is retained as an optional structural analysis. Temporal
neuronal roles use one signed incoming/outgoing profile per lag; `legacy_expanded`
features are explicit. The default role candidate has one group, while the primary
evaluator leaves groups and states disabled. Installed Leiden remains available.

State summaries compare K-means with a regularized diagonal Gaussian mixture,
including constant/no-state and continuous ridge prediction baselines. Scaling,
PCA, codebooks and forecast parameters are fitted within training data. State
counts/algorithms are selected on inner data; a frozen choice is evaluated on
outer future graph histories. Entire target histories begin after origin histories
end, and raw histories/targets are purged across splits. The default PCA dimension
is capped at eight. Invalid histories and recording/gap boundaries are excluded.

`pipeline/02_dynamics.py` now writes `state_evaluation.pkl` and a readable
`state_evaluation.json`. Insufficient raw-disjoint pairs yield an explicit
unavailable result. Full-series pictures require `states.analysis_scope=descriptive`
and a descriptive graph configuration. Prospective transition parameters use
training codes only; their output has no regime-discovery p-value.

Run `pipeline/10_state_calibration.py --profile smoke` with `.venv/bin/python`
for a bounded raw-data integration check. The development grid varies history
and dwell durations; `--neurons`, `--samples`, `--lag`, `--fps`, and `--stride`
support realistic dimension/timing checks. These are computational profiles,
not evidence that latent regimes have been recovered.

The neural-dynamics graph-selection objective remains the primary default. For
the separate endpoint-directed sensitivity, set
`evaluation.graph_selection_objective: endpoint_prediction` inside a copied raw
manifest. A runnable synthetic example is
`conf/evaluation-manifests/synthetic-endpoint-sensitivity.yaml`; both M0 and M1
use the new purged tuning boundaries in this sensitivity.

See [RECOMMENDATIONS_IMPLEMENTATION.md](RECOMMENDATIONS_IMPLEMENTATION.md) for
implementation scope, verification, and remaining scientific requirements.
