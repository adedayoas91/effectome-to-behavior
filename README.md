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

Connectivity notebook lanes cover c-GC, c-GC*, partial correlation, constant and windowed sparse
VAR, switching VAR, and smooth/fused TVVAR. Notebook stages are checkpointed; sequence estimators
checkpoint the complete sequence, while the primary evaluator resumes at outer-fold boundaries.

## Pipeline

```
00 data + temporal anchors → 01 signed/directed/weighted effectome
→ 02 global recurrent states + boundary-aware transitions
→ 03 signed/directed temporal communities → 04 manifold + anchor alignment
→ 05 leakage-safe linking → 06 attribution → 07 predictor-feature sensitivity
```

Each stage is a registry-backed module under `src/effectome/`, configured via a Hydra config
group under `conf/`, and exposed as a numbered entry point under `pipeline/`.

## Install

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"           # core + dev
uv pip install -e ".[full,dev]"      # + PCMCI, Leiden, CEBRA, UMAP (heavier)
```

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
