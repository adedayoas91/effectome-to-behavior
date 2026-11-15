# Dataset-grouped resumable analysis notebooks

The canonical adopted path is now `00_preprocessing/preprocess.ipynb` followed by
`09_cross_method/shared_transitions.ipynb` for each recording. It rebuilds raw
identity windows, fits one shared training-owned manifold, and compares all eight
methods on the same held-out events. Start with
[framework validation](../synthetic/README.md) before interpreting real recordings.
The existing method notebooks carry scope notices and remain optional individual
method analyses; they are not silently converted into shared-reference results.

The executable notebooks are grouped by independently analysed recording:

```text
notebooks/
├── c_elegans/
│   ├── 00_preprocessing/preprocess.ipynb
│   ├── 01_effectomes/<method>.ipynb
│   ├── 02_clustering/<method>.ipynb
│   ├── 03_probabilistic_states/<method>.ipynb
│   ├── 04_communities/<method>.ipynb
│   ├── 05_manifolds/<method>.ipynb
│   ├── 06_linking/<method>.ipynb
│   ├── 07_counterfactuals/<directed-method>.ipynb
│   └── 08_evaluation/primary_m0_m1.ipynb
└── v2a_rsns/
    ├── 220119_F2_run11/
    └── 220127_F4_run2/
```

Every recording repeats the numbered stage structure shown under `c_elegans`.
Method identity lives in the notebook filename, not another directory level.
The generated surface retains all eight implemented notebook lanes: fastC-GC,
fastC-GC*, partial correlation, constant and windowed sparse VAR, switching VAR,
and smooth/fused TVVAR. The numbered stage directory supplies the run order; the
method filename supplies the estimator identity. PCMCI+ and JPCMCI+ remain
registry-backed causal-structure-learning configurations, with JPCMCI+ reserved
for explicitly scoped pooled/context-aware workflows. V2a-RSN datasets 03 and 04
are intentionally not generated yet; add their source recording names to
`_build_notebooks.py` after their paths and schemas have been confirmed.

## Output layout

Every useful stage artifact is stored beneath the dataset and recording that
produced it:

```text
outputs/analysis/<dataset_id>/<recording_id>/<run_id>/
├── shared/
│   └── stages/{raw_recording,recording,windows,...}/
├── cgc/
│   └── stages/{windows_input,connectivity,connectivity.chunk-*,graph_states,...}/
├── cgc_star/
│   └── stages/{windows_input,connectivity,connectivity.chunk-*,graph_states,...}/
└── correlation_partial/
    └── stages/{windows_input,connectivity,connectivity.chunk-*,graph_states,...}/
```

Each stage directory contains `artifact.pkl` and `status.json`. The status file
records the state, timestamps, configuration and input signature; the artifact
contains the full typed analysis result, including anchors and provenance. Only
load pickle artifacts created by this trusted local project.

### Interactive 3D manifold trajectories

The tracked [`11_visualization/manifold_trajectory_3d.ipynb`](11_visualization/manifold_trajectory_3d.ipynb)
and `scripts/plot_manifold_trajectory.py` helper turn a completed three-dimensional
manifold checkpoint into a draggable Matplotlib view, a static PNG, and a rotating
GIF. The notebook provides start-step, segment-count, and active-effectome sliders
plus Previous/Next buttons. It shows the active effectome and the mean effectome
for the selected consecutive window beside the trajectory.

Temporal joins use endpoint-preserving centripetal Catmull--Rom curves rather
than straight chords. The curves pass through every observed manifold point,
with tangent arrows showing direction. They stop across recording gaps,
bad-frame boundaries, and cross-fit fold boundaries. Black rings mark behavior
or state changes.

From the package root, generate motif, graph-state, and HMM-state views with:

```bash
uv run python scripts/plot_manifold_trajectory.py \
  --run-root outputs/analysis/bundle-net-c-elegans/bundle-net-worm-0/reference \
  --method cgc \
  --color-by motif graph hmm \
  --show
```

Drag a native plot window to rotate it and scroll to zoom. The files are written beneath
`outputs/analysis-output/manifold-3d/`. Use `--arrow-stride N` only when showing
every arrow is visually too dense; the default is one arrow per step. GIF speed
and smoothness are controlled by `--frames`, `--fps`, and `--dpi`. The smoothing
is visual interpolation only; it does not create additional observations or
establish continuous-time dynamics.

The C. elegans suite currently analyses Bundle-Net worm 0 from the downloaded
`NoStim_Data.mat`; its endpoint is the eight-class locomotor motif. The source
commit and SHA-256 checksums are pinned in `conf/data/c_elegans.yaml`. The source
file contains five worms with different recorded neuron sets, so the remaining
four worms require separate recording namespaces rather than pooling.
Because locomotor motifs are categorical, their feature-sensitivity notebook
fits a manifold-dynamics surrogate and reports latent displacement; it does not
regress arbitrary motif integer codes as though their distances were
quantitative.

The C. elegans loader matches the eight published exclusions against the raw
per-worm `NeuronNames` before creating any analysis checkpoint. As in the
upstream BunDLe-Net code, unidentified numeric labels are retained. For worm 0,
six requested names are present, so all downstream methods use 103 of the 109
recorded neurons. The preprocessing notebook also checkpoints an exact
Bundle-Net reference lane. It starts from the upstream `deltaFOverF_bc` (already
bleach-corrected) field, applies the pinned fourth-order Butterworth filter once
forward and once on the reversed full recording without padding, and creates the
published paired inputs from overlapping 15-sample slices. The pairing uses each
16-sample block as `X[t:t+15]` and `X[t+1:t+16]`, with behavior at `t+15`, so a
recording of length `T` produces `T-15` pairs. These are saved as
`bundle_net_reference_recording` and `bundle_net_training_pairs`.
All manifold notebooks default to cross-fitted classical PCA on the main
recording. The pinned faithful BunDLe architecture and reference preprocessing
remain available for an optional retrospective reproduction: use
`conf/manifold/bunddle.yaml` with `bundle_net_reference_recording` in a separate
output namespace. That full-fit/noncausally filtered chart is not prospective.
Install the `bundle` extra on a platform supported by PyTorch. Python 3.14 on
Intel macOS has no published PyTorch wheel; use the PCA default there.

That exact filter is global and noncausal: every filtered time point can depend
on future samples, and its unpadded reversal retains endpoint transients. It is
therefore a retrospective reproduction/sensitivity artifact, not the input to
prospective c-GC claims. The main `recording` and `windows` checkpoints use the
same selected neuron cohort but retain the bleach-corrected signal without that
noncausal filter.

The first two V2a-RSN suites are bound to `220119_F2_run11` and
`220127_F4_run2`. Their configs use fluorescence, registered coordinates, tail
angle resampled to SCAPE frame times, and the union of the supplied emitter and
receiver cells. This makes the windowed c-GC run tractable and focuses it on the
annotated RSN roles, but it is a targeted analysis rather than coverage of every
detected cell. Use a separate `RUN_ID` for a later whole-recording or broader-cell
sensitivity analysis. Recordings 03 and 04 are present locally but deferred.

The shared windows retain unscaled, identity-preprocessed neural histories.
The primary evaluator and strict sensitivity notebook fit their scales on the
owned training partition and apply the frozen transform to held-out histories.
Per-window standardization is a separate legacy sensitivity profile and is
rejected by these revised entry points.

The primary notebook contract is 120 seconds of history and a 5.164-second
target/stride, resolved from each recording's sampling frequency. It yields
`349/15/15` frames and 186 windows for C. elegans, `723/31/31` and 94 windows
for V2a-01, and `636/27/27` and 95 windows for V2a-02. Adjacent histories still
share about 96% of their samples and are not independent replicates. Sparse V2a
bad frames remain on the native clock: c-GC/c-GC* mask only contaminated VAR
rows, while state/community/manifold transitions reset across their intervals.
The explicit `500/15/15` profile and strict all-bad-frames-as-boundaries policy
remain labeled sensitivities.

## Order and parallelism for each dataset

1. Run `00_preprocessing/preprocess.ipynb`. This is the only writer to the shared
   recording and raw-window checkpoints.
2. Run `08_evaluation/primary_m0_m1.ipynb` for the outer-owned primary comparison.
   The historical descriptive stages below are optional and their stitched
   outputs do not establish prospective validation.
3. Run the wanted method notebooks in `01_effectomes/` in parallel. Each writes
   to its own method namespace.
4. After a method's effectome completes, run its same-named notebooks under
   `02_clustering/`, `03_probabilistic_states/`, `04_communities/`, and
   `05_manifolds/` in parallel.
5. Run `06_linking/<method>.ipynb` after that method's clustering,
   community, and manifold notebooks. The HMM is a descriptive optional comparison and is not a computational
   dependency of linking.
6. Run `07_counterfactuals/<method>.ipynb` directly
   after preprocessing for a strict development sensitivity analysis. It builds
   a separate `outer_owned_features` checkpoint from raw windows, using a trailing
   evaluation block with full raw-history purging. Scaling, PCA and role centroids
   use training histories only. The constant VAR fits training histories once;
   the other audited directed lanes refit from issued causal histories. This
   notebook does not reuse the descriptive community, manifold or linking fits.
   Review the derived training/evaluation counts before expensive work, or supply
   explicit frozen index arrays. Candidate screening, surrogate fitting, matched
   control eligibility, dose settings and the declared complete multiplicity family
   remain tied to that outer scope. No passing candidate produces an unavailable
   result; the top screened-out neuron is never substituted.
   These notebooks report predictor-feature sensitivity, not biological interventions.
   Partial correlation stops at `06` because an undirected baseline cannot support
   directional candidate-driver claims.

The three recording-level `00_preprocessing/preprocess.ipynb` notebooks are independent and may
also run in parallel. Afterward, any ready stage in one recording is independent
of stages writing to another recording because each notebook writes under its own
`outputs/analysis/<dataset_id>/<recording_id>/...` namespace.

## Resume contract

- Keep `FORCE=False` for normal execution and resume. Every generated notebook
  starts by printing its `run_dir`, expected stages, current `completed`,
  `failed`, `running`, and `not started` states, and whether `FORCE` will
  recompute matching checkpoints.
- Completed stages are reused only when their configuration and declared input
  hashes match. Same-shaped changed window data also invalidates connectivity
  chunks through an in-memory content digest.
- Candidate and sensitivity cache identities include the outer partitions,
  fit scope, frozen candidate set, dose/control settings and hypothesis family.
  Changing an ownership argument requires rebuilding the corresponding owned bundle.
- Connectivity is checkpointed in independent chunks, so an interrupted run
  resumes from the first missing or invalid chunk.
- Atomic replacement prevents incomplete artifacts from appearing completed.
- A lock prevents concurrent writers to the same dataset/recording/run/method/stage.
  Different methods, stages, or recordings have independent locks.
- If a run is interrupted, rerun the same notebook from the top. Failed stage
  records are shown in the header and are recomputed automatically as long as no
  stale `.lock` remains from a still-running process. Remove a lock only after
  confirming the original process has stopped.

## Faster-machine run checklist

1. Copy the repository, `data/`, and any wanted `outputs/analysis/` checkpoints.
2. Install the locked environment with the `dev`, `csl`, `community`, and
   `notebooks` extras. All default notebook lanes use installed packages; the
   optional BunDLe reproduction additionally requires the `bundle` extra.
3. Open notebooks in the order above. Leave `FORCE=False` for resume.
4. Watch each notebook's first printed header before running expensive cells; it
   should point to the intended dataset/recording/method namespace.
5. For parallel execution, use different method or recording namespaces. Do not
   run two notebooks that write the same `run_id/method/stage`.

## Method boundaries

- The `c-GC` / `c-GC-star` directories now use installed fastC-GC / fastC-GC*
  with complete FFT circular-shift enumeration. Restart kernels on the Python
  3.14 `.venv` after migration. Legacy analytic and sampled circular-shift tests
  remain explicit sensitivity options. The fast estimator changes the screening
  method; old graph outputs are not interchangeable with new outputs.
- Lag-resolved signed ridge-VAR weights are primary; lag-collapsed matrices are
  downstream graph summaries.
- The probabilistic notebook fits a diagonal-Gaussian HMM. Its dwell diagnostic
  may justify a later HSMM but is not an HSMM fit.
- Prospective manifold cross-fitting currently supports classical PCA only.
- The prospective manifold outcome is the within-fold Euclidean displacement
  from the current anchor to the next valid anchor; its magnitude is invariant
  to fold-chart rotation/reflection. The exact BunDLe reference is descriptive.

## Installed backend environment — updated 14 November 2026

All 176 notebooks and the generator use Python 3.14 kernel metadata and a
shared setup check for fastcgc, Tigramite, scikit-learn, kmedoids, igraph, and
leidenalg. From the package root, run:

```bash
uv sync --python 3.14 --locked --extra dev --extra csl --extra community --extra notebooks
```

Select the project `.venv` kernel and restart it. Existing outputs were preserved;
they may reflect earlier code and are not evidence that the new backends have
been run on biological data. No upstream repository checkout or path injection
is needed. Internal project path discovery remains for opening notebooks from
their nested dataset directories.

PCMCI+ and JPCMCI+ are available through the installed Tigramite adapters and
`pcmci`/`jpcmci` configuration keys. They are not substituted for the existing
eight notebook lanes; JPCMCI+ remains reserved for explicitly scoped
pooled/context-aware analyses.
Leiden uses igraph/leidenalg, K-means and spectral clustering use scikit-learn,
and the optional distance-based k-medoids path uses the installed kmedoids solver.
Temporal role inference remains a distinct project-specific model.

The current [execution-order guide](../exec_order.md) includes the raw-calibration
notebook and distinguishes the primary experiment from optional summaries.
State notebooks now perform nested purged comparisons rather than force three
states; role notebooks preserve signed lag profiles. Full-series descriptive
outputs and existing stored cell outputs do not establish latent regimes.

State notebooks preserve the full raw-history purge. If a recording cannot
support the nested splits, they checkpoint `status="unavailable"` with the
reason, and linking reports the same limitation. No descriptive fit replaces
that missing evaluation. Both supplied V2a recordings currently have this
limitation at the default 120-second history; preprocessing, primary endpoint
evaluation, PCA, neuron roles, and owned feature sensitivity remain separate.
