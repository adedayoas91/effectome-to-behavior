# Empirical analysis notebooks

## C. elegans motif rollout

Run these notebooks in order with the project Python 3.14 `.venv` kernel.

1. [00 · Data and timing](c_elegans/00_data_and_timing.ipynb) verifies the
   pinned input files and freezes the chronological partitions and forecast timing.
   The default uses all five worms. Set `WORM_INDICES = (0,)` for a quick pilot.
2. [01 · Fit and forecast](c_elegans/01_fit_and_forecast.ipynb) trains the
   motif-probability readout, learns fastC-GC support on training windows, and
   compares pooled and trailing local rollouts with the tuned selector.
3. [02 · Score and disconnections](c_elegans/02_score_and_disconnections.ipynb)
   compares held-out multiclass Brier scores against pooled, local, decoded
   persistence, and training-prevalence forecasts. A nonempty
   `DISCONNECT_ROIS` requests paired model-based motif-probability shifts.

From the application repository, install the locked environment with
`uv sync --python 3.14 --locked --extra notebooks`. The three notebooks use
the installed fastC-GC backend and the categorical readout.

The default inputs are `NoStim_Data.mat`, `ClassIDs_279.mat`, and
`Order279.mat` in `data/c_elegans/bundle_net_raw/`. Set `DATA_ROOT` in
notebook 00 if the same files are stored elsewhere. The source and checksums
are pinned in [the C. elegans data configuration](../conf/data/c_elegans.yaml).

The workflow uses the source `deltaFOverF` traces before the added BunDLe-Net
filter. Each worm has its own readout, support, and forecast evaluation.
The targets are supplied neural-derived fictive motor motifs, and the
forecasts give probabilities for those motifs. Forecast scores describe
held-out annotation prediction. Optional disconnections describe probability
shifts in the fitted model.

Set the same `RUN_ID` in all three notebooks. The default output directory
is `outputs/celegans-rollout/motif-rollout-01/`, with `preflight.json` and
`summary.json` at the run level. Each `worm-<index>/` directory contains
`forecasts.json`, `scores.csv`, and `disconnections.json` when requested.
Use a fresh `RUN_ID` for changes to input settings, selected worms, or
the workflow configuration.

Open the [intervention explorer](../docs/celegans_intervention_explorer.md)
after notebook 01 to compare motif forecasts while switching outgoing
connections off or on for individual neurons or groups.

The older worm estimator, clustering, and visualization lanes remain in Git
history. Regenerate the three active notebooks with
`python -m notebooks._build_celegans_rollout`.

## V2a-RSN shared-transition notebooks

For each V2a recording, the shared-transition path is
`00_preprocessing/preprocess.ipynb` followed by
`09_cross_method/shared_transitions.ipynb`. It rebuilds raw identity windows,
fits one shared training-owned manifold, and compares all eight methods on
the same held-out events. Start with
[framework validation](../synthetic/README.md) before interpreting results.
Estimation and descriptive notebooks remain available per recording in
`v2a_rsns/220119_F2_run11/` and `v2a_rsns/220127_F4_run2/`.

The generated V2a notebooks retain fastC-GC, fastC-GC*, partial correlation,
constant and windowed sparse VAR, switching VAR, and smooth and fused TVVAR.
The numbered stage directory supplies execution order and the filename
identifies the estimator. PCMCI+ and JPCMCI+ remain registry-backed
configurations. V2a-RSN datasets 03 and 04 are deferred until their paths and
schemas have been confirmed.

### Output layout

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

From the package root, a saved legacy C. elegans checkpoint can still generate motif, graph-state, and HMM-state views:

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
`723/31/31` frames and 94 windows for V2a-01, and `636/27/27` frames and 95
windows for V2a-02. Adjacent histories still
share about 96% of their samples and are not independent replicates. Sparse V2a
bad frames remain on the native clock: c-GC/c-GC* mask only contaminated VAR
rows, while state/community/manifold transitions reset across their intervals.
The explicit `500/15/15` profile and strict all-bad-frames-as-boundaries policy
remain labeled sensitivities.

### Order and parallelism for each V2a recording

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

The two V2a recording-level `00_preprocessing/preprocess.ipynb` notebooks are independent and may
also run in parallel. Afterward, any ready stage in one recording is independent
of stages writing to another recording because each notebook writes under its own
`outputs/analysis/<dataset_id>/<recording_id>/...` namespace.

### Resume contract

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

### Faster-machine run checklist

1. Copy the repository, `data/`, and any wanted `outputs/analysis/` checkpoints.
2. Install the locked environment with the `dev`, `csl`, `community`, and
   `notebooks` extras. All default notebook lanes use installed packages; the
   optional BunDLe reproduction additionally requires the `bundle` extra.
3. Open notebooks in the order above. Leave `FORCE=False` for resume.
4. Watch each notebook's first printed header before running expensive cells; it
   should point to the intended dataset/recording/method namespace.
5. For parallel execution, use different method or recording namespaces. Do not
   run two notebooks that write the same `run_id/method/stage`.

### Method boundaries

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

### Installed backend environment

The V2a and shared legacy notebooks use Python 3.14 kernel metadata and a
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
