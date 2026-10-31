# Dataset-grouped resumable analysis notebooks

The executable notebooks are grouped by independently analysed recording:

```text
notebooks/
├── c_elegans/
│   ├── preprocessing/
│   ├── effectomes/{c-GC,c-GC-star,partial-correlation}/
│   ├── clustering/{c-GC,c-GC-star,partial-correlation}/
│   ├── probabilistic_states/{c-GC,c-GC-star,partial-correlation}/
│   ├── communities/{c-GC,c-GC-star,partial-correlation}/
│   ├── manifolds/{c-GC,c-GC-star,partial-correlation}/
│   ├── linking/{c-GC,c-GC-star,partial-correlation}/
│   └── counterfactuals/{c-GC,c-GC-star}/
└── v2a_rsns/
    ├── 220119_F2_run11/
    └── 220127_F4_run2/
```

Every recording repeats the same analysis-oriented structure shown under
`c_elegans`. The method directories make c-GC, c-GC*, and partial correlation
independently runnable, while the parent directories group notebooks by their
scientific purpose. V2a-RSN datasets 03 and 04 are intentionally not generated
yet; add their source recording names to `_build_notebooks.py` after their paths
and schemas have been confirmed.

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

The C. elegans suite currently analyses Bundle-Net worm 0 from the downloaded
`NoStim_Data.mat`; its endpoint is the eight-class locomotor motif. The source
commit and SHA-256 checksums are pinned in `conf/data/c_elegans.yaml`. The source
file contains five worms with different recorded neuron sets, so the remaining
four worms require separate recording namespaces rather than pooling.
Because locomotor motifs are categorical, their feature-sensitivity notebook
fits a manifold-dynamics surrogate and reports latent displacement; it does not
regress arbitrary motif integer codes as though their distances were
quantitative.

The C. elegans preprocessing notebook also checkpoints an exact Bundle-Net
reference lane. It starts from the upstream `deltaFOverF_bc` (already
bleach-corrected) field, matches the eight published exclusions against the raw
per-worm `NeuronNames`, applies the pinned fourth-order Butterworth filter once
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
prospective c-GC claims. The main `recording` and `windows` checkpoints retain
all recorded neurons and use the bleach-corrected signal without that filter.
This also avoids removing plausible command/interneuron drivers merely because
Bundle-Net excluded them for its representation-learning control.

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

1. Run `preprocessing/00_preprocess.ipynb`. This is the only writer to the shared
   recording and raw-window checkpoints.
2. Run `evaluation/08_primary_m0_m1.ipynb` for the outer-owned primary comparison.
   The historical descriptive stages below are optional and their stitched
   outputs do not establish prospective validation.
3. Run the three `effectomes/*/01_connectivity.ipynb` notebooks in parallel. They write to
   separate `cgc`, `cgc_star`, and `correlation_partial` method namespaces.
4. After a method's connectivity completes, run its notebooks under `clustering/`,
   `probabilistic_states/`, `communities/`, and `manifolds/` in parallel.
5. Run `linking/<method>/06_linking.ipynb` after that method's clustering,
   community, and manifold notebooks. The HMM is a descriptive optional comparison and is not a computational
   dependency of linking.
6. Run `counterfactuals/<method>/07_attribution_counterfactual.ipynb` directly
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

The three recording-level `preprocessing/00_preprocess.ipynb` notebooks are independent and may
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

## Installed backend environment — 31 October 2026

All 173 notebooks and the generator now use Python 3.14 kernel metadata and a
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
`pcmci`/`jpcmci` configuration keys. The existing notebook lane layout is retained.
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
