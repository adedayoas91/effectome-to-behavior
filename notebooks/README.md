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
Because locomotor motifs are categorical, their counterfactual notebook fits a
manifold-dynamics surrogate and reports latent displacement; it does not regress
arbitrary motif integer codes as though their distances were quantitative.

The C. elegans preprocessing notebook also checkpoints an exact Bundle-Net
reference lane. It starts from the upstream `deltaFOverF_bc` (already
bleach-corrected) field, matches the eight published exclusions against the raw
per-worm `NeuronNames`, applies the pinned fourth-order Butterworth filter once
forward and once on the reversed full recording without padding, and creates the
published paired inputs from overlapping 15-sample slices. The pairing uses each
16-sample block as `X[t:t+15]` and `X[t+1:t+16]`, with behavior at `t+15`, so a
recording of length `T` produces `T-15` pairs. These are saved as
`bundle_net_reference_recording` and `bundle_net_training_pairs`.

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

For primary effectome estimation, every dataset's temporal history is standardized
independently per neuron: within each window, each neural trace is centered and
divided by its own window standard deviation. Constant traces become zero. This
normalizes the signed ridge-VAR coefficients without changing the saved recording
or using samples outside the causal history. Raw activity magnitude remains
available from the unscaled `recording` artifact for separate covariate analyses.

With strict bad-frame boundaries, the 500/15/15 reference profile currently
produces 176 C. elegans windows, 51 V2a-01 windows, and 76 V2a-02 windows. These
are highly overlapping observations (97% shared history), not independent
replicates. The 500 samples also represent different physical durations across
species (about 172 s for worm 0 versus 83--94 s for the two fish). Keep this as
the reproducible reference/sensitivity profile; freeze a primary physical-time
profile only after the planned synthetic and held-out stability checks.

## Order and parallelism for each dataset

1. Run `preprocessing/00_preprocess.ipynb`. This is the only writer to the shared
   recording and standardized-window checkpoints.
2. Run the three `effectomes/*/01_connectivity.ipynb` notebooks in parallel. They write to
   separate `cgc`, `cgc_star`, and `correlation_partial` method namespaces.
3. After a method's connectivity completes, run its notebooks under `clustering/`,
   `probabilistic_states/`, `communities/`, and `manifolds/` in parallel.
4. Run `linking/<method>/06_linking.ipynb` after that method's clustering,
   community, and manifold notebooks. The HMM is scientifically required for the
   state-model comparison but is not a computational dependency of linking.
5. After linking, run `counterfactuals/<method>/07_attribution_counterfactual.ipynb`
   for c-GC and c-GC*.
   Partial correlation stops at `06` because an undirected baseline cannot support
   directional candidate-driver claims.

The three recording-level `preprocessing/00_preprocess.ipynb` notebooks are independent and may
also run in parallel. Afterward, any ready stage in one recording is independent
of stages writing to another recording because each notebook writes under its own
`outputs/analysis/<dataset_id>/<recording_id>/...` namespace.

## Resume contract

- Keep `FORCE=False` for normal execution and resume.
- Completed stages are reused only when their configuration and declared input
  hashes match. Same-shaped changed window data also invalidates connectivity
  chunks through an in-memory content digest.
- Connectivity is checkpointed in independent chunks, so an interrupted run
  resumes from the first missing or invalid chunk.
- Atomic replacement prevents incomplete artifacts from appearing completed.
- A lock prevents concurrent writers to the same dataset/recording/run/method/stage.
  Different methods, stages, or recordings have independent locks.

## Method boundaries

- c-GC and c-GC* use the analytic marginal/conditional screen by default; retain
  circular-shift inference as the autocorrelation-aware sensitivity analysis.
- Lag-resolved signed ridge-VAR weights are primary; lag-collapsed matrices are
  downstream graph summaries.
- The probabilistic notebook fits a diagonal-Gaussian HMM. Its dwell diagnostic
  may justify a later HSMM but is not an HSMM fit.
- Prospective manifold cross-fitting currently supports classical PCA only.
