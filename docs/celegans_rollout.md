# C. elegans motif rollout

The empirical pilot transfers the pooled and local neural mechanisms,
fixed fastC-GC support, and completed-forecast selector from the confirmed
synthetic rollout. Each NoStim worm gets a separate fit. The recordings
have different neuron sets and sampling rates, and their supplied targets
are eight neural-derived fictive motor motifs. The score measures prediction
of those annotations from held-out fluorescence histories. [NoStim source](https://github.com/akshey-kumar/BunDLe-Net/tree/cbefad93fe98b5860c7e219e5af9a7bb0a177143/data/raw) · [Kato et al.](https://pubmed.ncbi.nlm.nih.gov/26478179/) · [Synthetic tuning](effectome_rollout_tuning.md)

## Adjustments for these recordings

- **Use the supplied dF/F series.** Check the three pinned source hashes.
  Load `deltaFOverF` and retain the existing raw-name neuron exclusions and
  recording-local ROI IDs. Fit the neural scale on the mechanism training
  block. The prospective path skips the full-recording reversed filter
  and bleach-corrected trace field used in the historical replication.
  The source file is an upstream fluorescence export. The provenance of
  the supplied annotation may include temporal context beyond its sample.
  [Data configuration](../conf/data/c_elegans.yaml)
- **Split chronologically.** Use the first 35% of each recording to fit the
  readout, the next 30% to discover support and fit the pooled dynamics,
  and the remainder for forecasting. The first held-out origin follows
  one complete trailing history inside the final block. Training
  windows follow the native clock and use no motor-state or switch labels.
  The pooled fit uses the contiguous mechanism training block once,
  even though its discovery windows overlap.
- **Convert timing to native frames.** Use a 25.6-second trailing fit,
  102.4-second discovery histories stepped by 25.6 seconds, a 0.8-second
  horizon, 0.4-second terminal interval, and 1.6-second forecast cadence.
  Resolve each duration independently against the worm's sampling rate.
  For worm 0 this gives 74 history frames, a two-frame horizon, one
  terminal frame, and a five-frame cadence. Preflight saves the frame
  counts and the realized durations for each recording.
  [Pilot settings](../conf/behavior_rollout_pipeline/celegans.yaml)
- **Forecast categorical probabilities.** A pointwise supervised nonlinear
  readout uses a two-dimensional bottleneck and softmax over motifs seen
  in its fitting block. The numerical artifact stores the class order,
  fitted weights and optimizer diagnostics. The full observed neural
  state still evolves in the ridge-fitted additive tanh mechanism. Both
  candidates decode each simulated terminal frame before averaging across
  frames and draws. Missing training motifs retain probability zero and
  their appearance in the holdout contributes to the reported error.
- **Keep the forecast decision causal in time.** The tuned rule uses six
  completed forecasts, zero threshold, three consecutive losses to fall
  back, and one favorable recovery forecast. It compares the squared
  probability error of both forecasts against the frozen readout applied
  to the newly available neural target samples. The supplied test labels
  enter scoring after the forecasts are issued. The real recordings have
  no simulator switch times or paired intervention truth.
  [Forecast selection](effectome_rollout_online.md) · [fastC-GC](https://arxiv.org/abs/2610.37883)

## Run and interpret

Run the three [numbered notebooks](../notebooks/README.md#c-elegans-motif-rollout)
in `notebooks/c_elegans/` using the project Python 3.14 environment.
Notebook 00 verifies inputs and saves the run's preflight. Notebook 01
learns one support mask per worm and fits and forecasts with both mechanisms.
Notebook 02 compares multiclass Brier scores against the always-pooled,
always-local, decoded persistence and training prevalence baselines.
Decoded persistence repeats the motif distribution from the last observed
neural frame. The score is the sum of squared errors against the terminal
one-hot motif distribution or its mean across terminal samples, averaged
over origins. Origins overlap in time, so per-worm scores are descriptive.

The default run processes all five worms and leaves disconnection evaluation
off. To inspect effects, set `DISCONNECT_ROIS` in notebook 02 to source
ROI IDs listed in preflight. Each declared ROI is disconnected individually.
The comparison removes outgoing cross-neuron terms from the selected fit,
retains self-history and incoming terms, and uses paired innovations.
It measures a change in estimated motif probabilities in the fitted model.
ROI IDs are local to each recording, so an ID shared by two worms may denote
different neurons. Run a single worm for a named ROI comparison.

Use a new `RUN_ID` if the chosen recordings, data path, implementation, or
timing settings change. Re-running notebook 01 under the same configuration
reuses completed forecast artifacts. The notebook outputs are JSON and CSV
under `outputs/celegans-rollout/<RUN_ID>/`. Git history retains the retired estimator notebooks.

## Verification on the five bundled recordings

All three notebooks executed in order in fresh Python 3.14 Jupyter kernels.
The verification run saved 1,073 forecasts. An independent replay checked
the forecast timing, probability normalization, selector decisions and
per-origin scores. A recording-local ROI disconnection also replayed with
the same factual states and innovation draws in every worm.
[Verification summary](../outputs/celegans-rollout/verification-01/summary.json)

| Worm | Origins | Pooled Brier | Local Brier | Selected Brier | Decoded persistence | Training prevalence |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 205 | 0.9655 | 1.0883 | 1.0806 | 1.1411 | 0.8048 |
| 1 | 205 | 1.1976 | 1.1761 | 1.1939 | 1.2349 | 0.9855 |
| 2 | 200 | 1.1300 | 1.1836 | 1.2024 | 1.2807 | 0.9536 |
| 3 | 216 | 0.6712 | 0.9125 | 0.8891 | 0.9601 | 0.7140 |
| 4 | 247 | 1.0033 | 1.0026 | 0.9907 | 1.0616 | 0.7392 |

The selected forecasts improved on decoded persistence in all five worms.
Training prevalence beat the selected forecast in every worm. The readout
optimizer reached the configured iteration limit in worms 0, 1 and 2.
Worms 0, 1 and 3 lacked the `nostate` motif in readout training and encountered
it at 5, 4 and 8 evaluated origins, respectively. This pilot establishes
executable fits and reports their held-out errors under the frozen settings.
