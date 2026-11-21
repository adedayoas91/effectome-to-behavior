# C. elegans intervention explorer

The local browser app compares the selected model's all-on forecast with a
forecast after removing the outgoing connections of any set of recorded
neurons. Pick a worm and a held-out forecast origin, then use the neuron
switches to disconnect or restore individual ROIs. Multiple switches form
one joint intervention. Search by recording-local ROI ID, recorded label,
or canonical name. The outgoing-links filter shows ROIs with fitted targets.
Saved groups belong to the selected run and worm and are stored in the
browser. The two-dimensional trajectory shows where the selected origin
falls in the motif-supervised representation.

## Launch

Run from the application repository in the project Python 3.14 environment.

```sh
.venv/bin/python -m scripts.celegans_intervention_explorer
```

Open the local address printed by the command. The default port is 8765.
The app uses the completed `motif-rollout-01` notebook run or the
`verification-01` run when those are the available forecasts. Supply
`--run-root outputs/celegans-rollout/<RUN_ID>` to select another run.
A run needs notebooks 00 and 01 and the corresponding pinned source files.
The app reads those artifacts and runs the requested comparisons on demand.
Stop it with Ctrl+C. [C. elegans notebook workflow](../notebooks/README.md#c-elegans-motif-rollout)

## What the controls do

- **Forecast origin.** Use the last observed neural state and the exact
  pooled or local mechanism selected at that origin by the frozen policy.
  Refit the local parameters on the original trailing history when needed.
- **Connections off.** Zero the selected sources' outgoing cross-neuron
  weights. Self-history and incoming effects from outside the selected
  group still contribute to its activity. Restore a switch to restore
  that source's fitted outgoing weights.
- **Motif probabilities.** Start both rollouts at the same neural state and
  reuse the same standardized innovations. The original readout maps
  both simulated futures to motif probabilities. Report the modified
  forecast minus the all-on forecast in percentage points. A source
  with zero outgoing support leaves the forecast unchanged.

This is an intervention on the fitted dynamics for the supplied fictive
motor motifs. The current pilot's held-out scores provide the context for
interpreting these probability shifts. [Pilot results and provenance](celegans_rollout.md#verification-on-the-five-bundled-recordings)

## Example from the verification run

Choose worm 0 and sample 2142, the seventh forecast origin. Turn off ROI 3
(ADLR) and ROI 23 (RMER). The fitted local mechanism loses three outgoing
links. Its forecast of the slow motif changes from 37.8% to 24.1%,
a decrease of 13.70 percentage points. Restoring both switches recovers the
original forecast. The example illustrates the effect of a specified edit
at one observed origin.
