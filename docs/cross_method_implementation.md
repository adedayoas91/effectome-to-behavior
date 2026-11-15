# Shared transitions across effectome methods

Decision: 14 November 2026; scientific specification: workspace `response.md`.

## Implementation plan

1. Fit a single PCA coordinate system and geometric state vocabulary on raw
   training histories; reuse both for every estimator. Store fit indices and
   fitted parameters. Keep this exploratory vocabulary separate from validated
   dynamical regimes.
2. Build identical strictly future event pairs, with physical horizons, source
   continuity, invalid-frame exclusion, chronological ownership, and raw-history
   plus outcome purging. Retain stays and competing destinations.
3. Run the eight current estimators using existing factories and training-owned
   scaling. Fit constant sparse VAR on training histories only. Keep partial
   correlation undirected. Record individual failures without changing event IDs.
4. Fit baseline and graph-augmented transition predictors, then refit after
   neuron/group feature omission. Score held-out probabilities and continuous
   manifold displacement; distinguish empirical, predicted, and native graph
   transition matrices.
5. Report training-frozen graph/manifold correspondence and held-out diversity
   in both directions: several graph configurations in one manifold region,
   and several subsequent outcomes from one graph configuration. These are
   descriptive diagnostics requiring uncertainty and replication before a
   scientific claim, rather than automatic proof of circuit degeneracy.
6. Provide a CLI for raw windows/raw manifests and a bounded synthetic smoke
   run, machine-readable outputs, candidate tables, and documented interpretation.

## Verification requirements

- Changing held-out values cannot change fitted scaling, PCA, state centers,
  training graph parameters, or training correspondence maps.
- Events never cross recording, gap, invalid-frame, ownership, or outcome-support
  boundaries. Shared events and empirical matrix are identical across methods.
- Test deliberate many-to-one and one-to-many fixtures and a one-to-one control.
- Recover a planted useful graph-feature block and retain negative/competing
  origins in transition-specific scores; handle absent training classes.
- Constant graphs retain one native state; undirected associations never receive
  directed drive interpretations. State-count and insufficiency decisions are
  explicit in output.
- Run targeted tests, the complete existing suite, Ruff, mypy, compilation, and
  an executable smoke comparison across all eight lanes.

## Scope and interpretation

No new dependencies or existing-notebook regeneration are needed. The first
workflow is a frozen chronological development holdout per recording, not
independent-animal confirmation. Fixed hyperparameters are declared defaults;
no held-out tuning or p-values are introduced. Candidate feature importance does
not identify biological necessity. Publication-scale null calibration,
multi-animal replication, and intervention validation remain scientific gates.

## Run and inspect

Start with the four notebooks under `synthetic/`, in numbered order. They
separate planted truth from inferred graphs and compare a one-to-one control,
distinct circuits with matched stationary population marginals, and trajectories
from a fixed circuit with different initial conditions. The fixed-circuit case
can be explained by current activity/history; it is not an unexplained circuit
switch or proof of causal necessity. The fourth notebook inventories or executes
the repeated framework study, with a stationary no-switching control, declared
observation stresses, seed-level uncertainty, retained failures and manuscript
exports. See [the five-area delivery audit](framework_validation_audit.md).

From the package root, run all eight existing backends with:

```bash
.venv/bin/python pipeline/11_cross_method.py --profile smoke \
  --scenario matched_marginals --output-dir outputs/cross-method/smoke
```

The smoke profile uses short histories, a separately declared c-GC threshold of
0.05, and one switching-VAR component to exercise its path cheaply. It does not
calibrate production thresholds or switching-regime recovery. The development
profile loads the original estimator configurations, and failures (including
nonconvergence and thresholds unattainable at the selected history) are retained
as unavailable methods.

Use `--windows` with a trusted raw-window artifact, or `--raw-manifest` with an
existing raw-evaluation manifest, only after inspecting synthetic results. These
inputs require finite histories, identity preprocessing, no per-window scaling,
and correct temporal/source metadata. Use `--method <lane>` repeatedly to run a
declared subset. The CLI returns a nonzero status if any requested method fails.

Outputs include `comparison.json` (fitted reference, correspondence, paired
predictions, matrices, diagnostic metadata and family rankings),
`transition_events.csv`, and `neuron_group_contributions.csv`. Unsupported source
rows are unavailable, not fabricated probability rows. In the Python API, pass
`groups={"group_name": ["neuron_id", ...]}` for joint omission scores; IDs must
match the explicit or raw-manifest neuron panel.

## Initial engineering verification — 14 November 2026

- Complete regression suite: 464 passed, one optional skip.
- All three synthetic notebooks executed through their default code cells;
  their checked-in sources retain empty outputs and execution counts.
- Eight-method CLI smoke: all eight available on the same 41 training and 15
  held-out events, at a fixed 1.6-second horizon for the 1024-sample example.
- Ruff, mypy over 70 source files, Python compilation, and diff checks passed.
- These checks establish engineering behavior on bounded examples, not calibrated
  biological regimes, circuit degeneracy, intervention effects, or neuron necessity.
