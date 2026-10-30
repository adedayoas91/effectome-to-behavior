# Preparing and running analyses

Run these commands from the nested Python package (`effectome-to-behavior/`). Use its
existing `.venv`; no additional dependencies are required for the implemented baseline lanes.
Every dataset/run should have a separate output directory.

## Verified readiness on 30 October 2026

The full suite passed with **295 tests passed and one skipped**. Ruff passed for source,
pipeline, tests and notebook helpers; mypy passed for 64 source/entrypoint files. Python cells
in all 21 attribution notebooks compile. Executable checks covered primary CLI fitting and
resume, the five-model raw benchmark and resume, a synthetic six-evaluation-animal exact-null
cohort, and a 19-draw development-only raw conditional-null calibration smoke.

Real-data preflight produced:

| Recording | Neurons | Native Hz | Windows | Strictly future origins | Usable folds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Worm 0 | 109 | 2.9046296296296297 | 186 | 185 | 5 |
| 220119 F2 run 11 | 100 | 6.0268 | 94 | 86 | 5 |
| 220127 F4 run 2 | 92 | 5.2962 | 95 | 89 | 5 |

These are data/split checks, not biological results. No publication grid or full biological
evaluation was launched. The complete publication preflight resolved 1400 registered cells.

## Inspect raw inputs before fitting

```bash
.venv/bin/python pipeline/08_evaluate.py \
  paths.evaluation_manifest=conf/evaluation-manifests/c-elegans-worm0-development.yaml \
  paths.artifacts=outputs/development/worm0 \
  +preflight=true
```

Preflight loads the raw data, checks preprocessing ownership and endpoint vocabulary, and
constructs dependency-purged splits without fitting models. It saves `preflight.json` with
the native clock, dimensions, endpoint status and every available/unavailable split.
An unavailable split retains its reason. A run with no available folds exits unsuccessfully.

The available development manifests are:

| Manifest in `conf/evaluation-manifests/` | Endpoint | Intended use |
| --- | --- | --- |
| `synthetic-smoke.yaml` | Generated regime label | Small executable integration check |
| `c-elegans-worm0-development.yaml` | Supplied fictive neural-state annotation | Within-animal development, with annotation lineage unresolved |
| `v2a-220119-F2-run11-development.yaml` | Recorded tail angle | Within-animal development |
| `v2a-220127-F4-run2-development.yaml` | Recorded tail angle | Within-animal development |

The real-data manifests retain a 120-second history and 5.164-second target/cadence,
resolved at each recording's native sampling rate. They use training-owned scaling and
preserve missing samples and recording boundaries. The c-GC lag horizon and VAR lag order
are distinct parameters; the current primary decoder uses sparse VAR features.

## Start or resume primary development evaluation

Remove `+preflight=true` from the same command:

```bash
.venv/bin/python pipeline/08_evaluate.py \
  paths.evaluation_manifest=conf/evaluation-manifests/c-elegans-worm0-development.yaml \
  paths.artifacts=outputs/development/worm0
```

The command compares the activity/history baseline M0 with M1 including graph features.
It writes immutable prediction records in `evaluation.pkl`, a readable `evaluation-report.json`,
and the exact input `evaluation-manifest.yaml`. Development folds checkpoint independently
under `checkpoints/`. Rerunning the same command reuses matching completed folds; changed
inputs, settings or implementation invalidate their signatures. Use `+force=true` to recompute.
The supplied manifest controls evaluation settings; unrelated Hydra `evaluation.*` overrides
do not replace settings inside that manifest.

For an inexpensive installation/integration check, use `synthetic-smoke.yaml` and a fresh
output directory first. The real-data commands run the complete configured evaluation;
preflight does not estimate their final runtime or guarantee every fitted solver converges.

## Preflight and run raw-process benchmarks

```bash
.venv/bin/python pipeline/09_validate.py --profile publication \
  --preflight-output outputs/validation/publication-preflight.json

.venv/bin/python pipeline/09_validate.py --profile smoke --run \
  --output-dir outputs/validation/readiness-smoke
```

The default action only inspects the registered grid. The development smoke run is bounded
to one small raw recording and two neural forecast origins. It exercises the same primary
M0/M1 evaluator and all five predictive adapters: constant/windowed sparse VAR, smooth/fused
TVVAR and observed switching VAR. Its results are labeled `smoke_only`.

The full manifest contains 800 core/stress/nonlinear recordings plus 600 independent null
calibration recordings; optional grouping adds 120. The baseline uses the exact worm-0 clock
(`2.9046296296296297 Hz`), 3137 samples and two lags. Stress/nonlinear/calibration cells use
109 neurons. Independent graph and observation streams preserve paired method comparisons.
Appendix C and the machine manifest specify correlated innovations as the ninth stress condition;
the manuscript's alternative variance-only stress remains a documented discrepancy. Variance-only
fixed interactions are evaluated separately in the calibration suite.

Every attempted cell records configuration/input/implementation fingerprints, runtime and the
memory measurement method, prediction/forecast/recovery scores and failures. Cell files and
`report.json` live under the specified output directory. Rerunning the command reuses matching
attempts. Select suites with repeated `--suite`, shard by `--offset`/`--max-cells`, and use
`--no-resume` only for intentional recomputation.

To start the complete registered grid, first freeze the calibration tolerance and all tuning
settings in a saved copy of `conf/validation/revised.yaml`, then use `--config <saved-copy>`
with `--profile publication --run`. Calibration pass/fail remains unavailable for an incomplete
registered family. Independent observed-process projection truth must be requested before
interpreting fluorescence coefficient recovery; latent coefficient discrepancy is labeled
separately. Unsupported optional state/group metrics stay unavailable.

## Training-owned candidate and sensitivity notebooks

Each method's attribution notebook now starts from `recording.pkl` and identity-preprocessed,
unstandardized `windows.pkl`. It constructs a separate `outer_owned_features` stage, selects
candidates in the purged training partition, and evaluates predictor-feature edits on the held-out
partition. It freezes PCA, roles, matching variables, unique controls and the multiplicity family.
Changing the split, seed, dimensions, source inputs or implementation invalidates checkpoints.
The CLI stages `06_attribution.py` and `07_intervention.py` use the same producer.

No qualifying candidate produces an unavailable sensitivity result; it does not substitute the
top rejected neuron. An unattainable finite-control significance threshold also remains explicit.
These diagnostics are optional for the primary M0/M1 analysis.

## Scientific registration before confirmation

Development outputs do not establish population generalization. Confirmatory paired-performance
inference requires one frozen development fit and separate independent evaluation animals,
an explicitly declared common neuron panel, verified endpoint lineage, a frozen practical-effect
threshold, completed independent calibration and a registered complete multiplicity family.
The confirmation implementation checks finite animal/control-pool resolution before fitting.
Neither folds nor optimizer seeds increase the number of independent animals.

Real-data manifests deliberately leave these scientific checks unresolved. Do not set their
verification flags merely to bypass a guard. Worm fictive annotations must remain distinct
from observed behavior. The optional generative intervention model is outside this analysis
preparation; current perturbation outputs describe predictor-feature sensitivity.

A cohort manifest uses a list of loader mappings under `data`. Every record declares
`recording_id`, `animal_id`, and a frozen `neuron_panel` containing `mapping_id`,
`source_indices`, ordered unique `canonical_ids`, and a mapping `provenance` description.
Matching recording-local row numbers are insufficient to align animals. The current cohort
loader requires matching native sampling rates and window shapes; it does not silently resample.
External confirmatory sources also require `source_file_sha256` (or `downloaded_sha256`)
covering exactly every consumed file. The loader checks the actual bytes before fitting.

Register `evaluation.protocol: fixed_development`, disjoint `development_animal_ids` and
`evaluation_animal_ids`, `null_families: [animal_sign_flip]`, the sign-symmetry assumption,
and the complete `multiplicity_members`/`claim_to_test_map`. Compute the frozen declaration
hash using `effectome.workflows.evaluation.raw_manifest_source_hash`. The report's independent
source fingerprint additionally binds the actual model inputs. Exact or seeded plus-one Monte
Carlo sign flips act on animal-level paired loss differences; they do not test conditional graph
information or dynamic structure. Confirmation checkpoints as one whole cohort, preserving
resource-budget failures. Runtime limits and traced Python memory are reported post hoc;
they are not operating-system process limits.
