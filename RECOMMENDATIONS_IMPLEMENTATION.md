# Implementation plan following independent review

## Scientific scope

Keep community detection as an optional structural analysis. Neuronal roles,
whole-graph states, and assortative communities remain separate targets. The
primary activity/history-versus-graph prediction comparison does not require
any of them. Retain constant, continuous, and switching dynamical comparators.
No new claim of causal identification follows from clustering.

## Planned changes and verification

1. Extend the signed, lag-resolved state representation with a regularized,
   diagonal-covariance Gaussian mixture and soft held-out memberships. Keep
   K-means as a baseline. Use training-only scaling/PCA, include a one-state
   alternative, record effective representation/algorithm/distance separately,
   and retire misleading causal-kernel provenance. Verify future invariance,
   finite soft memberships, unequal-spread recovery, and schema handling.
2. Simplify role features to one incoming/outgoing signed representation, retain
   legacy features only as an explicitly named compatibility option, and make
   community-versus-role metadata precise. Existing role invariance and boundary
   tests protect the inference procedures. Add representation tests.
3. Separate descriptive full-series state pictures from a purged training-only
   state-analysis path. Compare candidate state counts/algorithms through inner
   future-feature prediction with a continuous baseline; score the frozen
   selected representation on outer held-out data. Report one-state/no-benefit
   outcomes and assignment uncertainty. Label-only transition nulls remain
   conditional sequence diagnostics, not evidence of neural regime discovery.
4. Add executable raw-data state calibration scenarios: fixed interactions,
   variance/kinetic changes, continuous drift, recurrent and nonrecurring
   dynamics, nonlinear dependence, and varying history/dwell ratios. Rerun graph
   estimation and state selection in each draw; record false-state claims,
   recovery, transition delay, uncertainty, and predictive value. Provide a
   bounded smoke profile and explicit larger development profiles.
5. Update configs, notebook entry points/generator, documentation, and manuscript
   method description to match implemented behavior. Preserve existing notebook
   outputs and paths. Run targeted regressions, installed-backend checks, Ruff,
   mypy, compilation and the full suite.

## Boundaries

The Markham dependence-contribution kernel remains a separately specified
research comparator, rather than being approximated by an RBF and called
causal. Treatment-response causal K-means requires an identified intervention
estimand and suitable data. Cohort expansion, endpoint lineage verification,
independent kinematic measurements, and publication-scale calibration require
scientific data/work beyond a code revision; no results will be invented.

No new dependencies are needed: scikit-learn supplies the mixture backend.
Existing user changes and scientific outputs are preserved.

## Implemented and verified

The code repository is on `dev`; fetching/pruning confirmed the deleted remote
`astra-revise` reference is gone. No branch was recreated. Existing uncommitted
changes and stored scientific outputs were preserved.

- State backend: regularized diagonal GMM, conditional soft membership/entropy,
  frozen transforms, truthful algorithm/geometry provenance, and one-state support.
- Nested state evaluator: raw-support-purged chronological inner/outer splits,
  wholly future graph histories, constant/continuous comparisons, training-only
  transition fitting, and explicit unavailable outcomes. Invalid anchors split
  continuity on both sides. Stage 3 writes a readable JSON evaluation summary.
- Roles: one signed incoming/outgoing vector per lag, zero self slots, no implicit
  duplication, explicit legacy representation, and schema checks. Structural
  Leiden community detection remains available; groups/states stay optional.
- Prediction: separate opt-in endpoint-directed graph/decoder selection, with
  matched raw-purged M0/M1 tuning boundaries. The original neural objective is
  preserved as the primary default.
- Calibration: seven raw scenarios, paired history/dwell designs, JSON/CSV
  manifests, CLI and a new resumable notebook. All 173 notebooks/543 code cells
  parse. Seventy-two existing notebooks were updated without erasing outputs.

Verification: the full suite passed **368 tests with 1 optional skip**; after the
independent review's invalid-anchor correction, all seven state-evaluation
regressions pass again, including the added case. Ruff passes, mypy passes over
68 source/entrypoint files, and compilation succeeds. The actual prospective
Stage 3 CLI and endpoint-directed raw-manifest CLI both execute successfully.
The manuscript rebuilds with all 34 cited keys resolved.

A seven-draw raw smoke run completed with all draws available; its report is
`outputs/state-calibration/review-smoke/state-calibration.json` (with CSV beside
it). Three of four no-discrete-regime scenarios selected a multi-code candidate
on inner data. This finite diagnostic is deliberately reported; no biological
regime claim is enabled. It demonstrates that adding a mixture competitor does
not establish regime recovery. Publication-scale calibration and independent
biological evidence remain required.

The Markham kernel and treatment-response causal K-means remain separate research
extensions, not implemented by relabeling ordinary RBF features. Reliability
weights are accepted only as externally supplied training-owned inputs; default
GMM responsibilities do not propagate upstream graph-estimation uncertainty.

## Main changed files

`dynamics/graph_states.py`, new `dynamics/state_evaluation.py`,
`dynamics/transitions.py`, `community/temporal.py`, `workflows/evaluation.py`,
`workflows/feature_ownership.py`, the dynamics exports, state/community/evaluation
configs, `pipeline/02_dynamics.py`, new `pipeline/10_state_calibration.py`, new
`experiments/state_calibration.py`, notebook helpers/generator and affected
notebooks, new regression files, README/run-order guides, and the adjacent
manuscript method/limitations/revision documentation and rebuilt PDF. No new
third-party dependencies were introduced for these review-driven changes.
