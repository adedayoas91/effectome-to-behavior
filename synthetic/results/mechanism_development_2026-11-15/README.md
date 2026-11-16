# Mechanism-readout development evidence

This model-matched, fully observed control is separate from the completed failed
latent-bistable driver study. See `docs/mechanism_development_findings.md` for the
results, rationale, limits and commands; methodological citations are verified
in `docs/driver_methodology_citation_audit.md`.

- `smoke/`: one development seed, six nodes, the initial 256-sample discovery history.
- `known-support-three-seed/`: known/empty support across seeds20--22 and eight nodes.
- `history-256/` and `history-1024/`: the paired, all-four-native-method history
  comparison on identical six-node recordings, with cutoffs and target unchanged.
- `history-comparison.json`: confirms identical training inputs, origins and oracle
  events across history settings. The two settings are not independent replicates.
- Each subdirectory retains the report, protocol, descriptive summaries and a ZIP
  of per-lane cell records and native evidence. Extract that ZIP into its study
  directory to restore `cells/`.
- `implementation-source.zip`: the evaluated implementation, configs, dependency
  locks, history launcher and cited rationale. Restore it in a clean application
  directory before attempting exact score reproduction.
- `provenance.json`: implementation identity, original-study hash and checksums.

All four native lanes achieved top-k recall1 in the longer-history driver control;
the shorter-history failures remain. One null seed cannot establish calibration:
its one-sided95% exact upper bound is.95. Known-support recovery across three
seeds is also development evidence. No independent confirmation, biological
necessity, latent-state robustness or calibrated nomination claim is made.
