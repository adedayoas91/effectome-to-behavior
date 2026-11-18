# Statistical appendix

This document exports frozen runner summaries and decisions without new fitting,
cutoff selection or alternative acceptance rules. The protocol and all exact
settings are in ../study-protocol.json; source and candidate ownership are recorded
in ../source-manifest.json. Raw independent-seed values are in seed-metrics.csv.

Primary analysis uses 20 positive seeds and 60 untouched null seeds, after 19
separate calibration-null seeds. Directions are first averaged within seed;
windows, candidates, trials and related method variants do not multiply n.
Known/empty-support controls and all unavailable/ineligible evidence remain explicit.
Means/SDs and 2,000-resample percentile intervals use seed means. Incomplete planned
sets receive no complete-set mean or interval. The separate available means are
conditional descriptive summaries, not replacements for the frozen complete-set gate.

Calibration orders global-null whole-seed maxima, with rank ceil((n+1)*.95), uses
strict exceedance and retains the prespecified 0.10 effect-size floor. It provides
marginal protection under exchangeability of the randomized scoring procedure,
not a 95% conditional guarantee for the realized cutoff. Fresh null-test exact
Clopper–Pearson bounds are per-method whole-seed events, not simultaneous coverage
or strong non-null FDR. Incomplete null sets receive no bound.

The 256/1024 sensitivity comparison pairs full fitting data, origins and oracle
probabilities within each positive seed. Native MAE sign tests exclude exact ties
and use one fixed four-method Holm family; missing contrasts remain p=1. They test
sign tendency, not mean magnitude, and cannot override primary acceptance gates.
Candidate group diagnostics in eligibility-diagnostics.json describe prespecified
interventions; they do not retrospectively replace the individual-ranking criterion.
Verified references appear in references.bib and the project citation audit.
