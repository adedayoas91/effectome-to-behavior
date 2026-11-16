# Grouping effectomes by identity and topology

Decision: 16 November 2026. Development implementation, after the runnable
mechanism checkpoint `8ce5ec6` was pushed to `dev`. Implementation lives on
`feat/effectome-graph-similarity`; the remote simulation checkpoint stays fixed.

## Reason for the change

Communities describe groups of neurons inside one graph. Effectome states group
whole graphs across windows. A separate graph-comparison geometry lets us test
whether recurring states reflect the particular neurons involved or recurring
circuit architecture. The existing prospective signed-lag path uses Euclidean
K-means or a Gaussian mixture, regardless of the legacy metric request. New
similarities therefore need an explicit fit/transform path, not only another
distance function in the legacy metric table.

The primary view preserves recording-local neuron identity: compare directed
edges at their registered physical lags using support Jaccard distance. The
secondary view compares topology through normalized WL subtree-count kernels.
The directed, lag-labelled refinement is a project extension of the published
WL kernel framework; the original paper does not validate our time-series
states or causal interpretation. See [Shervashidze et al. (2011)](https://jmlr.org/papers/v12/shervashidze11a.html).
References and claim boundaries are recorded in the
[citation audit](graph_similarity_citation_audit.md) and
[verified bibliography](graph_similarity_references.bib).

## Implementation and verification plan

1. Add strict binary-support comparison helpers. Preserve source-to-target
   direction, lag labels and the declared diagonal-exclusion policy. Report
   empty/empty Jaccard as undefined; use distance zero only as an explicit
   grouping convention.
2. Reuse the installed k-medoids solver with training-owned graph medoids. Freeze
   medoids, physical lags and node schema for held-out assignments. Reject
   incompatible feature standardization, PCA, feature weights and mixtures.
3. Add the two similarities as optional candidates in nested graph-summary
   evaluation. Every candidate predicts the same future signed-lag feature
   response, in training-fitted coordinates, against constant and continuous
   baselines. Inner data select the method and state count; outer data evaluate.
4. Add exact directed, attributed isomorphism and directed degree, reciprocity
   and triad descriptors as diagnostics. An identical WL signature is not proof
   of isomorphism; six-cycle/two-three-cycle controls must demonstrate this.
5. Provide notebook `07_graph_similarity.ipynb` and an inventory-first runner.
   Compare neuron-distinct chains, direction reversals, lag changes and
   edge-count-preserving perturbations. Keep graph-sampling seeds separate from
   independent neural-process seeds and retain unavailable settings.
6. Verify frozen fitting under held-out mutation, schema failures, empty graphs,
   deterministic assignment, direction/lag sensitivity, legacy regressions,
   numeric outputs, lint and types. Record measured results after execution.

## Design boundaries

For supports E and F, Jaccard similarity is `|E intersection F| / |E union F|`;
each edge is the ordered tuple `(physical lag, source identity, target identity)`.
Joint absences do not increase similarity. The denominator is zero for two
empty graphs, which supplies no positive recovery evidence. New metrics accept
only finite binary support. They do not silently threshold signed weights or
reinterpret native dependence scores as coefficients.

Topology comparisons permit neuron relabelling and therefore can merge graphs
with different initiating neurons. They must remain separately named from the
identity-preserving state. Exact isomorphism is a diagnostic rather than the
default grouping rule because a single uncertain edge breaks an exact match.
Directed motifs and degrees are summaries, not unique graph identifiers.

Do not symmetrize directed effectomes merely to apply a symmetric Laplacian or
SPD distance. Do not select state counts from outer-test performance or use a
high silhouette as evidence of behavioural relevance. Similarity cannot recover
edges missed by the upstream CSL procedure, identify biological drivers on its
own, or overturn the completed failed driver study.

Default scientific analyses and old result archives retain their existing
identities. No new dependency or replacement discovery algorithm is introduced.

## Use and ownership

The native four-CSL adapters now carry actual ROI order from `make_windows`,
scope identifiers by dataset, recording, animal, session and segment, and register
the exported positive lag in seconds. Unknown dataset/recording identities remain
unregistered. This preserves within-recording identity without claiming
anatomical alignment across animals. Older/manual window artifacts without ROI
registration remain unregistered: regenerate them or supply a verified schema.
No missing identities are filled using an assumed neuron correspondence.

```python
from effectome.dynamics import GraphStateConfig, fit_graph_states

identity = fit_graph_states(
    series,
    GraphStateConfig(
        n_states=3, metric="edge_jaccard", assignment="kmedoids",
        feature_representation="binary_lag_offdiag", standardize=False,
        analysis_scope="prospective",
    ),
    fit_indices=training_indices,
)
heldout_labels = identity.transform(heldout_series)
```

Use `metric="directed_wl"` for a separately named topology view; its K need not
match the identity view. The example K is illustrative, not an empirically
selected scientific state count. `states=graph_similarity` adds both methods to
the existing nested predictive-compression candidate panel in pipeline stage 02.
The shared response remains signed-lag coordinates fitted on training graphs.
The selected response transform is retained as `result.response_model`; a
medoid model itself exposes assignments rather than fitted Euclidean features.

The installed alternating k-medoids heuristic is reused, with a seeded
farthest-first initialization to avoid starting several medoids in the same
zero-distance class. This initialization is a project choice and gives no
global optimum guarantee. A requested K above the distinguishable training
classes is explicitly unavailable. Pairwise training distances require
quadratic storage in the number of graphs; the initial benchmark is bounded.
See [Schubert and Lenssen (2022)](https://doi.org/10.21105/joss.04183) and the
[solver documentation](https://python-kmedoids.readthedocs.io/en/latest/).

## Fit of e-means to this methodology

The requested [Bessa et al. preprint, version 1](https://arxiv.org/html/2611.09501v1)
introduces EMA optimization of Euclidean centroid quantization. Its embedding
benchmarks and scaling laws do not establish performance for small graph-state
panels; Section 3.3 explicitly discusses tiny/low-dimensional limitations.
We therefore classify it as a **planned optimizer baseline**, separate from the
implemented graph-similarity measures. No e-means experiment or speedup has been
measured in this project.

| Effectome representation | Potential e-means role | Comparison needed |
|---|---|---|
| Identity-ordered, directed lagged edge vectors | Optimize a vector codebook | Lloyd and mini-batch K-means on the same vectors |
| Normalized directed WL count vectors | Optimize a topology-feature codebook | Identical vocabulary, normalization and training split |
| Precomputed Jaccard distances | Requires an additional embedding/approximation | Retain graph medoids as the direct distance-based method |

For a fair optimizer comparison, freeze the graph representation first. Fit
vocabulary, scaling and PCA inside training data; shuffle training rows only.
WL normalization must retain the norm contribution from features unseen during
training, rather than dropping and renormalizing them. Fit final codebooks before
assigning either training or held-out graphs. Select K and settings through inner
validation and evaluate the same outer response and constant/continuous baselines.
Report held-out quantization error, group stability, dead groups and behavioural
prediction separately from runtime. Match both data-pass and wall-clock budgets,
including initialization, final assignment and hardware startup/transfer costs.

The official implementation offers a CPU `NumpyEmeans` path and optional GPU
backends. [The inspected source pin](https://github.com/facebookresearch/faiss/tree/2b648f0605f2ef3d14c4705350f8a0b214463878/contrib/e_means)
documents recipe differences from paper v1, including scaling coefficients and
initial mass. Its `train()` returns last-pass assignments; use `assign()` against
the final centroids for a frozen comparison. An eventual experiment must record
the exact source, resolved recipe, precision and backend. Faiss is not installed
here; no new dependency was added. Python 3.14 incompatibility has not been
established. This candidate remains unevaluated rather than being replaced by a
local approximation of the published algorithm.

Lower vector distortion would support a compression result. Causal driver
recovery still depends on support discovery, the endpoint and intervention
validation. Profile the full pipeline before assuming codebook optimization is
its runtime bottleneck.

## First executed structural controls

The smoke profile executed all 36 declared contrasts on graph-sampling seeds
0, 1 and 2. Each panel has 24 training and 24 separately sampled held-out graphs,
six neurons, two physical lags and exactly two directed cross-neuron edges per
graph. The four generating archetypes comprise two neuron-disjoint chains, an
outward star and an inward star. Their topology target has three classes. The
rewiring stress replaces one edge with probability 0.25 independently per graph,
in both training and evaluation, while preserving edge count and lag.

The table reports mean held-out adjusted Rand index across the three sampling
seeds. All K settings were declared; the held-out scores do not select a winner.

| Geometry | K | Clean identity ARI | Clean topology ARI | Rewired identity ARI | Rewired topology ARI |
|---|---:|---:|---:|---:|---:|
| Edge Jaccard | 2 | 0.303 | 0.511 | 0.266 | 0.464 |
| Edge Jaccard | 3 | 0.685 | 0.425 | 0.394 | 0.418 |
| Edge Jaccard | 4 | 1.000 | 0.685 | 0.635 | 0.379 |
| Directed WL | 2 | 0.303 | 0.511 | 0.008 | -0.033 |
| Directed WL | 3 | 0.685 | 1.000 | 0.188 | 0.287 |
| Directed WL | 4 | Unavailable | Unavailable | 0.352 | 0.511 |

There were 33 available fits. Clean WL K=4 was unavailable for all three seeds:
the selected kernel distinguishes only three training graph classes. The noisy
graphs introduce additional patterns; their K=4 fits are retained without
reinterpreting them as a fourth clean topology class. The negative noisy WL K=2
score and degraded performance of both views remain in the report.

The two clean views recover their respective designed distinctions at the
corresponding K, which does not make their semantic targets interchangeable.
Diagnostic probes also separate inward/outward direction and physical lag.
The six-cycle/two-three-cycle pair has zero WL distance while exact isomorphism
is false, demonstrating the kernel's incomplete topology discrimination.

These are planted structural inputs, not CSL-estimated neural graphs or causal
driver trials. The sampling seeds provide no biological or intervention
replication, confidence claim or calibrated significance result. A separate
regression fixture verifies that nested medoid selection can predict a nonlinear
future graph response while preserving common response coordinates and frozen
outer assignment; that software check is not a scientific validation study.

Versioned settings, all seed rows, unavailable reasons, figures and source
provenance are in
[`synthetic/results/graph_similarity_2026-11-16/`](../synthetic/results/graph_similarity_2026-11-16/).
The final source fingerprint is recorded there. Reproduce with:

```bash
.venv/bin/python pipeline/16_graph_similarity.py --profile smoke
.venv/bin/python pipeline/16_graph_similarity.py --profile smoke --run \
  --output-dir outputs/graph-similarity/fresh-run
```
