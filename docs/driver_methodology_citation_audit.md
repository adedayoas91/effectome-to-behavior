# Citation and rationale audit: driver-methodology revision

Verified 15 November 2026. This audit separates bibliographic verification,
external methodological support, local numerical evidence and new design choices.
No citation is used to imply that the revised implementation has passed recovery
or that its time-series intervention effects are automatically identifiable.

## Verified references and supported claims

| Key | Primary record and independent corroboration | Claim supported here | Limit on use |
|---|---|---|---|
| `williamson2023importance` | [Author manuscript/arXiv](https://arxiv.org/abs/2004.03683), [Crossref DOI](https://doi.org/10.1080/01621459.2021.2003200), [PubMed 37982008](https://pubmed.ncbi.nlm.nih.gov/37982008/) | Variable importance can be defined through a contrast in predictive performance with and without features. | This does not equate the project's graph-only omission score with an intervention effect; the project has not implemented the paper's full inference procedure. |
| `pearl2010causalinference` | [PMLR publisher record](https://proceedings.mlr.press/v6/pearl10a.html), [author's UCLA bibliography](https://bayes.cs.ucla.edu/csl_papers.html), [original proceedings PDF](https://jmlr.csail.mit.edu/proceedings/papers/v6/pearl10a/pearl10a.pdf) | Structural assignments, assumptions and intervention operators are needed to define causal intervention queries. | The proposed outgoing-edge operation must be defined explicitly; it is not a validated biological silencing intervention. |
| `peters2014additive` | [JMLR publisher record](https://jmlr.org/papers/v15/peters14a.html), [author manuscript/arXiv](https://arxiv.org/abs/1309.6779) | Additive-noise structural models illustrate how functional/noise assumptions affect causal identifiability. | The iid DAG results are not a theorem guaranteeing identifiability, finite-sample recovery or robustness of this time-series readout. |
| `cawley2010overfitting` | [JMLR publisher record](https://www.jmlr.org/papers/v11/cawley10a.html), [DBLP record](https://dblp.org/rec/journals/jmlr/CawleyT10.html) | Optimizing a noisy selection criterion can overfit model selection and bias subsequent performance assessment. | The chronological whole-trial inner split is the project's design choice for independent trials, not a specific procedure claimed from this paper. |
| `hoerl1970ridge` | [Publisher/DOI](https://doi.org/10.1080/00401706.1970.10488634), Crossref metadata, [original article copy](https://homepages.math.uic.edu/~lreyzin/papers/ridge.pdf) | Ridge regularization estimates coefficients with an explicit quadratic penalty. | The project's fixed nonlinear basis and structural interpretation are additional assumptions; Ridge alone does not identify causal mechanisms. |
| `clopper1934confidence` | [Oxford publisher issue](https://academic.oup.com/biomet/issue/26/4?browseBy=volume), [Crossref/publisher DOI](https://doi.org/10.1093/biomet/26.4.404), [original article scan](https://dornsife.usc.edu/sergey-lototsky/wp-content/uploads/sites/211/2025/06/Clopper-Pearson-Original.pdf) | Binomial-tail inversion gives exact frequentist confidence limits for a binomial event probability. | The one-sided bound is per method and independent null seed under a fixed nomination rule, not neuron-level FDR or simultaneous coverage across methods. |

No additional algorithm is attributed to these papers. The `tanh` mechanism,
known-support diagnostic sequence, empty-support check, shared random-number
construction and planned nomination calibration are explicit project design
choices. They are evaluated on their own terms rather than presented as results
proved by the cited literature. The rollout documentation must specify how its
paired innovations and target events are computed, so their meaning can be
checked without relying on a name or citation.

## Retrieval and metadata corrections

The BibTeX entries were retrieved programmatically, not reconstructed from memory:

- Williamson: Crossref DOI transformation endpoint
  `https://api.crossref.org/works/10.1080%2F01621459.2021.2003200/transform/application/x-bibtex`.
- Pearl: the publisher's `code#bibtex` block at the PMLR record above.
- Peters: `https://jmlr.org/papers/v15/peters14a.bib`.
- Cawley: `https://www.jmlr.org/papers/v11/cawley10a.bib`.
- Hoerl: Crossref DOI transformation endpoint
  `https://api.crossref.org/works/10.1080%2F00401706.1970.10488634/transform/application/x-bibtex`.
- Clopper--Pearson: Crossref DOI transformation endpoint
  `https://api.crossref.org/works/10.1093%2Fbiomet%2F26.4.404/transform/application/x-bibtex`.

Citation keys were normalized to the manuscript's author/year style. Unicode
page dashes were normalized to BibTeX `--`; Pearl's editor-name accent was encoded
for LaTeX. Clopper--Pearson's all-capital author/title strings were changed to
ordinary capitalization without changing names or words. Other bibliographic
content was retained from the retrieved records.

The Williamson Crossref export uses **2022**, its online-publication year. The
arXiv journal reference and PubMed record identify the final issue citation as
**2023, volume 118, issue 543, pages 1645--1658**; PubMed separately records the
article date as 5 January 2022. The manuscript and revision bibliography use the
final issue year **2023**, with the same DOI. This is a documented metadata
correction, not a second paper or an unverified year inferred from the volume.

SHA256 hashes of the retrieved source records:

| Retrieved record | SHA256 |
|---|---|
| Williamson Crossref BibTeX | `a13ec10acde3b6d3595ed8fc60ba196d31c4397a2fd8b8ecd3604ff55f46f951` |
| Pearl publisher HTML | `d9f676d444e830b97f5207561a1b4c5007123abb3f32a4de54de5d232d480ec4` |
| Peters publisher BibTeX | `6652d4b3b88647039abae2cff7bb5b926cc9cbdc63dd67645d822dcca95d8260` |
| Cawley publisher BibTeX | `9b5887c228d55fe8991f9d9354d3a8bfbceb5fd8888ef23052ebc5825e31ea8d` |
| Hoerl Crossref BibTeX | `067a13a09a5255eed11cb1a4b75cfdf4850899c78423aa488e8aa8ba63e47150` |
| Clopper--Pearson Crossref BibTeX | `700d5ecf1f9eb49f47222d67b564b7bf4bd828dbbe9c6c531ac5b1ea13338c8b` |
| Clopper--Pearson original article scan | `758de703e7d5d60c353396d24b65b8f2e535b8bc4027575c01d2e323cc5cab72` |

The raw records were downloaded to task-local temporary files; the normalized
records are retained in `driver_methodology_references.bib` and the manuscript
bibliography. There are no unverified citation placeholders in this revision.

## Exact null bound and method attribution

Clopper--Pearson's original discussion in Section 1 inverts binomial tails;
the note accompanying the confidence belts explicitly permits noncentral
intervals. The implementation uses the modern beta-quantile expression of a
one-sided upper limit: for `k<n`, `U=BetaQuantile(.95;k+1,n-k)` solves
`P[Binomial(n,U)<=k]=.05`; at `k=n`, `U=1`. This equivalence concerns the
binomial confidence procedure, not a Bayesian prior/posterior interpretation.
No exact bound is reported for an incomplete planned null seed set.

The data unit is one independent raw-process null seed with any false nominee
across that method's units/directions. Its binomial sampling assumptions and the
need to fix the nomination rule remain explicit. Neither an empty-support result
nor zero false nominees in a tiny development sample establishes a calibrated
causal discovery or driver-identification procedure.

Native discovery attribution is retained through the project's
[package installation/provenance record](../README.md#install),
[conventional runtime guide](../tools/conventional-cgc/README.md), and
[four-CSL migration record](four_csl_migration.md). The manuscript reuses its
existing `adedayo2025cgc` and `adedayo2026fastcgc` method entries; this revision
does not create substitute citations or claim a new discovery implementation.
The present citation pass verifies the six methodological/statistical references
above; the existing native-method provenance remains linked to its maintained
source and dependency pins.

## Project-specific evidence and inferences

The completed-study claims come from
`synthetic/results/driver_confirmation_v2/driver-validation.json`, its
`provenance.json` and `conclusion.md`, rather than from an external publication.
The report contains 560 completed/available method-recording runs and failed
acceptance decisions for all four native methods. It remains unchanged.

The 64-training-origin count, 48 baseline columns, 112 graph columns, zero graph
changes between independent trial origins, graph-only omissions and fixed
decoder penalty are implementation observations in `shared_transitions.py`,
`transition_synthesis.py`, `transition_prediction.py` and the scheduled-v2 config.
They motivate possible failure mechanisms. The claim that any one caused the
negative result is **not established**; the known-support comparison is intended
to investigate that distinction.

The favorable control's directly observed endpoint is narrower than the older
learned-manifold/bistable benchmark. No success there may be substituted for the
failed study, a calibrated candidate false-positive bound, hidden-confounder
identification or biological necessity. New confirmation requires a separately
frozen protocol and previously unused independent simulation seeds.

For the MDPI BibTeX style, Hoerl’s retrieved `Feb` macro is expanded to
`{February}`. Williamson’s online `Jan` field is omitted from the final-issue
2023 citation to avoid conflating online-publication timing with the issue date.
The source-record checksums above still refer to the unmodified retrieved records.

The optional publisher abstract field is omitted from the normalized Pearl
BibTeX entry; bibliographic metadata and the verification record are retained.
