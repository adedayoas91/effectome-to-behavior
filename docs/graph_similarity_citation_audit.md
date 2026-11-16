# Graph-similarity citation and attribution audit

Verified 16 November 2026 using the citation-only workflow. The accompanying
[bibliography](graph_similarity_references.bib) separates retrieved publication
records from API documentation. The scientific question is whether effectomes
recur in their neuron-specific wiring, their circuit topology, or both. The two
questions require distinct comparisons; this is a project design decision.

## Verified records and claim boundaries

| Key | Primary source and corroboration | Supported use | Limit |
|---|---|---|---|
| `shervashidze2011wl` | [JMLR publisher record](https://www.jmlr.org/papers/v12/shervashidze11a.html), [published PDF](https://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf), [author institutional record](https://www.biochem.mpg.de/8259425/graph-kernels), [DBLP metadata](https://dblp.org/rec/journals/jmlr/ShervashidzeSLMB11.html) | Iterative neighbourhood-label refinement and a kernel defined by counts of labels at successive refinement depths. | The article's graph definition is undirected. Separate incoming/outgoing multisets, physical-lag edge labels and the default refinement depth are explicit project extensions/settings. The citation does not establish intervention recovery or biological equivalence. |
| `hagberg2008networkx` | [NetworkX's citation recommendation](https://networkx.org/documentation/stable/#citing), [publisher's DOI redirect](https://conference.scipy.org/proceedings/scipy2008/paper_2/), [Crossref DOI](https://doi.org/10.25080/TCWV9851) | Attribution of the NetworkX software used for graph objects, exact matching and structural descriptors. | The 2008 software article does not certify the output of a later project-specific graph pipeline. |
| `networkxIsomorphismDocs` | [Official `is_isomorphic` reference](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.isomorphism.is_isomorphic.html), installed 3.6.1 docstring/source inspected | The API uses VF2 and accepts explicit node/edge attribute match functions, including directed and multigraph use. | Attributes are ignored when matching functions are omitted. The caller must preserve direction, lag labels and any identity/cell-type constraints explicitly. |
| `networkxTriadicCensusDocs` | [Official `triadic_census` reference](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.triads.triadic_census.html), installed 3.6.1 docstring/source inspected | A directed triad census has 16 structural categories. | Triad counts are graph descriptors, not unique identifiers or causal-effect estimates. Lag-specific counts and any union across lags must be labelled separately. |
| `scipyJaccardDocs` | [Official SciPy Jaccard definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html) | Boolean set encoding gives intersection/union similarity and its complementary distance. Joint absences do not increase similarity. | The project implements this definition over `(lag, source, target)` edges. The reporting and empty-state conventions below are declared separately; no historical Jaccard paper is invented. |
| `schubert2022kmedoids` | [JOSS publisher record](https://joss.theoj.org/papers/10.21105/joss.04183), [official package documentation](https://python-kmedoids.readthedocs.io/en/latest/#references), Crossref DOI metadata | Attribution of the existing Rust/Python k-medoids package. | The project wrapper uses `alternating`, not PAM, FasterPAM or silhouette optimisation. Citing the package does not imply that those other algorithms were executed. |
| `kmedoidsAlternatingDocs` | [Official alternating-algorithm reference](https://python-kmedoids.readthedocs.io/en/latest/#alternating-k-medoids-k-means-style), installed package metadata | Alternating assignment and medoid updates accept a precomputed distance matrix. | This is a heuristic, with no claim of a global minimum. The package documentation warns that it can produce poorer solutions than PAM on difficult data. |
| `bessa2026emeans` | [Versioned arXiv record](https://arxiv.org/abs/2611.09501v1), [v1 paper](https://arxiv.org/html/2611.09501v1), [authors' website and BibTeX](https://e-means.github.io/) | A mini-batch EMA optimization procedure for Euclidean k-means and its reported embedding-dataset experiments. | This is an arXiv preprint, submitted 6 November 2026. It does not define a graph distance or validate neural graph grouping, and its reported hardware speedups are not measurements of this project. |

## Definitions and project-specific choices

For binary directed lagged supports, let `E(G)` contain one tuple
`(physical_lag, source_id, target_id)` for each selected off-diagonal edge. For a
nonempty union, the primary similarity is
`J(G,H) = |E(G) intersect E(H)| / |E(G) union E(H)|` and grouping distance is
`1 - J(G,H)`. Neuron identity is retained through the registered graph schema.
Thus relabelling an initiating neuron can change the primary state even when
the unlabelled topology remains the same. This definition is compatible with
the [Boolean-set interpretation in SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html).

For two empty graphs, the project reports Jaccard similarity as unavailable
(`None`) because agreement on no detected edges is not positive recovery
evidence. Grouping needs a finite distance, so its declared empty/empty distance
is zero, and empty/nonempty distance is one. SciPy also defines zero distance
for two all-zero vectors. The project-specific distinction between an evidence
statement and a grouping convention must remain visible in reports.

The secondary topology view counts original and refined neighbourhood labels,
following Definition 4 and Equation (2) in
[Shervashidze et al.](https://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf),
Sections 2 and 3.2, printed pages 2543–2547. The project separates incoming and
outgoing lag-labelled neighbours. With feature-count vectors `phi(G)`, it uses
the normalized dot product `k = <phi(G),phi(H)> / (||phi(G)|| ||phi(H)||)` and
distance `sqrt(max(0, 2 - 2*k))`. This distance is the Euclidean distance between
the normalized feature vectors; the normalization is stated as the project's
choice, rather than a theorem about causal equivalence from the paper.

Section 2 of the WL paper explicitly explains that equal refined labels can
occur when non-isomorphic graphs have not been distinguished. A simple local
counterexample is one directed six-cycle versus two disjoint directed
three-cycles with uniform initial node labels: every node has one incoming and
one outgoing neighbour, so the refinement counts remain equal at every depth.
Their numbers of weak components differ, and NetworkX exact isomorphism returns
false. This constructed check demonstrates a limitation of the selected kernel;
it is not attributed as a specific example from the publication. A zero WL
distance, or matching hash, is therefore not an isomorphism certificate.

The choice to fit graph medoids on training windows, freeze held-out assignment,
retain density/edge-count controls, and compare identity and topology targets
separately is the project's evaluation protocol. None of the citations above
provides evidence that this pipeline improves transition-driver recovery. The
earlier failed confirmation and the separate mechanism-readout development
results retain their original identities.

## Scope of the user-supplied e-means paper

[Bessa et al., version 1](https://arxiv.org/html/2611.09501v1) optimize the
Euclidean squared-distance objective stated in Equation (1), using mini-batch
exponential moving-average counts and sums. This makes e-means a possible
future comparator for frozen vector representations of effectomes. Its
Euclidean means are not a drop-in optimizer for the primary edge-Jaccard
medoid objective; that applicability distinction is an inference from the two
objectives, rather than a claim made by the authors.

Section 3.3 reports fitted scaling laws and explicitly notes poorer suitability
for very low-dimensional or tiny datasets. Sections 4.1 and 4.3 report
experiments on large embedding datasets with specified CPU/GPU hardware. Those
results motivate a separately timed, training-owned baseline comparison; they
do not establish a speedup or grouping-quality improvement for this project.
Any future comparison must keep its feature transform, training/held-out split
and evaluation target fixed. No dependency or native estimator is replaced on
the strength of that citation. The versioned source remains the authority when
the unversioned arXiv record or implementation later changes.

## Retrieval, metadata normalization and software versions

Publication BibTeX was retrieved programmatically:

- WL: `https://www.jmlr.org/papers/v12/shervashidze11a.bib`.
- NetworkX: `https://api.crossref.org/works/10.25080%2FTCWV9851/transform/application/x-bibtex`.
- k-medoids: `https://api.crossref.org/works/10.21105%2Fjoss.04183/transform/application/x-bibtex`.
- e-means: `https://arxiv.org/bibtex/2611.09501`, checked against the version-1
  abstract/full text and the matching author-website BibTeX.

Keys were normalized to `shervashidze2011wl`, `hagberg2008networkx` and
`schubert2022kmedoids`; page dashes and HTTPS DOI URLs were normalized. Product
capitalization was protected for BibTeX. NetworkX's Crossref record gives June
2008, whereas its official citation recommendation states August 2008; the
month is omitted because the agreed year, authors, title, venue and pages
identify the same work. The optional month was also omitted from the JOSS
record. The WL publisher record is retained as volume 12, article number 77,
pages 2539–2561, year 2011; an older author-hosted draft with placeholder venue
metadata was not used.

The e-means key was normalized to `bessa2026emeans`. All seven authors and their
order were retained from the fetched arXiv record; accented names were encoded
with equivalent LaTeX accents. The URL was pinned to `2611.09501v1` and the
version/submission date was recorded explicitly. No journal/conference venue
or peer-review status was inferred. The arXiv landing page describes its DOI
as pending registration, so no DOI was added beyond the fetched BibTeX metadata.

Documentation entries were constructed from the retrieved page titles,
organisations, versions and access date; no publication year or DOI was
inferred for them. The online NetworkX stable pages identified themselves as
**3.7** on retrieval. The dependency lock and installed distribution identify
the evaluated package as **3.6.1**. Its `is_isomorphic` and `triadic_census`
docstrings/source were independently checked for the API claims above. No
dependency upgrade was made. The live SciPy manual identified itself as 1.18.0;
it supplies a mathematical definition, not a claim about the installed SciPy
version. The existing k-medoids distribution is pinned to **0.5.5**.

The old SciPy-conference BibTeX URL redirected to a nonresponsive host, so the
NetworkX publication record was obtained from Crossref using the DOI supplied
by the publisher's redirect page. The initial DBLP BibTeX request returned an
anti-bot HTML page and was not accepted as bibliographic data. Its web metadata
and the author's institutional record corroborate the successfully retrieved
JMLR record. These retrieval failures do not leave placeholder citations.

## Retrieval and installed-source hashes

SHA256 values identify the actual retrieved records. Raw downloads were kept in
task-local temporary files; normalized publication records and documentation
metadata are retained in the accompanying bibliography.

| Record | SHA256 |
|---|---|
| WL publisher BibTeX | `5650ee7fa4f2a32ecff493874d92cc4e6c7c43158f6b72d0320a07eea1e8b751` |
| WL publisher HTML | `442c134fc4fea7f3c48131f4632b931a516413af6b51d012be814c9c454ad998` |
| WL published PDF | `251455e12994924700a3931a0ba10cabb47dc1ad97690b5d05c2c55f94dfd8b5` |
| WL author institutional record | `5843186912e37bcd00438ccf62616eb96bddead44b045a439f378562cfd56ae5` |
| NetworkX Crossref BibTeX | `ff0b83454bba07396bb6cb49295bfe9303ec9f672006c4bf0e44bf5d7953d8a7` |
| NetworkX stable index/citation recommendation | `eb432f9eb9e6b35d214e1a3231067408dc0b2a6e544277c51c2d0a17d6ffd416` |
| NetworkX stable isomorphism documentation | `a8f65c891b60abc19b1f2579baddc541429a561fcad32ef3ac3e0f97652074a5` |
| NetworkX stable triadic-census documentation | `ee5633e48fdb7e37e676736eaca3fffc1a0e9c63c9349702a85284fbf399836b` |
| Installed 3.6.1 `isomorph.py` | `3b64d9b543def3d0ac671a0d721bfa17262c194efd916af0cb018505ea2792c1` |
| Installed 3.6.1 `triads.py` | `16a3ad0797e2e3493812803ff9b5754c1c229ac462840aab55d172367ab4867b` |
| k-medoids Crossref BibTeX | `9c145da4f04b147d60151bf999d3317336b7d734981f9fdb20f10634bc1a9c0d` |
| k-medoids JOSS publisher HTML | `04990eb6c2ca2a22636694845fe3684a5d14cb0c4d5fad740ff5dc448e1f8c78` |
| k-medoids package documentation | `337b6754508b31760f3461ab36f2ea3883dd1cef9a14531a8c2960e9a46ebc46` |
| SciPy Jaccard documentation | `57059b5d873f3326934e15ef4e74216370f277483f313f5812fce96713c2ae18` |
| e-means arXiv BibTeX | `dc87d05cc9449903a21a31ac9b009b55d42885fa5ad4a13dead8e793cfaf6d2a` |
| e-means version-1 HTML | `5b48bab8386b958d0afcbc46ddd7e819759877ce8f16cb6789d0a89cdc70b6fc` |
| e-means author website | `249d99bfc30c2d48b29bb4e9442702dbd90e59cee5f16bc27f15bd9fb841fa49` |

## Bibliography verification

All eight keys are unique and the bibliography parses and renders with the
repository's MDPI bibliography style. A standalone cached Tectonic build
produced a PDF containing all eight entries, including the accented e-means
author names, with no BibTeX errors or warnings. The narrow standalone page had
ordinary line-layout warnings; that check establishes reference syntax and
rendering, not final manuscript typography.
