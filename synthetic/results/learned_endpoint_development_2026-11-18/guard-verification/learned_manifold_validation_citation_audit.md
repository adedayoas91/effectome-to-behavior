# Learned behavioral manifold validation: citation audit

Verified on **18 November 2026**. This audit accompanies
[learned_manifold_validation_references.bib](learned_manifold_validation_references.bib).
The references motivate a validation design; they are not evidence that the
present implementation has passed it. No experiment was run for this audit.

## Verified claims and their limits

| Key | Primary content checked | Supported use | Limit on the claim |
| --- | --- | --- | --- |
| `schneider2023cebra` | [Nature article](https://www.nature.com/articles/s41586-023-06031-6), especially “Joint behavioural and neural embeddings” and decoding/consistency evaluations | CEBRA offers behavior-supervised, time-based self-supervised, and hybrid contrastive embeddings. Behavioral information can shape a neural representation during training; held-out decoding and consistency provide relevant evaluations. | A behavior-supervised embedding must be described as supervised. Decoding or reproducible geometry does not by itself demonstrate correct responses to interventions. |
| `sani2021psid` | [Nature Neuroscience article](https://www.nature.com/articles/s41593-020-00733-0), model description and Extended Data Fig. 1 | PSID prioritizes behaviorally relevant neural dynamics. Parameters use neural and behavioral training data; the learned model extracts latent states and decodes behavior from neural test data alone. This motivates behavioral and dynamical evaluation beyond variance explained. | PSID is a linear state-space identification method. Its results do not establish a nonlinear chart's causal interpretation or validity after a mechanism change. |
| `kumar2023bundle` | [bioRxiv v5](https://www.biorxiv.org/content/10.1101/2023.08.08.551978v5), Fig. 1, definitions, Theorems 1–2 and proof footnote 2; [official full-text XML](https://www.biorxiv.org/content/early/2026/10/08/2023.08.08.551978.source.xml) | Joint behavioral/dynamical learning and conditional causal-consistency theory. | The exact theory scope is specified below; it is not a finite-sample guarantee. |
| `scholkopf2021causalrepresentation` | [Author/publisher record](https://is.mpg.de/ei/en/publications/scholkopfetal21) and [full arXiv text](https://arxiv.org/html/2102.11107v1), sections on interventions and learning causal variables | A statistical representation and a causally meaningful representation have different requirements. The review identifies well-defined interventions on coarse-grained variables as a substantive problem. | This is conceptual background, not a theorem validating our encoder or proving every learned manifold inadequate. |
| `pearl2010causalinference` | [Official PMLR paper and BibTeX](https://proceedings.mlr.press/v6/pearl10a.html), reused from the maintained driver-methodology bibliography | Structural causal models distinguish observational prediction from effects of an explicitly specified intervention. This supports defining the intervention and target outcome before comparing effects. | It does not equate disconnecting outgoing mechanisms with clamping activity, or give causal semantics to an arbitrary latent coordinate. |
| `jolliffe2016pca` | [Royal Society publisher record](https://doi.org/10.1098/rsta.2015.0202), metadata verified 18 November 2026; direct full-text fetch returned 403 | Standard PCA background for the explicit affine bridge. | Variance retention is not behavioral or causal validity; no theorem from this review is invoked. |
| `lloyd1982quantization` | [Original IEEE paper hosted by Stanford](https://web.stanford.edu/class/ee398a/handouts/papers/Lloyd%20-%20Least%20Squares%20Q%20in%20PCM.pdf), first-page authorship and original centroid/partition formulation | Classical least-squares centroid/partition background for the geometry control. | Author is **Stuart P. Lloyd**; some secondary search metadata incorrectly gives a different first name. This is not a behavioral-state discovery guarantee. |
| `pedregosa2011sklearn` | [Official JMLR article and author list](https://www.jmlr.org/papers/v12/pedregosa11a.html) | Implementation citation for the installed PCA, KMeans and logistic-regression software. | Software citation, not evidence for the new scientific endpoint claim. |

## BunDLe-Net v5: exact theory scope

The allowed intervention family is **complete state resetting**,
`I_X = {do(X_t = x): x in X}`: all neural-state components jointly. The
corresponding latent state is `tau(x)`. This is not a theorem about arbitrary
outgoing-mechanism deletion.

The dSCM has independent exogenous **vectors across times**; it does not state
within-time component independence. Theorem 1 requires
`Pa(X_{t+1}) = {X_t}`, a surjective `tau`, and exact distributional consistency
for every state:

`P(tau(X_{t+1}) = y | X_t = x) = P(Y_{t+1} = y | Y_t = tau(x))`.

This is fiber sufficiency/strong lumpability (our characterization). Theorem 2
uses commuting probability-transition operators for **any initial
distribution**, not merely training trajectories. No explicit positivity or
observational-support assumption is supplied; empirical verification outside
observed support therefore remains unresolved. Proof footnote 2 restricts the
given proof to discrete state spaces and omits a continuous proof. Small
empirical training loss does not establish these exact conditions.

## What follows for the proposed validation

The following are **design decisions for this project**, informed by the papers,
rather than results attributed to those papers:

1. Fit the neural encoder, preprocessing, and behavioral/state readout using only
   training data; select settings on separate validation data. A behavioral
   training label is legitimate supervision. The claim being tested is whether
   the resulting representation generalizes, not whether behavior appeared
   without labels. Keep an explicitly unsupervised or time-only comparator when
   making a claim about the value of behavioral supervision.
2. Freeze that entire chart before testing. Apply the same map to factual and
   intervention futures from both the simulator and fitted neural-space model.
   Relearning the map separately after each intervention would change the
   measurement alongside the dynamics and make a difference of endpoint
   distributions ambiguous. Cross-seed alignment, when needed, must use
   training/validation anchors rather than test intervention outcomes.
3. Separate two questions. Comparing model and simulator intervention effects
   through the **same chart** checks mechanism accuracy at a common endpoint.
   Comparing simulator effects through that chart with independently defined
   simulator behavioral effects checks whether the endpoint measures the
   behavior of interest. A chart can agree with itself while omitting the true
   behavioral change; the second comparison is essential.
4. Evaluate held-out natural trajectories and held-out intervention trajectories,
   including behavioral decoding, dynamical prediction/consistency, state
   coverage, and failure outside training support. Observational success alone
   is insufficient evidence for this intervention family. A diagnosis of poor
   support must remain visible in the reported result.
5. Retain the neural-space structural intervention for the first bridge. Frozen
   chart outcomes require no autonomous latent Markov model; that stronger
   claim needs separate sufficiency and intervention-consistency tests.

## Metadata retrieval and normalization

BibTeX was retrieved programmatically from Crossref's DOI transform service for
CEBRA, PSID, BunDLe-Net, and the causal-representation review. For example:
`https://api.crossref.org/works/10.1038/s41586-023-06031-6/transform/application/x-bibtex`.
The other retrieved DOIs are `10.1038/s41593-020-00733-0`,
`10.1101/2023.08.08.551978`, and `10.1109/JPROC.2021.3058954`.
Titles, authors, and publication details were checked against the primary
records linked above. Keys, URL schemes, page-range dashes, and TeX accents were
normalized without inventing publication fields. Pearl's maintained entry was
checked against the official publisher's BibTeX block.

- **PSID year:** Crossref returns the online publication date, 9 November 2020.
  The final publisher issue is *Nature Neuroscience* **24**(1), 140–149 (2021).
  The bibliography uses final-issue year 2021 and key `sani2021psid`.
- **BunDLe-Net version:** the [official bioRxiv version API](https://api.biorxiv.org/details/biorxiv/10.1101/2023.08.08.551978)
  lists v1 (12 August 2023), v2 (8 September 2023), v3 (5 October 2023), v4
  (15 April 2024), and v5 (8 October 2026). Versions 1–4 use *BunDLe-Net:
  Neuronal Manifold Learning Meets Behaviour*. Versions 2–4 include Mauricio
  Gonzalez-Soto as a fifth author. Version 5 has the new title and four authors:
  Kumar, Gilra, Meunier, and Grosse-Wentrup. The new entry pins v5 and includes
  its date in a note, while retaining Crossref's initial-preprint year 2023 and
  existing citation key. Do not merge the v5 title with the v2–v4 author list,
  or silently attribute v5 theory to the earlier implementation/reference.
- **Causal-representation review:** the journal title is *Toward Causal
  Representation Learning*; the arXiv version uses *Towards*. Journal metadata
  and DOI are used, with the author's verified spelling Schölkopf.

No new dependency or bibliography-management service was used. Existing
manuscript bibliography entries were left unchanged; this separate file pins
the references relevant to the new validation proposal.

Validation: all five keys are unique, braces and whitespace checks pass, and a
standalone Tectonic/BibTeX document citing all entries compiled successfully.
The temporary compilation artifacts remain outside the repository.
