# Behavior-prioritized alternatives: reference and claim audit

Verified 18 November 2026. This audit accompanies
[the alternatives review](behavior_prioritized_alternatives.md) and
[its bibliography](behavior_prioritized_alternatives_references.bib).
It adds two references to the previously verified eight-entry manifold
bibliography. Existing metadata/version decisions remain recorded in
[the original audit](learned_manifold_validation_citation_audit.md).

## New references

| Key | Verification and retrieval | Supported statement | Boundary |
|---|---|---|---|
| `le2018supervisedautoencoders` | Official [NeurIPS abstract](https://proceedings.neurips.cc/paper/2018/hash/2a38a4a9316c49e5a833517c45d31070-Abstract.html), [metadata JSON](https://proceedings.neurips.cc/paper_files/paper/2018/file/2a38a4a9316c49e5a833517c45d31070-Metadata.json), [paper](https://proceedings.neurips.cc/paper_files/paper/2018/file/2a38a4a9316c49e5a833517c45d31070-Paper.pdf), and downloadable [BibTeX](https://proceedings.neurips.cc/paper_files/paper/2018/file/2a38a4a9316c49e5a833517c45d31070-Bibtex.bib) agree | A supervised autoencoder learns prediction and reconstruction jointly; the paper derives a linear-model stability result under assumptions and evaluates nonlinear architectures empirically | No theorem for our outgoing disconnections, arbitrary nonlinear encoders, or shifted intervention distribution. Authors are **Lei Le, Andrew Patterson and Martha White**. The official BibTeX leaves pages empty; that empty field was removed rather than filled from memory. |
| `fukumizu2004sufficientdimension` | Official [JMLR record](https://www.jmlr.org/papers/v5/fukumizu04a.html) and [full paper](https://www.jmlr.org/papers/volume5/fukumizu04a/fukumizu04a.pdf), checked against the [author publication list](https://www.ism.ac.jp/~fukumizu/papers/research.html) and [coauthor-hosted paper](https://www.di.ens.fr/~fbach/fukumizu04a.pdf) | Supervised dimension reduction seeks a low-dimensional subspace retaining the statistical neural/behavior relationship; kernel covariance operators characterize a conditional-independence objective | Observational sufficiency is not interventional invariance. This is the JMLR **2004, 5:73–99** paper, not its 2003 technical report or the later 2009 article. No DOI was invented. |

## Programmatic provenance

The SAE BibTeX was downloaded verbatim from the official NeurIPS export using
`curl -fL`. Only its key was normalized and its empty page field removed. The
official JSON provides a second structured metadata record; the PDF validates
the objective and theorem scope.

The older JMLR page supplies `citation_title`, repeated `citation_author`,
`citation_publication_date`, `citation_journal_title`, `citation_volume`,
`citation_firstpage` and `citation_lastpage` HTML metadata. It exposes no usable
BibTeX export at the tested paths. A Python `HTMLParser` extracted those exact
fields and serialized them to BibTeX; title/authors/year/pages were checked
against the author record. This is a documented conversion of retrieved
publisher metadata, not a memory-generated citation. The primary landing-page
URL is retained. A DBLP fetch returned a bot-check HTML page and was not used.

## Reused sources and implementation references

- `sani2021psid`: publisher/previous audit supports preferentially extracting
  behaviorally relevant dynamics in a linear state-space framework. The issue
  year is 2021, although online publication was 2020.
- `schneider2023cebra`: the prior primary-source audit verifies behavior/time/
  hybrid modes. Supervision is explicit and does not establish causal semantics.
- `kumar2023bundle`: retain the pinned **v5, 8 October 2026** title and four-author
  list. Its theory covers complete state resetting with exact consistency and
  neural Markov assumptions; the supplied proof is discrete-state. Do not apply
  that theorem automatically to outgoing-mechanism deletions or finite-sample
  chart fits. The local implementation and the paper version are separate
  objects whose inference contracts must be checked.
- `jolliffe2016pca`, `pearl2010causalinference`,
  `scholkopf2021causalrepresentation` and `pedregosa2011sklearn` retain their
  original verified metadata and claim limits. They provide statistical,
  structural-causal or implementation background, not new positive evidence.
- The official [scikit-learn cross-decomposition documentation](https://scikit-learn.org/stable/modules/cross_decomposition.html)
  directly supports the PLS/PCA distinction and installed implementation option.
  The standard software citation is `pedregosa2011sklearn`. This is not presented
  as the original historical PLS paper or as a guarantee under intervention.

## Project-specific reasoning

The scalar-score identity, mean-logit endpoint requirement, full-neural causal
model/frozen measurement design and revised distributed-null warning are
project-specific mathematical/design arguments. They are not attributed to the
papers as established experimental results. The completed scalar replay and PCA
dimension sweep are project development diagnostics documented in the probe
bundle. Their results are not attributed to the cited papers. Proposed nonlinear
encoder experiments remain unexecuted; replayed scalar events are not independent
new samples or evidence for one-dimensional intrinsic behavioral geometry.

Validation: all ten bibliography keys are unique. A standalone Tectonic/BibTeX
document using `\\nocite{*}` compiled successfully; its artifacts are under
`/tmp/effectome-alternatives-bibcheck/`.

No paper was imported into an external library or sent to another service for
bibliography management. No model dependency was installed.
