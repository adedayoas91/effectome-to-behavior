# effectome

**From dynamic effectomes to behavior.** Infer time-varying causal/effective connectivity from
larval-zebrafish whole-brain calcium imaging, model how the connectivity matrices transition as
a Markov process over recurring states, detect evolving neural communities, and link them to a
behavioral manifold to find the connectivity dynamics that drive behavior.

See the parent workspace `../goals.md` for the scientific goals and `../manuscript/draft.md`
for the manuscript.

## Pipeline

```
0 preprocess → 1 windowing (shifted segments) → 2 connectivity (PCMCI+ / Granger / correlation)
→ 3 dynamics (connectivity-state clustering + Markov transitions)
→ 4 community (Leiden / Markov-stability / temporal-multilayer)
→ 5 manifold (CEBRA / BundDLe-Net / classical) → 6 linking (decoding, lead-lag, behavior↔community)
```

Each stage is a registry-backed module under `src/effectome/`, configured via a Hydra config
group under `conf/`, and exposed as a numbered entry point under `pipeline/`.

## Install

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"           # core + dev
uv pip install -e ".[full,dev]"      # + PCMCI, Leiden, CEBRA, UMAP (heavier)
```

## Quickstart (synthetic data — no real dataset needed)

```bash
# Generate synthetic neural data with a known ground-truth causal graph and run the full chain
python pipeline/00_preprocess.py   data=synthetic
python pipeline/01_connectivity.py connectivity=granger
python pipeline/02_dynamics.py
python pipeline/03_community.py    community=leiden
python pipeline/04_manifold.py     manifold=classical
python pipeline/05_linking.py
```

Each script writes typed artifacts under `outputs/` (gitignored) that the next stage consumes.

## Tests

```bash
pytest                 # unit + synthetic ground-truth recovery tests
ruff check . && mypy src/
```

## Design conventions

- Hydra + frozen-dataclass configs; `set_seed` everywhere for reproducibility.
- Factory + Registry per module — add a method by registering it, never by branching on strings.
- Files kept to 200–400 lines; type hints and docstrings throughout; logging not `print`.

## Layout

```
src/effectome/
├── data_module/     # loaders, preprocessing, windowing (shifted segments)
├── connectivity/    # correlation, Granger, PCMCI+  (ConnectivityFactory)
├── dynamics/        # graph-state clustering + Markov/HMM transitions
├── community/       # static / multiscale / temporal community detection
├── manifold/        # CEBRA / BundDLe-Net / classical behavioral manifolds
├── linking/         # alignment, decoding, lead-lag, behavior↔community stats
├── viz/             # plotting helpers
└── utils/           # seed, io, synthetic ground-truth generator
```
