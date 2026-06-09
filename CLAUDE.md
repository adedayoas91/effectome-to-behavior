# CLAUDE.md — effectome (code repo)

Engineering conventions for the `effectome` package. The scientific context lives in
`../goals.md` and `../CLAUDE.md` (the research workspace).

## Architecture in one picture

Six stages, each a registry-backed module + Hydra config group + numbered pipeline script.
Stages pass typed artifacts through `outputs/` so each can run and be cached independently.

| Stage | Module | Config group | Script | Key classes |
|-------|--------|--------------|--------|-------------|
| 0 Preprocess | `data_module` | `data/`, `preprocess/` | `pipeline/00_preprocess.py` | `NeuralRecording`, loaders |
| 1 Windowing  | `data_module.windowing` | `preprocess/` | (part of 00/01) | `Window`, `WindowedSegments` |
| 2 Connectivity | `connectivity` | `connectivity/` | `pipeline/01_connectivity.py` | `ConnectivityEstimator`, `ConnectivitySeries` |
| 3 Dynamics | `dynamics` | `states/` | `pipeline/02_dynamics.py` | `GraphStateModel`, `TransitionModel` |
| 4 Community | `community` | `community/` | `pipeline/03_community.py` | `CommunityDetector`, `CommunitySeries` |
| 5 Manifold | `manifold` | `manifold/` | `pipeline/04_manifold.py` | `ManifoldEmbedder` |
| 6 Linking | `linking` | `linking/` | `pipeline/05_linking.py` | decoders, lead-lag, behavior↔community |

## Registry pattern (follow exactly)

Every pluggable family exposes `register_X(name)` + `XFactory(name) -> Type`. Example:

```python
from effectome.connectivity import ConnectivityFactory, register_connectivity

@register_connectivity("granger")
class GrangerConnectivity(ConnectivityEstimator):
    def __init__(self, cfg: ConnectivityConfig): ...
    def estimate(self, segment: np.ndarray) -> np.ndarray: ...   # returns N×N matrix
```

The pipeline resolves the class via the factory using `cfg.connectivity.name`. **Never** add
`if name == "granger"` branches in pipeline code.

## Config rules

- Configs are **frozen dataclasses** (`@dataclass(frozen=True)`) mirrored by YAML in `conf/`.
- Model/estimator `__init__` takes a single typed `cfg` object.
- No hardcoded hyperparameters in code — everything flows from `conf/`.

## Reproducibility (non-negotiable)

- Call `effectome.utils.seed.set_seed(cfg.seed)` at the top of every pipeline script.
- Hydra writes the resolved config + overrides to `outputs/<date>/<time>/.hydra/`.
- Any behavioral claim requires a surrogate null (time-shuffle / phase-randomize /
  configuration-model) and cross-validation. See `linking/stats.py`.

## Data contracts

- `NeuralRecording`: `traces` (N×T float32), `time` (T,), `coords` (N×3 or None),
  `neuron_ids` (N,), `behavior` (dict[str, T-array]), `fps` (float).
- `ConnectivitySeries`: `matrices` (K×N×N), `window_starts` (K,), `method` (str), `directed` (bool).
- Keep these stable; downstream stages depend on them.

## Testing

- `utils/synthetic.py` generates multivariate time series from a **known** causal graph
  (linear VAR + nonlinear options) plus synthetic behavior. Use it for ground-truth recovery
  tests (connectivity AUROC, state recovery) so tests don't need the real dataset.
- Run `pytest`, `ruff check .`, `mypy src/` before committing.

## File-size discipline

Keep modules 200–400 lines. If a file grows past ~400, split by responsibility (e.g.
`connectivity/granger.py` vs `connectivity/pcmci.py`) rather than adding sections.
