# Isolated behavior-encoder runtime

The main package remains on Python 3.14. The continuous, pointwise
`bunddle_behavior` experiment trains in a separate Python 3.12 process and exports
numerical weights for frozen NumPy inference. The original categorical/window
`bunddle` component and its optional runtime are separate.

From the application repository:

```sh
uv sync --project tools/behavior-encoder --python 3.12 --locked
tools/behavior-encoder/.venv/bin/python tools/behavior-encoder/worker.py --probe
```

Intel macOS uses PyTorch 2.2.2, the last official Intel wheel family; see the
[official deprecation notice](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690).
Other platforms use PyTorch 2.9.0; Linux defaults to the official CPU index to
avoid installing an unused CUDA stack. NumPy 1.26.4 preserves compatibility with
the older Intel wheel. No package is substituted for the nonlinear method.

The worker accepts finite numerical NPZ inputs with pickle disabled, a JSON
configuration and a JSON output path. It uses whole-trial inner validation,
records selection and fixed-epoch refitting, and exports encoder/readout
parameters plus training and environment provenance. The main process checks
Torch/NumPy inference parity. An explicitly supplied compatible runtime can be
selected through the manifold configuration; its actual version is recorded.
Changing source, runtime or training settings requires a new run namespace.

This is a BunDLe-inspired continuous pointwise adaptation, not the faithful
categorical temporal-window reference. Latent dynamics are an auxiliary loss;
causal intervention simulation remains in the observed neural state space.
