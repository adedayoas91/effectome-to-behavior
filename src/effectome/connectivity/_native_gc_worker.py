"""Standalone transport for the official conventional GC package (no effectome imports).

Run with the Python environment pinned under tools/conventional-cgc. All windows
share one process to amortize the upstream Numba compiler startup. No causal
structure algorithm is implemented here: support is the native graph's nonzeros.
"""

from __future__ import annotations

import importlib
import json
import sys
import traceback
from contextlib import redirect_stdout
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np


def run_job(input_path: Path, output_path: Path, metadata_path: Path) -> None:
    with np.load(input_path, allow_pickle=False) as payload:
        segments = np.asarray(payload["segments"], dtype=np.float64)
        options = json.loads(str(payload["options"]))
    module_name = options["upstream_module"]
    if module_name not in {"causalized_gc", "d_CSL"}:
        raise ValueError("conventional upstream_module must be causalized_gc or d_CSL")
    module = importlib.import_module(module_name)
    provenance = {
        "module": module_name, "module_file": module.__file__,
        "runtime_python": sys.executable, "runtime_version": sys.version,
    }
    distribution = "causalized-gc" if module_name == "causalized_gc" else "gc-extension"
    installed = metadata.distribution(distribution)
    provenance.update({"distribution": distribution, "version": installed.version})
    direct_url = installed.read_text("direct_url.json")
    if direct_url:
        provenance["direct_url"] = json.loads(direct_url)
    output: dict[str, Any] = {}
    matrices, windows = [], []
    for index, segment in enumerate(segments):
        seed = options["seed"] + index
        np.random.seed(seed)
        kwargs = {
            "n_perm": options["n_perm"], "n_pasts": 1, "n_lags": 1,
            "temporal": True, "method": options["method"],
        }
        if module_name == "causalized_gc":
            kwargs.update({"backend": "cpu", "random_state": seed})
        elif kwargs["method"] == "cgc_star":
            kwargs["method"] = "fcgc"
        estimator = module.GcStar(**kwargs)
        estimator.fit(segment.T, verbose=options["verbose"])
        native = np.asarray(estimator.get_connectivity_matrix(
            simulation=True, alpha=options["alpha"], beta=options["beta"],
        ), dtype=np.float64)
        if native.shape != (segment.shape[1], segment.shape[1]) or not np.isfinite(native).all():
            raise RuntimeError("conventional native graph has invalid shape or nonfinite weights")
        support = (native != 0).astype(np.float64)
        np.fill_diagonal(support, 0.0)
        matrices.append(support)
        native_key = f"window_{index}__native_matrix"
        output[native_key] = native
        evidence_keys = {"native_matrix": native_key}
        for name in ("corr_", "pVal_corr_", "inv_corr_", "pVal_inv_corr_", "conn_mat"):
            value = getattr(estimator, name, None)
            if value is not None:
                key = f"window_{index}__{name}"
                output[key] = np.asarray(value)
                evidence_keys[name] = key
        windows.append({
            "evidence_keys": evidence_keys,
            "requested_seed": seed,
            "seed_policy": "requested_only; upstream_CPU_threaded_NumPy/Numba_rng_not_controlled",
            "support_extraction": "native_graph_nonzeros; simulation=True excludes tau0; n_lags=1",
        })
    output["matrices"] = np.stack(matrices)
    np.savez(output_path, **output)
    metadata_path.write_text(json.dumps({"upstream_provenance": provenance, "windows": windows}))


def main() -> None:
    if sys.argv[1:] != ["--serve"]:
        run_job(*map(Path, sys.argv[1:]))
        return
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("shutdown"):
                return
            with redirect_stdout(sys.stderr):
                run_job(*(Path(request[key]) for key in ("input", "output", "metadata")))
            response: dict[str, Any] = {"ok": True}
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            response = {"ok": False, "error_type": type(error).__name__, "error": str(error)}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
