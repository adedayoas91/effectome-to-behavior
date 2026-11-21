"""Explore fitted outgoing disconnections in a completed C. elegans rollout."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from numbers import Integral
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

import numpy as np
from threadpoolctl import threadpool_limits

from effectome.data_module.schema import NeuralRecording
from effectome.experiments.behavior_rollout_timing_kernels import fit_timing_mechanisms, timing_forecasts
from effectome.experiments.celegans_motif_readout import MotifReadout
from effectome.experiments.celegans_rollout_data import WormPlan
from effectome.experiments.manifold_validation import _verify_artifacts
from effectome.experiments.structural_mechanism import AdditiveMechanism, MechanismRolloutConfig
from effectome.workflows.celegans_rollout import (
    _collect_worm,
    _histories,
    _load_preflight,
    _probabilities,
    _read,
    _rollout_config,
)
from effectome.workflows.resume import content_fingerprint

REPO = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).with_name("celegans_explorer")
logger = logging.getLogger(__name__)


def _ordered_ids(values: Sequence[int]) -> tuple[int, ...]:
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in values):
        raise ValueError("ROI IDs must be nonnegative integers.")
    result = tuple(sorted(map(int, values)))
    if len(set(result)) != len(result):
        raise ValueError("Choose each ROI at most once.")
    return result


@dataclass(frozen=True)
class WormReplay:
    recording: NeuralRecording
    plan: WormPlan
    neural: np.ndarray
    readout: MotifReadout
    pooled: AdditiveMechanism
    config: MechanismRolloutConfig
    selected: np.ndarray
    saved_probabilities: np.ndarray

    @classmethod
    def from_case(
        cls,
        recording: NeuralRecording,
        plan: WormPlan,
        config: MechanismRolloutConfig,
        case: dict[str, Any],
    ) -> WormReplay:
        if not np.array_equal(case["origins"], plan.origins):
            raise ValueError("Saved forecasts do not match the held-out origins in preflight.")
        if case["motif_names"] != recording.metadata["state_names"]:
            raise ValueError("Saved forecasts do not match the recording's motif vocabulary.")
        selected = np.asarray(case["selected_local"])
        if selected.shape != plan.origins.shape or selected.dtype != np.bool_:
            raise ValueError("Saved mechanism choices do not match the held-out forecast grid.")
        readout = MotifReadout.from_dict(case["readout"])
        if readout.n_classes != len(case["motif_names"]):
            raise ValueError("The motif readout width does not match the forecast vocabulary.")
        scale = np.asarray(case["neural_scale"], dtype=float)
        mean = np.asarray(case["neural_mean"], dtype=float)
        if (
            scale.shape != mean.shape
            or mean.shape != (recording.n_neurons,)
            or not np.isfinite(mean).all()
            or not np.all(np.isfinite(scale) & (scale > 0))
        ):
            raise ValueError("The forecast artifact has invalid training scales.")
        neural = (recording.traces.T.astype(float) - mean) / scale
        saved = _probabilities(case["forecasts"]["selected"], (len(plan.origins), readout.n_classes))
        return cls(
            recording, plan, neural, readout, AdditiveMechanism(**case["pooled_fit"]), config, selected, saved
        )

    @property
    def motifs(self) -> list[str]:
        return self.recording.metadata["state_names"]

    def model_at(self, position: int, ridge_alpha: float) -> AdditiveMechanism:
        if self.selected[position]:
            origins = self.plan.origins[position : position + 1]
            observed_history = _histories(self.neural, origins, self.plan.history)
            return fit_timing_mechanisms(observed_history, self.pooled.support, ridge_alpha)[0]
        return self.pooled

    def forecast(
        self, model: AdditiveMechanism, origin: int, disconnected: tuple[int, ...] = ()
    ) -> np.ndarray:
        return timing_forecasts(
            (model,),
            self.neural[origin : origin + 1],
            np.array([origin]),
            self.readout,
            self.config,
            output_width=self.readout.n_classes,
            disconnected=disconnected,
        )[0]

    def metadata(self, index: int) -> dict[str, Any]:
        names = self.recording.metadata
        roi_ids = self.recording.neuron_ids.tolist()
        counts = self.pooled.support.sum(axis=1).tolist()
        neurons = [
            {"id": roi, "name": name, "raw_name": raw, "outgoing_edges": count}
            for roi, name, raw, count in zip(
                roi_ids,
                names["canonical_neuron_names"],
                names["raw_neuron_names"],
                counts,
                strict=True,
            )
        ]
        samples = self.plan.origins
        coords = self.readout.encode(self.neural[samples]).tolist()
        recording_motif = self.recording.behavior["motif"]
        origins = [
            {
                "sample": int(sample),
                "seconds": float(self.recording.time[sample]),
                "model": "local" if local else "pooled",
                "code": code,
                "observed_motif": int(recording_motif[sample]),
            }
            for sample, local, code in zip(samples, self.selected, coords, strict=True)
        ]
        return {
            "id": index,
            "label": self.recording.identity.recording_id,
            "motifs": self.motifs,
            "neurons": neurons,
            "origins": origins,
        }


class Explorer:
    """Read one run and replay its selected mechanism at held-out forecast origins."""

    def __init__(self, repo: Path, run_root: Path) -> None:
        self.repo = repo
        self.run_root = run_root
        self.preflight = _load_preflight(repo, run_root)
        self.settings = self.preflight["protocol"]["settings"]
        self.inventory = {
            item["worm_index"]: item
            for item in self.preflight["worms"]
            if (run_root / f"worm-{item['worm_index']}" / "forecasts.json").is_file()
        }
        if not self.inventory:
            raise FileNotFoundError(
                f"No completed forecasts in {run_root}. Run 01_fit_and_forecast.ipynb first."
            )
        self._worm = lru_cache(maxsize=5)(self._load_worm)
        self._factual = lru_cache(maxsize=32)(self._replay_selected)

    def describe(self) -> dict[str, Any]:
        return {
            "run_id": self.run_root.name,
            "worms": [
                {"id": index, "label": entry["recording_id"]} for index, entry in self.inventory.items()
            ],
        }

    def _load_worm(self, index: int) -> WormReplay:
        inventory = self.inventory.get(index)
        if inventory is None:
            raise ValueError("Choose a worm with completed forecasts in this run.")
        folder = self.run_root / f"worm-{index}"
        case = _read(folder / "forecasts.json", "01_fit_and_forecast.ipynb")
        if case["signature"] != content_fingerprint((self.preflight["signature"], inventory)):
            raise ValueError(f"Worm {index} forecasts belong to a different preflight.")
        _verify_artifacts(folder, case["artifacts"])
        recording, plan = _collect_worm(self.repo, inventory, self.preflight)
        return WormReplay.from_case(recording, plan, _rollout_config(plan, self.settings, index), case)

    def worm(self, index: int) -> dict[str, Any]:
        return self._worm(index).metadata(index)

    def _replay_selected(self, index: int, position: int) -> tuple[AdditiveMechanism, np.ndarray]:
        worm = self._worm(index)
        model = worm.model_at(position, self.settings["rollout"]["ridge_alpha"])
        factual = worm.forecast(model, int(worm.plan.origins[position]))
        if not np.allclose(factual, worm.saved_probabilities[position], rtol=0, atol=1e-10):
            raise RuntimeError("The selected mechanism could not reproduce its saved factual forecast.")
        return model, factual

    def intervene(self, index: int, origin: int, off: Sequence[int]) -> dict[str, Any]:
        if isinstance(origin, bool) or not isinstance(origin, Integral):
            raise ValueError("Choose an integer forecast origin from the held-out grid.")
        origin = int(origin)
        worm = self._worm(index)
        origins = worm.plan.origins
        position = int(np.searchsorted(origins, origin))
        if position == len(origins) or origins[position] != origin:
            raise ValueError("Choose a forecast origin from the held-out grid.")
        selected_ids = _ordered_ids(off)
        lookup = {int(roi): position for position, roi in enumerate(worm.recording.neuron_ids)}
        missing = set(selected_ids) - lookup.keys()
        if missing:
            raise ValueError(f"Recording-local ROI IDs not found: {sorted(missing)}")
        removed = tuple(lookup[roi] for roi in selected_ids)
        model, factual = self._factual(index, position)
        disconnected = worm.forecast(model, origin, removed) if removed else factual
        return {
            "worm": index,
            "origin": origin,
            "seconds": float(worm.recording.time[origin]),
            "model": "local" if worm.selected[position] else "pooled",
            "draws": worm.config.n_rollouts,
            "horizon_seconds": worm.config.horizon / worm.recording.fps,
            "target_seconds": worm.config.target_length / worm.recording.fps,
            "motifs": worm.motifs,
            "off": list(selected_ids),
            "factual": factual.tolist(),
            "disconnected": disconnected.tolist(),
            "delta_pp": (100 * (disconnected - factual)).tolist(),
            "removed_edges": int(model.support[list(removed), :].sum()),
        }


def _integer_query(query: dict[str, list[str]], name: str) -> int:
    values = query.get(name, [])
    if len(values) != 1 or not values[0].isascii() or not values[0].isdigit() or len(values[0]) > 12:
        raise ValueError(f"Choose a valid {name} integer.")
    return int(values[0])


def _off_query(query: dict[str, list[str]]) -> tuple[int, ...]:
    values = query.get("off", [""])
    if len(values) != 1:
        raise ValueError("Send one list of disconnected ROI IDs.")
    if not values[0]:
        return ()
    fields = values[0].split(",")
    if not all(field.isascii() and field.isdigit() and len(field) <= 12 for field in fields):
        raise ValueError("Use comma-separated integer ROI IDs.")
    return _ordered_ids(tuple(map(int, fields)))


def handler_for(explorer: Explorer) -> type[BaseHTTPRequestHandler]:
    """Serve the explorer and its read-only local JSON routes."""
    assets = {
        "/": ("text/html; charset=utf-8", (ASSETS / "index.html").read_bytes()),
        "/app.js": ("text/javascript; charset=utf-8", (ASSETS / "app.js").read_bytes()),
        "/app.css": ("text/css; charset=utf-8", (ASSETS / "app.css").read_bytes()),
    }

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: HTTPStatus, media: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", media)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
                "img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
            self._send(status, "application/json; charset=utf-8", json.dumps(body, allow_nan=False).encode())

        def do_GET(self) -> None:
            port = cast(HTTPServer, self.server).server_port
            if self.headers.get("Host") not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
                self._json(HTTPStatus.FORBIDDEN, {"error": "Open the explorer using the local address."})
                return
            try:
                request = urlsplit(self.path)
                if request.path in assets:
                    self._send(HTTPStatus.OK, *assets[request.path])
                elif request.path == "/api/run":
                    self._json(HTTPStatus.OK, explorer.describe())
                elif request.path == "/api/worm":
                    query = parse_qs(request.query, max_num_fields=3)
                    self._json(HTTPStatus.OK, explorer.worm(_integer_query(query, "worm")))
                elif request.path == "/api/intervene":
                    query = parse_qs(request.query, keep_blank_values=True, max_num_fields=3)
                    payload = explorer.intervene(
                        _integer_query(query, "worm"), _integer_query(query, "origin"), _off_query(query)
                    )
                    self._json(HTTPStatus.OK, payload)
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "Page not found."})
            except ValueError as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except (OSError, RuntimeError, KeyError) as error:
                logger.warning("Explorer request failed: %s", error)
                if not isinstance(error, (BrokenPipeError, ConnectionResetError)):
                    self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

        def log_message(self, format: str, *args: Any) -> None:
            logger.debug(format, *args)

    return Handler


def _default_run(repo: Path) -> Path:
    root = repo / "outputs" / "celegans-rollout"
    primary = root / "motif-rollout-01"
    verification = root / "verification-01"
    if any(primary.glob("worm-*/forecasts.json")) or not any(verification.glob("worm-*/forecasts.json")):
        return primary
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        help="directory containing preflight.json and worm forecasts",
    )
    parser.add_argument("--port", type=int, default=8765, help="local HTTP port (0 chooses a free port)")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        explorer = Explorer(REPO, (args.run_root or _default_run(REPO)).resolve())
        with (
            threadpool_limits(limits=1),
            HTTPServer(("127.0.0.1", args.port), handler_for(explorer)) as server,
        ):
            print(f"Open http://127.0.0.1:{server.server_port}/  ({explorer.run_root})", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
