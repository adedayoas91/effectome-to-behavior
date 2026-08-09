"""Fingerprint-validated, atomic checkpoints for long-running analyses."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")
U = TypeVar("U")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str, field: str) -> str:
    if not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"{field} must contain only letters, digits, '.', '_' or '-'")
    return value


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))  # type: ignore[arg-type]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        digest = hashlib.sha256(np.ascontiguousarray(value).view(np.uint8).tobytes()).hexdigest()
        return {"array_sha256": digest, "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dependency_fingerprints(paths: Mapping[str, str | Path] | None) -> dict[str, dict[str, Any]]:
    fingerprints: dict[str, dict[str, Any]] = {}
    for name, raw_path in sorted((paths or {}).items()):
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint dependency is not a file: {path}")
        fingerprints[name] = {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": _file_digest(path),
        }
    return fingerprints


def _signature(config: Any, dependencies: Mapping[str, str | Path] | None) -> tuple[str, dict[str, Any]]:
    payload = {
        "config": _jsonable(config),
        "dependencies": _dependency_fingerprints(dependencies),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def content_fingerprint(value: Any) -> str:
    """Return a stable digest for arrays, dataclasses, mappings, and scalar config values."""
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_pickle(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary = Path(handle.name)
    os.replace(temporary, path)


@dataclass(frozen=True)
class StageRecord:
    stage: str
    status: str
    signature: str
    artifact_path: str
    started_at: str
    completed_at: str | None = None
    error: str | None = None
    reused: bool = False


class ResumableRun:
    """Checkpoint namespace for one dataset/run/method combination.

    A completed artifact is reused only when its configuration and every declared
    input file have the same content fingerprint. Separate method namespaces avoid
    write contention when notebooks run in parallel.
    """

    def __init__(self, root: str | Path, run_id: str, method: str) -> None:
        self.root = Path(root).resolve()
        self.run_id = _safe_name(run_id, "run_id")
        self.method = _safe_name(method, "method")
        self.run_dir = self.root / self.run_id / self.method
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.last_record: StageRecord | None = None

    def _paths(self, stage: str) -> tuple[Path, Path, Path]:
        safe_stage = _safe_name(stage, "stage")
        stage_dir = self.run_dir / "stages" / safe_stage
        return stage_dir / "artifact.pkl", stage_dir / "status.json", stage_dir / ".lock"

    def _load_record(self, status_path: Path) -> dict[str, Any] | None:
        if not status_path.is_file():
            return None
        with status_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def completed_stages(self) -> list[str]:
        stages_dir = self.run_dir / "stages"
        if not stages_dir.exists():
            return []
        completed: list[str] = []
        for status_path in stages_dir.glob("*/status.json"):
            record = self._load_record(status_path)
            if record and record.get("status") == "completed":
                completed.append(status_path.parent.name)
        return sorted(completed)

    def execute(
        self,
        stage: str,
        compute: Callable[[], T],
        *,
        config: Any,
        dependencies: Mapping[str, str | Path] | None = None,
        force: bool = False,
    ) -> T:
        artifact_path, status_path, lock_path = self._paths(stage)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        signature, signature_payload = _signature(config, dependencies)
        existing = self._load_record(status_path)
        if (
            not force
            and existing is not None
            and existing.get("status") == "completed"
            and existing.get("signature") == signature
            and artifact_path.is_file()
        ):
            with artifact_path.open("rb") as handle:
                value = pickle.load(handle)
            self.last_record = StageRecord(
                stage=stage,
                status="completed",
                signature=signature,
                artifact_path=str(artifact_path),
                started_at=str(existing["started_at"]),
                completed_at=existing.get("completed_at"),
                reused=True,
            )
            return value

        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(
                f"stage '{stage}' is already running in {self.run_id}/{self.method}; "
                "remove the lock only after confirming that process has stopped"
            ) from exc
        started_at = _utc_now()
        try:
            os.write(descriptor, f"pid={os.getpid()} started_at={started_at}\n".encode())
            os.close(descriptor)
            running = {
                "stage": stage,
                "status": "running",
                "signature": signature,
                "signature_payload": signature_payload,
                "artifact_path": str(artifact_path),
                "started_at": started_at,
                "completed_at": None,
                "error": None,
            }
            _atomic_json(running, status_path)
            value = compute()
            _atomic_pickle(value, artifact_path)
            completed_at = _utc_now()
            complete = {**running, "status": "completed", "completed_at": completed_at}
            _atomic_json(complete, status_path)
            self.last_record = StageRecord(
                stage=stage,
                status="completed",
                signature=signature,
                artifact_path=str(artifact_path),
                started_at=started_at,
                completed_at=completed_at,
                reused=False,
            )
            return value
        except Exception as exc:
            completed_at = _utc_now()
            error = f"{type(exc).__name__}: {exc}"
            failed = {
                "stage": stage,
                "status": "failed",
                "signature": signature,
                "signature_payload": signature_payload,
                "artifact_path": str(artifact_path),
                "started_at": started_at,
                "completed_at": completed_at,
                "error": error,
            }
            _atomic_json(failed, status_path)
            self.last_record = StageRecord(
                stage=stage,
                status="failed",
                signature=signature,
                artifact_path=str(artifact_path),
                started_at=started_at,
                completed_at=completed_at,
                error=error,
            )
            raise
        finally:
            lock_path.unlink(missing_ok=True)

    def execute_chunks(
        self,
        stage: str,
        chunks: Sequence[U],
        compute_chunk: Callable[[U], T],
        combine: Callable[[list[T]], T],
        *,
        config: Any,
        dependencies: Mapping[str, str | Path] | None = None,
        force: bool = False,
    ) -> T:
        """Checkpoint each chunk independently, then checkpoint their combination."""
        results: list[T] = []
        chunk_artifacts: dict[str, Path] = {}
        for index, chunk in enumerate(chunks):
            chunk_stage = f"{stage}.chunk-{index:05d}"

            def compute_current(current: U = chunk) -> T:
                return compute_chunk(current)

            result = self.execute(
                chunk_stage,
                compute_current,
                config={"stage_config": config, "chunk_index": index, "chunk": _jsonable(chunk)},
                dependencies=dependencies,
                force=force,
            )
            results.append(result)
            chunk_artifacts[f"chunk_{index:05d}"] = self._paths(chunk_stage)[0]
        return self.execute(
            stage,
            lambda: combine(results),
            config={"stage_config": config, "n_chunks": len(chunks)},
            dependencies=chunk_artifacts,
            force=force,
        )


__all__ = ["ResumableRun", "StageRecord", "content_fingerprint"]
