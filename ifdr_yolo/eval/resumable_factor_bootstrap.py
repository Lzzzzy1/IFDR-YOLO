"""Checkpointed, deterministic F0-relative factor bootstrap execution.

The formal factor-repair statistic is intentionally kept in
``factor_repair_gate``.  This module only owns execution state: it prepares a
shared F0 draw cache, evaluates one candidate in deterministic blocks, and
persists enough identity to reject an unsafe resume before any new work.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time
from typing import Any, TextIO

import numpy as np

from ifdr_yolo.eval import factor_repair_evidence as _evidence_module
from ifdr_yolo.eval import factor_repair_gate as _gate_module
from ifdr_yolo.eval.factor_repair_gate import (
    FACTOR_GATE_BOOTSTRAP_PERCENTILES,
    FACTOR_GATE_BOOTSTRAP_REPLICATES,
    FACTOR_GATE_BOOTSTRAP_SEED,
    PRIMARY_ENDPOINTS,
    FactorRepairEvidence,
    PairedDelta,
    composite_mechanism_score,
    paired_image_cluster_replicate,
    paired_resample_indices,
    recompute_endpoints,
)


CHECKPOINT_SCHEMA = "resumable-factor-bootstrap/v1"
REFERENCE_CACHE_CONDITION = "F0"
DEFAULT_CHECKPOINT_INTERVAL = 100
MAX_CHECKPOINT_WALL_SECONDS = 300.0
DEFAULT_PROGRESS_FILENAME = "progress.json"
DEFAULT_REFERENCE_CHECKPOINT = "F0.reference.json"
_ESTIMAND = "candidate_minus_F0_composite_four_registered_endpoints"
_RNG_SCHEME = "sha256-derived-default_rng-index-draw-v1"


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=repr)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    return repr(value)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"unable to hash code identity file: {path}") from error


def _source_hash(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a 64-hex SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a 64-hex SHA256") from error
    return value.lower()


def _module_path(module: object) -> Path:
    source = inspect.getsourcefile(module)
    if source is None:
        raise ValueError(f"unable to resolve code identity for {module!r}")
    return Path(source).resolve(strict=True)


def _code_hashes(extra_paths: Sequence[Path] = ()) -> dict[str, str]:
    paths = {
        "runner": _module_path(__import__(__name__, fromlist=["_module_path"])),
        "factor_repair_gate": _module_path(_gate_module),
        "factor_repair_evidence": _module_path(_evidence_module),
    }
    for index, path in enumerate(extra_paths):
        resolved = Path(path).expanduser().resolve(strict=True)
        paths[f"extra_{index}"] = resolved
    return {name: _file_digest(path) for name, path in sorted(paths.items())}


def _atomic_write(path: Path, payload: bytes, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace and (path.exists() or path.is_symlink()):
        raise ValueError(f"refusing to overwrite existing checkpoint: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            finally:
                os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_checkpoint(path: Path, payload: Mapping[str, object], *, initial: bool = False) -> None:
    encoded = _canonical_bytes(payload) + b"\n"
    # Validate the temporary bytes before replacing the visible checkpoint.
    # This keeps a serialization/identity regression from publishing a file
    # that cannot be resumed, while preserving atomic replacement semantics.
    try:
        decoded = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint serialization validation failed") from error
    if not isinstance(decoded, dict) or not isinstance(decoded.get("identity"), dict):
        raise ValueError("checkpoint serialization validation failed")
    if decoded.get("identity_sha256") != _digest(decoded["identity"]):
        raise ValueError("checkpoint identity digest validation failed")
    completed = decoded.get("completed")
    replicates = decoded.get("replicates")
    if not isinstance(completed, int) or isinstance(completed, bool) or not isinstance(replicates, list) or completed != len(replicates):
        raise ValueError("checkpoint serialization replicate validation failed")
    _atomic_write(path, encoded, replace=not initial)


def _write_progress(path: Path, payload: Mapping[str, object]) -> None:
    # Progress is observational state, so replacement is intentional.  The
    # checkpoint itself remains the source of truth for resume.
    _atomic_write(path, _canonical_bytes(payload) + b"\n", replace=True)


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is missing or malformed: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _safe_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _coerce_endpoint_draw(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("reference endpoint draw must be a mapping")
    result: dict[str, float] = {}
    for name in PRIMARY_ENDPOINTS:
        if name not in value:
            raise ValueError(f"reference endpoint draw is missing {name}")
        result[name] = _safe_float(value[name], f"reference endpoint {name}")
    # Validate the registered bounds and endpoint set through the production
    # composite helper; no alternate statistic is introduced here.
    composite_mechanism_score(result)
    return result


def _evidence_attr(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _evidence_identity(
    candidate: object,
    reference: object,
    *,
    source_file_sha256: Mapping[str, str | None] | None = None,
) -> dict[str, object]:
    if source_file_sha256 is not None and not {"candidate", "reference"}.issubset(source_file_sha256):
        raise ValueError("source_file_sha256 must include candidate and reference")
    candidate_hash = str(_evidence_attr(candidate, "evidence_sha256", ""))
    reference_hash = str(_evidence_attr(reference, "evidence_sha256", ""))
    condition = str(_evidence_attr(candidate, "condition", ""))
    reference_condition = str(_evidence_attr(reference, "condition", ""))
    image_ids_raw = _evidence_attr(reference, "image_ids", ())
    candidate_ids_raw = _evidence_attr(candidate, "image_ids", ())
    image_ids = tuple(sorted(str(item) for item in image_ids_raw))
    candidate_ids = tuple(sorted(str(item) for item in candidate_ids_raw))
    image_hash = str(_evidence_attr(reference, "image_ids_hash", ""))
    candidate_image_hash = str(_evidence_attr(candidate, "image_ids_hash", ""))
    if not image_ids or image_ids != candidate_ids or not image_hash or image_hash != candidate_image_hash:
        raise ValueError("candidate/F0 evidence image IDs mismatch")
    if not candidate_hash or not reference_hash:
        raise ValueError("candidate/F0 evidence hashes are required")
    if condition not in {"F1", "F2", "F3"}:
        raise ValueError("candidate condition must be F1, F2, or F3")
    if reference_condition != "F0":
        raise ValueError("reference evidence condition must be F0")
    candidate_checkpoint = _evidence_attr(candidate, "checkpoint", {})
    reference_checkpoint = _evidence_attr(reference, "checkpoint", {})
    # Evidence producers are expected to bind ``evidence_sha256`` to their
    # full canonical payload.  Include the observable input payload as a
    # second guard as well: small test/third-party records sometimes carry a
    # legacy placeholder hash, and resuming such a record must still fail
    # closed if an endpoint or raw-observation value changed.
    def input_digest(value: object, source_hash: str | None) -> str:
        if source_hash is not None:
            # The formal CLI already streamed the source bytes once.  Reusing
            # that identity avoids serializing a ~93 MB raw payload for every
            # candidate while retaining a fail-closed file hash guard.
            return _digest(
                {
                    "condition": _evidence_attr(value, "condition", ""),
                    "evidence_sha256": _evidence_attr(value, "evidence_sha256", ""),
                    "source_file_sha256": source_hash,
                }
            )
        return _digest(
            {
                "condition": _evidence_attr(value, "condition", ""),
                "image_ids": tuple(sorted(str(item) for item in (_evidence_attr(value, "image_ids", ()) or ()))),
                "image_ids_hash": _evidence_attr(value, "image_ids_hash", ""),
                "endpoints": _evidence_attr(value, "endpoints", {}),
                "raw_observations": _evidence_attr(value, "raw_observations", None),
                "endpoint_rows": _evidence_attr(value, "endpoint_rows", None),
            }
        )
    def checkpoint_hash(value: object) -> str | None:
        if isinstance(value, Mapping):
            raw = value.get("sha256", value.get("hash"))
            return None if raw is None else str(raw)
        return None
    source_hashes = {
        "candidate": _source_hash(
            None if source_file_sha256 is None else source_file_sha256.get("candidate"),
            "source_file_sha256.candidate",
        ),
        "reference": _source_hash(
            None if source_file_sha256 is None else source_file_sha256.get("reference"),
            "source_file_sha256.reference",
        ),
    }
    return {
        "candidate_condition": condition,
        "reference_condition": reference_condition,
        "candidate_evidence_sha256": candidate_hash,
        "reference_evidence_sha256": reference_hash,
        "evidence_canonical_sha256": {"candidate": candidate_hash, "reference": reference_hash},
        "source_file_sha256": source_hashes,
        "candidate_input_sha256": input_digest(candidate, source_hashes["candidate"]),
        "reference_input_sha256": input_digest(reference, source_hashes["reference"]),
        "candidate_checkpoint_sha256": checkpoint_hash(candidate_checkpoint),
        "reference_checkpoint_sha256": checkpoint_hash(reference_checkpoint),
        "image_ids": image_ids,
        "image_ids_hash": image_hash,
        "image_count": len(image_ids),
    }


def _statistical_identity(total_replicates: int) -> dict[str, object]:
    return {
        "stage": "development",
        "total_replicates": int(total_replicates),
        "bootstrap_seed": FACTOR_GATE_BOOTSTRAP_SEED,
        "percentiles": tuple(float(value) for value in FACTOR_GATE_BOOTSTRAP_PERCENTILES),
        "percentile_method": "linear",
        "endpoint_names": tuple(PRIMARY_ENDPOINTS),
        "estimand": _ESTIMAND,
        "rng_schedule": {
            "scheme": _RNG_SCHEME,
            "seed": FACTOR_GATE_BOOTSTRAP_SEED,
            "draw_key": "(seed, stage, image_ids_hash, replicate_index)",
            "next_replicate_index_origin": 0,
        },
    }


def _build_identity(
    candidate: object,
    reference: object,
    *,
    total_replicates: int,
    reference_cache_sha256: str | None,
    source_file_sha256: Mapping[str, str | None] | None = None,
    extra_code_paths: Sequence[Path] = (),
) -> dict[str, object]:
    identity = _evidence_identity(candidate, reference, source_file_sha256=source_file_sha256)
    statistical = _statistical_identity(total_replicates)
    identity.update(
        {
            "schema": CHECKPOINT_SCHEMA,
            "statistical_config": statistical,
            "statistical_config_sha256": _digest(statistical),
            "reference_cache_sha256": reference_cache_sha256,
            "code_hashes": _code_hashes(extra_code_paths),
        }
    )
    return identity


def _validate_checkpoint_identity(payload: Mapping[str, object], expected: Mapping[str, object]) -> None:
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint identity mismatch: schema")
    persisted_identity = payload.get("identity")
    if not isinstance(persisted_identity, Mapping) or _jsonable(persisted_identity) != _jsonable(expected):
        raise ValueError("checkpoint identity mismatch")
    if payload.get("identity_sha256") != _digest(expected):
        raise ValueError("checkpoint identity mismatch: digest")
    # Keep the high-value identity fields visible at the checkpoint root as
    # well as inside the canonical identity object.  Validate both copies so
    # hand-edited code/config/RNG fields fail closed rather than being hidden
    # behind an untouched nested object.
    for key in (
        "candidate_condition",
        "reference_condition",
        "candidate_evidence_sha256",
        "reference_evidence_sha256",
        "evidence_canonical_sha256",
        "source_file_sha256",
        "candidate_input_sha256",
        "reference_input_sha256",
        "image_ids_hash",
        "image_count",
        "statistical_config_sha256",
        "code_hashes",
        "endpoint_names",
    ):
        expected_value: object = expected.get(key)
        if key == "endpoint_names":
            expected_value = expected["statistical_config"]["endpoint_names"]  # type: ignore[index]
        if key in payload and _jsonable(payload[key]) != _jsonable(expected_value):
            raise ValueError(f"checkpoint identity mismatch: {key}")
    persisted_rng = payload.get("rng_schedule")
    expected_rng = expected["statistical_config"]["rng_schedule"]  # type: ignore[index]
    if not isinstance(persisted_rng, Mapping) or not isinstance(expected_rng, Mapping):
        raise ValueError("checkpoint identity mismatch: rng_schedule")
    for key in ("scheme", "seed", "draw_key", "image_ids_hash"):
        expected_value = expected_rng.get(key)
        if key == "image_ids_hash":
            expected_value = expected.get("image_ids_hash")
        if key in persisted_rng and _jsonable(persisted_rng.get(key)) != _jsonable(expected_value):
            raise ValueError(f"checkpoint identity mismatch: rng_schedule.{key}")
    next_index = persisted_rng.get("next_replicate_index")
    if next_index != payload.get("completed"):
        raise ValueError("checkpoint RNG next index does not match completed count")


def _validate_replicates(payload: Mapping[str, object], total: int) -> tuple[float, ...]:
    raw = payload.get("replicates")
    if not isinstance(raw, list):
        raise ValueError("checkpoint replicate vector is malformed")
    values = tuple(_safe_float(item, "checkpoint replicate") for item in raw)
    completed = payload.get("completed")
    if not isinstance(completed, int) or isinstance(completed, bool) or completed < 0 or completed > total:
        raise ValueError("checkpoint completed count is malformed")
    if completed != len(values):
        raise ValueError("checkpoint completed count does not match replicate vector")
    return values


def _validate_reference_replicates(payload: Mapping[str, object], total: int) -> tuple[Mapping[str, float], ...]:
    raw = payload.get("replicates")
    if not isinstance(raw, list):
        raise ValueError("reference checkpoint replicate vector is malformed")
    values = tuple(_coerce_endpoint_draw(item) for item in raw)
    completed = payload.get("completed")
    if not isinstance(completed, int) or isinstance(completed, bool) or completed < 0 or completed > total:
        raise ValueError("reference checkpoint completed count is malformed")
    if completed != len(values):
        raise ValueError("reference checkpoint completed count does not match replicate vector")
    return values


def _progress_payload(
    *,
    condition: str,
    completed: int,
    total: int,
    elapsed: float,
    checkpoint_path: Path,
    state: str,
) -> dict[str, object]:
    rate = completed / elapsed if completed > 0 and elapsed > 0.0 else None
    eta = (total - completed) / rate if rate and rate > 0.0 else None
    return {
        "schema": CHECKPOINT_SCHEMA,
        "condition": condition,
        "completed": int(completed),
        "total": int(total),
        "percent": (100.0 * completed / total) if total else 100.0,
        "elapsed_seconds": float(max(0.0, elapsed)),
        "replicate_rate": None if rate is None else float(rate),
        "eta_seconds": None if eta is None else float(eta),
        "checkpoint_path": str(checkpoint_path),
        "state": state,
        "eta_is_estimate": bool(rate and rate > 0.0),
        "timestamp": time.time(),
    }


def _emit_progress(
    payload: Mapping[str, object],
    *,
    progress_path: Path,
    callback: Callable[[Mapping[str, object]], object] | None,
    stream: TextIO | None,
) -> None:
    _write_progress(progress_path, payload)
    if stream is not None:
        stream.write(
            "factor-bootstrap "
            f"{payload['condition']} {payload['completed']}/{payload['total']} "
            f"{float(payload['percent']):.2f}% "
            f"rate={payload['replicate_rate']} eta={payload['eta_seconds']} "
            f"checkpoint={payload['checkpoint_path']}\n"
        )
        stream.flush()
    if callback is not None:
        callback(payload)


@dataclass(frozen=True)
class ReferenceDrawCache:
    """Immutable sequence of endpoint draws shared by F1/F2/F3."""

    draws: tuple[Mapping[str, float], ...]
    identity_sha256: str
    checkpoint_path: Path

    def __len__(self) -> int:
        return len(self.draws)

    def __iter__(self):
        return iter(self.draws)

    def __getitem__(self, index: int) -> Mapping[str, float]:
        return self.draws[index]


@dataclass(frozen=True)
class ResumablePairedDelta(PairedDelta):
    """Paired delta plus its ordered, persisted replicate vector."""

    replicates: tuple[float, ...]
    checkpoint_path: Path
    completed: int
    total_replicates: int
    elapsed_seconds: float
    replicate_rate: float | None

    def __post_init__(self) -> None:
        super().__post_init__()
        values = tuple(_safe_float(value, "replicate") for value in self.replicates)
        if len(values) != int(self.total_replicates) or int(self.completed) != len(values):
            raise ValueError("resumable paired delta replicate count is inconsistent")
        object.__setattr__(self, "replicates", values)
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        object.__setattr__(self, "completed", int(self.completed))
        object.__setattr__(self, "total_replicates", int(self.total_replicates))
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))
        object.__setattr__(self, "replicate_rate", None if self.replicate_rate is None else float(self.replicate_rate))


def _cache_from_argument(reference_draws: object | None) -> tuple[tuple[Mapping[str, float], ...] | None, str | None]:
    if reference_draws is None:
        return None, None
    if isinstance(reference_draws, ReferenceDrawCache):
        draws = tuple(reference_draws.draws)
        identity = reference_draws.identity_sha256
    else:
        try:
            draws = tuple(reference_draws)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError("reference_draws must be a sequence of endpoint mappings") from error
        identity = None
    normalized = tuple(_coerce_endpoint_draw(draw) for draw in draws)
    computed = _digest(normalized)
    if identity is not None and str(identity) != computed:
        raise ValueError("reference draw cache identity mismatch")
    return normalized, computed


def _reference_identity(
    reference: object,
    *,
    total_replicates: int,
    source_file_sha256: str | None = None,
    extra_code_paths: Sequence[Path] = (),
) -> dict[str, object]:
    image_ids = tuple(sorted(str(item) for item in (_evidence_attr(reference, "image_ids", ()) or ())))
    image_hash = str(_evidence_attr(reference, "image_ids_hash", ""))
    evidence_hash = str(_evidence_attr(reference, "evidence_sha256", ""))
    if not image_ids or not image_hash or not evidence_hash:
        raise ValueError("reference evidence hashes and image IDs are required")
    statistical = _statistical_identity(total_replicates)
    source_hash = _source_hash(source_file_sha256, "source_file_sha256.reference")
    input_identity = (
        {
            "condition": _evidence_attr(reference, "condition", ""),
            "evidence_sha256": evidence_hash,
            "source_file_sha256": source_hash,
        }
        if source_hash is not None
        else {
            "condition": _evidence_attr(reference, "condition", ""),
            "image_ids": image_ids,
            "image_ids_hash": image_hash,
            "endpoints": _evidence_attr(reference, "endpoints", {}),
            "raw_observations": _evidence_attr(reference, "raw_observations", None),
            "endpoint_rows": _evidence_attr(reference, "endpoint_rows", None),
        }
    )
    input_digest = _digest(input_identity)
    identity = {
        "candidate_condition": REFERENCE_CACHE_CONDITION,
        "reference_condition": REFERENCE_CACHE_CONDITION,
        "candidate_evidence_sha256": evidence_hash,
        "reference_evidence_sha256": evidence_hash,
        "evidence_canonical_sha256": {"candidate": evidence_hash, "reference": evidence_hash},
        "source_file_sha256": {"candidate": source_hash, "reference": source_hash},
        "candidate_input_sha256": input_digest,
        "reference_input_sha256": input_digest,
        "candidate_checkpoint_sha256": None,
        "reference_checkpoint_sha256": None,
        "image_ids": image_ids,
        "image_ids_hash": image_hash,
        "image_count": len(image_ids),
        "schema": CHECKPOINT_SCHEMA,
        "statistical_config": statistical,
        "statistical_config_sha256": _digest(statistical),
        "code_hashes": _code_hashes(extra_code_paths),
    }
    return identity


def build_shared_reference_draws(
    reference: object,
    output_dir: str | Path,
    *,
    total_replicates: int = FACTOR_GATE_BOOTSTRAP_REPLICATES,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    checkpoint_wall_time_seconds: float = MAX_CHECKPOINT_WALL_SECONDS,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    stop_after: int | None = None,
    progress_callback: Callable[[Mapping[str, object]], object] | None = None,
    progress_stream: TextIO | None = None,
    source_file_sha256: str | None = None,
    code_paths: Sequence[Path] = (),
) -> ReferenceDrawCache:
    """Prepare one deterministic F0 endpoint draw per replicate.

    The cache is itself checkpointed so candidate resumes never have to
    silently recompute the shared reference statistic.
    """

    if not isinstance(total_replicates, int) or isinstance(total_replicates, bool) or total_replicates <= 0:
        raise ValueError("total_replicates must be a positive integer")
    if not isinstance(checkpoint_interval, int) or isinstance(checkpoint_interval, bool) or checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be a positive integer")
    if (
        isinstance(checkpoint_wall_time_seconds, bool)
        or not isinstance(checkpoint_wall_time_seconds, (int, float))
        or not math.isfinite(float(checkpoint_wall_time_seconds))
        or float(checkpoint_wall_time_seconds) <= 0.0
        or float(checkpoint_wall_time_seconds) > MAX_CHECKPOINT_WALL_SECONDS
    ):
        raise ValueError(f"checkpoint_wall_time_seconds must be in (0, {MAX_CHECKPOINT_WALL_SECONDS}]")
    if stop_after is not None and (not isinstance(stop_after, int) or isinstance(stop_after, bool) or stop_after < 0):
        raise ValueError("stop_after must be a non-negative integer")
    output = Path(output_dir).expanduser().resolve(strict=False)
    checkpoints = Path(checkpoint_dir).expanduser().resolve(strict=False) if checkpoint_dir is not None else output / "checkpoints"
    path = checkpoints / DEFAULT_REFERENCE_CHECKPOINT
    identity = _reference_identity(
        reference,
        total_replicates=total_replicates,
        source_file_sha256=source_file_sha256,
        extra_code_paths=code_paths,
    )
    identity_sha = _digest(identity)
    if path.exists() or path.is_symlink():
        if not resume:
            raise ValueError(f"checkpoint exists; explicit resume is required: {path}")
        payload = _read_json(path, "reference checkpoint")
        _validate_checkpoint_identity(payload, identity)
        values = _validate_reference_replicates(payload, total_replicates)
        state = payload.get("state")
        if state not in {"running", "complete"}:
            raise ValueError("reference checkpoint state is malformed")
    else:
        values = ()
        state = "running"
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "state": state,
            "identity": identity,
            "identity_sha256": identity_sha,
            "condition": REFERENCE_CACHE_CONDITION,
            "candidate_condition": REFERENCE_CACHE_CONDITION,
            "reference_condition": REFERENCE_CACHE_CONDITION,
            "candidate_evidence_sha256": identity["candidate_evidence_sha256"],
            "reference_evidence_sha256": identity["reference_evidence_sha256"],
            "evidence_canonical_sha256": identity["evidence_canonical_sha256"],
            "source_file_sha256": identity["source_file_sha256"],
            "candidate_input_sha256": identity["candidate_input_sha256"],
            "reference_input_sha256": identity["reference_input_sha256"],
            "image_ids_hash": identity["image_ids_hash"],
            "image_count": identity["image_count"],
            "statistical_config_sha256": identity["statistical_config_sha256"],
            "code_hashes": identity["code_hashes"],
            "endpoint_names": list(PRIMARY_ENDPOINTS),
            "total": total_replicates,
            "completed": 0,
            "replicates": [],
            "rng_schedule": {
                **identity["statistical_config"]["rng_schedule"],  # type: ignore[index]
                "image_ids_hash": identity["image_ids_hash"],
                "next_replicate_index": 0,
            },
            "started_at": time.time(),
            "updated_at": time.time(),
            "elapsed_seconds": 0.0,
            "replicate_rate": None,
            "last_saved_at": time.time(),
            "completed_range": [0, 0],
            "next_replicate_index": 0,
            "output_paths": {
                "checkpoint": str(path),
                "progress": str(output / DEFAULT_PROGRESS_FILENAME),
            },
            "version": 1,
            "checkpoint_interval": checkpoint_interval,
            "checkpoint_wall_time_seconds": float(checkpoint_wall_time_seconds),
        }
        _write_checkpoint(path, payload, initial=True)
    if state == "complete":
        if len(values) != total_replicates:
            raise ValueError("complete reference checkpoint has incomplete replicates")
        return ReferenceDrawCache(tuple(_coerce_endpoint_draw(item) for item in values), _digest(values), path)

    # ``elapsed_seconds`` is cumulative compute time.  A resume starts a new
    # monotonic segment; persisted wall-clock timestamps are provenance only
    # and must never charge downtime to the bootstrap.
    base_elapsed = _safe_float(payload.get("elapsed_seconds", 0.0), "elapsed_seconds")
    segment_started_monotonic = time.monotonic()
    last_saved_monotonic = segment_started_monotonic
    mutable = list(values)
    start_index = len(mutable)
    if stop_after is not None and start_index >= stop_after:
        raise RuntimeError("interrupted")
    while start_index < total_replicates:
        indices = paired_resample_indices(
            stage="development",
            image_ids_hash=str(_evidence_attr(reference, "image_ids_hash", "")),
            image_count=len(tuple(_evidence_attr(reference, "image_ids", ()) or ())),
            replicate_index=start_index,
        )
        endpoint_draw = _coerce_endpoint_draw(recompute_endpoints(reference, indices))
        mutable.append(endpoint_draw)
        start_index += 1
        now_monotonic = time.monotonic()
        should_commit = (
            start_index % checkpoint_interval == 0
            or start_index == total_replicates
            or now_monotonic - last_saved_monotonic >= float(checkpoint_wall_time_seconds)
        )
        if should_commit:
            elapsed = base_elapsed + max(0.0, now_monotonic - segment_started_monotonic)
            mutable_payload = {
                **payload,
                "state": "running",
                "completed": start_index,
                "replicates": list(mutable),
                "updated_at": time.time(),
                "elapsed_seconds": elapsed,
                "replicate_rate": (start_index / elapsed) if elapsed > 0.0 else None,
                "last_saved_at": time.time(),
                "completed_range": [0, start_index],
                "next_replicate_index": start_index,
                "rng_schedule": {
                    **identity["statistical_config"]["rng_schedule"],  # type: ignore[index]
                    "image_ids_hash": identity["image_ids_hash"],
                    "next_replicate_index": start_index,
                },
                "output_paths": {
                    "checkpoint": str(path),
                    "progress": str(output / DEFAULT_PROGRESS_FILENAME),
                },
                "version": 1,
                "checkpoint_interval": checkpoint_interval,
                "checkpoint_wall_time_seconds": float(checkpoint_wall_time_seconds),
            }
            _write_checkpoint(path, mutable_payload)
            last_saved_monotonic = now_monotonic
            _emit_progress(
                _progress_payload(
                    condition=REFERENCE_CACHE_CONDITION,
                    completed=start_index,
                    total=total_replicates,
                    elapsed=elapsed,
                    checkpoint_path=path,
                    state="running",
                ),
                progress_path=output / DEFAULT_PROGRESS_FILENAME,
                callback=progress_callback,
                stream=progress_stream,
            )
        if stop_after is not None and start_index >= stop_after:
            # A non-boundary interruption intentionally leaves the last
            # committed block intact; the deterministic index resumes it.
            raise RuntimeError("interrupted")
    elapsed = base_elapsed + max(0.0, time.monotonic() - segment_started_monotonic)
    complete_payload = {
        **payload,
        "state": "complete",
        "completed": total_replicates,
        "replicates": list(mutable),
        "updated_at": time.time(),
        "elapsed_seconds": elapsed,
        "replicate_rate": (total_replicates / elapsed) if elapsed > 0.0 else None,
        "last_saved_at": time.time(),
        "completed_range": [0, total_replicates],
        "next_replicate_index": total_replicates,
        "rng_schedule": {
            **identity["statistical_config"]["rng_schedule"],  # type: ignore[index]
            "image_ids_hash": identity["image_ids_hash"],
            "next_replicate_index": total_replicates,
        },
        "output_paths": {
            "checkpoint": str(path),
            "progress": str(output / DEFAULT_PROGRESS_FILENAME),
        },
        "version": 1,
        "checkpoint_interval": checkpoint_interval,
        "checkpoint_wall_time_seconds": float(checkpoint_wall_time_seconds),
    }
    _write_checkpoint(path, complete_payload)
    _emit_progress(
        _progress_payload(
            condition=REFERENCE_CACHE_CONDITION,
            completed=total_replicates,
            total=total_replicates,
            elapsed=elapsed,
            checkpoint_path=path,
            state="complete",
        ),
        progress_path=output / DEFAULT_PROGRESS_FILENAME,
        callback=progress_callback,
        stream=progress_stream,
    )
    return ReferenceDrawCache(tuple(mutable), _digest(mutable), path)


def run_resumable_factor_bootstrap(
    candidate: object,
    reference: object,
    output_dir: str | Path,
    *,
    condition: str | None = None,
    total_replicates: int = FACTOR_GATE_BOOTSTRAP_REPLICATES,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    checkpoint_wall_time_seconds: float = MAX_CHECKPOINT_WALL_SECONDS,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    stop_after: int | None = None,
    reference_draws: object | None = None,
    progress_callback: Callable[[Mapping[str, object]], object] | None = None,
    progress_stream: TextIO | None = None,
    source_file_sha256: Mapping[str, str | None] | None = None,
    code_paths: Sequence[Path] = (),
) -> ResumablePairedDelta:
    """Run one candidate-minus-F0 bootstrap with exact checkpoint resume."""

    if not isinstance(total_replicates, int) or isinstance(total_replicates, bool) or total_replicates <= 0:
        raise ValueError("total_replicates must be a positive integer")
    if not isinstance(checkpoint_interval, int) or isinstance(checkpoint_interval, bool) or checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be a positive integer")
    if (
        isinstance(checkpoint_wall_time_seconds, bool)
        or not isinstance(checkpoint_wall_time_seconds, (int, float))
        or not math.isfinite(float(checkpoint_wall_time_seconds))
        or float(checkpoint_wall_time_seconds) <= 0.0
        or float(checkpoint_wall_time_seconds) > MAX_CHECKPOINT_WALL_SECONDS
    ):
        raise ValueError(f"checkpoint_wall_time_seconds must be in (0, {MAX_CHECKPOINT_WALL_SECONDS}]")
    if stop_after is not None and (not isinstance(stop_after, int) or isinstance(stop_after, bool) or stop_after < 0):
        raise ValueError("stop_after must be a non-negative integer")
    candidate_name = str(condition if condition is not None else _evidence_attr(candidate, "condition", ""))
    if candidate_name != str(_evidence_attr(candidate, "condition", candidate_name)):
        raise ValueError("condition does not match candidate evidence")
    # Validate the immutable pair before touching a checkpoint.  The public
    # helper also preserves arbitrary producer callbacks for raw evidence.
    candidate_hash = str(_evidence_attr(candidate, "evidence_sha256", ""))
    reference_hash = str(_evidence_attr(reference, "evidence_sha256", ""))
    if not bool(_evidence_attr(candidate, "complete", True)):
        raise ValueError("incomplete candidate evidence")
    if not bool(_evidence_attr(reference, "complete", True)):
        raise ValueError("incomplete F0 evidence")
    identity_cache, cache_sha = _cache_from_argument(reference_draws)
    output = Path(output_dir).expanduser().resolve(strict=False)
    checkpoints = Path(checkpoint_dir).expanduser().resolve(strict=False) if checkpoint_dir is not None else output / "checkpoints"
    path = checkpoints / f"{candidate_name}.json"
    identity = _build_identity(
        candidate,
        reference,
        total_replicates=total_replicates,
        reference_cache_sha256=cache_sha,
        source_file_sha256=source_file_sha256,
        extra_code_paths=code_paths,
    )
    identity_sha = _digest(identity)
    if identity_cache is not None and len(identity_cache) != total_replicates:
        raise ValueError("reference draw cache length does not match total replicates")
    if path.exists() or path.is_symlink():
        if not resume:
            raise ValueError(f"checkpoint exists; explicit resume is required: {path}")
        payload = _read_json(path, "candidate checkpoint")
        _validate_checkpoint_identity(payload, identity)
        values = _validate_replicates(payload, total_replicates)
        state = payload.get("state")
        if state not in {"running", "complete"}:
            raise ValueError("candidate checkpoint state is malformed")
    else:
        values = ()
        state = "running"
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "state": state,
            "identity": identity,
            "identity_sha256": identity_sha,
            "candidate": candidate_name,
            "reference": "F0",
            "candidate_condition": identity["candidate_condition"],
            "reference_condition": identity["reference_condition"],
            "candidate_evidence_sha256": identity["candidate_evidence_sha256"],
            "reference_evidence_sha256": identity["reference_evidence_sha256"],
            "evidence_canonical_sha256": identity["evidence_canonical_sha256"],
            "source_file_sha256": identity["source_file_sha256"],
            "candidate_input_sha256": identity["candidate_input_sha256"],
            "reference_input_sha256": identity["reference_input_sha256"],
            "image_ids_hash": identity["image_ids_hash"],
            "image_count": identity["image_count"],
            "statistical_config_sha256": identity["statistical_config_sha256"],
            "code_hashes": identity["code_hashes"],
            "endpoint_names": list(PRIMARY_ENDPOINTS),
            "total": total_replicates,
            "completed": 0,
            "replicates": [],
            "rng_schedule": {
                **identity["statistical_config"]["rng_schedule"],  # type: ignore[index]
                "image_ids_hash": identity["image_ids_hash"],
                "next_replicate_index": 0,
            },
            "started_at": time.time(),
            "updated_at": time.time(),
            "elapsed_seconds": 0.0,
            "replicate_rate": None,
            "last_saved_at": time.time(),
            "completed_range": [0, 0],
            "next_replicate_index": 0,
            "output_paths": {
                "checkpoint": str(path),
                "progress": str(output / DEFAULT_PROGRESS_FILENAME),
            },
            "version": 1,
            "checkpoint_interval": checkpoint_interval,
            "checkpoint_wall_time_seconds": float(checkpoint_wall_time_seconds),
        }
        _write_checkpoint(path, payload, initial=True)
    candidate_endpoints = {
        name: _safe_float(value, f"candidate endpoint {name}")
        for name, value in (
            (_evidence_attr(candidate, "endpoints", {}) or {}).items()
            if isinstance(_evidence_attr(candidate, "endpoints", {}), Mapping)
            else ()
        )
        if name in PRIMARY_ENDPOINTS
    }
    reference_endpoints = {
        name: _safe_float(value, f"reference endpoint {name}")
        for name, value in (
            (_evidence_attr(reference, "endpoints", {}) or {}).items()
            if isinstance(_evidence_attr(reference, "endpoints", {}), Mapping)
            else ()
        )
        if name in PRIMARY_ENDPOINTS
    }
    point = composite_mechanism_score(candidate_endpoints) - composite_mechanism_score(reference_endpoints)
    if state == "complete":
        if len(values) != total_replicates:
            raise ValueError("complete candidate checkpoint has incomplete replicates")
        array = np.asarray(values, dtype=np.float64)
        ci = tuple(float(value) for value in np.quantile(array, FACTOR_GATE_BOOTSTRAP_PERCENTILES, method="linear"))
        rate_value = payload.get("replicate_rate")
        return ResumablePairedDelta(
            point=point,
            ci95=(ci[0], ci[1]),
            candidate_endpoints=candidate_endpoints,
            candidate_evidence_sha256=candidate_hash,
            replicates=values,
            checkpoint_path=path,
            completed=total_replicates,
            total_replicates=total_replicates,
            elapsed_seconds=_safe_float(payload.get("elapsed_seconds", 0.0), "elapsed_seconds"),
            replicate_rate=None if rate_value is None else _safe_float(rate_value, "replicate_rate"),
        )

    # Keep elapsed as accumulated compute time across restarts.  ``started_at``
    # is retained only for provenance; wall-clock downtime is excluded.
    base_elapsed = _safe_float(payload.get("elapsed_seconds", 0.0), "elapsed_seconds")
    segment_started_monotonic = time.monotonic()
    last_saved_monotonic = segment_started_monotonic
    mutable = list(values)
    index = len(mutable)
    if stop_after is not None and index >= stop_after:
        raise RuntimeError("interrupted")
    while index < total_replicates:
        indices = paired_resample_indices(
            stage="development",
            image_ids_hash=str(_evidence_attr(reference, "image_ids_hash", "")),
            image_count=len(tuple(_evidence_attr(reference, "image_ids", ()) or ())),
            replicate_index=index,
        )
        draw = identity_cache[index] if identity_cache is not None else None
        value = paired_image_cluster_replicate(
            candidate,
            reference,
            index,
            indices=indices,
            reference_draw=draw,
        )
        mutable.append(_safe_float(value, "paired replicate"))
        index += 1
        now_monotonic = time.monotonic()
        should_commit = (
            index % checkpoint_interval == 0
            or index == total_replicates
            or now_monotonic - last_saved_monotonic >= float(checkpoint_wall_time_seconds)
        )
        if should_commit:
            elapsed = base_elapsed + max(0.0, now_monotonic - segment_started_monotonic)
            mutable_payload = {
                **payload,
                "state": "running",
                "completed": index,
                "replicates": list(mutable),
                "updated_at": time.time(),
                "elapsed_seconds": elapsed,
                "replicate_rate": (index / elapsed) if elapsed > 0.0 else None,
                "next_replicate_index": index,
                "rng_schedule": {
                    **identity["statistical_config"]["rng_schedule"],  # type: ignore[index]
                    "image_ids_hash": identity["image_ids_hash"],
                    "next_replicate_index": index,
                },
                "last_saved_at": time.time(),
                "completed_range": [0, index],
                "output_paths": {
                    "checkpoint": str(path),
                    "progress": str(output / DEFAULT_PROGRESS_FILENAME),
                },
                "version": 1,
                "checkpoint_interval": checkpoint_interval,
                "checkpoint_wall_time_seconds": float(checkpoint_wall_time_seconds),
            }
            _write_checkpoint(path, mutable_payload)
            last_saved_monotonic = now_monotonic
            _emit_progress(
                _progress_payload(
                    condition=candidate_name,
                    completed=index,
                    total=total_replicates,
                    elapsed=elapsed,
                    checkpoint_path=path,
                    state="running",
                ),
                progress_path=output / DEFAULT_PROGRESS_FILENAME,
                callback=progress_callback,
                stream=progress_stream,
            )
        if stop_after is not None and index >= stop_after:
            raise RuntimeError("interrupted")
    elapsed = base_elapsed + max(0.0, time.monotonic() - segment_started_monotonic)
    array = np.asarray(mutable, dtype=np.float64)
    ci = tuple(float(value) for value in np.quantile(array, FACTOR_GATE_BOOTSTRAP_PERCENTILES, method="linear"))
    complete_payload = {
        **payload,
        "state": "complete",
        "completed": total_replicates,
        "replicates": list(mutable),
        "updated_at": time.time(),
        "elapsed_seconds": elapsed,
        "replicate_rate": (total_replicates / elapsed) if elapsed > 0.0 else None,
        "next_replicate_index": total_replicates,
        "rng_schedule": {
            **identity["statistical_config"]["rng_schedule"],  # type: ignore[index]
            "image_ids_hash": identity["image_ids_hash"],
            "next_replicate_index": total_replicates,
        },
        "point": point,
        "ci95": list(ci),
        "candidate_endpoints": candidate_endpoints,
        "last_saved_at": time.time(),
        "completed_range": [0, total_replicates],
        "next_replicate_index": total_replicates,
        "output_paths": {
            "checkpoint": str(path),
            "progress": str(output / DEFAULT_PROGRESS_FILENAME),
        },
        "version": 1,
        "checkpoint_interval": checkpoint_interval,
        "checkpoint_wall_time_seconds": float(checkpoint_wall_time_seconds),
    }
    _write_checkpoint(path, complete_payload)
    _emit_progress(
        _progress_payload(
            condition=candidate_name,
            completed=total_replicates,
            total=total_replicates,
            elapsed=elapsed,
            checkpoint_path=path,
            state="complete",
        ),
        progress_path=output / DEFAULT_PROGRESS_FILENAME,
        callback=progress_callback,
        stream=progress_stream,
    )
    return ResumablePairedDelta(
        point=point,
        ci95=(ci[0], ci[1]),
        candidate_endpoints=candidate_endpoints,
        candidate_evidence_sha256=candidate_hash,
        replicates=tuple(mutable),
        checkpoint_path=path,
        completed=total_replicates,
        total_replicates=total_replicates,
        elapsed_seconds=elapsed,
        replicate_rate=(total_replicates / elapsed) if elapsed > 0.0 else None,
    )


__all__ = [
    "CHECKPOINT_SCHEMA",
    "DEFAULT_CHECKPOINT_INTERVAL",
    "ReferenceDrawCache",
    "ResumablePairedDelta",
    "build_shared_reference_draws",
    "run_resumable_factor_bootstrap",
]
