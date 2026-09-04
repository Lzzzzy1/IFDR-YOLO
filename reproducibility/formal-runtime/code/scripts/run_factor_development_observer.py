"""Run the fixed development seed-17 factor observer for one condition.

This entry point deliberately does orchestration only.  It binds one
calibration ``last.pt`` checkpoint to its role/hash manifest, constructs the
existing three-view factor-observer manifest over the canonical 371-image
development split, and writes resumable raw observations plus audit artifacts.
The statistical gate itself remains in :mod:`ifdr_yolo.eval.natural_factor_audit`.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.data.natural_degradation import (
    NaturalDegradationRecord,
    load_natural_degradation_records,
)
from ifdr_yolo.data.replay_sampler import sha256_canonical
from ifdr_yolo.eval.natural_factor_audit import (
    DEFAULT_INTERVENTION_SEVERITIES,
    NaturalFactorGateDecision,
    audit_natural_factors,
)

if TYPE_CHECKING:
    from ifdr_yolo.eval.factor_observer import FactorObservationManifest


DEVELOPMENT_SEED = 17
DEVELOPMENT_IMAGE_COUNT = 371
DEVELOPMENT_AUDIT_SEED = 20260805
VIEW_ROLES = ("target", "background", "natural")
AUDIT_CONFIDENCE = 0.95
MONOTONIC_THRESHOLD = 0.80
CHECKPOINT_NAME = "last.pt"
CHECKPOINT_ROLE = "calibration_last"


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite value cannot be serialized")
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
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
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (_canonical_json(_json_safe(payload)) + "\n").encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _sha256_file(path: Path) -> str:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"required file is missing or empty: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json_object(value: object, field: str) -> dict[str, object]:
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        path = Path(value).expanduser().resolve(strict=False)
        if not path.is_file():
            raise ValueError(f"{field} is missing: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"{field} is malformed: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must contain a JSON object")
    return payload


def _checkpoint_entry(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    if any(name in value for name in ("path", "sha256", "hash", "role", "checkpoint_role")):
        return value
    for name in ("primary_checkpoint", "calibration_last", "primary", "last_checkpoint", "last"):
        nested = value.get(name)
        if isinstance(nested, Mapping):
            return nested
    nested_roles = value.get("checkpoint_roles")
    if isinstance(nested_roles, Mapping):
        return _checkpoint_entry(nested_roles)
    return None


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def resolve_calibration_checkpoint(
    checkpoint: str | Path | Mapping[str, object],
    checkpoint_roles: str | Path | Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Resolve and hash-bind exactly one calibration ``last.pt`` checkpoint."""

    roles_payload = _read_json_object(checkpoint_roles, "checkpoint_roles")
    role_entry = _checkpoint_entry(roles_payload)
    if role_entry is None:
        raise ValueError("checkpoint_roles must contain calibration_last role/hash")
    checkpoint_entry: Mapping[str, object]
    if isinstance(checkpoint, Mapping):
        checkpoint_entry = checkpoint
    else:
        checkpoint_entry = {"path": checkpoint}
    supplied_role = checkpoint_entry.get("role", checkpoint_entry.get("checkpoint_role"))
    if supplied_role is not None and supplied_role not in {"primary", CHECKPOINT_ROLE}:
        raise ValueError("calibration checkpoint role must be primary/calibration_last")
    raw_path = checkpoint_entry.get("path", role_entry.get("path"))
    if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
        raise ValueError("calibration checkpoint path is required")
    path = Path(raw_path).expanduser().resolve(strict=False)
    if path.name != CHECKPOINT_NAME:
        raise ValueError("calibration evidence requires last.pt, never best.pt")
    actual_hash = _sha256_file(path)
    role = role_entry.get("role", role_entry.get("checkpoint_role"))
    checkpoint_role = role_entry.get("checkpoint_role")
    if role is not None and role not in {"primary", CHECKPOINT_ROLE}:
        raise ValueError("calibration checkpoint role must be primary/calibration_last")
    if checkpoint_role is not None and checkpoint_role != CHECKPOINT_ROLE:
        raise ValueError("calibration checkpoint role must be calibration_last")
    expected_hashes: list[str] = []
    for source in (checkpoint_entry, role_entry):
        raw_hash = source.get("sha256", source.get("hash"))
        if raw_hash is not None:
            expected_hashes.append(_require_sha256(raw_hash, "calibration checkpoint sha256"))
    if not expected_hashes:
        raise ValueError("checkpoint_roles must include calibration checkpoint hash")
    if len(set(expected_hashes)) != 1 or actual_hash != expected_hashes[0]:
        raise ValueError("calibration checkpoint hash mismatch")
    if role_entry.get("path") is not None:
        role_path = Path(str(role_entry["path"])).expanduser().resolve(strict=False)
        relative_role_path = Path(str(role_entry["path"]))
        if relative_role_path.is_absolute() or relative_role_path.as_posix() != CHECKPOINT_NAME:
            if role_path != path:
                raise ValueError("checkpoint_roles path does not match calibration checkpoint")
    resolved = {
        "path": str(path),
        "role": "primary",
        "checkpoint_role": CHECKPOINT_ROLE,
        "sha256": actual_hash,
    }
    return resolved, dict(_json_safe(roles_payload))  # type: ignore[arg-type]


def load_development_ids(path: str | Path) -> tuple[str, ...]:
    """Load the immutable 371-image development ID protocol."""

    path = Path(path).expanduser().resolve(strict=False)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"unable to read development IDs: {path}") from exc
    if len(lines) != DEVELOPMENT_IMAGE_COUNT:
        raise ValueError(
            f"development protocol requires exactly {DEVELOPMENT_IMAGE_COUNT} image IDs; got {len(lines)}"
        )
    ids: list[str] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(lines, start=1):
        if raw != raw.strip() or not raw:
            raise ValueError(f"invalid development image ID at {path}:{line_number}")
        image_id = raw.strip()
        if len(image_id) != 6 or not image_id.isdigit():
            raise ValueError(f"invalid KITTI image ID at {path}:{line_number}: {image_id!r}")
        if image_id in seen:
            raise ValueError(f"duplicate development image ID: {image_id}")
        seen.add(image_id)
        ids.append(image_id)
    return tuple(ids)


def _same_json_file(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"refusing to overwrite non-regular artifact: {path}")
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"existing artifact is malformed: {path}") from exc
        if _canonical_json(existing) != _canonical_json(_json_safe(payload)):
            raise ValueError(f"existing artifact does not match current protocol: {path}")
        return
    _atomic_write_json(path, payload)


def ensure_output_reusable(output_dir: str | Path, scientific_identity: Mapping[str, object]) -> None:
    """Reject complete or identity-mismatched outputs while allowing resume."""

    output = Path(output_dir).expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    if status_path.exists() or status_path.is_symlink():
        if status_path.is_symlink() or not status_path.is_file():
            raise ValueError("refusing to overwrite non-regular status artifact")
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError("status.json is malformed") from exc
        if not isinstance(status, Mapping):
            raise ValueError("status.json must contain an object")
        if status.get("status") == "complete":
            raise ValueError("refusing to overwrite complete output")
    provenance_path = output / "provenance.json"
    if provenance_path.exists() or provenance_path.is_symlink():
        if provenance_path.is_symlink() or not provenance_path.is_file():
            raise ValueError("refusing to overwrite non-regular provenance artifact")
        try:
            existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError("provenance.json is malformed") from exc
        if not isinstance(existing, Mapping) or _canonical_json(existing.get("scientific_identity")) != _canonical_json(_json_safe(scientific_identity)):
            raise ValueError("provenance scientific identity mismatch; refusing to resume")


def _resolve_image_path(image_dir: Path, image_id: str) -> Path:
    path = image_dir / f"{image_id}.png"
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"development image PNG is missing or empty: {path}")
    return path


def _checkpoint_arg(value: str) -> Path:
    """Accept a direct path or the existing ``17=PATH`` spelling."""

    if "=" in value:
        seed, raw_path = value.split("=", 1)
        if seed != str(DEVELOPMENT_SEED):
            raise argparse.ArgumentTypeError("development checkpoint must be seed 17")
        value = raw_path
    if not value.strip():
        raise argparse.ArgumentTypeError("checkpoint path must not be empty")
    return Path(value)


def _write_manifest(output_dir: Path, manifest: FactorObservationManifest) -> None:
    manifest_path = output_dir / "manifest.json"
    hash_path = output_dir / "manifest.sha256"
    _same_json_file(manifest_path, manifest.to_dict())
    expected_hash = manifest.hash()
    if hash_path.exists() or hash_path.is_symlink():
        if hash_path.is_symlink() or not hash_path.is_file():
            raise ValueError("manifest.sha256 is not a regular file")
        if hash_path.read_text(encoding="ascii").strip() != expected_hash:
            raise ValueError("manifest.sha256 does not match current manifest")
    else:
        _atomic_write_bytes(hash_path, (expected_hash + "\n").encode("ascii"))


def _summary_payload(
    rows: Sequence[object],
    decision: NaturalFactorGateDecision,
    *,
    condition: str,
    image_ids: Sequence[str],
    image_ids_hash: str,
    checkpoint: Mapping[str, object],
    manifest: FactorObservationManifest,
    selected_interventions: Sequence[tuple[str, int]],
) -> dict[str, object]:
    natural_ids = sorted({str(getattr(row, "image_id")) for row in rows if getattr(row, "intervention_kind", None) == "natural"})
    class_counts: Counter[str] = Counter(
        str(getattr(row, "class_name", None) or getattr(row, "class_id", ""))
        for row in rows
        if getattr(row, "intervention_kind", None) == "natural"
    )
    return {
        "schema_version": 1,
        "stage": "development",
        "condition": condition,
        "seed": DEVELOPMENT_SEED,
        "views": list(VIEW_ROLES),
        "image_count": len(image_ids),
        "image_ids": list(image_ids),
        "image_ids_hash": image_ids_hash,
        "natural_image_count": len(natural_ids),
        "row_count": len(rows),
        "manifest_sha256": manifest.hash(),
        "expected_observation_count": manifest.expected_observation_count,
        "intervention_object_count": len(selected_interventions),
        "natural_class_counts": dict(sorted(class_counts.items())),
        "checkpoint_sha256": checkpoint["sha256"],
        "gate_passed": bool(decision.passed),
    }


def write_audit_artifacts(
    output_dir: str | Path,
    *,
    condition: str,
    rows: Sequence[object],
    decision: NaturalFactorGateDecision,
    image_ids: Sequence[str],
    image_ids_hash: str,
    checkpoint: Mapping[str, object],
    manifest: FactorObservationManifest,
    selected_interventions: Sequence[tuple[str, int]],
) -> None:
    """Persist the raw-audit decision and summary without replacing mismatches."""

    output = Path(output_dir).expanduser().resolve(strict=False)
    audit_payload = dict(decision.to_dict())
    audit_payload.update(
        {
            "schema_version": 1,
            "stage": "development",
            "condition": condition,
            "seed": DEVELOPMENT_SEED,
            "views": list(VIEW_ROLES),
            "image_ids": list(image_ids),
            "image_ids_hash": image_ids_hash,
            "checkpoint_sha256": checkpoint["sha256"],
            "manifest_sha256": manifest.hash(),
        }
    )
    _same_json_file(output / "audit_decision.json", audit_payload)
    _same_json_file(
        output / "summary.json",
        _summary_payload(
            rows,
            decision,
            condition=condition,
            image_ids=image_ids,
            image_ids_hash=image_ids_hash,
            checkpoint=checkpoint,
            manifest=manifest,
            selected_interventions=selected_interventions,
        ),
    )


def _run(args: argparse.Namespace) -> int:
    if int(getattr(args, "seed", DEVELOPMENT_SEED)) != DEVELOPMENT_SEED:
        raise ValueError("development observer seed is frozen to 17")
    if tuple(getattr(args, "views", VIEW_ROLES)) != VIEW_ROLES:
        raise ValueError("development observer views are frozen to target/background/natural")
    if args.input_size <= 0 or args.transform_batch_size <= 0 or args.bootstrap_replicates < 2:
        raise ValueError("input-size and transform-batch-size must be positive; bootstrap-replicates >= 2")

    output_dir = args.output_dir.expanduser().resolve(strict=False)
    image_ids = load_development_ids(args.development_ids)
    image_ids = tuple(sorted(image_ids))
    image_ids_hash = sha256_canonical(list(image_ids))
    checkpoint, roles_payload = resolve_calibration_checkpoint(args.checkpoint, args.checkpoint_roles)
    # cv2/torch-backed observer modules are imported only after cheap protocol
    # and checkpoint identity validation, keeping those checks usable in
    # CPU-only tooling and unit tests.
    from ifdr_yolo.eval.factor_observer import (
        DEFAULT_REQUIRED_NODES,
        FactorObservationJournal,
        build_factor_observation_manifest,
    )
    from ifdr_yolo.eval.factor_observer_runtime import load_ifdr_checkpoint, run_factor_observer
    from scripts.audit_natural_factors import _load_observation_rows, select_intervention_objects

    metadata_path = args.metadata_jsonl.expanduser().resolve(strict=False)
    metadata_sha256 = _sha256_file(metadata_path)
    load_result = load_natural_degradation_records(metadata_path)
    records_by_image: dict[str, list[NaturalDegradationRecord]] = defaultdict(list)
    for record in load_result.records:
        records_by_image[record.image_id].append(record)
    image_dir = args.image_dir.expanduser().resolve(strict=False)
    selected_records: list[NaturalDegradationRecord] = []
    image_paths: dict[str, str] = {}
    for image_id in image_ids:
        records = records_by_image.get(image_id, [])
        if not records:
            raise ValueError(f"development image has no training metadata objects: {image_id}")
        path = _resolve_image_path(image_dir, image_id)
        image_paths[image_id] = str(path.resolve(strict=False))
        selected_records.extend(records)
    intervention_identities = select_intervention_objects(
        selected_records,
        image_ids,
        audit_seed=DEVELOPMENT_AUDIT_SEED,
    )
    if not intervention_identities:
        raise ValueError("development observer requires at least one selected intervention object")
    manifest = build_factor_observation_manifest(
        selected_records,
        image_paths,
        intervention_identities,
        str(checkpoint["sha256"]),
        DEVELOPMENT_SEED,
        required_nodes=DEFAULT_REQUIRED_NODES,
        input_size=args.input_size,
    )
    if manifest.image_ids != image_ids:
        raise ValueError("observer manifest image IDs do not match development protocol")
    scientific = {
        "stage": "development",
        "protocol": "factor_development_seed17_three_view_v1",
        "condition": args.condition,
        "seed": DEVELOPMENT_SEED,
        "views": list(VIEW_ROLES),
        "image_ids": list(image_ids),
        "image_ids_hash": image_ids_hash,
        "metadata_sha256": metadata_sha256,
        "checkpoint": checkpoint,
        "checkpoint_roles": roles_payload,
        "manifest_sha256": manifest.hash(),
        "required_nodes": list(DEFAULT_REQUIRED_NODES),
        "input_size": int(args.input_size),
        "audit_seed": DEVELOPMENT_AUDIT_SEED,
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "registered_severities": list(DEFAULT_INTERVENTION_SEVERITIES),
        "confidence": AUDIT_CONFIDENCE,
        "monotonic_threshold": MONOTONIC_THRESHOLD,
    }
    ensure_output_reusable(output_dir, scientific)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(output_dir, manifest)
    runtime = {
        "device": str(args.device),
        "transform_batch_size": int(args.transform_batch_size),
        "python_executable": sys.executable,
        "paths": {
            "metadata_jsonl": str(metadata_path),
            "development_ids": str(args.development_ids.expanduser().resolve(strict=False)),
            "image_dir": str(image_dir),
            "checkpoint": str(checkpoint["path"]),
            "checkpoint_roles": str(args.checkpoint_roles.expanduser().resolve(strict=False)),
        },
    }
    _same_json_file(
        output_dir / "provenance.json",
        {"schema_version": 1, "scientific_identity": scientific, "runtime": runtime},
    )
    status_path = output_dir / "status.json"
    _atomic_write_json(status_path, {"schema_version": 1, "status": "running"})
    try:
        journal = FactorObservationJournal(
            manifest,
            output_dir / "observations.jsonl",
            output_dir / "progress.json",
        )
        loaded = load_ifdr_checkpoint(checkpoint["path"], device=args.device)
        loaded_hash = getattr(loaded, "checkpoint_sha256", None)
        if loaded_hash != checkpoint["sha256"]:
            raise ValueError("loaded checkpoint hash does not match checkpoint_roles")
        run_factor_observer(
            loaded,
            manifest,
            journal,
            transform_batch_size=args.transform_batch_size,
        )
        FactorObservationJournal(
            manifest,
            output_dir / "observations.jsonl",
            output_dir / "progress.json",
        )
        rows = _load_observation_rows(output_dir / "observations.jsonl")
        if tuple(sorted({row.image_id for row in rows})) != image_ids:
            raise ValueError("raw observation image IDs do not match development protocol")
        decision = audit_natural_factors(
            rows,
            required_seeds=(DEVELOPMENT_SEED,),
            required_nodes=DEFAULT_REQUIRED_NODES,
            monotonic_threshold=MONOTONIC_THRESHOLD,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=DEVELOPMENT_AUDIT_SEED,
            confidence=AUDIT_CONFIDENCE,
            expected_intervention_severities=DEFAULT_INTERVENTION_SEVERITIES,
        )
        write_audit_artifacts(
            output_dir,
            condition=args.condition,
            rows=rows,
            decision=decision,
            image_ids=image_ids,
            image_ids_hash=image_ids_hash,
            checkpoint=checkpoint,
            manifest=manifest,
            selected_interventions=intervention_identities,
        )
        _atomic_write_json(status_path, {"schema_version": 1, "status": "complete"})
        return 0
    except Exception as exc:
        _atomic_write_json(
            status_path,
            {
                "schema_version": 1,
                "status": "failed",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    try:
        import torch

        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        default_device = "cpu"
    parser = argparse.ArgumentParser(
        description="Run fixed-seed development factor observer for one F0-F3 condition."
    )
    parser.add_argument("--condition", choices=("F0", "F1", "F2", "F3"), required=True)
    parser.add_argument("--checkpoint", type=_checkpoint_arg, required=True, help="calibration last.pt")
    parser.add_argument("--checkpoint-roles", type=Path, required=True)
    parser.add_argument("--development-ids", type=Path, required=True)
    parser.add_argument("--metadata-jsonl", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--transform-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.set_defaults(seed=DEVELOPMENT_SEED, views=VIEW_ROLES, audit_seed=DEVELOPMENT_AUDIT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except Exception as exc:
        print(f"development factor observer failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEVELOPMENT_AUDIT_SEED",
    "DEVELOPMENT_IMAGE_COUNT",
    "DEVELOPMENT_SEED",
    "VIEW_ROLES",
    "build_parser",
    "ensure_output_reusable",
    "load_development_ids",
    "main",
    "resolve_calibration_checkpoint",
    "write_audit_artifacts",
]
