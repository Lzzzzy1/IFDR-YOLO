"""Resumable, provenance-bound CLI for the natural-factor evidence gate.

This module intentionally owns only orchestration.  Manifest construction,
exactly-once JSONL journaling, checkpoint loading, and the statistical gate
remain implemented by :mod:`ifdr_yolo` and are called through their public
APIs here.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.data.natural_degradation import (
    NaturalDegradationLoadResult,
    NaturalDegradationRecord,
    load_natural_degradation_records,
)
from ifdr_yolo.eval.factor_observer import (
    DEFAULT_REQUIRED_NODES,
    FactorObservationJournal,
    FactorObservationManifest,
    build_factor_observation_manifest,
)
from ifdr_yolo.eval.factor_observer_runtime import (
    load_ifdr_checkpoint,
    run_factor_observer,
)
from ifdr_yolo.eval.natural_factor_audit import (
    DEFAULT_INTERVENTION_SEVERITIES,
    NaturalFactorObservation,
    audit_natural_factors,
)


REQUIRED_SEEDS = (17, 29, 41)
CLASS_NAMES = ("Car", "Pedestrian", "Cyclist")
SELECTION_STRATEGY = "joint_degradation_max_one_per_class_v1"
AUDIT_CONFIDENCE = 0.95
MONOTONIC_THRESHOLD = 0.80


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
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
    _atomic_write_bytes(path, (_canonical_json(_json_safe(payload)) + "\n").encode("utf-8"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"unable to read file: {path}") from exc
    if not raw:
        raise ValueError(f"file is empty: {path}")
    return _sha256_bytes(raw)


def _normalized_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def load_strict_ids(path: str | Path) -> tuple[str, ...]:
    """Load canonical six-digit KITTI IDs without silently dropping blanks."""

    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"unable to read ID file: {path}") from exc
    if not lines:
        raise ValueError(f"ID file is empty: {path}")
    image_ids: list[str] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(lines, start=1):
        if raw != raw.strip():
            raise ValueError(f"image_id has surrounding whitespace at {path}:{line_number}")
        image_id = raw.strip()
        if not image_id:
            raise ValueError(f"blank image_id at {path}:{line_number}")
        if len(image_id) != 6 or not image_id.isdigit():
            raise ValueError(f"invalid KITTI image ID at {path}:{line_number}: {image_id!r}")
        if image_id in seen:
            raise ValueError(f"duplicate image ID in split file: {path}: {image_id}")
        seen.add(image_id)
        image_ids.append(image_id)
    return tuple(image_ids)


def validate_split_ids(train_ids: Sequence[str], val_ids: Sequence[str]) -> None:
    train = tuple(train_ids)
    val = tuple(val_ids)
    if not train:
        raise ValueError("train IDs must not be empty")
    if len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ValueError("train/val IDs must not contain duplicates")
    overlap = sorted(set(train) & set(val))
    if overlap:
        raise ValueError(f"train/val overlap: {overlap[:5]}")


def select_audit_image_ids(
    train_ids: Sequence[str],
    val_ids: Sequence[str] = (),
    *,
    audit_seed: int = 20260804,
) -> tuple[str, ...]:
    """Select deterministic 20% train-only image clusters."""

    if isinstance(audit_seed, bool) or not isinstance(audit_seed, int) or audit_seed < 0:
        raise ValueError("audit_seed must be a non-negative integer")
    validate_split_ids(train_ids, val_ids)
    train = tuple(train_ids)
    val_set = set(val_ids)
    if any(image_id in val_set for image_id in train):
        raise ValueError("audit selection cannot contain validation IDs")
    count = max(1, math.floor(0.20 * len(train)))
    ranked = sorted(
        train,
        key=lambda image_id: (
            hashlib.sha256(f"{int(audit_seed)}{image_id}".encode("utf-8")).hexdigest(),
            image_id,
        ),
    )
    return tuple(sorted(ranked[:count]))


def select_intervention_objects(
    records: Iterable[NaturalDegradationRecord],
    audit_image_ids: Sequence[str],
    *,
    audit_seed: int = 20260804,
) -> tuple[tuple[str, int], ...]:
    """Choose at most one object per image/class by joint natural degradation."""

    if isinstance(audit_seed, bool) or not isinstance(audit_seed, int) or audit_seed < 0:
        raise ValueError("audit_seed must be a non-negative integer")
    selected_images = set(audit_image_ids)
    grouped: dict[tuple[str, str], list[NaturalDegradationRecord]] = defaultdict(list)
    for record in records:
        if not isinstance(record, NaturalDegradationRecord):
            raise ValueError("records must contain NaturalDegradationRecord values")
        if record.image_id in selected_images:
            if record.class_name not in CLASS_NAMES:
                raise ValueError(f"unsupported training class: {record.class_name!r}")
            grouped[(record.image_id, record.class_name)].append(record)
    chosen: list[tuple[str, int]] = []
    for (image_id, class_name), candidates in sorted(grouped.items()):
        if not candidates:
            continue

        def rank(record: NaturalDegradationRecord) -> tuple[float, str, int]:
            joint = 1.0 - (1.0 - float(record.sampling_score)) * (
                1.0 - float(record.visibility_score)
            )
            tie = hashlib.sha256(
                f"{int(audit_seed)}\0{image_id}\0{class_name}\0{record.object_id}".encode(
                    "utf-8"
                )
            ).hexdigest()
            return (-joint, tie, int(record.object_id))

        winner = min(candidates, key=rank)
        chosen.append((image_id, int(winner.object_id)))
    return tuple(sorted(chosen))


def _parse_checkpoint(value: str) -> tuple[int, Path]:
    if not isinstance(value, str) or "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be SEED=PATH")
    raw_seed, raw_path = value.split("=", 1)
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checkpoint seed must be an integer") from exc
    if not raw_path.strip():
        raise argparse.ArgumentTypeError("checkpoint path must not be empty")
    return seed, Path(raw_path)


class _SeverityAction(argparse.Action):
    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: str, option_string: str | None = None) -> None:
        current = getattr(namespace, self.dest, None)
        if current is None or current == DEFAULT_INTERVENTION_SEVERITIES:
            current = []
        else:
            current = list(current)
        raw_values = values.split(",") if isinstance(values, str) and "," in values else (values,)
        try:
            current.extend(float(value) for value in raw_values)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentError(self, "severity must be numeric") from exc
        setattr(namespace, self.dest, current)


def _validate_severities(values: Sequence[float]) -> tuple[float, ...]:
    canonical = tuple(float(value) for value in values)
    expected = tuple(DEFAULT_INTERVENTION_SEVERITIES)
    if len(canonical) != len(expected) or any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
        for left, right in zip(canonical, expected)
    ):
        raise ValueError(
            "registered severities are frozen to (0.25, 0.5, 0.75, 1.0)"
        )
    return expected


def build_parser() -> argparse.ArgumentParser:
    try:
        import torch

        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        default_device = "cpu"
    parser = argparse.ArgumentParser(
        description="Run a resumable natural-factor IFDR evidence audit."
    )
    parser.add_argument("--checkpoint", action="append", required=True, metavar="SEED=PATH")
    parser.add_argument("--metadata-jsonl", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--val-ids", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--transform-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--audit-seed", type=int, default=20260804)
    parser.add_argument(
        "--registered-severity",
        dest="registered_severities",
        action=_SeverityAction,
        metavar="LEVEL",
        default=DEFAULT_INTERVENTION_SEVERITIES,
        help="registered intervention severity (repeat exactly 0.25, 0.5, 0.75, 1.0)",
    )
    return parser


def _checkpoint_hashes(specs: Sequence[str]) -> tuple[dict[int, Path], dict[int, str]]:
    parsed = [_parse_checkpoint(item) for item in specs]
    if {seed for seed, _ in parsed} != set(REQUIRED_SEEDS) or len(parsed) != len(REQUIRED_SEEDS):
        raise ValueError("checkpoints must contain exactly one path for seeds 17, 29, and 41")
    paths: dict[int, Path] = {}
    hashes: dict[int, str] = {}
    for seed, raw_path in parsed:
        if seed in paths:
            raise ValueError(f"duplicate checkpoint seed: {seed}")
        path = raw_path.expanduser().resolve(strict=False)
        if not path.is_file():
            raise ValueError(f"checkpoint path does not exist: {path}")
        digest = _sha256_file(path)
        paths[seed] = path
        hashes[seed] = digest
    return paths, hashes


def _split_selection_hash(train_ids: Sequence[str], val_ids: Sequence[str], audit_ids: Sequence[str], seed: int) -> str:
    payload = {
        "audit_seed": int(seed),
        "train_ids": list(train_ids),
        "val_ids": list(val_ids),
        "audit_ids": list(audit_ids),
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parents[1],
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _manifest_hash_file(path: Path, manifest: FactorObservationManifest) -> None:
    hash_path = path.with_name("manifest.sha256")
    if (
        path.exists()
        or path.is_symlink()
        or hash_path.exists()
        or hash_path.is_symlink()
    ):
        _validate_existing_manifest(path.parent, manifest)
        return
    _atomic_write_json(path, manifest.to_dict())
    _atomic_write_bytes(
        hash_path,
        (manifest.hash() + "\n").encode("ascii"),
    )


def _validate_existing_manifest(seed_dir: Path, manifest: FactorObservationManifest) -> None:
    manifest_path = seed_dir / "manifest.json"
    hash_path = seed_dir / "manifest.sha256"
    manifest_present = manifest_path.exists() or manifest_path.is_symlink()
    hash_present = hash_path.exists() or hash_path.is_symlink()
    if not manifest_present and not hash_present:
        return
    if (
        manifest_path.is_symlink()
        or hash_path.is_symlink()
        or not manifest_path.is_file()
        or not hash_path.is_file()
    ):
        raise ValueError(f"seed manifest files are incomplete: {seed_dir}")
    try:
        raw_manifest = manifest_path.read_bytes()
        raw_hash = hash_path.read_bytes()
        payload = json.loads(
            raw_manifest.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, UnicodeDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"seed manifest is malformed: {seed_dir}") from exc
    try:
        matches = _canonical_json(payload) == _canonical_json(manifest.to_dict())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"seed manifest is malformed: {seed_dir}") from exc
    if not matches:
        raise ValueError(f"seed manifest does not match current manifest: {seed_dir}")
    try:
        existing_hash = raw_hash.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(f"seed manifest hash is malformed: {seed_dir}") from exc
    if existing_hash != manifest.hash():
        raise ValueError(f"seed manifest hash does not match current manifest: {seed_dir}")


def _load_observation_rows(
    path: Path,
    *,
    expected_observation_ids: set[str] | None = None,
) -> tuple[NaturalFactorObservation, ...]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"observation JSONL is missing or empty: {path}")
    rows: list[NaturalFactorObservation] = []
    seen: set[str] = set()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.endswith(b"\n"):
                raise ValueError(f"observation JSONL has unterminated line: {path}:{line_number}")
            try:
                payload = json.loads(raw.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            except (UnicodeDecodeError, ValueError, TypeError) as exc:
                raise ValueError(f"malformed observation JSON at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError("observation rows must be JSON objects")
            observation_id = payload.get("observation_id")
            if not isinstance(observation_id, str) or observation_id in seen:
                raise ValueError("observation rows contain duplicate or invalid observation_id")
            seen.add(observation_id)
            try:
                row = NaturalFactorObservation(
                    seed=payload["seed"],
                    node_id=payload["node_id"],
                    image_id=payload["image_id"],
                    object_id=payload["object_id"],
                    class_id=payload["class_id"],
                    class_name=payload.get("class_name"),
                    box_height=payload["box_height"],
                    region_role=payload["region_role"],
                    intervention_kind=payload["intervention_kind"],
                    intervention_factor=payload.get("intervention_factor"),
                    intervention_severity=payload["intervention_severity"],
                    pair_id=payload.get("pair_id"),
                    natural_sampling=payload["natural_sampling"],
                    natural_visibility=payload["natural_visibility"],
                    predicted_sampling=payload["predicted_sampling"],
                    predicted_visibility=payload["predicted_visibility"],
                    branch_weights=tuple(payload["branch_weights"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid observation row at {path}:{line_number}") from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"observation JSONL is empty: {path}")
    if expected_observation_ids is not None and seen != set(expected_observation_ids):
        missing = sorted(set(expected_observation_ids) - seen)
        extra = sorted(seen - set(expected_observation_ids))
        raise ValueError(
            f"root observation identities do not match manifests (missing={missing[:5]}, extra={extra[:5]})"
        )
    return tuple(rows)


def _rebuild_root_observations(output_dir: Path, seeds: Sequence[int]) -> Path:
    root = output_dir / "observations.jsonl"
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = root.with_name(f".{root.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as destination:
            for seed in seeds:
                path = output_dir / f"seed-{seed}" / "observations.jsonl"
                if not path.is_file() or path.stat().st_size <= 0:
                    raise ValueError(f"seed observation JSONL is missing or empty: {path}")
                source_size = 0
                last_byte = b""
                with path.open("rb") as source:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        source_size += len(chunk)
                        last_byte = chunk[-1:]
                        destination.write(chunk)
                if source_size == 0 or last_byte != b"\n":
                    raise ValueError(f"seed observation JSONL is not newline terminated: {path}")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, root)
        try:
            directory_fd = os.open(str(root.parent), os.O_RDONLY)
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
    return root


def _observation_summary(
    rows: Sequence[NaturalFactorObservation],
    *,
    selected_interventions: Sequence[tuple[str, int]],
    load_result: NaturalDegradationLoadResult,
    decision: object,
) -> dict[str, object]:
    natural = [row for row in rows if row.intervention_kind == "natural"]
    images = sorted({row.image_id for row in natural})
    objects = sorted({(row.image_id, row.object_id) for row in natural})
    seed_counts: Counter[str] = Counter(str(row.seed) for row in rows)
    node_counts: Counter[str] = Counter(str(row.node_id) for row in rows)
    class_counts: Counter[str] = Counter(row.class_name or str(row.class_id) for row in natural)
    seed_node_class: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for row in rows:
        seed_node_class[str(row.seed)][str(row.node_id)][
            row.class_name or str(row.class_id)
        ] += 1
    if hasattr(decision, "to_dict"):
        gate = decision.to_dict()
    elif isinstance(decision, Mapping):
        gate = dict(decision)
    else:
        gate = {"passed": bool(getattr(decision, "gate_passed", False))}
    gate_passed = bool(gate.get("passed", gate.get("gate_passed", False)))
    factors = gate.get("factors", {})
    sampling_pass = bool(isinstance(factors, Mapping) and factors.get("sampling", {}).get("passed", False))
    visibility_pass = bool(isinstance(factors, Mapping) and factors.get("visibility", {}).get("passed", False))
    return {
        "schema_version": 1,
        "images": len(images),
        "objects": len(objects),
        "intervention_objects": len(selected_interventions),
        "row_count": len(rows),
        "per_seed_counts": dict(sorted(seed_counts.items())),
        "per_node_counts": dict(sorted(node_counts.items())),
        "per_class_counts": dict(sorted(class_counts.items())),
        "per_seed_node_class_counts": {
            seed: {
                node: dict(sorted(classes.items()))
                for node, classes in sorted(nodes.items())
            }
            for seed, nodes in sorted(seed_node_class.items())
        },
        "skipped_non_training_count": load_result.skipped_non_training_count,
        "invalid_depth_count": load_result.invalid_depth_count,
        "gate_passed": gate_passed,
        "sampling_pass": sampling_pass,
        "visibility_pass": visibility_pass,
    }


def _scientific_identity(
    *,
    metadata_sha256: str,
    train_sha256: str,
    val_sha256: str,
    checkpoint_sha256: Mapping[int, str],
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    audit_ids: Sequence[str],
    audit_seed: int,
    intervention_identities: Sequence[tuple[str, int]],
    intervention_class_counts: Mapping[str, int],
    manifest_hashes: Mapping[int, str],
    input_size: int,
    bootstrap_replicates: int,
    severities: Sequence[float],
    confidence: float,
    monotonic_threshold: float,
    implementation_git_commit: str | None,
) -> dict[str, object]:
    return {
        "metadata_sha256": metadata_sha256,
        "train_ids_sha256": train_sha256,
        "val_ids_sha256": val_sha256,
        "checkpoint_sha256": {str(seed): checkpoint_sha256[seed] for seed in REQUIRED_SEEDS},
        "train_ids": list(train_ids),
        "val_ids": list(val_ids),
        "audit_ids": list(audit_ids),
        "audit_seed": int(audit_seed),
        "split_selection_sha256": _split_selection_hash(train_ids, val_ids, audit_ids, audit_seed),
        "intervention_selection_strategy": SELECTION_STRATEGY,
        "intervention_identities": [[image_id, object_id] for image_id, object_id in intervention_identities],
        "intervention_class_counts": dict(sorted(intervention_class_counts.items())),
        "manifest_sha256": {str(seed): manifest_hashes[seed] for seed in REQUIRED_SEEDS},
        "required_seeds": list(REQUIRED_SEEDS),
        "required_nodes": list(DEFAULT_REQUIRED_NODES),
        "input_size": int(input_size),
        "bootstrap_replicates": int(bootstrap_replicates),
        "registered_severities": list(severities),
        "confidence": float(confidence),
        "monotonic_threshold": float(monotonic_threshold),
        "implementation_git_commit": implementation_git_commit,
    }


def _validate_existing_provenance(path: Path, scientific: Mapping[str, object]) -> None:
    if not (path.exists() or path.is_symlink()):
        return
    if path.is_symlink() or not path.is_file():
        raise ValueError("provenance.json is not a regular file")
    try:
        existing = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("provenance.json is malformed") from exc
    if not isinstance(existing, dict):
        raise ValueError("provenance scientific identity mismatch; refusing to resume")
    try:
        matches = _canonical_json(existing.get("scientific_identity")) == _canonical_json(dict(scientific))
    except (TypeError, ValueError) as exc:
        raise ValueError("provenance.json is malformed") from exc
    if not matches:
        raise ValueError("provenance scientific identity mismatch; refusing to resume")


def _check_or_write_provenance(path: Path, scientific: Mapping[str, object], runtime: Mapping[str, object]) -> None:
    _validate_existing_provenance(path, scientific)
    payload = {
        "schema_version": 1,
        "scientific_identity": dict(scientific),
        "runtime": dict(runtime),
        "git_commit": _git_commit(),
    }
    _atomic_write_json(path, payload)


def _run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    provenance_path = output_dir / "provenance.json"
    protect_existing = provenance_path.exists() or provenance_path.is_symlink()
    status_started = False
    try:
        if args.input_size <= 0 or args.transform_batch_size <= 0 or args.bootstrap_replicates < 2:
            raise ValueError("input-size, transform-batch-size must be positive and bootstrap-replicates >= 2")
        severities = _validate_severities(args.registered_severities)
        train_ids = load_strict_ids(args.train_ids)
        val_ids = load_strict_ids(args.val_ids)
        validate_split_ids(train_ids, val_ids)
        audit_ids = select_audit_image_ids(train_ids, val_ids, audit_seed=args.audit_seed)
        checkpoint_paths, checkpoint_hashes = _checkpoint_hashes(args.checkpoint)
        metadata_path = args.metadata_jsonl.expanduser().resolve(strict=False)
        metadata_sha256 = _sha256_file(metadata_path)
        load_result = load_natural_degradation_records(metadata_path)
        records_by_image: dict[str, list[NaturalDegradationRecord]] = defaultdict(list)
        for record in load_result.records:
            records_by_image[record.image_id].append(record)
        selected_records: list[NaturalDegradationRecord] = []
        image_paths: dict[str, str] = {}
        image_dir = args.image_dir.expanduser().resolve(strict=False)
        for image_id in audit_ids:
            records = records_by_image.get(image_id, [])
            if not records:
                raise ValueError(f"selected train image has no training metadata objects: {image_id}")
            image_path = image_dir / f"{image_id}.png"
            if not image_path.is_file() or image_path.stat().st_size <= 0:
                raise ValueError(f"selected train image PNG is missing or empty: {image_path}")
            image_paths[image_id] = _normalized_path(image_path)
            selected_records.extend(records)
        if not selected_records:
            raise ValueError("audit requires at least one training object")
        intervention_identities = select_intervention_objects(
            selected_records, audit_ids, audit_seed=args.audit_seed
        )
        class_counts = Counter(
            next(record.class_name for record in selected_records if (record.image_id, record.object_id) == identity)
            for identity in intervention_identities
        )
        manifests: dict[int, FactorObservationManifest] = {}
        manifest_hashes: dict[int, str] = {}
        for seed in REQUIRED_SEEDS:
            manifest = build_factor_observation_manifest(
                selected_records,
                image_paths,
                intervention_identities,
                checkpoint_hashes[seed],
                seed,
                required_nodes=DEFAULT_REQUIRED_NODES,
                input_size=args.input_size,
            )
            manifests[seed] = manifest
            manifest_hashes[seed] = manifest.hash()
        implementation_git_commit = _git_commit()
        scientific = _scientific_identity(
            metadata_sha256=metadata_sha256,
            train_sha256=_sha256_file(args.train_ids.expanduser().resolve(strict=False)),
            val_sha256=_sha256_file(args.val_ids.expanduser().resolve(strict=False)),
            checkpoint_sha256=checkpoint_hashes,
            train_ids=train_ids,
            val_ids=val_ids,
            audit_ids=audit_ids,
            audit_seed=args.audit_seed,
            intervention_identities=intervention_identities,
            intervention_class_counts=class_counts,
            manifest_hashes=manifest_hashes,
            input_size=args.input_size,
            bootstrap_replicates=args.bootstrap_replicates,
            severities=severities,
            confidence=AUDIT_CONFIDENCE,
            monotonic_threshold=MONOTONIC_THRESHOLD,
            implementation_git_commit=implementation_git_commit,
        )
        _validate_existing_provenance(provenance_path, scientific)
        for seed in REQUIRED_SEEDS:
            _validate_existing_manifest(output_dir / f"seed-{seed}", manifests[seed])

        runtime = {
            "device": str(args.device),
            "transform_batch_size": int(args.transform_batch_size),
            "python_executable": sys.executable,
            "paths": {
                "metadata_jsonl": _normalized_path(metadata_path),
                "train_ids": _normalized_path(args.train_ids),
                "val_ids": _normalized_path(args.val_ids),
                "image_dir": _normalized_path(image_dir),
                "checkpoints": {
                    str(seed): _normalized_path(checkpoint_paths[seed])
                    for seed in REQUIRED_SEEDS
                },
            },
        }

        _atomic_write_json(status_path, {"schema_version": 1, "status": "running"})
        status_started = True
        _check_or_write_provenance(provenance_path, scientific, runtime)
        for seed in REQUIRED_SEEDS:
            seed_dir = output_dir / f"seed-{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            manifest = manifests[seed]
            _manifest_hash_file(seed_dir / "manifest.json", manifest)
            journal = FactorObservationJournal(
                manifest,
                seed_dir / "observations.jsonl",
                seed_dir / "progress.json",
            )
            loaded = load_ifdr_checkpoint(checkpoint_paths[seed], device=args.device)
            loaded_hash = getattr(loaded, "checkpoint_sha256", None)
            if loaded_hash is not None and loaded_hash != checkpoint_hashes[seed]:
                raise ValueError(f"loaded checkpoint hash mismatch for seed {seed}")
            run_factor_observer(
                loaded,
                manifest,
                journal,
                transform_batch_size=args.transform_batch_size,
            )
            # Re-open after the runner returns so root reconstruction only
            # consumes a seed file that still validates against its manifest
            # and exactly-once progress journal.
            FactorObservationJournal(
                manifest,
                seed_dir / "observations.jsonl",
                seed_dir / "progress.json",
            )
        root_observations = _rebuild_root_observations(output_dir, REQUIRED_SEEDS)
        expected_root_ids = {
            observation_id
            for manifest in manifests.values()
            for observation_id in manifest.expected_observation_ids
        }
        rows = _load_observation_rows(
            root_observations,
            expected_observation_ids=expected_root_ids,
        )
        decision = audit_natural_factors(
            rows,
            required_seeds=REQUIRED_SEEDS,
            required_nodes=DEFAULT_REQUIRED_NODES,
            monotonic_threshold=MONOTONIC_THRESHOLD,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.audit_seed,
            confidence=AUDIT_CONFIDENCE,
            expected_intervention_severities=severities,
        )
        gate_payload = decision.to_dict() if hasattr(decision, "to_dict") else dict(decision)
        summary_payload = _observation_summary(
            rows,
            selected_interventions=intervention_identities,
            load_result=load_result,
            decision=decision,
        )
        _atomic_write_json(output_dir / "summary.json", summary_payload)
        _atomic_write_json(output_dir / "gate.json", gate_payload)
        _check_or_write_provenance(provenance_path, scientific, runtime)
        required_files = (
            root_observations,
            output_dir / "summary.json",
            output_dir / "gate.json",
            provenance_path,
            status_path,
        )
        if any(not path.is_file() or path.stat().st_size <= 0 for path in required_files):
            raise ValueError("audit did not produce all required non-empty artifacts")
        _atomic_write_json(status_path, {"schema_version": 1, "status": "complete"})
        return 0
    except Exception as exc:
        if status_started or not protect_existing:
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except Exception as exc:
        print(f"natural factor audit failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_CONFIDENCE",
    "MONOTONIC_THRESHOLD",
    "REQUIRED_SEEDS",
    "SELECTION_STRATEGY",
    "build_parser",
    "load_strict_ids",
    "main",
    "select_audit_image_ids",
    "select_intervention_objects",
    "validate_split_ids",
]
