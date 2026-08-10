"""Select one development factor-repair candidate against F0.

The command is deliberately CPU-only: it loads four persisted evidence
bundles, checks their immutable shared image cluster, delegates paired
resampling and candidate selection to the registered gate APIs, and writes a
single-use selection record plus a human-readable mechanism table.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.data.replay_sampler import sha256_canonical
from ifdr_yolo.eval.factor_repair_evidence import (
    load_factor_repair_evidence,
    validate_shared_image_identity,
)
from ifdr_yolo.eval.factor_repair_gate import (
    PRIMARY_ENDPOINTS,
    select_repair_against_f0,
)
from ifdr_yolo.eval.resumable_factor_bootstrap import (
    DEFAULT_CHECKPOINT_INTERVAL,
    build_shared_reference_draws,
    run_resumable_factor_bootstrap,
)

from ifdr_yolo.eval.factor_repair_gate import paired_image_cluster_delta


DEVELOPMENT_SEED = 17
CONDITIONS = ("F0", "F1", "F2", "F3")
FORMAL_REPLICATES = 10_000


def _stream_file_sha256(path: Path) -> str:
    """Hash one registered evidence source without materializing its bytes."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"unable to hash evidence source file: {path}") from error
    return digest.hexdigest()


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


def _atomic_create(path: Path, payload: bytes) -> None:
    """Create one artifact atomically and refuse any existing destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to overwrite existing artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
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
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_once(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_create(path, (_canonical_json(_json_safe(payload)) + "\n").encode("utf-8"))


def _write_final_artifact(path: Path, payload: bytes, expected_sha256: str) -> None:
    """Create an output once, or verify an already-created identical payload."""

    if path.exists() or path.is_symlink():
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError(f"unable to validate existing final artifact: {path}") from error
        if actual != expected_sha256:
            raise ValueError(f"final artifact hash mismatch: {path}")
        return
    _atomic_create(path, payload)


def _finalization_identity(
    evidences: Mapping[str, object],
    source_file_hashes: Mapping[str, str],
    image_ids_hash: str,
    *,
    selected_condition: str | None,
) -> dict[str, object]:
    return {
        "evidence_canonical_sha256": {
            condition: str(getattr(evidences[condition], "evidence_sha256")) for condition in CONDITIONS
        },
        "source_file_sha256": dict(source_file_hashes),
        "image_ids_hash": image_ids_hash,
        "selected_condition": selected_condition,
    }


def _read_finalization(path: Path, expected_identity: Mapping[str, object]) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"finalization journal is missing or malformed: {path}") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("finalization journal is malformed")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("finalization journal identity is missing")
    for key in ("evidence_canonical_sha256", "source_file_sha256", "image_ids_hash"):
        if _json_safe(identity.get(key)) != _json_safe(expected_identity.get(key)):
            raise ValueError(f"finalization identity mismatch: {key}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("finalization journal artifacts are missing")
    return dict(payload)


def _write_finalization(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (_canonical_json(_json_safe(payload)) + "\n").encode("utf-8")
    _atomic_replace(path, encoded)


def _atomic_replace(path: Path, payload: bytes) -> None:
    """Atomically replace a mirror state file after validating JSON bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    # The three final artifacts include one CSV.  JSON support files remain
    # validated before replacement; the CSV is validated by its manifest hash
    # and exact byte copy instead.
    if path.suffix.lower() != ".csv":
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"mirror payload is not valid JSON: {path.name}") from error
        if not isinstance(decoded, Mapping):
            raise ValueError(f"mirror payload must be an object: {path.name}")
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
            except OSError:
                pass
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _mirror_publish(
    mirror_dir: Path,
    *,
    stage: str,
    output_dir: Path,
    checkpoint_dir: Path,
    conditions: Mapping[str, object] | None = None,
    selected_condition: str | None = None,
    resume: bool,
    workers: int,
    final_artifacts: Mapping[str, bytes] | None = None,
) -> None:
    """Persist a small independent mirror with a last-written commit marker."""

    artifact_names = ("selection_decision.json", "mechanism_table.json", "mechanism_table.csv")
    if stage == "complete":
        if not isinstance(final_artifacts, Mapping) or set(final_artifacts) != set(artifact_names):
            raise ValueError("final_artifacts are required for the complete mirror stage")
        if any(not isinstance(final_artifacts[name], bytes) for name in artifact_names):
            raise ValueError("final_artifacts must contain exact byte payloads")
    mirror_dir.mkdir(parents=True, exist_ok=True)
    generation = str(time.time_ns())
    def support_payload(payload: Mapping[str, object]) -> dict[str, object]:
        body = {**payload, "generation": generation}
        body["payload_sha256"] = sha256_canonical(body)
        return body

    summary = {
        "schema_version": 1,
        "stage": stage,
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "selected_condition": selected_condition,
        "conditions": dict(conditions or {}),
        "workers": int(workers),
        "formal_replicates": FORMAL_REPLICATES,
        "resume": bool(resume),
        "updated_at": time.time(),
    }
    checkpoint_index = {
        "schema_version": 1,
        "stage": stage,
        "checkpoint_dir": str(checkpoint_dir),
        "paths": {
            condition: str(checkpoint_dir / ("F0.reference.json" if condition == "F0" else f"{condition}.json"))
            for condition in CONDITIONS
        },
        "states": dict(conditions or {}),
    }
    progress = {
        "schema_version": 1,
        "stage": stage,
        "conditions": dict(conditions or {}),
        "selected_condition": selected_condition,
        "updated_at": summary["updated_at"],
    }
    resume_text = (
        "Factor bootstrap mirror\n"
        f"stage={stage}\n"
        f"checkpoint_dir={checkpoint_dir}\n"
        f"output_dir={output_dir}\n"
        "resume command: rerun the formal CLI with --resume and the same input evidence, "
        "checkpoint-dir, and mirror-dir.\n"
    )
    support = {
        "checkpoint_index.json": support_payload(checkpoint_index),
        "progress.json": support_payload(progress),
        "summary.json": support_payload(summary),
        "resume.txt": support_payload({
            "schema_version": 1,
            "kind": "resume-instructions",
            "text": resume_text,
        }),
    }
    file_hashes: dict[str, str] = {}
    for name, payload in support.items():
        encoded = (_canonical_json(_json_safe(payload)) + "\n").encode("utf-8")
        _atomic_replace(mirror_dir / name, encoded)
        file_hashes[name] = hashlib.sha256(encoded).hexdigest()
    artifact_hashes: dict[str, dict[str, object]] = {}
    if stage == "complete":
        assert final_artifacts is not None
        for name in artifact_names:
            encoded = final_artifacts[name]
            _atomic_replace(mirror_dir / name, encoded)
            artifact_hashes[name] = {
                "path": str(mirror_dir / name),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "generation": generation,
            }
    # The manifest is deliberately written last.  It is the only commit
    # marker and proves that all support files belong to one generation.
    manifest = {
        "schema_version": 1,
        "kind": "factor-bootstrap-mirror",
        "stage": stage,
        "formal_replicates": FORMAL_REPLICATES,
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "mirror_dir": str(mirror_dir),
        "resume_required_for_existing_state": True,
        "generation": generation,
        "files": file_hashes,
        "artifacts": artifact_hashes,
        "updated_at": summary["updated_at"],
    }
    manifest_bytes = (_canonical_json(_json_safe(manifest)) + "\n").encode("utf-8")
    _atomic_replace(mirror_dir / "manifest.json", manifest_bytes)
    _validate_mirror_commit(mirror_dir)


def _validate_mirror_commit(mirror_dir: Path) -> dict[str, object]:
    """Validate the last manifest and every generation/hash it commits."""

    manifest = json.loads((mirror_dir / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("files"), Mapping):
        raise ValueError("mirror manifest is malformed")
    generation = manifest.get("generation")
    if not isinstance(generation, str) or not generation:
        raise ValueError("mirror manifest generation is missing")
    for name, expected_hash in manifest["files"].items():
        path = mirror_dir / str(name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("generation") != generation:
            raise ValueError(f"mirror generation mismatch: {name}")
        if payload.get("payload_sha256") != sha256_canonical({key: value for key, value in payload.items() if key != "payload_sha256"}):
            raise ValueError(f"mirror payload hash mismatch: {name}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"mirror file hash mismatch: {name}")
    if manifest.get("stage") == "complete":
        artifact_entries = manifest.get("artifacts")
        artifact_names = ("selection_decision.json", "mechanism_table.json", "mechanism_table.csv")
        if not isinstance(artifact_entries, Mapping) or set(artifact_entries) != set(artifact_names):
            raise ValueError("complete mirror artifacts are missing")
        for name in artifact_names:
            entry = artifact_entries.get(name)
            if not isinstance(entry, Mapping) or entry.get("generation") != generation:
                raise ValueError(f"mirror artifact generation mismatch: {name}")
            path = mirror_dir / name
            try:
                actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as error:
                raise ValueError(f"mirror artifact is missing: {name}") from error
            if actual_hash != entry.get("sha256"):
                raise ValueError(f"mirror artifact hash mismatch: {name}")
    return dict(manifest)


def _load_bundle(condition: str, path: Path) -> object:
    evidence = load_factor_repair_evidence(path)
    if getattr(evidence, "condition", None) != condition:
        raise ValueError(f"{condition} evidence condition does not match input artifact: {path}")
    if getattr(evidence, "stage", "development") != "development":
        raise ValueError(f"{condition} evidence must be development-stage evidence")
    if not bool(getattr(evidence, "complete", False)):
        raise ValueError(f"incomplete {condition} evidence")
    if getattr(evidence, "endpoint_samples", None) is not None:
        raise ValueError("point endpoint samples are not admissible evidence")
    if not callable(getattr(evidence, "recompute_endpoints", None)):
        raise ValueError(f"{condition} evidence has no raw image-cluster recompute capability")
    gate = getattr(evidence, "absolute_gate", None)
    gate_stage = getattr(gate, "stage", None)
    if gate_stage is not None and gate_stage != "development":
        raise ValueError(f"{condition} absolute gate must be development-stage")
    raw_rows = getattr(evidence, "raw_observations", None)
    if not isinstance(raw_rows, (tuple, list)) or not raw_rows:
        raise ValueError(f"{condition} evidence is missing raw observations")
    seeds = {row.get("seed") for row in raw_rows if isinstance(row, Mapping)}
    if seeds != {DEVELOPMENT_SEED}:
        raise ValueError(f"{condition} evidence must contain seed 17 raw observations only")
    return evidence


def _gate_payload(evidence: object) -> dict[str, object]:
    gate = getattr(evidence, "absolute_gate", None)
    if gate is None:
        raise ValueError("evidence absolute gate is missing")
    to_dict = getattr(gate, "to_dict", None)
    payload = to_dict() if callable(to_dict) else gate
    if not isinstance(payload, Mapping):
        raise ValueError("evidence absolute gate is malformed")
    return dict(_json_safe(payload))  # type: ignore[arg-type]


def _endpoint_payload(evidence: object) -> dict[str, float]:
    raw = getattr(evidence, "endpoints", None)
    if not isinstance(raw, Mapping):
        raise ValueError("evidence endpoints are missing")
    result: dict[str, float] = {}
    for name in PRIMARY_ENDPOINTS:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"evidence endpoint is missing or non-finite: {name}")
        result[name] = float(value)
    return result


def _failure_reasons(
    evidence: object,
    *,
    delta: object | None,
    selected: bool,
) -> list[str]:
    reasons: list[str] = []
    if not bool(getattr(evidence, "absolute_gate_passed", False)):
        gate = _gate_payload(evidence)
        failures = gate.get("failures", ())
        if isinstance(failures, (tuple, list)):
            reasons.extend(f"absolute_gate:{item}" for item in failures)
        if not reasons:
            reasons.append("absolute_gate:failed")
    if delta is None and getattr(evidence, "condition", None) != "F0":
        reasons.append("paired_delta:unavailable")
    elif delta is not None and not selected and bool(getattr(evidence, "absolute_gate_passed", False)):
        ci = tuple(getattr(delta, "ci95"))
        if len(ci) == 2 and float(ci[0]) <= 0.0:
            reasons.append("selector:paired_ci_lower_not_positive")
        else:
            reasons.append("selector:not_selected_by_existing_selector")
    return reasons


def _condition_row(
    condition: str,
    evidence: object,
    *,
    paired: object | None,
    selected: bool,
) -> dict[str, object]:
    row: dict[str, object] = {
        "condition": condition,
        "stage": "development",
        "seed": DEVELOPMENT_SEED,
        "absolute_gate_passed": bool(getattr(evidence, "absolute_gate_passed", False)),
        "complete": bool(getattr(evidence, "complete", False)),
        "evidence_sha256": str(getattr(evidence, "evidence_sha256")),
        "absolute_gate": _gate_payload(evidence),
        "endpoints": _endpoint_payload(evidence),
        "delta_s_point": None,
        "delta_s_ci95": None,
        "paired_candidate_endpoints": None,
        "promoted": bool(selected),
    }
    if paired is not None:
        ci = tuple(float(value) for value in getattr(paired, "ci95"))
        row["delta_s_point"] = float(getattr(paired, "point"))
        row["delta_s_ci95"] = [ci[0], ci[1]]
        row["paired_candidate_endpoints"] = dict(_json_safe(getattr(paired, "candidate_endpoints")))
    row["failure_reasons"] = _failure_reasons(evidence, delta=paired, selected=selected)
    return row


def _mechanism_csv(table: Mapping[str, object]) -> bytes:
    fields = [
        "condition",
        "absolute_gate_passed",
        "complete",
        "evidence_sha256",
        *PRIMARY_ENDPOINTS,
        "delta_s_point",
        "delta_s_ci95_lower",
        "delta_s_ci95_upper",
        "promoted",
        "failure_reasons",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    conditions = table.get("conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("mechanism table conditions are missing")
    for condition in CONDITIONS:
        row = conditions.get(condition)
        if not isinstance(row, Mapping):
            raise ValueError(f"mechanism table row is missing: {condition}")
        endpoints = row.get("endpoints")
        if not isinstance(endpoints, Mapping):
            raise ValueError(f"mechanism table endpoints are missing: {condition}")
        ci = row.get("delta_s_ci95")
        ci_values = tuple(ci) if isinstance(ci, (tuple, list)) else (None, None)
        reasons = row.get("failure_reasons", ())
        writer.writerow(
            {
                "condition": condition,
                "absolute_gate_passed": bool(row.get("absolute_gate_passed", False)),
                "complete": bool(row.get("complete", False)),
                "evidence_sha256": row.get("evidence_sha256", ""),
                **{name: endpoints.get(name) for name in PRIMARY_ENDPOINTS},
                "delta_s_point": row.get("delta_s_point"),
                "delta_s_ci95_lower": ci_values[0],
                "delta_s_ci95_upper": ci_values[1],
                "promoted": bool(row.get("promoted", False)),
                "failure_reasons": "|".join(str(item) for item in reasons),
            }
        )
    return output.getvalue().encode("utf-8")


def _validate_mirror_location(output_dir: Path, checkpoint_dir: Path, mirror_dir: Path) -> None:
    """Require a distinct mirror sibling rather than a nested/ancestor path."""

    if (
        mirror_dir == output_dir
        or mirror_dir.is_relative_to(output_dir)
        or mirror_dir == checkpoint_dir
        or mirror_dir.is_relative_to(checkpoint_dir)
        or output_dir.is_relative_to(mirror_dir)
        or checkpoint_dir.is_relative_to(mirror_dir)
    ):
        raise ValueError("mirror-dir must not equal, contain, or be contained by output-dir/checkpoint-dir")


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_paths = tuple(output_dir / name for name in ("selection_decision.json", "mechanism_table.json", "mechanism_table.csv"))
    resume = bool(getattr(args, "resume", False))
    if getattr(args, "mirror_dir", None) is None:
        raise ValueError("--mirror-dir is required and must be a separate persistent location")
    early_workers = int(getattr(args, "workers", getattr(args, "worker_count", 1)))
    if early_workers != 1:
        raise ValueError("workers must be exactly 1; parallel candidate execution is not enabled")
    checkpoint_dir_arg = getattr(args, "checkpoint_dir", None)
    checkpoint_dir = (
        Path(checkpoint_dir_arg).expanduser().resolve(strict=False)
        if checkpoint_dir_arg is not None
        else output_dir / "checkpoints"
    )
    mirror_dir = Path(args.mirror_dir).expanduser().resolve(strict=False)
    _validate_mirror_location(output_dir, checkpoint_dir, mirror_dir)
    if not resume and any(path.exists() or path.is_symlink() for path in output_paths):
        raise ValueError(f"refusing to overwrite existing selection output: {output_dir}")
    if not resume and (output_dir / "finalization.json").exists():
        raise ValueError("finalization journal exists; explicit resume is required")
    paths = {
        "F0": Path(args.f0).expanduser().resolve(strict=False),
        "F1": Path(args.f1).expanduser().resolve(strict=False),
        "F2": Path(args.f2).expanduser().resolve(strict=False),
        "F3": Path(args.f3).expanduser().resolve(strict=False),
    }
    evidences = {condition: _load_bundle(condition, path) for condition, path in paths.items()}
    source_file_hashes = {condition: _stream_file_sha256(path) for condition, path in paths.items()}
    image_ids, image_ids_hash = validate_shared_image_identity(*[evidences[c] for c in CONDITIONS])
    finalization_path = output_dir / "finalization.json"
    base_final_identity = _finalization_identity(evidences, source_file_hashes, image_ids_hash, selected_condition=None)
    f0 = evidences["F0"]
    candidates = [evidences[c] for c in ("F1", "F2", "F3")]
    checkpoint_interval = int(getattr(args, "checkpoint_interval", DEFAULT_CHECKPOINT_INTERVAL))
    checkpoint_wall_time_seconds = float(getattr(args, "checkpoint_wall_time_seconds", 300.0))
    workers = int(getattr(args, "workers", getattr(args, "worker_count", 1)))
    if workers != 1:
        raise ValueError("workers must be exactly 1; parallel candidate execution is not enabled")
    if resume and any(path.exists() or path.is_symlink() for path in output_paths):
        if not finalization_path.exists():
            raise ValueError("existing final artifacts require a finalization journal for --resume")
        journal = _read_finalization(finalization_path, base_final_identity)
        # A complete, hash-verified bundle is idempotent.  Repair the mirror
        # milestone before returning so a prior mirror interruption is safe.
        if journal.get("state") == "complete":
            artifacts = journal["artifacts"]
            artifact_names = ("selection_decision.json", "mechanism_table.json", "mechanism_table.csv")
            if all(
                (output_dir / name).is_file()
                and isinstance(artifacts.get(name), Mapping)
                and hashlib.sha256((output_dir / name).read_bytes()).hexdigest() == str(artifacts[name]["sha256"])
                for name in artifact_names
            ):
                final_artifact_bytes = {name: (output_dir / name).read_bytes() for name in artifact_names}
                _mirror_publish(
                    mirror_dir,
                    stage="complete",
                    output_dir=output_dir,
                    checkpoint_dir=checkpoint_dir,
                    conditions={condition: "complete" for condition in CONDITIONS} | {"selection": "complete"},
                    selected_condition=journal.get("identity", {}).get("selected_condition"),
                    resume=True,
                    workers=workers,
                    final_artifacts=final_artifact_bytes,
                )
                return tuple(output_dir / name for name in artifact_names)  # type: ignore[return-value]
    if mirror_dir.exists() and any(mirror_dir.iterdir()) and not resume:
        raise ValueError(f"mirror state exists; explicit resume is required: {mirror_dir}")

    states: dict[str, object] = {condition: "pending" for condition in CONDITIONS}
    _mirror_publish(
        mirror_dir,
        stage="initialized",
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        conditions=states,
        resume=resume,
        workers=workers,
    )
    reference_cache = build_shared_reference_draws(
        f0,
        output_dir,
        total_replicates=FORMAL_REPLICATES,
        checkpoint_interval=checkpoint_interval,
        checkpoint_wall_time_seconds=checkpoint_wall_time_seconds,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        source_file_sha256=source_file_hashes["F0"],
        progress_stream=sys.stdout,
        code_paths=(Path(__file__).resolve(),),
    )
    states["F0"] = "complete"
    _mirror_publish(
        mirror_dir,
        stage="reference_complete",
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        conditions=states,
        resume=resume,
        workers=workers,
    )
    paired: dict[str, object] = {}
    for candidate in candidates:
        condition = str(getattr(candidate, "condition"))
        paired[condition] = run_resumable_factor_bootstrap(
            candidate,
            f0,
            output_dir,
            condition=condition,
            total_replicates=FORMAL_REPLICATES,
            checkpoint_interval=checkpoint_interval,
            checkpoint_wall_time_seconds=checkpoint_wall_time_seconds,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            reference_draws=reference_cache,
            progress_stream=sys.stdout,
            source_file_sha256={
                "candidate": source_file_hashes[condition],
                "reference": source_file_hashes["F0"],
            },
            code_paths=(Path(__file__).resolve(),),
        )
        states[condition] = "complete"
        _mirror_publish(
            mirror_dir,
            stage=f"{condition}_complete",
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            conditions=states,
            resume=resume,
            workers=workers,
        )

    # Selection consumes only the already checkpointed candidate deltas.  No
    # second 10,000-replicate bootstrap is allowed here.
    selection = select_repair_against_f0(f0, candidates, paired_deltas=paired)
    selected_condition = getattr(selection, "selected_condition", None) if selection is not None else None
    if selected_condition is not None and selected_condition not in {"F1", "F2", "F3"}:
        raise ValueError("existing selector returned an invalid candidate condition")
    if selection is not None:
        verify_digest = getattr(selection, "verify_digest", None)
        if callable(verify_digest) and not verify_digest():
            raise ValueError("existing selector returned an invalid decision digest")
        # Keep the selected record exactly as emitted by the pure selector.
        # This preserves the historical artifact semantics even when a
        # producer supplies a richer PairedDelta subclass for the checkpoint.
        endpoint_table = getattr(selection, "endpoint_table", None)
        candidate_endpoints = (
            endpoint_table.get(selected_condition)
            if isinstance(endpoint_table, Mapping) and selected_condition is not None
            else None
        )
        if candidate_endpoints is None and selected_condition is not None:
            candidate_endpoints = getattr(evidences[selected_condition], "endpoints", None)
        if isinstance(candidate_endpoints, Mapping) and selected_condition is not None:
            paired[selected_condition] = SimpleNamespace(
                point=float(getattr(selection, "delta_s_point")),
                ci95=tuple(float(value) for value in getattr(selection, "delta_s_ci95")),
                candidate_endpoints=dict(candidate_endpoints),
                candidate_evidence_sha256=str(getattr(evidences[selected_condition], "evidence_sha256")),
            )

    table_conditions: dict[str, object] = {}
    table_conditions["F0"] = _condition_row("F0", f0, paired=None, selected=False)
    for candidate in candidates:
        condition = str(getattr(candidate, "condition"))
        table_conditions[condition] = _condition_row(
            condition,
            candidate,
            paired=paired.get(condition),
            selected=condition == selected_condition,
        )
    selection_payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "development",
        "seed": DEVELOPMENT_SEED,
        "reference_condition": "F0",
        "selected_condition": selected_condition,
        "image_ids": list(image_ids),
        "image_ids_hash": image_ids_hash,
        "evidence_sha256": {
            condition: str(getattr(evidences[condition], "evidence_sha256"))
            for condition in CONDITIONS
        },
        "selection": None if selection is None else selection.to_dict(),
        "failure_reasons": {
            condition: table_conditions[condition]["failure_reasons"]
            for condition in CONDITIONS
            if table_conditions[condition]["failure_reasons"]
        },
    }
    if selection is not None:
        selection_payload.update(selection.to_dict())
    selection_payload["selection_sha256"] = sha256_canonical(selection_payload)
    table_payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "development",
        "seed": DEVELOPMENT_SEED,
        "image_ids": list(image_ids),
        "image_ids_hash": image_ids_hash,
        "selected_condition": selected_condition,
        "selection_sha256": selection_payload["selection_sha256"],
        "conditions": table_conditions,
    }
    states["selection"] = "ready"
    _mirror_publish(
        mirror_dir,
        stage="selection_ready",
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        conditions=states,
        selected_condition=selected_condition,
        resume=resume,
        workers=workers,
    )
    selection_path = output_dir / "selection_decision.json"
    mechanism_path = output_dir / "mechanism_table.json"
    csv_path = output_dir / "mechanism_table.csv"
    selection_bytes = (_canonical_json(_json_safe(selection_payload)) + "\n").encode("utf-8")
    mechanism_bytes = (_canonical_json(_json_safe(table_payload)) + "\n").encode("utf-8")
    csv_bytes = _mechanism_csv(table_payload)
    final_artifact_bytes = {
        "selection_decision.json": selection_bytes,
        "mechanism_table.json": mechanism_bytes,
        "mechanism_table.csv": csv_bytes,
    }
    final_identity = _finalization_identity(
        evidences,
        source_file_hashes,
        image_ids_hash,
        selected_condition=selected_condition,
    )
    artifacts = {
        "selection_decision.json": {"path": str(selection_path), "sha256": hashlib.sha256(selection_bytes).hexdigest()},
        "mechanism_table.json": {"path": str(mechanism_path), "sha256": hashlib.sha256(mechanism_bytes).hexdigest()},
        "mechanism_table.csv": {"path": str(csv_path), "sha256": hashlib.sha256(csv_bytes).hexdigest()},
    }
    existing_journal = _read_finalization(finalization_path, base_final_identity) if finalization_path.exists() else None
    if existing_journal is not None:
        old_identity = existing_journal.get("identity")
        if not isinstance(old_identity, Mapping) or _json_safe(old_identity) != _json_safe(final_identity):
            raise ValueError("finalization identity mismatch: selected condition or payload")
        if _json_safe(existing_journal.get("artifacts")) != _json_safe(artifacts):
            raise ValueError("finalization payload hash mismatch")
    else:
        _write_finalization(
            finalization_path,
            {
                "schema_version": 1,
                "state": "pending",
                "identity": final_identity,
                "artifacts": artifacts,
                "updated_at": time.time(),
            },
        )
    # Publish the final mirror commit marker before promoting user-visible
    # artifacts.  A mirror failure therefore leaves only a pending journal;
    # no result can be mistaken for a promoted selection.
    states["selection"] = "complete"
    _mirror_publish(
        mirror_dir,
        stage="complete",
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        conditions=states,
        selected_condition=selected_condition,
        resume=resume,
        workers=workers,
        final_artifacts=final_artifact_bytes,
    )
    _write_final_artifact(selection_path, selection_bytes, artifacts["selection_decision.json"]["sha256"])
    _write_final_artifact(mechanism_path, mechanism_bytes, artifacts["mechanism_table.json"]["sha256"])
    _write_final_artifact(csv_path, csv_bytes, artifacts["mechanism_table.csv"]["sha256"])
    _write_finalization(
        finalization_path,
        {
            "schema_version": 1,
            "state": "complete",
            "identity": final_identity,
            "artifacts": artifacts,
            "updated_at": time.time(),
        },
    )
    return selection_path, mechanism_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select one F0-relative development factor-repair evidence candidate.")
    parser.add_argument("--f0", "--f0-evidence", dest="f0", required=True, type=Path)
    parser.add_argument("--f1", "--f1-evidence", dest="f1", required=True, type=Path)
    parser.add_argument("--f2", "--f2-evidence", dest="f2", required=True, type=Path)
    parser.add_argument("--f3", "--f3-evidence", dest="f3", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--mirror-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument("--checkpoint-wall-time-seconds", type=float, default=300.0)
    parser.add_argument("--workers", "--worker-count", dest="workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        selection_path, mechanism_path, csv_path = run(args)
    except Exception as exc:
        print(f"factor repair evidence selection failed: {exc}", file=sys.stderr)
        return 1
    print(f"selection={selection_path}")
    print(f"mechanism_table={mechanism_path}")
    print(f"mechanism_csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run"]
