from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Mapping, Sequence


PROTOCOL_FIELDS = (
    "fit_ids_sha256",
    "development_ids_sha256",
    "seed",
    "epochs",
    "imgsz",
    "batch",
    "workers",
    "amp",
    "deterministic",
    "checkpoint_role",
    "evaluator",
    "prediction_args",
)

REQUIRED_CANDIDATE_FIELDS = (
    "name",
    "asset_exists",
    "audit_status",
    "run_identity_sha256",
    "checkpoint_sha256",
    "code_sha256",
    "config_sha256",
    "model_sha256",
    "source_weight_sha256",
    "fit_ids_sha256",
    "development_ids_sha256",
    "actual_train_ids_sha256",
    "train_cache_sha256",
    "data_content_manifest_sha256",
    "seed",
    "epochs",
    "imgsz",
    "batch",
    "workers",
    "amp",
    "deterministic",
    "checkpoint_role",
    "initialization",
    "augmentation",
    "runtime",
    "evaluator",
    "evaluator_source_sha256",
    "prediction_args",
    "evidence",
)

BOUND_FIELDS = (
    "run_identity_sha256",
    "checkpoint_sha256",
    "code_sha256",
    "config_sha256",
    "model_sha256",
    "source_weight_sha256",
    "actual_train_ids_sha256",
    "train_cache_sha256",
    "data_content_manifest_sha256",
    "initialization",
    "augmentation",
    "runtime",
    "evaluator_source_sha256",
    "evidence",
)

MATRIX_COLUMNS = ("reference", "candidate", "status", "reason")


def _require_fields(value: Mapping[str, object], fields: Sequence[str], *, label: str) -> None:
    for field in fields:
        if field not in value:
            raise ValueError(f"{label} missing required identity field: {field}")


def _candidate_status(active: Mapping[str, object], candidate: Mapping[str, object]) -> tuple[str, list[str]]:
    name = str(candidate["name"])
    if name == "R" and not candidate["asset_exists"] and str(candidate["audit_status"]).startswith("NO_GO"):
        return "NO_GO", ["registered audit did not authorize repair R"]
    if not candidate["asset_exists"]:
        return "MISSING", ["registered candidate asset does not exist"]

    reasons: list[str] = []
    for field in BOUND_FIELDS:
        value = candidate[field]
        if value is None or value == "" or value == []:
            reasons.append(f"unbound identity field: {field}")
    for field in PROTOCOL_FIELDS:
        if candidate[field] != active[field]:
            reasons.append(f"protocol mismatch: {field}")
    if candidate["actual_train_ids_sha256"] != active["fit_ids_sha256"]:
        reasons.append("actual train IDs do not equal registered fit IDs")
    return ("MISMATCHED", reasons) if reasons else ("VALID_MATCHED", [])


def build_ledger(
    active: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], list[dict[str, str]]]:
    _require_fields(active, PROTOCOL_FIELDS, label="active protocol")
    if not candidates:
        raise ValueError("candidate inventory is empty")

    records: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        _require_fields(candidate, REQUIRED_CANDIDATE_FIELDS, label="candidate")
        name = str(candidate["name"])
        if name in records:
            raise ValueError(f"duplicate candidate: {name}")
        status, reasons = _candidate_status(active, candidate)
        records[name] = {
            **dict(candidate),
            "status": status,
            "status_reasons": reasons,
            "may_enter_seed0_subtraction": status == "VALID_MATCHED",
        }

    reference = records.get("P3P5_CONTROL")
    matrix: list[dict[str, str]] = []
    for name, record in records.items():
        if name == "P3P5_CONTROL":
            continue
        valid = reference is not None and reference["status"] == "VALID_MATCHED" and record["status"] == "VALID_MATCHED"
        matrix.append({
            "reference": "P3P5_CONTROL",
            "candidate": name,
            "status": "VALID_MAIN" if valid else "FORBIDDEN_SUBTRACTION",
            "reason": (
                "both candidates satisfy the active matched protocol"
                if valid
                else "one or both candidates are not VALID_MATCHED under the active protocol"
            ),
        })

    required = {"P3P5_CONTROL", "PLAIN_P2", "DCLI", "R"}
    ready = required.issubset(records) and all(records[name]["status"] == "VALID_MATCHED" for name in required)
    ledger: dict[str, object] = {
        "schema_version": 1,
        "scientific_role": "identity and forbidden-subtraction gate; contains no performance estimate",
        "active_protocol": dict(active),
        "candidates": records,
        "valid_matched_candidates": sorted(name for name, record in records.items() if record["status"] == "VALID_MATCHED"),
        "decision": "READY_SEED0_FAIR_COMPARISON" if ready else "NO_GO_SEED0_FAIR_COMPARISON",
        "training_authorized": False,
        "development_subtraction_authorized": ready,
    }
    return ledger, matrix


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _csv_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=MATRIX_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def publish_ledger(
    primary: Path,
    mirror: Path,
    ledger: Mapping[str, object],
    matrix: Sequence[Mapping[str, str]],
) -> None:
    payloads = {
        "DATA_EVALUATION_LEDGER.json": _json_bytes(ledger),
        "protocol_matrix.csv": _csv_bytes(matrix),
    }
    manifest = {
        "schema_version": 1,
        "files": {
            name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in payloads.items()
        },
    }
    manifest_payload = _json_bytes(manifest)
    for root in (primary, mirror):
        for name, payload in payloads.items():
            _atomic_write(root / name, payload)
        _atomic_write(root / "manifest.json", manifest_payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    ledger, matrix = build_ledger(source["active_protocol"], source["candidates"])
    publish_ledger(args.primary, args.mirror, ledger, matrix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
