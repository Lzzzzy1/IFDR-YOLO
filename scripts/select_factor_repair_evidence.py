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
import io
import json
import math
import os
from pathlib import Path
import sys
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
    paired_image_cluster_delta,
    select_repair_against_f0,
)


DEVELOPMENT_SEED = 17
CONDITIONS = ("F0", "F1", "F2", "F3")


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


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_paths = tuple(output_dir / name for name in ("selection_decision.json", "mechanism_table.json", "mechanism_table.csv"))
    if any(path.exists() or path.is_symlink() for path in output_paths):
        raise ValueError(f"refusing to overwrite existing selection output: {output_dir}")
    paths = {
        "F0": Path(args.f0).expanduser().resolve(strict=False),
        "F1": Path(args.f1).expanduser().resolve(strict=False),
        "F2": Path(args.f2).expanduser().resolve(strict=False),
        "F3": Path(args.f3).expanduser().resolve(strict=False),
    }
    evidences = {condition: _load_bundle(condition, path) for condition, path in paths.items()}
    image_ids, image_ids_hash = validate_shared_image_identity(*[evidences[c] for c in CONDITIONS])
    f0 = evidences["F0"]
    candidates = [evidences[c] for c in ("F1", "F2", "F3")]
    selection = select_repair_against_f0(f0, candidates)
    selected_condition = getattr(selection, "selected_condition", None) if selection is not None else None
    if selected_condition is not None and selected_condition not in {"F1", "F2", "F3"}:
        raise ValueError("existing selector returned an invalid candidate condition")
    if selection is not None:
        verify_digest = getattr(selection, "verify_digest", None)
        if callable(verify_digest) and not verify_digest():
            raise ValueError("existing selector returned an invalid decision digest")

    paired: dict[str, object] = {}
    if selection is not None:
        # select_repair_against_f0 has already called the registered paired
        # bootstrap for its chosen candidate.  Reuse that immutable result so
        # a 10,000-replicate draw is not repeated unnecessarily.
        endpoint_table = getattr(selection, "endpoint_table", None)
        candidate_endpoints = (
            endpoint_table.get(selected_condition)
            if isinstance(endpoint_table, Mapping) and selected_condition is not None
            else None
        )
        if isinstance(candidate_endpoints, Mapping):
            paired[selected_condition] = SimpleNamespace(
                point=float(getattr(selection, "delta_s_point")),
                ci95=tuple(float(value) for value in getattr(selection, "delta_s_ci95")),
                candidate_endpoints=dict(candidate_endpoints),
                candidate_evidence_sha256=str(getattr(evidences[selected_condition], "evidence_sha256")),
            )
    for candidate in candidates:
        condition = str(getattr(candidate, "condition"))
        if not bool(getattr(candidate, "complete", False)) or condition in paired:
            continue
        paired[condition] = paired_image_cluster_delta(candidate, f0)

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
    selection_path = output_dir / "selection_decision.json"
    mechanism_path = output_dir / "mechanism_table.json"
    csv_path = output_dir / "mechanism_table.csv"
    _write_json_once(selection_path, selection_payload)
    try:
        _write_json_once(mechanism_path, table_payload)
        _atomic_create(csv_path, _mechanism_csv(table_payload))
    except Exception:
        # Do not replace or repair an already-created decision.  A failed
        # table write is left visible for the caller to remove/review.
        raise
    return selection_path, mechanism_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select one F0-relative development factor-repair evidence candidate.")
    parser.add_argument("--f0", "--f0-evidence", dest="f0", required=True, type=Path)
    parser.add_argument("--f1", "--f1-evidence", dest="f1", required=True, type=Path)
    parser.add_argument("--f2", "--f2-evidence", dest="f2", required=True, type=Path)
    parser.add_argument("--f3", "--f3-evidence", dest="f3", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
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
