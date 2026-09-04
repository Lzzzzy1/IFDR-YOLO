"""Small, fail-closed acceptance report for the six registered YOLO evidence gates.

The script only reads frozen local evidence.  It does not train, connect to a
server, or turn an internal evaluator into an official-devkit result.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CLASSES = ("Pedestrian", "Cyclist")
OFFICIAL_DEVKIT_SHA256 = "CE0B76B69C0C5F89690A0D65B7302BBBDB962A0C7E8ABA6EFC7050D1B04B4CF1"
SCHEMA_VERSION = 1


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_bytes(path, _canonical(payload))


def _source(path: str | Path | None, *, role: str, required: bool = False) -> dict[str, object]:
    if path is None:
        return {"role": role, "path": None, "exists": False, "sha256": None, "status": "MISSING"}
    resolved = Path(path).expanduser().resolve()
    exists = resolved.is_file() or resolved.is_dir()
    if required and not exists:
        raise FileNotFoundError(f"required evidence source is missing: {resolved}")
    return {
        "role": role,
        "path": str(resolved),
        "exists": exists,
        "size": resolved.stat().st_size if resolved.is_file() else (sum(item.stat().st_size for item in resolved.rglob("*") if item.is_file()) if resolved.is_dir() else None),
        "sha256": sha256_file(resolved) if resolved.is_file() else (_directory_sha256(resolved) if resolved.is_dir() else None),
        "status": "AVAILABLE" if exists else "MISSING",
    }


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(child.read_bytes())
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence must be an object: {path}")
    return payload


def build_data_use_ledger(spec: Mapping[str, object]) -> dict[str, object]:
    """Build the permanent fit/development/confirmation/test usage ledger."""

    development = spec.get("development")
    development_map = dict(development) if isinstance(development, Mapping) else {}
    # ponytail: one immutable role prevents accidental relabelling of the used
    # 371 images; a separate confirmation set must be supplied explicitly.
    development_map.update({
        "role": "development_route_selection_permanent",
        "permanent": True,
        "count": 371,
    })
    confirmation = spec.get("confirmation")
    confirmation_status = "AVAILABLE" if isinstance(confirmation, Mapping) and confirmation.get("status") == "AVAILABLE" else "NONE"
    test = dict(spec.get("test") or {}) if isinstance(spec.get("test"), Mapping) else {}
    test_status = str(test.get("status", "BLOCKED"))
    sets = {
        "fit": {
            **(dict(spec.get("fit") or {}) if isinstance(spec.get("fit"), Mapping) else {}),
            "role": "training",
            "count": 3341,
        },
        "development": development_map,
        "historical_exposed": {
            **(dict(spec.get("historical_exposed") or {}) if isinstance(spec.get("historical_exposed"), Mapping) else {}),
            "role": "development_exposed",
            "count": 3769,
            "confirmation_eligible": False,
        },
        "test": {
            **test,
            "role": "official_kitti_test_hidden_labels",
            "count": 7518,
            "status": test_status,
        },
    }
    blockers = []
    if confirmation_status != "AVAILABLE":
        blockers.append("no untouched labeled confirmation set is registered")
    if test_status != "AVAILABLE":
        blockers.append("official KITTI test submission is blocked until method freeze and submission")
    sources = []
    for key, value in sets.items():
        if isinstance(value, Mapping):
            sources.append(_source(value.get("path"), role=f"ledger:{key}", required=False))
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "sets": sets,
        "sources": sources,
        "independent_confirmation": {
            "status": confirmation_status,
            "role": "independent_confirmation" if confirmation_status == "AVAILABLE" else "NONE",
            "reason": "registered 371 development images were used for route selection",
        },
        "test_route": {
            "status": test_status,
            "role": "official_kitti_test_hidden_labels",
            "count": 7518,
            "blocked_until": "frozen method and official submission",
        },
        "blockers": blockers,
    }


def _normalise_class_metrics(value: Mapping[str, object]) -> dict[str, object]:
    valid_gt = value.get("valid_gt", value.get("num_valid_gt"))
    tp = value.get("tp", value.get("true_positives"))
    fp = value.get("fp", value.get("false_positives"))
    fn = value.get("fn")
    if fn is None and isinstance(valid_gt, int) and isinstance(tp, int):
        fn = valid_gt - tp
    return {
        "ap40": value.get("ap40"),
        "valid_gt": valid_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _metrics_from_payload(payload: Mapping[str, object], side: str | None) -> tuple[dict[str, object], dict[str, object]]:
    observed = payload.get("observed")
    if isinstance(observed, Mapping):
        moderate = observed.get("moderate")
        if not isinstance(moderate, Mapping):
            raise ValueError("observed evidence has no moderate section")
        selected = moderate.get(side or "candidate")
        if not isinstance(selected, Mapping):
            raise ValueError("observed evidence side is missing")
        return {name: _normalise_class_metrics(selected[name]) for name in CLASSES if isinstance(selected.get(name), Mapping)} | {"macro": selected.get("macro")}, dict(observed)
    classes = payload.get("classes")
    if not isinstance(classes, Mapping):
        raise ValueError("metrics evidence has no classes section")
    result: dict[str, object] = {}
    for name in CLASSES:
        class_payload = classes.get(name)
        if not isinstance(class_payload, Mapping):
            continue
        moderate = class_payload.get("moderate")
        if isinstance(moderate, Mapping):
            result[name] = _normalise_class_metrics(moderate)
    result["macro"] = payload.get("moderate_macro_ap_r40")
    if result["macro"] is None and all(isinstance(result.get(name), Mapping) and isinstance(result[name].get("ap40"), (int, float)) for name in CLASSES):
        result["macro"] = sum(float(result[name]["ap40"]) for name in CLASSES) / len(CLASSES)  # type: ignore[index]
    return result, payload


def _positive_images(label_dir: Path | None, split_path: Path | None) -> dict[str, int | None]:
    if label_dir is None or split_path is None or not label_dir.is_dir() or not split_path.is_file():
        return {name: None for name in CLASSES}
    try:
        from ifdr_yolo.data.kitti_parser import parse_kitti_file
        from ifdr_yolo.data.splits import load_ids
        from ifdr_yolo.eval.kitti_ap40 import is_valid_ground_truth
        from ifdr_yolo.data.kitti_types import Difficulty
    except ImportError:
        return {name: None for name in CLASSES}
    counts = {name: 0 for name in CLASSES}
    for image_id in load_ids(split_path):
        objects = parse_kitti_file(label_dir / f"{image_id}.txt")
        for name in CLASSES:
            if any(is_valid_ground_truth(obj, name, Difficulty.MODERATE) for obj in objects):
                counts[name] += 1
    return counts


def _strata_from_payload(raw: Mapping[str, object]) -> dict[str, object]:
    strata = raw.get("strata")
    if not isinstance(strata, Mapping):
        return {}
    selected: dict[str, object] = {}
    for axis, values in strata.items():
        if not isinstance(values, Mapping):
            continue
        for name in ("small_25_40", "far_gt_40m"):
            value = values.get(name)
            if isinstance(value, Mapping):
                selected.setdefault(str(axis), {})[name] = value
    return selected


def _paired_ci_from_payload(payload: Mapping[str, object]) -> dict[str, object]:
    bootstrap = payload.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        return {"source": None, "available": False, "macro": None}
    moderate = bootstrap.get("moderate")
    if not isinstance(moderate, Mapping):
        return {"source": None, "available": False, "macro": None}
    macro = moderate.get("macro")
    delta = macro.get("delta") if isinstance(macro, Mapping) else moderate.get("delta")
    if not isinstance(delta, Mapping):
        return {"source": None, "available": False, "macro": None}
    return {
        "source": "bootstrap.moderate.macro.delta",
        "available": isinstance(delta.get("ci_lower"), (int, float)) and isinstance(delta.get("ci_upper"), (int, float)),
        "macro": {"lower": delta.get("ci_lower"), "upper": delta.get("ci_upper")},
    }


def build_denominator_audit(results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    incomplete: list[str] = []
    for index, spec in enumerate(results):
        name = str(spec.get("name", f"result_{index}"))
        source_path = spec.get("path")
        sources = [_source(source_path, role=f"{name}:result", required=False)]
        payload: Mapping[str, object]
        if isinstance(spec.get("payload"), Mapping):
            payload = spec["payload"]  # type: ignore[assignment]
        elif isinstance(spec.get("observed"), Mapping):
            payload = {"observed": spec["observed"], "bootstrap": spec.get("bootstrap")}
        elif source_path is not None and Path(source_path).is_file():
            payload = _load_json(source_path)
        else:
            payload = {}
        try:
            metrics, raw = _metrics_from_payload(payload, spec.get("side") if isinstance(spec.get("side"), str) else None)
        except ValueError:
            metrics, raw = {}, {}
        split_path = Path(spec["split_path"]) if spec.get("split_path") else None
        label_dir = Path(spec["ground_truth_dir"]) if spec.get("ground_truth_dir") else None
        positive = spec.get("positive_images") if isinstance(spec.get("positive_images"), Mapping) else _positive_images(label_dir, split_path)
        if label_dir is not None:
            sources.append(_source(label_dir, role=f"{name}:ground_truth_dir"))
        if split_path is not None:
            sources.append(_source(split_path, role=f"{name}:split"))
        classes: dict[str, object] = {}
        missing: list[str] = []
        for class_name in CLASSES:
            value = metrics.get(class_name)
            if not isinstance(value, Mapping):
                missing.append(f"{class_name}.moderate")
                value = {}
            class_metrics = dict(value)
            class_metrics["positive_images"] = positive.get(class_name) if isinstance(positive, Mapping) else None
            for field in ("ap40", "valid_gt", "positive_images", "tp", "fp", "fn"):
                if class_metrics.get(field) is None:
                    missing.append(f"{class_name}.moderate.{field}")
            classes[class_name] = {"moderate": class_metrics}
        macro = metrics.get("macro")
        if macro is None:
            missing.append("moderate.macro")
        strata = _strata_from_payload(raw)
        for axis, key in (("height", "small_25_40"), ("depth", "far_gt_40m")):
            if key not in (strata.get(axis) or {}):
                missing.append(f"strata.{axis}.{key}")
        paired_ci = spec.get("paired_ci") if isinstance(spec.get("paired_ci"), Mapping) else _paired_ci_from_payload(payload)
        if not isinstance(paired_ci, Mapping) or paired_ci.get("available") is not True:
            missing.append("paired_ci")
        row = {
            "name": name,
            "status": "PASS" if not missing else "INCOMPLETE",
            "classes": classes,
            "moderate_macro_ap40": macro,
            "strata": strata,
            "paired_ci": paired_ci,
            "sources": sources,
            "missing_fields": missing,
            "cyclist_moderate_gt_expected": 55,
            "cyclist_moderate_gt_verified": isinstance(classes.get("Cyclist"), Mapping) and classes["Cyclist"]["moderate"].get("valid_gt") == 55,  # type: ignore[index]
        }
        rows.append(row)
        if missing:
            incomplete.append(name)
    return {
        "status": "PASS" if not incomplete else "INCOMPLETE",
        "results": rows,
        "incomplete_results": incomplete,
        "saturation_note": "Cyclist Moderate denominator is expected to be 55 GT; high AP is saturation-sensitive and must retain paired uncertainty.",
    }


def build_official_reconciliation(spec: Mapping[str, object]) -> dict[str, object]:
    zip_path = spec.get("official_zip")
    zip_source = _source(zip_path, role="official KITTI devkit zip", required=False)
    expected = str(spec.get("official_zip_sha256", OFFICIAL_DEVKIT_SHA256)).upper()
    actual = str(zip_source.get("sha256") or "").upper()
    sha_ok = bool(actual) and actual == expected == OFFICIAL_DEVKIT_SHA256
    tool = spec.get("official_tool") if isinstance(spec.get("official_tool"), Mapping) else {}
    runnable = bool(tool.get("available")) and bool(tool.get("executable"))
    adapter = _source(spec.get("adapter_source"), role="official evaluator output adapter", required=False)
    prediction = _source(spec.get("prediction_source"), role="fixed predictions for reconciliation", required=False)
    sources = [zip_source, adapter, prediction]
    status = "PASS" if sha_ok and runnable else "UNRESOLVED"
    return {
        "status": status,
        "severity": None if status == "PASS" else "P0",
        "official_zip_sha256": (actual or expected).lower(),
        "official_zip_sha256_expected": OFFICIAL_DEVKIT_SHA256,
        "tool_source_identity": {
            "zip": zip_source,
            "compiler": tool.get("compiler"),
            "executable": tool.get("executable"),
            "available": runnable,
        },
        "internal_evaluator": spec.get("internal_evaluator", "ifdr_yolo.eval.kitti_ap40.evaluate_class"),
        "internal_evaluator_compatibility": "internal_only_not_official_equivalent",
        "fixed_prediction_binding": prediction,
        "reconciliation_entry": {
            "command": "run official KITTI devkit adapter on the same fixed predictions and labels; compare GT counts, PR samples, and AP_R40",
            "gt_counts": "PENDING" if not runnable else "UNVERIFIED",
            "pr_samples": "PENDING" if not runnable else "UNVERIFIED",
            "ap_r40": "PENDING" if not runnable else "UNVERIFIED",
            "reproduce": True,
        },
        "blocker": "official C++ compiler/runnable devkit subset is unavailable; do not claim equivalence" if not runnable else None,
        "sources": sources,
    }


def build_storage_preflight(config: Mapping[str, object], *, free_bytes: int | None = None) -> dict[str, object]:
    checkpoint = config.get("checkpoint_bytes") if isinstance(config.get("checkpoint_bytes"), Mapping) else {}
    last = int(checkpoint.get("last", 0))
    best = int(checkpoint.get("best", 0))
    periodic = int(checkpoint.get("periodic", 0))
    log_peak = int(config.get("log_peak_bytes", 0))
    prediction_peak = int(config.get("prediction_peak_bytes", 0))
    mirror_peak = int(config.get("mirror_peak_bytes", 0))
    headroom = int(config.get("headroom_bytes", 0))
    forecast = {
        "last_best_checkpoint_bytes": last + best,
        "periodic_checkpoint_bytes": periodic,
        "log_peak_bytes": log_peak,
        "prediction_peak_bytes": prediction_peak,
        "mirror_peak_bytes": mirror_peak,
        "headroom_bytes": headroom,
    }
    forecast["required_bytes"] = sum(forecast.values())
    if free_bytes is None:
        workspace = Path(str(config.get("workspace", "."))).expanduser()
        free_bytes = shutil.disk_usage(workspace).free
    retention = dict(config.get("retention") or {}) if isinstance(config.get("retention"), Mapping) else {}
    retention_ok = retention.get("save_period") == -1 and set(retention.get("retain", ())) <= {"last.pt", "best.pt"} and periodic == 0
    resume = dict(config.get("resume_validation") or {}) if isinstance(config.get("resume_validation"), Mapping) else {"status": "UNKNOWN"}
    reasons = []
    if not retention_ok:
        reasons.append("retention must use save_period=-1 and retain only last.pt/best.pt")
    if int(free_bytes) < int(forecast["required_bytes"]):
        reasons.append("free space is below forecast peak plus required headroom")
    if resume.get("status") not in {"PASS", "UNKNOWN"}:
        reasons.append("latest-checkpoint resumability validation failed")
    return {
        "status": "PASS" if not reasons else "BLOCKED",
        "free_bytes": int(free_bytes),
        "forecast": forecast,
        "retention": {**retention, "save_period": retention.get("save_period"), "contract": "last.pt and best.pt only; no periodic epoch archives"},
        "resume_validation": resume,
        "blockers": reasons,
        "sources": [_source(path, role="storage:source", required=False) for path in config.get("sources", ())] if isinstance(config.get("sources"), Sequence) and not isinstance(config.get("sources"), (str, bytes)) else [],
    }


def _row_identity(row: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(row.get(key) for key in ("split_sha256", "data_use_role", "seed", "epochs", "imgsz", "checkpoint_role", "evaluator"))


def build_protocol_matrix(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    allowed: list[dict[str, object]] = []
    blocked: list[str] = []
    required = ("name", "model_role", "split_sha256", "data_use_role", "seed", "epochs", "imgsz", "checkpoint_role", "evaluator")
    identity_errors = [
        f"row {index} missing protocol identity fields: {','.join(key for key in required if row.get(key) is None)}"
        for index, row in enumerate(rows)
        if any(row.get(key) is None for key in required)
    ]
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            left_name, right_name = str(left.get("name")), str(right.get("name"))
            names = {left_name, right_name}
            role_pair = {str(left.get("model_role")), str(right.get("model_role"))}
            same_identity = _row_identity(left) == _row_identity(right)
            family = None
            if names == {"p3p5", "p2"} and {left.get("epochs"), right.get("epochs")} == {300} and {left.get("seed"), right.get("seed")} == {17} and same_identity:
                family = "p3p5_vs_p2"
            elif role_pair <= {"ifdr_c", "ifdr_b", "ifdr_ab"} and same_identity and left.get("seed") == right.get("seed") and left.get("epochs") == right.get("epochs") == 30:
                family = "c_b_ab_same_seed"
            if family:
                allowed.append({"left": left_name, "right": right_name, "family": family})
            else:
                blocked.append(f"{left_name} vs {right_name}: cross-split/evaluator/epoch/weight-role or unregistered comparison")
    if any("57" in str(row.get("metric", "")) or "96" in str(row.get("metric", "")) for row in rows):
        blocked.append("57-vs-96 cross-split comparison is explicitly prohibited")
    return {
        "status": "PASS" if rows and not identity_errors else "BLOCKED",
        "allowed_comparisons": allowed,
        "allowed_families": {"p3p5_vs_p2": "300-epoch seed17 last.pt internal evaluator only", "c_b_ab_same_seed": "same 30-epoch seed family only"},
        "blocked_comparisons": blocked,
        "identity_errors": identity_errors,
        "prohibitions": ["57-vs-96", "cross-split", "cross-evaluator", "cross-weight-role"],
        "sources": [_source(path, role="protocol:source", required=False) for row in rows for path in (row.get("source_paths", ()) if isinstance(row.get("source_paths"), Sequence) and not isinstance(row.get("source_paths"), (str, bytes)) else ())],
    }


def build_independent_acceptance(spec: Mapping[str, object]) -> dict[str, object]:
    registered = spec.get("preregistered") if isinstance(spec.get("preregistered"), Mapping) else {}
    seeds = [int(seed) for seed in registered.get("seeds", (17, 29, 41))]
    comparisons = [str(value) for value in registered.get("comparisons", ("B-C", "AB-C", "AB-B"))]
    seed_results = spec.get("seed_results") if isinstance(spec.get("seed_results"), Mapping) else {}
    ci_payload = spec.get("paired_ci") if isinstance(spec.get("paired_ci"), Mapping) else {}
    effects: dict[str, dict[str, object]] = {}
    summaries: dict[str, object] = {}
    for seed in seeds:
        raw = seed_results.get(str(seed), seed_results.get(seed, {})) if isinstance(seed_results, Mapping) else {}
        effects[str(seed)] = {comparison: (raw.get(comparison) if isinstance(raw, Mapping) else None) for comparison in comparisons}
    for comparison in comparisons:
        values = [effects[str(seed)][comparison] for seed in seeds]
        numeric = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        ci = ci_payload.get(comparison) if isinstance(ci_payload, Mapping) and isinstance(ci_payload.get(comparison), Mapping) else {}
        if len(numeric) == len(seeds) == 3 and not isinstance(ci.get("lower"), (int, float)):
            center = statistics.mean(numeric)
            margin = 4.302652729911275 * statistics.stdev(numeric) / math.sqrt(3)
            ci = {"lower": center - margin, "upper": center + margin, "source": "paired_seed_t_interval_df2", "confidence": 0.95}
        lower = ci.get("lower") if isinstance(ci, Mapping) else None
        complete = len(numeric) == len(seeds) and isinstance(lower, (int, float))
        summaries[comparison] = {
            "deltas": values,
            "missing_seeds": [seed for seed, value in zip(seeds, values) if not isinstance(value, (int, float))],
            "direction_consistent": all(value > 0 for value in numeric) if len(numeric) == len(seeds) else "UNKNOWN",
            "mean": sum(numeric) / len(numeric) if numeric else None,
            "paired_ci": dict(ci) if isinstance(ci, Mapping) else {"lower": None, "upper": None},
            "stable_positive": bool(complete and all(value > 0 for value in numeric) and sum(numeric) / len(numeric) > 0 and float(lower) > 0),
        }
    stable_b = bool(summaries.get("B-C", {}).get("stable_positive"))
    stable_joint = bool(summaries.get("AB-B", {}).get("stable_positive"))
    if stable_b and stable_joint:
        branch = "all_comparisons_stable"
    elif stable_b:
        branch = "dcli_only_stable"
    elif stable_joint:
        branch = "joint_only_stable"
    else:
        branch = "neither_stable"
    no_harm = dict(spec.get("no_harm") or {}) if isinstance(spec.get("no_harm"), Mapping) else {"status": "UNKNOWN"}
    blockers = [comparison for comparison, summary in summaries.items() if not summary["stable_positive"]]
    if no_harm.get("status") not in {"PASS", "UNKNOWN"}:
        blockers.append("no_harm")
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "effects": effects,
        "summaries": summaries,
        "retention_rule": {
            "requires_all_three_seed_deltas_positive": True,
            "requires_positive_mean": True,
            "requires_paired_ci_lower_positive": True,
            "minimum_meaningful_effect_ap": float(registered.get("minimum_meaningful_effect_ap", 1.0)),
        },
        "no_harm": {"limits": {"per_class_drop_max_ap": 1.0, "near_large_drop_max_ap": 0.5}, **no_harm},
        "frozen_conclusions": {
            "all_comparisons_stable": "retain_dcli_and_fusion",
            "dcli_only_stable": "retain_dcli_reject_fusion",
            "joint_only_stable": "add_fusion_only_A_and_full_interaction_before_attribution",
            "neither_stable": "stop_old_module_route",
        },
        "selected_conclusion": branch,
        "blockers": blockers,
        "preserved_negative_zero_positive": True,
        "frozen_hypotheses": spec.get("frozen_hypotheses", {}),
        "stopping_gate": spec.get("stopping_gate", {}),
        "sources": [_source(path, role="acceptance:source", required=False) for path in spec.get("source_paths", ())] if isinstance(spec.get("source_paths"), Sequence) and not isinstance(spec.get("source_paths"), (str, bytes)) else [],
    }


def _resolve_paths(value: object, root: Path) -> object:
    if isinstance(value, str) and not Path(value).is_absolute() and ("/" in value or "\\" in value):
        return str((root / value).resolve())
    if isinstance(value, list):
        return [_resolve_paths(item, root) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_paths(item, root) for key, item in value.items()}
    return value


def run_gates(config_path: Path, output_dir: Path, *, free_bytes: int | None = None) -> dict[str, object]:
    config_path = config_path.resolve()
    config = _load_json(config_path)
    config = _resolve_paths(config, config_path.parent)  # type: ignore[assignment]
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        existing_manifest = output_dir / "manifest.json"
        if not existing_manifest.is_file():
            raise FileExistsError(f"refusing to overwrite existing evidence report: {output_dir}")
        try:
            previous = _load_json(existing_manifest)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise FileExistsError(f"refusing to overwrite unrecognised evidence report: {output_dir}") from error
        if previous.get("schema_version") != SCHEMA_VERSION or not any(path.name == "evidence_gates.json" for path in output_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite unrecognised evidence report: {output_dir}")
    data_spec = config.get("data_use_ledger") if isinstance(config.get("data_use_ledger"), Mapping) else {}
    ledger = build_data_use_ledger(data_spec)
    denominator_spec = config.get("denominator_results") if isinstance(config.get("denominator_results"), Sequence) and not isinstance(config.get("denominator_results"), (str, bytes)) else []
    denominator = build_denominator_audit(denominator_spec)  # type: ignore[arg-type]
    official_spec = config.get("official_reconciliation") if isinstance(config.get("official_reconciliation"), Mapping) else {}
    official = build_official_reconciliation(official_spec)
    storage_spec = config.get("storage_preflight") if isinstance(config.get("storage_preflight"), Mapping) else {}
    storage = build_storage_preflight(storage_spec, free_bytes=free_bytes)
    protocol_spec = config.get("protocol_rows") if isinstance(config.get("protocol_rows"), Sequence) and not isinstance(config.get("protocol_rows"), (str, bytes)) else []
    protocol = build_protocol_matrix(protocol_spec)  # type: ignore[arg-type]
    acceptance_spec = config.get("independent_acceptance") if isinstance(config.get("independent_acceptance"), Mapping) else {}
    acceptance = build_independent_acceptance(acceptance_spec)
    gates = {
        "data_use_ledger": ledger,
        "denominator_saturation": denominator,
        "official_evaluator_reconciliation": official,
        "storage_preflight_retention": storage,
        "protocol_identity_matrix": protocol,
        "independent_result_acceptance": acceptance,
    }
    blockers = [name for name, gate in gates.items() if gate.get("status") not in {"PASS"}]
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "gates": gates,
        "blocked_gates": blockers,
        "overall_status": "PASS" if not blockers else "BLOCKED",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evidence_gates.json"
    csv_path = output_dir / "evidence_gates.csv"
    _atomic_json(json_path, report)
    csv_rows = ["gate,status,severity,blocker\n"]
    for name, gate in gates.items():
        severity = gate.get("severity", "")
        reasons = gate.get("blockers", gate.get("blocker", ""))
        if isinstance(reasons, list):
            reasons = "; ".join(str(value) for value in reasons)
        csv_rows.append(f"{name},{gate.get('status','')},{severity},{str(reasons).replace(',', ';')}\n")
    _atomic_bytes(csv_path, "".join(csv_rows).encode("utf-8"))
    source_records: list[dict[str, object]] = [{"role": "frozen_config", "path": str(config_path), "sha256": sha256_file(config_path)}]
    def collect(value: object) -> None:
        if isinstance(value, Mapping):
            if "path" in value and "sha256" in value and "role" in value:
                source_records.append({"role": value.get("role"), "path": value.get("path"), "sha256": value.get("sha256"), "exists": value.get("exists")})
            for child in value.values():
                collect(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for child in value:
                collect(child)
    collect(gates)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "overall_status": report["overall_status"],
        "artifacts": [
            {"name": json_path.name, "path": str(json_path), "sha256": sha256_file(json_path)},
            {"name": csv_path.name, "path": str(csv_path), "sha256": sha256_file(csv_path)},
        ],
        "sources": source_records,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the six local YOLO evidence acceptance gates")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--free-bytes", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_gates(args.config, args.output_dir, free_bytes=args.free_bytes)
    print(f"yolo_evidence_gates={report['overall_status']} blocked={len(report['blocked_gates'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
