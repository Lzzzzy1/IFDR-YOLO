from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


FROZEN_SPLIT_SHA256 = (
    "b1b6b6ee7e5398e93868fab407a2e8a86a53c753667002ef9b8381734ef2cda8"
)
FROZEN_SPLIT_COUNT = 371
FROZEN_BASELINE_MACRO = 85.9182635491459
FROZEN_THRESHOLDS = {
    "overall": (">=", 1.1),
    "small": (">", 0.0),
    "far": (">", 0.0),
    "near": (">=", 0.0),
    "large": (">=", 0.0),
}
SLICE_PATHS = {
    "small": ("height", "small_25_40"),
    "far": ("depth", "far_gt_40m"),
    "near": ("depth", "near_0_20m"),
    "large": ("height", "large_gt_80"),
}
CLASSES = ("Pedestrian", "Cyclist")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _macro_from_overall(metrics: Mapping[str, object]) -> tuple[float, dict[str, float]]:
    try:
        classes = metrics["classes"]
        values = {
            class_name: float(classes[class_name]["moderate"]["ap40"])
            for class_name in CLASSES
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("overall metrics lack Pedestrian/Cyclist Moderate AP40") from error
    macro = sum(values.values()) / len(values)
    explicit = metrics.get("moderate_macro_ap_r40")
    if explicit is not None and abs(float(explicit) - macro) > 1e-9:
        raise ValueError("explicit Moderate macro does not match per-class AP40")
    return macro, values


def _validate_split(payload: Mapping[str, object], label: str) -> None:
    if int(payload.get("split_count", -1)) != FROZEN_SPLIT_COUNT:
        raise ValueError(f"{label} split count is not {FROZEN_SPLIT_COUNT}")
    if payload.get("split_sha256") != FROZEN_SPLIT_SHA256:
        raise ValueError(f"{label} split SHA-256 is not frozen DEV371")


def _validate_stratified_run(run: Mapping[str, object], label: str) -> None:
    if run.get("metric") != "KITTI_2D_CONDITIONAL_AP40":
        raise ValueError(f"{label} does not use frozen conditional AP40")
    if run.get("base_difficulty") != "hard":
        raise ValueError(f"{label} does not use frozen HARD base difficulty")


def _macro_from_slice(run: Mapping[str, object], gate_name: str) -> float:
    axis, slice_name = SLICE_PATHS[gate_name]
    try:
        classes = run["slices"][axis][slice_name]["classes"]
        values = [float(classes[class_name]["ap40"]) for class_name in CLASSES]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing frozen {gate_name} slice") from error
    return sum(values) / len(values)


def _gate_pass(delta: float, operator: str, threshold: float) -> bool:
    if operator == ">=":
        return delta >= threshold or math.isclose(
            delta,
            threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    if operator == ">":
        return delta > threshold
    raise ValueError(f"unsupported gate operator: {operator}")


def _engineering_pass(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().upper() == "PASS"
    return False


def compute_gate(
    *,
    baseline_metrics: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
    stratified_report: Mapping[str, object],
    baseline_run: str,
    candidate_run: str,
    engineering_checks: Mapping[str, object],
    enforce_frozen_baseline: bool = True,
) -> dict[str, object]:
    _validate_split(baseline_metrics, "baseline overall")
    _validate_split(candidate_metrics, "candidate overall")
    _validate_split(stratified_report, "stratified report")
    if stratified_report.get("evaluator") != "ifdr_yolo.stratified_ap40":
        raise ValueError("stratified report evaluator identity is not frozen")
    try:
        baseline_slices = stratified_report["runs"][baseline_run]
        candidate_slices = stratified_report["runs"][candidate_run]
    except (KeyError, TypeError) as error:
        raise ValueError("stratified report lacks requested run names") from error
    _validate_stratified_run(baseline_slices, baseline_run)
    _validate_stratified_run(candidate_slices, candidate_run)

    baseline_macro, baseline_classes = _macro_from_overall(baseline_metrics)
    candidate_macro, candidate_classes = _macro_from_overall(candidate_metrics)
    if enforce_frozen_baseline and abs(baseline_macro - FROZEN_BASELINE_MACRO) > 1e-9:
        raise ValueError(
            "baseline macro does not match frozen PLAIN_P2: "
            f"{baseline_macro} != {FROZEN_BASELINE_MACRO}"
        )

    values: dict[str, dict[str, float]] = {
        "overall": {
            "baseline": baseline_macro,
            "candidate": candidate_macro,
            "delta": candidate_macro - baseline_macro,
        }
    }
    for gate_name in SLICE_PATHS:
        baseline_value = _macro_from_slice(baseline_slices, gate_name)
        candidate_value = _macro_from_slice(candidate_slices, gate_name)
        values[gate_name] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": candidate_value - baseline_value,
        }

    gates: dict[str, dict[str, object]] = {}
    for gate_name in ("overall", "small", "far", "near", "large"):
        operator, threshold = FROZEN_THRESHOLDS[gate_name]
        delta = values[gate_name]["delta"]
        gates[gate_name] = {
            **values[gate_name],
            "operator": operator,
            "threshold": threshold,
            "pass": _gate_pass(delta, operator, threshold),
        }
    failed_gates = [name for name, payload in gates.items() if not payload["pass"]]
    normalized_engineering = {
        name: _engineering_pass(value)
        for name, value in sorted(engineering_checks.items())
    }
    failed_engineering = [
        name for name, passed in normalized_engineering.items() if not passed
    ]
    scientific_pass = not failed_gates
    engineering_pass = bool(normalized_engineering) and not failed_engineering
    decision = "GO" if scientific_pass and engineering_pass else "NO_GO"
    return {
        "schema_version": 1,
        "decision": decision,
        "scientific_pass": scientific_pass,
        "engineering_pass": engineering_pass,
        "failed_gates": failed_gates,
        "failed_engineering_checks": failed_engineering,
        "gates": gates,
        "engineering_checks": normalized_engineering,
        "overall_metric": "Pedestrian/Cyclist Moderate macro AP_R40",
        "slice_metric": "KITTI_2D_CONDITIONAL_AP40",
        "slice_base_difficulty": "hard",
        "slice_report_role": "frozen_gate",
        "classes": list(CLASSES),
        "baseline_run": baseline_run,
        "candidate_run": candidate_run,
        "baseline_overall_classes": baseline_classes,
        "candidate_overall_classes": candidate_classes,
        "split_count": FROZEN_SPLIT_COUNT,
        "split_sha256": FROZEN_SPLIT_SHA256,
        "endpoint": "epoch15_last.pt_only",
        "local_result_role": "directional_screen_not_formal",
    }


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply the predeclared v213 epoch-15 scientific and engineering gate."
    )
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--stratified-report", type=Path, required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--engineering-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    engineering_payload = _load_json(args.engineering_evidence)
    checks = engineering_payload.get("checks", engineering_payload)
    if not isinstance(checks, Mapping):
        parser.error("engineering evidence must contain an object named checks")
    report = compute_gate(
        baseline_metrics=_load_json(args.baseline_metrics),
        candidate_metrics=_load_json(args.candidate_metrics),
        stratified_report=_load_json(args.stratified_report),
        baseline_run=args.baseline_run,
        candidate_run=args.candidate_run,
        engineering_checks=checks,
    )
    report["provenance"] = {
        "baseline_metrics": {
            "path": str(args.baseline_metrics.resolve()),
            "sha256": _sha256(args.baseline_metrics),
        },
        "candidate_metrics": {
            "path": str(args.candidate_metrics.resolve()),
            "sha256": _sha256(args.candidate_metrics),
        },
        "stratified_report": {
            "path": str(args.stratified_report.resolve()),
            "sha256": _sha256(args.stratified_report),
        },
        "engineering_evidence": {
            "path": str(args.engineering_evidence.resolve()),
            "sha256": _sha256(args.engineering_evidence),
        },
        "gate_evaluator": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"decision={report['decision']}")
    print(f"failed_gates={','.join(report['failed_gates']) or 'none'}")
    print(
        "failed_engineering_checks="
        f"{','.join(report['failed_engineering_checks']) or 'none'}"
    )
    print(f"gate_report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
