from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ENDPOINTS = ("overall", "small", "far", "near", "large")
CLASSES = ("Pedestrian", "Cyclist")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(left: object, right: object) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)


def _historical_values(run: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    values = {"overall": float(run["overall"]["macro_ap_r40"])}
    values.update(
        {name: float(run["slices_macro_ap_r40"][name]) for name in ENDPOINTS[1:]}
    )
    classes = {
        name: float(run["overall"]["classes"][name]["ap_r40"])
        for name in CLASSES
    }
    return values, classes


def _gate_result(delta: dict[str, float]) -> tuple[str, list[str], dict[str, bool]]:
    checks = {
        "overall": delta["overall"] > 1.1 or _close(delta["overall"], 1.1),
        "small": delta["small"] > 0.0,
        "far": delta["far"] > 0.0,
        "near": delta["near"] > 0.0 or _close(delta["near"], 0.0),
        "large": delta["large"] > 0.0 or _close(delta["large"], 0.0),
    }
    failed = [name for name in ENDPOINTS if not checks[name]]
    return ("GO" if not failed else "NO_GO"), failed, checks


def compare_endpoints(
    *, historical: dict[str, Any], frozen_gate: dict[str, Any]
) -> dict[str, Any]:
    if frozen_gate.get("baseline_run") != "PLAIN":
        raise ValueError("frozen gate is not bound to PLAIN")
    if frozen_gate.get("candidate_run") != "ANCHORED":
        raise ValueError("frozen gate candidate is not ANCHORED")
    historical_runs = historical.get("runs")
    if not isinstance(historical_runs, dict) or not {
        "PLAIN",
        "B0",
        "B1",
    }.issubset(historical_runs):
        raise ValueError("historical analysis lacks PLAIN/B0/B1")

    values_by_run: dict[str, dict[str, float]] = {}
    classes_by_run: dict[str, dict[str, float]] = {}
    for name in ("PLAIN", "B0", "B1"):
        values_by_run[name], classes_by_run[name] = _historical_values(
            historical_runs[name]
        )

    gate_entries = frozen_gate.get("gates")
    if not isinstance(gate_entries, dict):
        raise ValueError("frozen gate lacks endpoint entries")
    for endpoint in ENDPOINTS:
        if not _close(
            values_by_run["PLAIN"][endpoint], gate_entries[endpoint]["baseline"]
        ):
            raise ValueError(f"PLAIN binding mismatch at {endpoint}")
    for class_name in CLASSES:
        if not _close(
            classes_by_run["PLAIN"][class_name],
            frozen_gate["baseline_overall_classes"][class_name],
        ):
            raise ValueError(f"PLAIN binding mismatch at class {class_name}")

    values_by_run["ANCHORED"] = {
        endpoint: float(gate_entries[endpoint]["candidate"])
        for endpoint in ENDPOINTS
    }
    classes_by_run["ANCHORED"] = {
        class_name: float(frozen_gate["candidate_overall_classes"][class_name])
        for class_name in CLASSES
    }
    baseline = values_by_run["PLAIN"]

    runs: dict[str, Any] = {}
    for name in ("PLAIN", "B0", "B1", "ANCHORED"):
        delta = {
            endpoint: values_by_run[name][endpoint] - baseline[endpoint]
            for endpoint in ENDPOINTS
        }
        if name == "PLAIN":
            decision = "BASELINE"
            failed: list[str] = []
            checks: dict[str, bool] = {}
        else:
            decision, failed, checks = _gate_result(delta)
        runs[name] = {
            "values": values_by_run[name],
            "overall_classes": classes_by_run[name],
            "delta_vs_plain": delta,
            "decision": decision,
            "failed_gates": failed,
            "gate_checks": checks,
        }

    anchored_minus_b1 = {
        endpoint: values_by_run["ANCHORED"][endpoint]
        - values_by_run["B1"][endpoint]
        for endpoint in ENDPOINTS
    }
    anchored_minus_b1_classes = {
        class_name: classes_by_run["ANCHORED"][class_name]
        - classes_by_run["B1"][class_name]
        for class_name in CLASSES
    }
    ranks = {
        endpoint: [
            {"run": name, "ap_r40": values_by_run[name][endpoint]}
            for name in sorted(
                values_by_run,
                key=lambda candidate: values_by_run[candidate][endpoint],
                reverse=True,
            )
        ]
        for endpoint in ENDPOINTS
    }

    return {
        "schema_version": 1,
        "scope": historical.get("scope"),
        "local_result_role": "directional_screen_not_formal",
        "metric": "Moderate Pedestrian/Cyclist macro AP_R40 plus frozen HARD conditional slices",
        "split_count": frozen_gate.get("split_count"),
        "split_sha256": frozen_gate.get("split_sha256"),
        "frozen_gate_contract": {
            "overall": ">= +1.1",
            "small": "> 0",
            "far": "> 0",
            "near": ">= 0",
            "large": ">= 0",
        },
        "runs": runs,
        "anchored_minus_b1": anchored_minus_b1,
        "anchored_minus_b1_overall_classes": anchored_minus_b1_classes,
        "anchored_vs_b1_finding": (
            "ANCHORED_BELOW_B1_ON_ALL_FIVE_ENDPOINTS"
            if all(value < 0.0 for value in anchored_minus_b1.values())
            else "MIXED"
        ),
        "ranks": ranks,
    }


def _write_json_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv_new(path: Path, report: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run",
        "decision",
        "failed_gates",
        *ENDPOINTS,
        *(f"delta_{name}" for name in ENDPOINTS),
        "pedestrian_overall",
        "cyclist_overall",
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, run in report["runs"].items():
            row: dict[str, object] = {
                "run": name,
                "decision": run["decision"],
                "failed_gates": ";".join(run["failed_gates"]),
                "pedestrian_overall": run["overall_classes"]["Pedestrian"],
                "cyclist_overall": run["overall_classes"]["Cyclist"],
            }
            row.update(run["values"])
            row.update(
                {f"delta_{key}": value for key, value in run["delta_vs_plain"].items()}
            )
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare PLAIN/B0/B1/ANCHORED under one frozen local screen."
    )
    parser.add_argument("--historical-analysis", type=Path, required=True)
    parser.add_argument("--frozen-gate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    historical_path = args.historical_analysis.resolve()
    gate_path = args.frozen_gate.resolve()
    report = compare_endpoints(
        historical=_read_json(historical_path),
        frozen_gate=_read_json(gate_path),
    )
    report["provenance"] = {
        "historical_analysis": {
            "path": str(historical_path),
            "sha256": _sha256(historical_path),
        },
        "frozen_gate": {"path": str(gate_path), "sha256": _sha256(gate_path)},
    }
    _write_json_new(args.output_json.resolve(), report)
    _write_csv_new(args.output_csv.resolve(), report)
    print(
        f"endpoint_comparison={report['anchored_vs_b1_finding']} "
        f"anchored_decision={report['runs']['ANCHORED']['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
