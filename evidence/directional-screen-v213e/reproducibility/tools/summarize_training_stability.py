from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


GENERIC_METRICS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)
LR_COLUMNS = ("lr/pg0", "lr/pg1", "lr/pg2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_run(path: Path, expected_epochs: int) -> tuple[dict[str, Any], list[dict[str, float]]]:
    resolved = path.resolve()
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {resolved}")
        required = {"epoch", "time", *GENERIC_METRICS}
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"missing required result columns: {resolved}")
        rows: list[dict[str, float]] = []
        for row_number, raw in enumerate(reader, 2):
            parsed: dict[str, float] = {}
            for key, value in raw.items():
                try:
                    number = float(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{resolved}:{row_number}: non-numeric {key}") from error
                if not math.isfinite(number):
                    raise ValueError(f"{resolved}:{row_number}: non-finite {key}")
                parsed[key] = number
            rows.append(parsed)
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, expected_epochs + 1)):
        raise ValueError(f"epoch sequence mismatch: {resolved}: {epochs}")
    times = [row["time"] for row in rows]
    if any(current <= previous for previous, current in zip(times, times[1:])):
        raise ValueError(f"training time is not strictly increasing: {resolved}")
    final = rows[-1]
    loss_columns = [
        name
        for name in rows[0]
        if name.startswith("train/") or name.startswith("val/")
    ]
    generic_trajectory = [
        {"epoch": int(row["epoch"]), **{key: row[key] for key in GENERIC_METRICS}}
        for row in rows
    ]
    best_row = max(rows, key=lambda row: row["metrics/mAP50-95(B)"])
    summary = {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "rows": len(rows),
        "epoch_sequence_complete": True,
        "all_values_finite": True,
        "time_strictly_increasing": True,
        "duration_seconds": final["time"],
        "final_generic_metrics": {key: final[key] for key in GENERIC_METRICS},
        "final_losses": {key: final[key] for key in loss_columns},
        "best_generic_map50_95_descriptive_only": {
            "epoch": int(best_row["epoch"]),
            "value": best_row["metrics/mAP50-95(B)"],
        },
        "generic_metric_trajectory": generic_trajectory,
    }
    return summary, rows


def summarize_trajectories(
    *,
    runs: dict[str, Path],
    expected_epochs: int,
    reference_name: str | None,
    candidate_name: str | None,
) -> dict[str, Any]:
    if not runs or expected_epochs <= 0:
        raise ValueError("runs and a positive expected epoch count are required")
    summaries: dict[str, Any] = {}
    parsed: dict[str, list[dict[str, float]]] = {}
    for name, path in runs.items():
        summaries[name], parsed[name] = _load_run(path, expected_epochs)

    available_lr_columns = tuple(
        column
        for column in LR_COLUMNS
        if all(column in rows[0] for rows in parsed.values())
    )
    lr_identical = all(
        [
            [row[column] for row in rows]
            for rows in parsed.values()
        ][1:]
        == [
            [row[column] for row in next(iter(parsed.values()))]
            for _ in range(len(parsed) - 1)
        ]
        for column in available_lr_columns
    ) if len(parsed) > 1 else True

    comparison: dict[str, Any] | None = None
    if reference_name is not None or candidate_name is not None:
        if reference_name not in parsed or candidate_name not in parsed:
            raise ValueError("comparison names must reference supplied runs")
        reference = parsed[reference_name][-1]
        candidate = parsed[candidate_name][-1]
        common = sorted(set(reference) & set(candidate) - {"epoch", "time"})
        comparison = {
            "reference": reference_name,
            "candidate": candidate_name,
            "final_delta": {key: candidate[key] - reference[key] for key in common},
        }

    return {
        "schema_version": 1,
        "role": "training_stability_and_generic_metrics_not_frozen_ap40_selection",
        "expected_epochs": expected_epochs,
        "all_training_stability_checks_pass": True,
        "lr_columns_compared": list(available_lr_columns),
        "lr_schedule_identical": lr_identical,
        "runs": summaries,
        "comparison": comparison,
    }


def _parse_run_spec(specification: str) -> tuple[str, Path]:
    try:
        name, raw_path = specification.split("=", 1)
    except ValueError as error:
        raise ValueError("run must use NAME=RESULTS_CSV") from error
    if not name or not raw_path:
        raise ValueError("run specification contains an empty field")
    return name, Path(raw_path)


def _write_json_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and compare YOLO training trajectories.")
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--expected-epochs", type=int, required=True)
    parser.add_argument("--reference-name")
    parser.add_argument("--candidate-name")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs: dict[str, Path] = {}
    for specification in args.run:
        name, path = _parse_run_spec(specification)
        if name in runs:
            raise ValueError(f"duplicate run name: {name}")
        runs[name] = path
    report = summarize_trajectories(
        runs=runs,
        expected_epochs=args.expected_epochs,
        reference_name=args.reference_name,
        candidate_name=args.candidate_name,
    )
    _write_json_new(args.output.resolve(), report)
    print(
        f"training_stability={report['all_training_stability_checks_pass']} "
        f"runs={len(runs)} lr_identical={report['lr_schedule_identical']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
