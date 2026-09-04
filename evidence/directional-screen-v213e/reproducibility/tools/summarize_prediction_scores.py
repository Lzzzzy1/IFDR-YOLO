from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


CLASS_NAMES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_ids(path: Path) -> tuple[str, ...]:
    image_ids = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not image_ids or len(set(image_ids)) != len(image_ids):
        raise ValueError("split must contain unique non-empty image IDs")
    return image_ids


def _parse_run(
    prediction_dir: Path,
    image_ids: tuple[str, ...],
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    prediction_dir = prediction_dir.resolve()
    actual = {path.stem for path in prediction_dir.glob("*.txt") if path.is_file()}
    if actual != set(image_ids):
        raise ValueError(
            f"prediction file set does not match split: {prediction_dir}; "
            f"missing={len(set(image_ids) - actual)} extra={len(actual - set(image_ids))}"
        )
    scores: dict[str, list[float]] = {name: [] for name in CLASS_NAMES.values()}
    images: dict[str, set[str]] = {name: set() for name in CLASS_NAMES.values()}
    manifest = hashlib.sha256()
    for image_id in image_ids:
        path = prediction_dir / f"{image_id}.txt"
        file_hash = _sha256(path)
        manifest.update(f"{image_id}.txt {file_hash}\n".encode("utf-8"))
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            fields = raw.split()
            if len(fields) != 6:
                raise ValueError(f"{path}:{line_number}: expected six YOLO fields")
            try:
                class_id = int(fields[0])
                coordinates = [float(value) for value in fields[1:5]]
                confidence = float(fields[5])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid numeric field") from error
            if class_id not in CLASS_NAMES:
                raise ValueError(f"{path}:{line_number}: unknown class ID {class_id}")
            if not all(math.isfinite(value) for value in (*coordinates, confidence)):
                raise ValueError(f"{path}:{line_number}: non-finite prediction")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"{path}:{line_number}: confidence outside [0, 1]")
            class_name = CLASS_NAMES[class_id]
            scores[class_name].append(confidence)
            images[class_name].add(image_id)

    classes: dict[str, Any] = {}
    for class_name in CLASS_NAMES.values():
        values = scores[class_name]
        quantiles = {f"q{int(probability * 100):02d}": _quantile(values, probability) for probability in QUANTILES}
        threshold_counts = {
            f"{threshold:g}": sum(value >= threshold for value in values)
            for threshold in thresholds
        }
        classes[class_name] = {
            "detections": len(values),
            "images_with_detection": len(images[class_name]),
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "mean": fmean(values) if values else None,
            "median": quantiles["q50"],
            "quantiles": quantiles,
            "count_at_or_above": threshold_counts,
            "fraction_at_or_above": {
                key: value / len(values) if values else None
                for key, value in threshold_counts.items()
            },
        }
    return {
        "prediction_dir": str(prediction_dir),
        "prediction_file_count": len(image_ids),
        "prediction_manifest_sha256": manifest.hexdigest(),
        "classes": classes,
    }


def summarize_runs(
    *,
    runs: dict[str, Path],
    split_path: Path,
    reference_name: str | None,
    candidate_name: str | None,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one run is required")
    if not thresholds or any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise ValueError("thresholds must be within [0, 1]")
    image_ids = _split_ids(split_path)
    run_reports = {
        name: _parse_run(path, image_ids, thresholds) for name, path in runs.items()
    }
    comparison: dict[str, Any] | None = None
    if reference_name is not None or candidate_name is not None:
        if reference_name not in run_reports or candidate_name not in run_reports:
            raise ValueError("comparison names must reference supplied runs")
        comparison_classes: dict[str, Any] = {}
        for class_name in CLASS_NAMES.values():
            reference = run_reports[reference_name]["classes"][class_name]
            candidate = run_reports[candidate_name]["classes"][class_name]
            comparison_classes[class_name] = {
                "detection_count_delta": candidate["detections"] - reference["detections"],
                "images_with_detection_delta": candidate["images_with_detection"] - reference["images_with_detection"],
                "mean_score_delta": (
                    candidate["mean"] - reference["mean"]
                    if candidate["mean"] is not None and reference["mean"] is not None
                    else None
                ),
                "median_score_delta": (
                    candidate["median"] - reference["median"]
                    if candidate["median"] is not None and reference["median"] is not None
                    else None
                ),
                "count_at_or_above_delta": {
                    key: candidate["count_at_or_above"][key] - value
                    for key, value in reference["count_at_or_above"].items()
                },
            }
        comparison = {
            "reference": reference_name,
            "candidate": candidate_name,
            "classes": comparison_classes,
        }
    return {
        "schema_version": 1,
        "role": "descriptive_raw_post_nms_score_distribution_not_ap",
        "class_map": {str(key): value for key, value in CLASS_NAMES.items()},
        "split_path": str(split_path.resolve()),
        "split_count": len(image_ids),
        "split_sha256": _sha256(split_path.resolve()),
        "thresholds": list(thresholds),
        "runs": run_reports,
        "comparison": comparison,
    }


def _parse_run_spec(specification: str) -> tuple[str, Path]:
    try:
        name, raw_path = specification.split("=", 1)
    except ValueError as error:
        raise ValueError("run must use NAME=PREDICTION_DIR") from error
    if not name or not raw_path:
        raise ValueError("run specification contains an empty field")
    return name, Path(raw_path)


def _write_json_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _write_csv_new(path: Path, report: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run", "class_name", "detections", "images_with_detection", "minimum", "q10", "q25", "median", "q75", "q90", "q95", "q99", "maximum", "mean"]
    fields.extend(f"count_ge_{threshold:g}" for threshold in report["thresholds"])
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run_name, run in report["runs"].items():
            for class_name, summary in run["classes"].items():
                row = {
                    "run": run_name,
                    "class_name": class_name,
                    "detections": summary["detections"],
                    "images_with_detection": summary["images_with_detection"],
                    "minimum": summary["minimum"],
                    "median": summary["median"],
                    "maximum": summary["maximum"],
                    "mean": summary["mean"],
                    **{
                        key: value
                        for key, value in summary["quantiles"].items()
                        if key != "q50"
                    },
                }
                row.update({f"count_ge_{key}": value for key, value in summary["count_at_or_above"].items()})
                writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize raw post-NMS YOLO confidence distributions.")
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--reference-name")
    parser.add_argument("--candidate-name")
    parser.add_argument("--threshold", type=float, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs: dict[str, Path] = {}
    for specification in args.run:
        name, path = _parse_run_spec(specification)
        if name in runs:
            raise ValueError(f"duplicate run name: {name}")
        runs[name] = path
    report = summarize_runs(
        runs=runs,
        split_path=args.split,
        reference_name=args.reference_name,
        candidate_name=args.candidate_name,
        thresholds=tuple(sorted(set(args.threshold))),
    )
    _write_json_new(args.output_json.resolve(), report)
    _write_csv_new(args.output_csv.resolve(), report)
    print(f"prediction_score_summary=complete runs={len(runs)} split={report['split_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
