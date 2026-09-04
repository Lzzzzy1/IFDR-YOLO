from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence


def _bootstrap_source_root(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-root", type=Path)
    args, _ = parser.parse_known_args(argv)
    if args.source_root is not None:
        sys.path.insert(0, str(args.source_root.resolve()))


_bootstrap_source_root(sys.argv[1:])

from PIL import Image  # noqa: E402

from ifdr_yolo.data.kitti_types import (  # noqa: E402
    Detection,
    Difficulty,
    KittiObject,
)
from ifdr_yolo.data.splits import load_ids, sha256_file  # noqa: E402
from ifdr_yolo.eval.kitti_ap40 import evaluate_class  # noqa: E402
from ifdr_yolo.eval.prediction_io import (  # noqa: E402
    load_kitti_ground_truth,
    load_yolo_predictions,
)
from ifdr_yolo.eval.stratified_ap40 import (  # noqa: E402
    KITTI_RESEARCH_SLICES,
    TargetSlice,
)


DEFAULT_CLASSES = ("Pedestrian", "Cyclist")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_prediction_ids(
    prediction_dir: Path,
    image_ids: tuple[str, ...],
) -> None:
    if not prediction_dir.is_dir():
        raise FileNotFoundError(
            f"prediction directory does not exist: {prediction_dir}"
        )
    actual = {path.stem for path in prediction_dir.glob("*.txt")}
    expected = set(image_ids)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "prediction IDs do not match split: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )


def evaluate_moderate_slices(
    *,
    gt_by_image: dict[str, tuple[KittiObject, ...]],
    detections_by_image: dict[str, tuple[Detection, ...]],
    classes: tuple[str, ...] = DEFAULT_CLASSES,
    target_slices: tuple[TargetSlice, ...] = KITTI_RESEARCH_SLICES,
) -> dict[str, object]:
    if not classes:
        raise ValueError("at least one evaluation class is required")
    grouped: dict[str, dict[str, object]] = {}
    for target_slice in target_slices:
        class_metrics: dict[str, object] = {}
        ap_values: list[float] = []
        for class_name in classes:
            metrics = evaluate_class(
                gt_by_image=gt_by_image,
                detections_by_image=detections_by_image,
                class_name=class_name,
                difficulty=Difficulty.MODERATE,
                valid_selector=target_slice.matches,
            )
            ap_values.append(metrics.ap40)
            class_metrics[class_name] = {
                "ap40": metrics.ap40,
                "num_valid_gt": metrics.num_valid_gt,
                "true_positives": metrics.true_positives,
                "false_positives": metrics.false_positives,
                "ignored_detections": metrics.ignored_detections,
            }
        grouped.setdefault(target_slice.axis, {})[target_slice.name] = {
            "definition": asdict(target_slice),
            "classes": class_metrics,
            "macro_ap40": sum(ap_values) / len(ap_values),
        }
    return {
        "schema_version": 1,
        "metric": "KITTI_2D_MODERATE_CONDITIONAL_AP40",
        "base_difficulty": Difficulty.MODERATE.value,
        "classes": list(classes),
        "slices": grouped,
    }


def evaluate_runs(
    *,
    run_prediction_dirs: dict[str, Path],
    label_dir: Path,
    image_dir: Path,
    split_path: Path,
    source_root: Path,
) -> dict[str, object]:
    if not run_prediction_dirs:
        raise ValueError("at least one prediction run is required")
    image_ids = load_ids(split_path)
    image_sizes: dict[str, tuple[int, int]] = {}
    for image_id in image_ids:
        image_path = image_dir / f"{image_id}.png"
        if not image_path.is_file():
            raise FileNotFoundError(
                f"evaluation image does not exist: {image_path}"
            )
        with Image.open(image_path) as image:
            image_sizes[image_id] = image.size
    ground_truth = load_kitti_ground_truth(label_dir, image_ids)

    runs: dict[str, object] = {}
    for run_name, prediction_dir in sorted(run_prediction_dirs.items()):
        _validate_prediction_ids(prediction_dir, image_ids)
        predictions = load_yolo_predictions(prediction_dir, image_sizes)
        evaluation = evaluate_moderate_slices(
            gt_by_image=ground_truth,
            detections_by_image=predictions,
        )
        evaluation["prediction_dir"] = str(prediction_dir.resolve())
        runs[run_name] = evaluation

    evaluator_path = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "evaluator": "independent_moderate_stratified_ap40",
        "interpretation": "descriptive_only_not_frozen_v213_gate",
        "split_sha256": sha256_file(split_path),
        "split_count": len(image_ids),
        "source_root": str(source_root.resolve()),
        "evaluator_sha256": _sha256(evaluator_path),
        "kitti_ap40_sha256": _sha256(
            source_root / "ifdr_yolo" / "eval" / "kitti_ap40.py"
        ),
        "stratified_selector_sha256": _sha256(
            source_root / "ifdr_yolo" / "eval" / "stratified_ap40.py"
        ),
        "runs": runs,
    }


def write_report(output_dir: Path, report: dict[str, object]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "moderate_stratified_ap40.json"
    csv_path = output_dir / "moderate_stratified_ap40.csv"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "run",
                "axis",
                "slice",
                "class",
                "ap40",
                "macro_ap40",
                "num_valid_gt",
                "true_positives",
                "false_positives",
                "ignored_detections",
            )
        )
        for run_name, run in sorted(report["runs"].items()):
            for axis, slices in sorted(run["slices"].items()):
                for slice_name, payload in sorted(slices.items()):
                    for class_name, metrics in sorted(
                        payload["classes"].items()
                    ):
                        writer.writerow(
                            (
                                run_name,
                                axis,
                                slice_name,
                                class_name,
                                metrics["ap40"],
                                payload["macro_ap40"],
                                metrics["num_valid_gt"],
                                metrics["true_positives"],
                                metrics["false_positives"],
                                metrics["ignored_detections"],
                            )
                        )
    return json_path, csv_path


def _parse_run(value: str) -> tuple[str, Path]:
    name, separator, path_text = value.partition("=")
    if not separator or not name.strip() or not path_text.strip():
        raise argparse.ArgumentTypeError("run must use NAME=PREDICTION_DIR")
    return name.strip(), Path(path_text.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Emit descriptive Moderate-valid Pedestrian/Cyclist AP_R40 "
            "slices without replacing the frozen HARD-conditional gate."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = dict(args.run)
    if len(runs) != len(args.run):
        parser.error("run names must be unique")
    report = evaluate_runs(
        run_prediction_dirs=runs,
        label_dir=args.label_dir,
        image_dir=args.image_dir,
        split_path=args.split,
        source_root=args.source_root,
    )
    json_path, csv_path = write_report(args.output_dir, report)
    print(f"moderate_stratified_json={json_path}")
    print(f"moderate_stratified_csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
