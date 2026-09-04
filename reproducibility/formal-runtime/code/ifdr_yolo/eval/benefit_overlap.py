"""Fixed-image KITTI Moderate benefit/harm overlap diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random

from PIL import Image

from ifdr_yolo.data.kitti_types import Detection, Difficulty, EVAL_CLASSES, KittiObject
from ifdr_yolo.eval.kitti_ap40 import (
    CLASS_IOU_THRESHOLDS,
    DIFFICULTY_RULES,
    GroundTruthStatus,
    box_iou,
    classify_ground_truth,
)
from ifdr_yolo.eval.paired_bootstrap import paired_bootstrap_ap40
from ifdr_yolo.eval.prediction_io import load_kitti_ground_truth, load_yolo_predictions


SCHEMA_VERSION = 1
DEFAULT_CLASSES = ("Pedestrian", "Cyclist")
DEFAULT_BOOTSTRAP_ITERATIONS = 1000
DEFAULT_BOOTSTRAP_SEED = 17
AP_NOTE = "AP cannot be computed by summing object counts; it is evaluated separately."


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_bytes(path, (_canonical(value) + "\n").encode("utf-8"))


def _ids(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(values)
    if not result:
        raise ValueError("image_ids must not be empty")
    if len(set(result)) != len(result):
        raise ValueError("image_ids must not contain duplicates")
    for image_id in result:
        if not isinstance(image_id, str) or len(image_id) != 6 or not image_id.isdigit():
            raise ValueError(f"invalid KITTI image ID: {image_id!r}")
    return result


def _files(
    directory: Path,
    suffix: str,
    image_ids: tuple[str, ...],
    label: str,
    *,
    missing_as_empty: bool = False,
) -> dict[str, Path | None]:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {directory}")
    paths = {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == suffix
    }
    extra = sorted(set(paths) - set(image_ids))
    if extra:
        raise ValueError(f"{label} has extra image IDs: {extra}")
    missing = sorted(set(image_ids) - set(paths))
    if missing and not missing_as_empty:
        raise ValueError(f"{label} is missing image IDs: {missing}")
    return {image_id: paths.get(image_id) for image_id in image_ids}


def _file_record(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    return {"path": str(path.resolve()), "sha256": _sha256_file(path), "size": path.stat().st_size}


def _intersection_over_detection(detection: Detection, region: KittiObject) -> float:
    left, right = detection.bbox, region.bbox
    width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    return width * height / left.area if left.area > 0.0 else 0.0


def _identity(image_id: str, index: int, obj: KittiObject) -> str:
    return f"{image_id}:{obj.kind}:{index}"


def _match(
    image_id: str,
    objects: tuple[KittiObject, ...],
    detections: tuple[Detection, ...],
    class_name: str,
) -> dict[str, object]:
    valid: list[tuple[str, KittiObject]] = []
    ignored: list[KittiObject] = []
    dontcare = [obj for obj in objects if obj.kind == "DontCare"]
    for index, obj in enumerate(objects):
        status = classify_ground_truth(obj, class_name, Difficulty.MODERATE)
        if status is GroundTruthStatus.VALID:
            valid.append((_identity(image_id, index, obj), obj))
        elif status is GroundTruthStatus.IGNORED:
            ignored.append(obj)

    threshold = CLASS_IOU_THRESHOLDS[class_name]
    min_height = DIFFICULTY_RULES[Difficulty.MODERATE][0]
    used_valid = [False] * len(valid)
    used_ignored = [False] * len(ignored)
    tp_ids: list[str] = []
    fp = duplicates = ignored_detections = 0
    ranked = sorted((d for d in detections if d.kind == class_name), key=lambda d: d.score, reverse=True)
    for detection in ranked:
        best = -1
        best_iou = threshold
        for index, (_, obj) in enumerate(valid):
            if not used_valid[index]:
                overlap = box_iou(detection.bbox, obj.bbox)
                if overlap > best_iou:
                    best, best_iou = index, overlap
        if best >= 0:
            used_valid[best] = True
            tp_ids.append(valid[best][0])
            continue

        best = -1
        best_iou = threshold
        for index, obj in enumerate(ignored):
            if not used_ignored[index]:
                overlap = box_iou(detection.bbox, obj.bbox)
                if overlap > best_iou:
                    best, best_iou = index, overlap
        if best >= 0:
            used_ignored[best] = True
            ignored_detections += 1
            continue

        if (
            detection.bbox.height < min_height
            or any(_intersection_over_detection(detection, obj) > threshold for obj in dontcare)
        ):
            ignored_detections += 1
            continue
        fp += 1
        duplicates += any(
            used_valid[index] and box_iou(detection.bbox, obj.bbox) > threshold
            for index, (_, obj) in enumerate(valid)
        )
    return {
        "tp_ids": sorted(tp_ids),
        "tp": len(tp_ids),
        "fp": fp,
        "duplicates": int(duplicates),
        "ignored": ignored_detections,
    }


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss == 0.0 or right_ss == 0.0:
        return None
    return sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right)) / math.sqrt(left_ss * right_ss)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def _object_bootstrap(rows: Mapping[str, Mapping[str, object]], image_ids: tuple[str, ...], iterations: int, seed: int) -> dict[str, object]:
    names = ("a_only", "b_only", "overlap", "a_harm", "b_harm", "a_fp_delta", "b_fp_delta", "a_duplicate_delta", "b_duplicate_delta", "rescue_jaccard", "overlap_min_rescue_ratio")
    samples = {name: [] for name in names}
    randomizer = random.Random(seed)
    for _ in range(iterations):
        counts = {name: 0.0 for name in names[:-2]}
        for image_id in (randomizer.choice(image_ids) for _ in image_ids):
            row = rows[image_id]
            base, a, b = (set(row[name]["tp_ids"]) for name in ("base", "A", "B"))
            a_rescue, b_rescue = a - base, b - base
            counts["a_only"] += len(a_rescue - b_rescue)
            counts["b_only"] += len(b_rescue - a_rescue)
            counts["overlap"] += len(a_rescue & b_rescue)
            counts["a_harm"] += len(base - a)
            counts["b_harm"] += len(base - b)
            counts["a_fp_delta"] += row["A"]["fp"] - row["base"]["fp"]
            counts["b_fp_delta"] += row["B"]["fp"] - row["base"]["fp"]
            counts["a_duplicate_delta"] += row["A"]["duplicates"] - row["base"]["duplicates"]
            counts["b_duplicate_delta"] += row["B"]["duplicates"] - row["base"]["duplicates"]
        union = counts["a_only"] + counts["b_only"] + counts["overlap"]
        minimum = min(counts["a_only"] + counts["overlap"], counts["b_only"] + counts["overlap"])
        samples["rescue_jaccard"].append(counts["overlap"] / union if union else 0.0)
        samples["overlap_min_rescue_ratio"].append(counts["overlap"] / minimum if minimum else 0.0)
        for name, value in counts.items():
            samples[name].append(value)
    return {
        "metric": "object_identity_image_cluster_bootstrap",
        "seed": seed,
        "iterations": iterations,
        "confidence": 0.95,
        "ci95": {name: {"lower": _percentile(values, 0.025), "upper": _percentile(values, 0.975)} for name, values in samples.items()},
    }


def _ap_bootstrap(gt: Mapping[str, tuple[KittiObject, ...]], base: Mapping[str, tuple[Detection, ...]], candidate: Mapping[str, tuple[Detection, ...]], class_name: str, iterations: int, seed: int) -> dict[str, object]:
    try:
        result = paired_bootstrap_ap40(
            gt_by_image=dict(gt), reference_by_image=dict(base), candidate_by_image=dict(candidate),
            class_name=class_name, difficulty=Difficulty.MODERATE, iterations=iterations, seed=seed,
        )
    except (ValueError, RuntimeError) as error:
        return {"available": False, "reason": str(error), "metric": "KITTI_MODERATE_AP40_paired_image_bootstrap"}
    return {
        "available": True,
        "metric": "KITTI_MODERATE_AP40_paired_image_bootstrap",
        "reference_ap40": result.reference_ap40,
        "candidate_ap40": result.candidate_ap40,
        "difference_ap40": result.difference_ap40,
        "ci95": {"lower": result.ci_lower, "upper": result.ci_upper},
        "iterations": result.iterations,
        "seed": result.seed,
    }


def _manifest(ids: tuple[str, ...], image_ids_path: Path | None, image_paths: Mapping[str, Path], label_paths: Mapping[str, Path], prediction_paths: Mapping[str, Mapping[str, Path | None]], sizes: Mapping[str, tuple[int, int]]) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "image_ids": list(ids),
        "image_ids_sha256": _sha256_bytes(_canonical(list(ids)).encode()),
        "image_sizes": {image_id: list(sizes[image_id]) for image_id in ids},
        "images": {image_id: _file_record(image_paths[image_id]) for image_id in ids},
        "labels": {image_id: _file_record(label_paths[image_id]) for image_id in ids},
        "predictions": {
            condition: {
                "missing_as_empty": True,
                "files": {image_id: _file_record(prediction_paths[condition][image_id]) for image_id in ids},
            }
            for condition in ("P2", "A", "B")
        },
    }
    if image_ids_path is not None:
        result["image_ids_file"] = _file_record(image_ids_path)
    result["manifest_sha256"] = _sha256_bytes(_canonical(result).encode())
    return result


def _input_hash(manifest: Mapping[str, object], image_id: str) -> str:
    return _sha256_bytes(_canonical({
        "image": manifest["images"][image_id], "label": manifest["labels"][image_id],
        **{condition: manifest["predictions"][condition]["files"][image_id] for condition in ("P2", "A", "B")},
    }).encode())


def _journal(path: Path, manifest_sha: str, ids: tuple[str, ...]) -> dict[str, dict[str, object]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical({"schema_version": SCHEMA_VERSION, "kind": "manifest", "manifest_sha256": manifest_sha}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return {}
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    if lines and not raw.endswith(b"\n"):
        lines.pop()  # tolerate a process dying during the final append
    if not lines:
        raise ValueError("benefit-overlap journal is empty")
    header = json.loads(lines[0])
    if header.get("kind") != "manifest" or header.get("manifest_sha256") != manifest_sha:
        raise ValueError("benefit-overlap journal manifest mismatch")
    result: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(lines[1:], 2):
        record = json.loads(line)
        image_id = record.get("image_id")
        if record.get("kind") != "image" or image_id not in ids or image_id in result:
            raise ValueError(f"benefit-overlap journal identity mismatch at line {line_number}")
        result[image_id] = record
    return result


def _append_journal(path: Path, record: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_canonical(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _totals(rows: Mapping[str, Mapping[str, object]], condition: str) -> dict[str, int]:
    return {name: sum(int(row[condition][name]) for row in rows.values()) for name in ("tp", "fp", "duplicates", "ignored")}


def _class_result(rows: Mapping[str, Mapping[str, object]], ids: tuple[str, ...], gt: Mapping[str, tuple[KittiObject, ...]], predictions: Mapping[str, Mapping[str, tuple[Detection, ...]]], class_name: str, iterations: int, seed: int) -> dict[str, object]:
    identity_sets = {condition: {value for row in rows.values() for value in row[condition]["tp_ids"]} for condition in ("base", "A", "B")}
    base_ids, a_ids, b_ids = identity_sets["base"], identity_sets["A"], identity_sets["B"]
    a_rescue, b_rescue = a_ids - base_ids, b_ids - base_ids
    overlap, a_only, b_only = a_rescue & b_rescue, a_rescue - b_rescue, b_rescue - a_rescue
    base, a, b = (_totals(rows, condition) for condition in ("base", "A", "B"))
    changes = {
        condition: {
            "tp_delta": candidate["tp"] - base["tp"],
            "fp_delta": candidate["fp"] - base["fp"],
            "new_fp": max(0, candidate["fp"] - base["fp"]),
            "duplicate_delta": candidate["duplicates"] - base["duplicates"],
        }
        for condition, candidate in (("A", a), ("B", b))
    }
    correlations = {}
    for condition in ("A", "B"):
        correlations[condition] = _pearson(
            [row[condition]["tp"] - row["base"]["tp"] for row in rows.values()],
            [row[condition]["fp"] - row["base"]["fp"] for row in rows.values()],
        )
    union = a_rescue | b_rescue
    minimum = min(len(a_rescue), len(b_rescue))
    return {
        "base": base,
        "A": a,
        "B": b,
        "rescue": {
            "A_only": len(a_only), "B_only": len(b_only), "overlap": len(overlap), "union": len(union),
            "jaccard": len(overlap) / len(union) if union else 0.0,
            "overlap_min_rescue_ratio": len(overlap) / minimum if minimum else 0.0,
            "A_only_identities": sorted(a_only), "B_only_identities": sorted(b_only), "overlap_identities": sorted(overlap),
        },
        "harm": {"A": len(base_ids - a_ids), "B": len(base_ids - b_ids), "A_identities": sorted(base_ids - a_ids), "B_identities": sorted(base_ids - b_ids)},
        "changes": changes,
        "paired_delta_correlation": correlations,
        "bootstrap": {
            "object_overlap": _object_bootstrap(rows, ids, iterations, seed),
            "ap40": {
                "A_vs_P2": _ap_bootstrap(gt, predictions["P2"], predictions["A"], class_name, iterations, seed),
                "B_vs_P2": _ap_bootstrap(gt, predictions["P2"], predictions["B"], class_name, iterations, seed),
            },
        },
    }


def analyze_benefit_overlap(*, image_ids: Sequence[str], image_dir: Path, label_dir: Path, p2_dir: Path, a_dir: Path, b_dir: Path, class_names: Sequence[str] = DEFAULT_CLASSES, bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS, bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED, journal_path: Path | None = None, image_ids_path: Path | None = None, output_json: Path | None = None, output_csv: Path | None = None, max_images: int | None = None) -> dict[str, object]:
    ids = _ids(image_ids)
    classes = tuple(class_names)
    if not classes or len(set(classes)) != len(classes) or any(name not in EVAL_CLASSES for name in classes):
        raise ValueError("class_names must be unique KITTI evaluation classes")
    if isinstance(bootstrap_iterations, bool) or not isinstance(bootstrap_iterations, int) or bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be a positive integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a non-negative integer")
    if max_images is not None and (isinstance(max_images, bool) or not isinstance(max_images, int) or max_images < 0):
        raise ValueError("max_images must be a non-negative integer or None")
    image_paths = _files(Path(image_dir), ".png", ids, "image")
    label_paths = _files(Path(label_dir), ".txt", ids, "ground-truth label")
    prediction_paths = {
        "P2": _files(Path(p2_dir), ".txt", ids, "P2 prediction", missing_as_empty=True),
        "A": _files(Path(a_dir), ".txt", ids, "A prediction", missing_as_empty=True),
        "B": _files(Path(b_dir), ".txt", ids, "B prediction", missing_as_empty=True),
    }
    image_sizes: dict[str, tuple[int, int]] = {}
    for image_id in ids:
        with Image.open(image_paths[image_id]) as image:  # type: ignore[arg-type]
            image_sizes[image_id] = image.size
    manifest = _manifest(ids, image_ids_path, image_paths, label_paths, prediction_paths, image_sizes)  # type: ignore[arg-type]
    ground_truth = load_kitti_ground_truth(Path(label_dir).resolve(), ids)
    predictions = {condition: load_yolo_predictions(Path(directory).resolve(), image_sizes) for condition, directory in (("P2", p2_dir), ("A", a_dir), ("B", b_dir))}
    journal_records = _journal(Path(journal_path), manifest["manifest_sha256"], ids) if journal_path is not None else {}
    per_class: dict[str, dict[str, dict[str, object]]] = {name: {} for name in classes}
    processed = 0
    for image_id in ids:
        record = journal_records.get(image_id)
        if record is not None:
            if record.get("input_sha256") != _input_hash(manifest, image_id):
                raise ValueError(f"benefit-overlap journal input identity mismatch: {image_id}")
            payload = record.get("classes")
            if not isinstance(payload, Mapping):
                raise ValueError(f"invalid journal class payload: {image_id}")
            for class_name in classes:
                if class_name not in payload or not isinstance(payload[class_name], Mapping):
                    raise ValueError(f"benefit-overlap journal class mismatch: {image_id}")
                per_class[class_name][image_id] = dict(payload[class_name])
            continue
        if max_images is not None and processed >= max_images:
            raise InterruptedError("benefit-overlap interrupted after max_images")
        class_payload = {
            class_name: {
                "base": _match(image_id, ground_truth[image_id], predictions["P2"][image_id], class_name),
                "A": _match(image_id, ground_truth[image_id], predictions["A"][image_id], class_name),
                "B": _match(image_id, ground_truth[image_id], predictions["B"][image_id], class_name),
            }
            for class_name in classes
        }
        for class_name in classes:
            per_class[class_name][image_id] = class_payload[class_name]
        if journal_path is not None:
            _append_journal(Path(journal_path), {"schema_version": SCHEMA_VERSION, "kind": "image", "image_id": image_id, "input_sha256": _input_hash(manifest, image_id), "classes": class_payload})
        processed += 1
    if set(per_class[classes[0]]) != set(ids):
        raise InterruptedError("benefit-overlap journal is incomplete")
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "metric": "KITTI_MODERATE_OBJECT_IDENTITY_BENEFIT_OVERLAP",
        "ap_note": AP_NOTE,
        "image_ids": list(ids),
        "manifest": manifest,
        "classes": {class_name: _class_result(per_class[class_name], ids, ground_truth, predictions, class_name, bootstrap_iterations, bootstrap_seed) for class_name in classes},
        "per_image": per_class,
    }
    if output_json is not None:
        write_benefit_overlap_json(Path(output_json), result)
    if output_csv is not None:
        write_benefit_overlap_csv(Path(output_csv), result)
    return result


def write_benefit_overlap_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_json(Path(path), payload)


def write_benefit_overlap_csv(path: Path, payload: Mapping[str, object]) -> None:
    classes = payload.get("classes")
    manifest = payload.get("manifest")
    if not isinstance(classes, Mapping):
        raise ValueError("benefit-overlap payload classes must be a mapping")
    fields = ("manifest_sha256", "class_name", "base_tp", "a_tp", "b_tp", "base_fp", "a_fp", "b_fp", "a_only_rescue", "b_only_rescue", "overlap_rescue", "a_harm", "b_harm", "rescue_jaccard", "overlap_min_rescue_ratio", "a_fp_delta", "b_fp_delta", "a_duplicate_delta", "b_duplicate_delta", "a_tp_fp_delta_correlation", "b_tp_fp_delta_correlation", "ap40_a_ci_lower", "ap40_a_ci_upper", "ap40_b_ci_lower", "ap40_b_ci_upper")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    manifest_sha = manifest.get("manifest_sha256") if isinstance(manifest, Mapping) else None
    for class_name in sorted(classes):
        metrics = classes[class_name]
        ap = metrics["bootstrap"]["ap40"]
        a_ci = ap["A_vs_P2"].get("ci95", {}) if ap["A_vs_P2"].get("available") else {}
        b_ci = ap["B_vs_P2"].get("ci95", {}) if ap["B_vs_P2"].get("available") else {}
        writer.writerow({
            "manifest_sha256": manifest_sha, "class_name": class_name,
            "base_tp": metrics["base"]["tp"], "a_tp": metrics["A"]["tp"], "b_tp": metrics["B"]["tp"],
            "base_fp": metrics["base"]["fp"], "a_fp": metrics["A"]["fp"], "b_fp": metrics["B"]["fp"],
            "a_only_rescue": metrics["rescue"]["A_only"], "b_only_rescue": metrics["rescue"]["B_only"], "overlap_rescue": metrics["rescue"]["overlap"],
            "a_harm": metrics["harm"]["A"], "b_harm": metrics["harm"]["B"], "rescue_jaccard": metrics["rescue"]["jaccard"], "overlap_min_rescue_ratio": metrics["rescue"]["overlap_min_rescue_ratio"],
            "a_fp_delta": metrics["changes"]["A"]["fp_delta"], "b_fp_delta": metrics["changes"]["B"]["fp_delta"], "a_duplicate_delta": metrics["changes"]["A"]["duplicate_delta"], "b_duplicate_delta": metrics["changes"]["B"]["duplicate_delta"],
            "a_tp_fp_delta_correlation": metrics["paired_delta_correlation"]["A"], "b_tp_fp_delta_correlation": metrics["paired_delta_correlation"]["B"],
            "ap40_a_ci_lower": a_ci.get("lower"), "ap40_a_ci_upper": a_ci.get("upper"), "ap40_b_ci_lower": b_ci.get("lower"), "ap40_b_ci_upper": b_ci.get("upper"),
        })
    _atomic_bytes(Path(path), output.getvalue().encode("utf-8"))


__all__ = ["AP_NOTE", "DEFAULT_CLASSES", "DEFAULT_BOOTSTRAP_ITERATIONS", "DEFAULT_BOOTSTRAP_SEED", "analyze_benefit_overlap", "write_benefit_overlap_csv", "write_benefit_overlap_json"]
