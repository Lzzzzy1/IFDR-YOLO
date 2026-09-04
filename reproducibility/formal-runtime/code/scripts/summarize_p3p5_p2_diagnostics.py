"""CPU-only paired diagnostics for the registered P3--P5 and plain-P2 runs.

The formal result is deliberately narrow: the same image-cluster draw is used
for both prediction directories, KITTI's existing AP40 evaluator is reused,
and the paired delta (candidate P2 minus reference P3--P5) is recorded per
replicate.  The output is resumable and identity-bound; this module never
trains a model or changes an evaluator threshold.
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
import random
import socket
import statistics
import sys
import time

from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.data.kitti_types import Detection, Difficulty, KittiObject
from ifdr_yolo.data.splits import load_ids, sha256_file
from ifdr_yolo.eval.kitti_ap40 import evaluate_class
from ifdr_yolo.eval.prediction_io import load_kitti_ground_truth, load_yolo_predictions
from ifdr_yolo.eval.stratified_ap40 import KITTI_RESEARCH_SLICES, TargetSlice


CLASSES = ("Pedestrian", "Cyclist")
FORMAL_REPLICATES = 10_000
DEFAULT_SEED = 17
DEFAULT_CHECKPOINT_INTERVAL = 25
CHECKPOINT_WALL_SECONDS = 240.0
SCHEMA_VERSION = 1
BOOTSTRAP_SLICES = tuple(
    target_slice
    for target_slice in KITTI_RESEARCH_SLICES
    if target_slice.name in {"small_25_40", "far_gt_40m"}
)

# These are the registered artifacts in the current clean 3341/371 protocol.
REGISTERED = {
    "reference_run_identity": "1cf263d0e1b46b8fc810b6ca347f5c0c4745df9250e8cb12698adbec395258da",
    "reference_checkpoint_sha256": "b03ebcfbbde5212195e6a6e57d93637e33443c55515d9c8f8828ce77954fc150",
    "candidate_run_identity": "cf166d1a9b0460f66a5d8901393ef7f332e03c74be509dbe9cd0885c710d447f",
    "candidate_checkpoint_sha256": "c2f9c4c1b7be8697e64352ee2b0a99e5634d1a285b88017be44d71090e7d592c",
    "development_split_sha256": "b1b6b6ee7e5398e93868fab407a2e8a86a53c753667002ef9b8381734ef2cda8",
    "development_count": 371,
    "label_manifest_sha256": "1f1aad3f5157a44cb49ba2920094855a2b48478143f549f671d3d3ed56155192",
    "image_dimensions_sha256": "6cf246e180c1495dfe059c4baffd725c92be2d5cf06c86737b2c19ef2e57f5d0",
    "reference_prediction_manifest_sha256": "3fe40d93871da2fdb71d2accd52daee0caee4bfa445a554bf58008633e580cb4",
    "candidate_prediction_manifest_sha256": "d9424e9fb0764b482e3b6564312e360a3b53fecef355f480f2790e4183dda56c",
}
REGISTERED_OBSERVED = {
    "reference": {"Pedestrian": 93.66667792085089, "Cyclist": 99.87068965517241},
    "candidate": {"Pedestrian": 93.2109883887592, "Cyclist": 97.14285714285714},
}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(path, (_canonical(value) + "\n").encode("utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_prediction_dir(directory: Path, image_ids: tuple[str, ...]) -> tuple[str, list[dict[str, object]]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"prediction directory does not exist: {directory}")
    expected = set(image_ids)
    files = {path.stem: path for path in directory.glob("*.txt") if path.is_file()}
    if set(files) != expected:
        raise ValueError(
            f"prediction IDs do not match split: missing={sorted(expected - set(files))[:5]}, "
            f"extra={sorted(set(files) - expected)[:5]}"
        )
    entries = []
    digest = hashlib.sha256()
    for image_id in image_ids:
        path = files[image_id]
        file_hash = _file_sha256(path)
        entries.append({"image_id": image_id, "name": path.name, "size": path.stat().st_size, "sha256": file_hash})
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest(), entries


def _hash_image_dimensions(image_dir: Path, image_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for image_id in image_ids:
        path = image_dir / f"{image_id}.png"
        if not path.is_file():
            raise FileNotFoundError(f"evaluation image does not exist: {path}")
        with Image.open(path) as image:
            digest.update(f"{image_id}:{image.width}x{image.height}\n".encode("utf-8"))
    return digest.hexdigest()


def _hash_labels(label_dir: Path, image_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for image_id in image_ids:
        path = label_dir / f"{image_id}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"evaluation label does not exist: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # ``os.kill(pid, 0)`` reports success for some exited Windows PIDs.
        # Query the process exit code instead so an interrupted run can resume.
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if os.name != "nt" else value


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile from no values")
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _rng_state_json(generator: random.Random) -> object:
    return json.loads(json.dumps(generator.getstate()))


def _tupleify(value: object) -> object:
    if isinstance(value, list):
        return tuple(_tupleify(item) for item in value)
    return value


def _restore_rng_state(generator: random.Random, state: object) -> None:
    restored = _tupleify(state)
    if not isinstance(restored, tuple):
        raise ValueError("checkpoint RNG state is invalid")
    generator.setstate(restored)


def resolve_replicates(*, benchmark: bool, requested: int | None) -> int:
    if requested is not None and (isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0):
        raise ValueError("replicates must be a positive integer")
    if benchmark:
        if requested is None:
            raise ValueError("benchmark mode requires an explicit replicate count")
        return requested
    if requested is not None and requested != FORMAL_REPLICATES:
        raise ValueError(f"formal mode is fixed at {FORMAL_REPLICATES} replicates; use --benchmark for a screen")
    return FORMAL_REPLICATES


def _implementation_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = {
        "scripts/summarize_p3p5_p2_diagnostics.py": Path(__file__).resolve(),
        "ifdr_yolo/eval/kitti_ap40.py": root / "ifdr_yolo/eval/kitti_ap40.py",
        "ifdr_yolo/eval/prediction_io.py": root / "ifdr_yolo/eval/prediction_io.py",
        "ifdr_yolo/eval/stratified_ap40.py": root / "ifdr_yolo/eval/stratified_ap40.py",
        "ifdr_yolo/data/kitti_types.py": root / "ifdr_yolo/data/kitti_types.py",
        "ifdr_yolo/data/kitti_parser.py": root / "ifdr_yolo/data/kitti_parser.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _identity(
    *,
    split: Path,
    label_dir: Path,
    image_dir: Path,
    image_ids: tuple[str, ...],
    reference_dir: Path,
    candidate_dir: Path,
    reference_run_identity: str,
    reference_checkpoint_sha256: str,
    candidate_run_identity: str,
    candidate_checkpoint_sha256: str,
) -> dict[str, object]:
    reference_aggregate, reference_files = _hash_prediction_dir(reference_dir, image_ids)
    candidate_aggregate, candidate_files = _hash_prediction_dir(candidate_dir, image_ids)
    return {
        "protocol": "3341_fit_371_development",
        "split_sha256": sha256_file(split),
        "split_count": len(image_ids),
        "label_manifest_sha256": _hash_labels(label_dir, image_ids),
        "image_dimensions_sha256": _hash_image_dimensions(image_dir, image_ids),
        "prediction_manifest_sha256": {"reference": reference_aggregate, "candidate": candidate_aggregate},
        "prediction_manifests": {
            "reference": {"directory": str(reference_dir.resolve()), "aggregate_sha256": reference_aggregate, "files": reference_files},
            "candidate": {"directory": str(candidate_dir.resolve()), "aggregate_sha256": candidate_aggregate, "files": candidate_files},
        },
        "registered_runs": {
            "reference": {"run_identity": reference_run_identity, "last_sha256": reference_checkpoint_sha256},
            "candidate": {"run_identity": candidate_run_identity, "last_sha256": candidate_checkpoint_sha256},
        },
        "implementation_sha256": _implementation_identity(),
    }


def _assert_registered_identity(identity: Mapping[str, object], *, strict: bool) -> None:
    if not strict:
        return
    if identity["split_count"] != REGISTERED["development_count"]:
        raise ValueError("registered development split must contain exactly 371 images")
    if identity["split_sha256"] != REGISTERED["development_split_sha256"]:
        raise ValueError("development split identity mismatch")
    for key in ("label_manifest_sha256", "image_dimensions_sha256"):
        if identity[key] != REGISTERED[key]:
            raise ValueError(f"registered {key} mismatch")
    prediction_hashes = identity.get("prediction_manifest_sha256")
    if not isinstance(prediction_hashes, Mapping):
        raise ValueError("prediction manifest identity is malformed")
    if prediction_hashes.get("reference") != REGISTERED["reference_prediction_manifest_sha256"]:
        raise ValueError("registered reference prediction manifest mismatch")
    if prediction_hashes.get("candidate") != REGISTERED["candidate_prediction_manifest_sha256"]:
        raise ValueError("registered candidate prediction manifest mismatch")
    runs = identity["registered_runs"]
    if not isinstance(runs, Mapping):
        raise ValueError("registered run identity is malformed")
    for side in ("reference", "candidate"):
        expected = {
            "reference": (REGISTERED["reference_run_identity"], REGISTERED["reference_checkpoint_sha256"]),
            "candidate": (REGISTERED["candidate_run_identity"], REGISTERED["candidate_checkpoint_sha256"]),
        }[side]
        actual = runs[side]
        if not isinstance(actual, Mapping):
            raise ValueError(f"registered {side} run identity is malformed")
        if (actual.get("run_identity"), actual.get("last_sha256")) != expected:
            raise ValueError(f"registered {side} run identity mismatch")


def _resample_ground_truth(sampled_ids: Sequence[str], ground_truth: Mapping[str, tuple[object, ...]]) -> dict[str, tuple[object, ...]]:
    return {f"bootstrap_{index:06d}": ground_truth[source_id] for index, source_id in enumerate(sampled_ids)}


def _resample_predictions(sampled_ids: Sequence[str], predictions: Mapping[str, tuple[Detection, ...]]) -> dict[str, tuple[Detection, ...]]:
    return {
        f"bootstrap_{index:06d}": tuple(
            Detection(f"bootstrap_{index:06d}", detection.kind, detection.score, detection.bbox)
            for detection in predictions[source_id]
        )
        for index, source_id in enumerate(sampled_ids)
    }


def _draw_sha(sampled_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sampled_ids).encode("utf-8")).hexdigest()


def _ap_values(gt: Mapping[str, tuple[KittiObject, ...]], predictions: Mapping[str, tuple[Detection, ...]], difficulty: Difficulty, target_slice: TargetSlice | None = None) -> dict[str, float]:
    values = {
        class_name: evaluate_class(
            dict(gt), dict(predictions), class_name, difficulty,
            valid_selector=target_slice.matches if target_slice is not None else None,
        ).ap40
        for class_name in CLASSES
    }
    values["macro"] = statistics.fmean(values.values())
    return values


def _delta_values(reference: Mapping[str, float], candidate: Mapping[str, float]) -> dict[str, float]:
    return {name: float(candidate[name]) - float(reference[name]) for name in (*CLASSES, "macro")}


def _replicate(
    index: int,
    image_ids: tuple[str, ...],
    ground_truth: Mapping[str, tuple[KittiObject, ...]],
    reference: Mapping[str, tuple[Detection, ...]],
    candidate: Mapping[str, tuple[Detection, ...]],
    generator: random.Random,
) -> dict[str, object]:
    sampled_ids = generator.choices(image_ids, k=len(image_ids))
    sampled_gt = _resample_ground_truth(sampled_ids, ground_truth)
    sampled_reference = _resample_predictions(sampled_ids, reference)
    sampled_candidate = _resample_predictions(sampled_ids, candidate)
    moderate_reference = _ap_values(sampled_gt, sampled_reference, Difficulty.MODERATE)
    moderate_candidate = _ap_values(sampled_gt, sampled_candidate, Difficulty.MODERATE)
    row: dict[str, object] = {
        "replicate": index,
        "draw_sha256": _draw_sha(sampled_ids),
        "moderate": {
            "reference": moderate_reference,
            "candidate": moderate_candidate,
            "delta": _delta_values(moderate_reference, moderate_candidate),
        },
    }
    strata: dict[str, dict[str, object]] = {}
    for target_slice in BOOTSTRAP_SLICES:
        reference_values = _ap_values(sampled_gt, sampled_reference, Difficulty.HARD, target_slice)
        candidate_values = _ap_values(sampled_gt, sampled_candidate, Difficulty.HARD, target_slice)
        strata.setdefault(target_slice.axis, {})[target_slice.name] = {
            "reference": reference_values,
            "candidate": candidate_values,
            "delta": _delta_values(reference_values, candidate_values),
            "definition": {
                "axis": target_slice.axis,
                "lower": target_slice.lower,
                "upper": target_slice.upper,
                "include_lower": target_slice.include_lower,
                "include_upper": target_slice.include_upper,
            },
        }
    row["strata"] = strata
    row["moderate_macro_delta"] = moderate_candidate["macro"] - moderate_reference["macro"]
    return row


def _observed_errors(
    ground_truth: Mapping[str, tuple[KittiObject, ...]],
    predictions: Mapping[str, tuple[Detection, ...]],
    class_name: str,
) -> dict[str, object]:
    metrics = evaluate_class(dict(ground_truth), dict(predictions), class_name, Difficulty.MODERATE)
    localization_values = [1.0 - float(iou) for iou in metrics.matched_ious if float(iou) > 0.0]
    if len(localization_values) != metrics.true_positives:
        raise ValueError("evaluator TP/IoU count mismatch")
    totals = {
        "valid_gt": metrics.num_valid_gt,
        "tp": metrics.true_positives,
        "fp": metrics.false_positives,
        "fn": metrics.num_valid_gt - metrics.true_positives,
        "ignored": metrics.ignored_detections,
    }
    localization = {
        "n": len(localization_values),
        "mean": statistics.fmean(localization_values) if localization_values else None,
        "median": statistics.median(localization_values) if localization_values else None,
        "p95": _quantile(localization_values, 0.95) if localization_values else None,
    }
    return {**totals, "localization_error": localization}


def _error_delta(reference: Mapping[str, object], candidate: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {
        key: int(candidate[key]) - int(reference[key]) for key in ("valid_gt", "tp", "fp", "fn", "ignored")
    }
    reference_loc = reference["localization_error"]
    candidate_loc = candidate["localization_error"]
    if not isinstance(reference_loc, Mapping) or not isinstance(candidate_loc, Mapping):
        raise ValueError("localization error payload is malformed")
    result["localization_error"] = {
        key: (float(candidate_loc[key]) - float(reference_loc[key]) if candidate_loc[key] is not None and reference_loc[key] is not None else None)
        for key in ("mean", "median", "p95")
    }
    return result


def _observed_strata(
    ground_truth: Mapping[str, tuple[KittiObject, ...]],
    reference: Mapping[str, tuple[Detection, ...]],
    candidate: Mapping[str, tuple[Detection, ...]],
) -> dict[str, object]:
    result: dict[str, dict[str, object]] = {}
    for target_slice in KITTI_RESEARCH_SLICES:
        ref = _ap_values(ground_truth, reference, Difficulty.HARD, target_slice)
        cand = _ap_values(ground_truth, candidate, Difficulty.HARD, target_slice)
        counts: dict[str, dict[str, dict[str, int]]] = {"reference": {}, "candidate": {}}
        for side, detections in (("reference", reference), ("candidate", candidate)):
            for class_name in CLASSES:
                metrics = evaluate_class(dict(ground_truth), dict(detections), class_name, Difficulty.HARD, valid_selector=target_slice.matches)
                counts[side][class_name] = {
                    "num_valid_gt": metrics.num_valid_gt,
                    "tp": metrics.true_positives,
                    "fp": metrics.false_positives,
                    "fn": metrics.num_valid_gt - metrics.true_positives,
                    "ignored": metrics.ignored_detections,
                }
        result.setdefault(target_slice.axis, {})[target_slice.name] = {
            "definition": {
                "axis": target_slice.axis,
                "lower": target_slice.lower,
                "upper": target_slice.upper,
                "include_lower": target_slice.include_lower,
                "include_upper": target_slice.include_upper,
                "base_difficulty": Difficulty.HARD.value,
            },
            "reference": ref,
            "candidate": cand,
            "delta": _delta_values(ref, cand),
            "counts": counts,
        }
    return result


def _observed(
    ground_truth: Mapping[str, tuple[KittiObject, ...]],
    reference: Mapping[str, tuple[Detection, ...]],
    candidate: Mapping[str, tuple[Detection, ...]],
) -> dict[str, object]:
    reference_ap = _ap_values(ground_truth, reference, Difficulty.MODERATE)
    candidate_ap = _ap_values(ground_truth, candidate, Difficulty.MODERATE)
    reference_metrics = {class_name: {"ap40": reference_ap[class_name], **_observed_errors(ground_truth, reference, class_name)} for class_name in CLASSES}
    candidate_metrics = {class_name: {"ap40": candidate_ap[class_name], **_observed_errors(ground_truth, candidate, class_name)} for class_name in CLASSES}
    delta: dict[str, object] = {}
    for class_name in CLASSES:
        delta[class_name] = {
            "ap40": candidate_ap[class_name] - reference_ap[class_name],
            **_error_delta(reference_metrics[class_name], candidate_metrics[class_name]),
        }
    delta["macro"] = candidate_ap["macro"] - reference_ap["macro"]
    return {
        "moderate": {"difficulty": Difficulty.MODERATE.value, "reference": {**reference_metrics, "macro": reference_ap["macro"]}, "candidate": {**candidate_metrics, "macro": candidate_ap["macro"]}, "delta": delta},
        "strata": _observed_strata(ground_truth, reference, candidate),
    }


def _assert_registered_observed(observed: Mapping[str, object], *, strict: bool) -> None:
    if not strict:
        return
    moderate = observed["moderate"]
    if not isinstance(moderate, Mapping):
        raise ValueError("observed moderate payload is malformed")
    for side in ("reference", "candidate"):
        values = moderate[side]
        if not isinstance(values, Mapping):
            raise ValueError(f"observed {side} payload is malformed")
        for class_name in CLASSES:
            actual = float(values[class_name]["ap40"])
            expected = REGISTERED_OBSERVED[side][class_name]
            if abs(actual - expected) > 1e-9:
                raise ValueError(f"registered observed {side}/{class_name} AP mismatch: {actual} != {expected}")


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "ci_lower": _quantile(values, 0.025),
        "ci_upper": _quantile(values, 0.975),
    }


def _bootstrap_summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    def side_values(scope: str, key: str, side: str) -> list[float]:
        return [float(record[scope][side][key]) for record in records]  # type: ignore[index]

    moderate: dict[str, object] = {}
    for key in (*CLASSES, "macro"):
        moderate[key] = {
            "reference": _summary(side_values("moderate", key, "reference")),
            "candidate": _summary(side_values("moderate", key, "candidate")),
            "delta": _summary([float(record["moderate"]["delta"][key]) for record in records]),  # type: ignore[index]
        }
    strata: dict[str, dict[str, object]] = {}
    for target_slice in BOOTSTRAP_SLICES:
        target = strata.setdefault(target_slice.axis, {})
        scope = f"{target_slice.axis}/{target_slice.name}"
        del scope
        target[target_slice.name] = {}
        for key in (*CLASSES, "macro"):
            values = [record["strata"][target_slice.axis][target_slice.name] for record in records]  # type: ignore[index]
            target[target_slice.name][key] = {
                "reference": _summary([float(value["reference"][key]) for value in values]),
                "candidate": _summary([float(value["candidate"][key]) for value in values]),
                "delta": _summary([float(value["delta"][key]) for value in values]),
            }
    return {"confidence": 0.95, "paired": True, "moderate": moderate, "strata": strata}


def _journal_header(identity: Mapping[str, object], config: Mapping[str, object]) -> dict[str, object]:
    return {"schema_version": SCHEMA_VERSION, "kind": "records", "identity": identity, "config": config}


def _load_records(path: Path, identity: Mapping[str, object], config: Mapping[str, object]) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty records journal: {path}")
    header = json.loads(lines[0])
    if header != _journal_header(identity, config):
        raise ValueError("records journal identity mismatch")
    records: list[dict[str, object]] = []
    for line in lines[1:]:
        if line.strip():
            record = json.loads(line)
            if record.get("replicate") != len(records):
                raise ValueError("records journal replicate order mismatch")
            records.append(record)
    return records


def _append_record(path: Path, record: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_records_header(path: Path, identity: Mapping[str, object], config: Mapping[str, object]) -> None:
    _atomic_bytes(path, (_canonical(_journal_header(identity, config)) + "\n").encode("utf-8"))


def _records_bytes(identity: Mapping[str, object], config: Mapping[str, object], records: Sequence[Mapping[str, object]]) -> bytes:
    lines = [_canonical(_journal_header(identity, config))]
    lines.extend(_canonical(record) for record in records)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _rewrite_records(path: Path, identity: Mapping[str, object], config: Mapping[str, object], records: Sequence[Mapping[str, object]]) -> None:
    _atomic_bytes(path, _records_bytes(identity, config, records))


def _checkpoint_payload(
    *, state: str, identity: Mapping[str, object], config: Mapping[str, object], records: Sequence[Mapping[str, object]], generator: random.Random,
    elapsed_seconds: float, started_at_unix: float, owner: Mapping[str, object], output_dir: Path, mirror_dir: Path,
) -> dict[str, object]:
    remaining = int(config["replicates"]) - len(records)
    rate = len(records) / elapsed_seconds if elapsed_seconds > 0 else 0.0
    return {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "identity": identity,
        "config": config,
        "owner": dict(owner),
        "completed": len(records),
        "next_replicate": len(records),
        "total_replicates": config["replicates"],
        "remaining": remaining,
        "eta_seconds": remaining / rate if rate > 0 else None,
        "elapsed_seconds": elapsed_seconds,
        "started_at_unix": started_at_unix,
        "last_saved_at_unix": time.time(),
        "rng_state": _rng_state_json(generator),
        "records_sha256": _file_sha256(output_dir / "records.jsonl") if (output_dir / "records.jsonl").is_file() else None,
        "output_paths": {"records": "records.jsonl", "checkpoint": "checkpoint.json", "mirror_checkpoint": "checkpoint.json"},
    }


def _write_checkpoint(payload: Mapping[str, object], output_dir: Path, mirror_dir: Path) -> None:
    _atomic_json(output_dir / "checkpoint.json", payload)
    _atomic_json(mirror_dir / "checkpoint.json", payload)
    line = _canonical({
        "state": payload["state"], "completed": payload["completed"], "next_replicate": payload["next_replicate"],
        "total_replicates": payload["total_replicates"], "remaining": payload["remaining"], "elapsed_seconds": payload["elapsed_seconds"],
        "eta_seconds": payload["eta_seconds"], "rss_bytes": _rss_bytes(), "saved_at_unix": payload["last_saved_at_unix"],
    }) + "\n"
    for directory in (output_dir, mirror_dir):
        with (directory / "progress.log").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())


def _load_checkpoint(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid checkpoint: {path}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    return payload


def _csv_payload(records: Sequence[Mapping[str, object]]) -> str:
    fields = ["replicate", "draw_sha256"]
    for scope in ("moderate", "strata"):
        if scope == "moderate":
            fields.extend(f"moderate_{side}_{key}" for side in ("reference", "candidate", "delta") for key in (*CLASSES, "macro"))
        else:
            for target_slice in BOOTSTRAP_SLICES:
                fields.extend(f"{target_slice.name}_{side}_{key}" for side in ("reference", "candidate", "delta") for key in (*CLASSES, "macro"))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in records:
        row = {"replicate": record["replicate"], "draw_sha256": record["draw_sha256"]}
        for side in ("reference", "candidate", "delta"):
            for key in (*CLASSES, "macro"):
                row[f"moderate_{side}_{key}"] = record["moderate"][side][key]  # type: ignore[index]
        for target_slice in BOOTSTRAP_SLICES:
            payload = record["strata"][target_slice.axis][target_slice.name]  # type: ignore[index]
            for side in ("reference", "candidate", "delta"):
                for key in (*CLASSES, "macro"):
                    row[f"{target_slice.name}_{side}_{key}"] = payload[side][key]
        writer.writerow(row)
    return output.getvalue()


def _publish_final(payload: Mapping[str, object], records: Sequence[Mapping[str, object]], output_dir: Path, mirror_dir: Path) -> None:
    diagnostics_bytes = (_canonical(payload) + "\n").encode("utf-8")
    csv_bytes = _csv_payload(records).encode("utf-8")
    for directory in (output_dir, mirror_dir):
        _atomic_bytes(directory / "diagnostics.json", diagnostics_bytes)
        _atomic_bytes(directory / "diagnostics.csv", csv_bytes)
        _atomic_bytes(directory / "records.jsonl", (directory / "records.jsonl").read_bytes())
    primary_records_sha = _file_sha256(output_dir / "records.jsonl")
    mirror_records_sha = _file_sha256(mirror_dir / "records.jsonl")
    if primary_records_sha != mirror_records_sha:
        raise ValueError("primary and mirror records SHA mismatch before manifest publish")
    artifacts = {
        "diagnostics.json": {"sha256": hashlib.sha256(diagnostics_bytes).hexdigest(), "size": len(diagnostics_bytes)},
        "diagnostics.csv": {"sha256": hashlib.sha256(csv_bytes).hexdigest(), "size": len(csv_bytes)},
        "records.jsonl": {
            "primary_sha256": primary_records_sha,
            "mirror_sha256": mirror_records_sha,
            "size": (output_dir / "records.jsonl").stat().st_size,
        },
    }
    manifest = {"schema_version": SCHEMA_VERSION, "state": "complete", "identity": payload["identity"], "config": payload["config"], "artifacts": artifacts}
    for directory in (output_dir, mirror_dir):
        _atomic_json(directory / "manifest.json", manifest)


def run_diagnostics(
    *,
    split: Path,
    label_dir: Path,
    image_dir: Path,
    reference_dir: Path,
    candidate_dir: Path,
    output_dir: Path,
    mirror_dir: Path,
    replicates: int = FORMAL_REPLICATES,
    seed: int = DEFAULT_SEED,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    mode: str = "formal",
    resume: bool = False,
    stop_after: int | None = None,
    reference_run_identity: str = REGISTERED["reference_run_identity"],
    reference_checkpoint_sha256: str = REGISTERED["reference_checkpoint_sha256"],
    candidate_run_identity: str = REGISTERED["candidate_run_identity"],
    candidate_checkpoint_sha256: str = REGISTERED["candidate_checkpoint_sha256"],
    strict_registered_identities: bool = True,
) -> dict[str, object]:
    if mode not in {"formal", "benchmark"}:
        raise ValueError("mode must be formal or benchmark")
    replicates = resolve_replicates(benchmark=mode == "benchmark", requested=replicates)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(checkpoint_interval, bool) or not isinstance(checkpoint_interval, int) or not 0 < checkpoint_interval <= DEFAULT_CHECKPOINT_INTERVAL:
        raise ValueError(f"checkpoint_interval must be within 1..{DEFAULT_CHECKPOINT_INTERVAL}")
    if stop_after is not None and (isinstance(stop_after, bool) or not isinstance(stop_after, int) or stop_after <= 0 or stop_after > replicates):
        raise ValueError("stop_after must be within the replicate range")
    split, label_dir, image_dir = Path(split).resolve(), Path(label_dir).resolve(), Path(image_dir).resolve()
    reference_dir, candidate_dir = Path(reference_dir).resolve(), Path(candidate_dir).resolve()
    output_dir, mirror_dir = Path(output_dir).resolve(), Path(mirror_dir).resolve()
    if _paths_overlap(output_dir, mirror_dir):
        raise ValueError("output and mirror directories must be distinct and non-overlapping")
    image_ids = load_ids(split)
    if not image_ids:
        raise ValueError("development split is empty")
    image_sizes = {}
    for image_id in image_ids:
        path = image_dir / f"{image_id}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image_sizes[image_id] = image.size
    identity = _identity(split=split, label_dir=label_dir, image_dir=image_dir, image_ids=image_ids, reference_dir=reference_dir, candidate_dir=candidate_dir, reference_run_identity=reference_run_identity, reference_checkpoint_sha256=reference_checkpoint_sha256, candidate_run_identity=candidate_run_identity, candidate_checkpoint_sha256=candidate_checkpoint_sha256)
    _assert_registered_identity(identity, strict=strict_registered_identities)
    config = {"replicates": replicates, "seed": seed, "mode": mode, "checkpoint_interval": checkpoint_interval, "difficulty": Difficulty.MODERATE.value, "classes": list(CLASSES), "bootstrap_slices": [target_slice.name for target_slice in BOOTSTRAP_SLICES], "confidence": 0.95, "estimand": "candidate_plain_P2_minus_reference_P3P5_same_image_cluster"}
    ground_truth = load_kitti_ground_truth(label_dir, image_ids)
    reference = load_yolo_predictions(reference_dir, image_sizes)
    candidate = load_yolo_predictions(candidate_dir, image_sizes)
    observed = _observed(ground_truth, reference, candidate)
    _assert_registered_observed(observed, strict=strict_registered_identities)
    output_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, mirror_checkpoint_path = output_dir / "checkpoint.json", mirror_dir / "checkpoint.json"
    records_path, mirror_records_path = output_dir / "records.jsonl", mirror_dir / "records.jsonl"
    owner = {"hostname": socket.gethostname(), "pid": os.getpid()}
    generator = random.Random(seed)
    records: list[dict[str, object]] = []
    elapsed_base, started_at_unix = 0.0, time.time()
    checkpoint: dict[str, object] | None = None
    if checkpoint_path.exists() or mirror_checkpoint_path.exists() or records_path.exists() or mirror_records_path.exists():
        if not resume:
            raise ValueError("existing diagnostic checkpoint requires --resume")
        if not checkpoint_path.is_file() or not mirror_checkpoint_path.is_file() or not records_path.is_file() or not mirror_records_path.is_file():
            raise ValueError("primary and mirror checkpoint/journal must both exist")
        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint != _load_checkpoint(mirror_checkpoint_path):
            raise ValueError("primary and mirror checkpoints differ")
        if checkpoint.get("identity") != identity or checkpoint.get("config") != config:
            raise ValueError("checkpoint identity or config mismatch")
        saved_owner = checkpoint.get("owner")
        if not isinstance(saved_owner, Mapping):
            raise ValueError("checkpoint owner is invalid")
        if saved_owner.get("hostname") == owner["hostname"] and isinstance(saved_owner.get("pid"), int) and saved_owner["pid"] != owner["pid"] and _pid_alive(saved_owner["pid"]):
            raise RuntimeError("another live owner is using this diagnostic output")
        records = _load_records(records_path, identity, config)
        mirror_records = _load_records(mirror_records_path, identity, config)
        completed_value = checkpoint.get("completed")
        next_value = checkpoint.get("next_replicate")
        if isinstance(completed_value, bool) or not isinstance(completed_value, int) or completed_value < 0 or completed_value > replicates:
            raise ValueError("checkpoint completed count is invalid")
        if next_value != completed_value:
            raise ValueError("checkpoint next replicate does not equal completed count")
        completed = completed_value
        expected_records_sha = checkpoint.get("records_sha256")
        if not isinstance(expected_records_sha, str):
            raise ValueError("checkpoint records SHA is missing")
        if len(records) < completed or len(mirror_records) < completed:
            raise ValueError("records journal is shorter than checkpoint")
        prefix_records = records[:completed]
        mirror_prefix_records = mirror_records[:completed]
        if prefix_records != mirror_prefix_records:
            raise ValueError("primary and mirror checkpoint prefixes differ")
        prefix_sha = hashlib.sha256(_records_bytes(identity, config, prefix_records)).hexdigest()
        mirror_prefix_sha = hashlib.sha256(_records_bytes(identity, config, mirror_prefix_records)).hexdigest()
        if prefix_sha != expected_records_sha or mirror_prefix_sha != expected_records_sha:
            raise ValueError("checkpoint records SHA mismatch")
        # A crash can append a fully written record to only one journal after
        # the last checkpoint.  Keep the verified prefix and atomically trim
        # both sides before resuming from the checkpoint RNG state.
        records = prefix_records
        _rewrite_records(records_path, identity, config, records)
        _rewrite_records(mirror_records_path, identity, config, records)
        if _file_sha256(records_path) != expected_records_sha or _file_sha256(mirror_records_path) != expected_records_sha or records_path.read_bytes() != mirror_records_path.read_bytes():
            raise ValueError("truncated records journal SHA mismatch")
        _restore_rng_state(generator, checkpoint["rng_state"])
        elapsed_base, started_at_unix = float(checkpoint["elapsed_seconds"]), float(checkpoint["started_at_unix"])
    else:
        if any(output_dir.iterdir()) or any(mirror_dir.iterdir()):
            raise FileExistsError("output directory is not empty")
        _write_records_header(records_path, identity, config)
        _write_records_header(mirror_records_path, identity, config)
        _write_checkpoint(_checkpoint_payload(state="running", identity=identity, config=config, records=records, generator=generator, elapsed_seconds=0.0, started_at_unix=started_at_unix, owner=owner, output_dir=output_dir, mirror_dir=mirror_dir), output_dir, mirror_dir)
    run_started = time.monotonic()
    last_checkpoint = run_started

    def elapsed() -> float:
        return elapsed_base + time.monotonic() - run_started

    def save(state: str) -> None:
        _write_checkpoint(_checkpoint_payload(state=state, identity=identity, config=config, records=records, generator=generator, elapsed_seconds=elapsed(), started_at_unix=started_at_unix, owner=owner, output_dir=output_dir, mirror_dir=mirror_dir), output_dir, mirror_dir)

    while len(records) < replicates:
        record = _replicate(len(records), tuple(image_ids), ground_truth, reference, candidate, generator)
        records.append(record)
        _append_record(records_path, record)
        _append_record(mirror_records_path, record)
        if stop_after is not None and len(records) >= stop_after:
            save("running")
            raise RuntimeError("interrupted after requested replicate count")
        if len(records) % checkpoint_interval == 0 or time.monotonic() - last_checkpoint >= CHECKPOINT_WALL_SECONDS:
            save("running")
            last_checkpoint = time.monotonic()
    save("finalizing")
    payload = {"schema_version": SCHEMA_VERSION, "state": "complete", "metric": "KITTI_AP_R40_PAIRED_IMAGE_BOOTSTRAP", "identity": identity, "config": config, "observed": observed, "bootstrap": _bootstrap_summary(records), "replicates": records}
    save("complete")
    _publish_final(payload, records, output_dir, mirror_dir)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumable paired P3-P5 versus plain-P2 diagnostics")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mirror-dir", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference-run-identity", default=REGISTERED["reference_run_identity"])
    parser.add_argument("--reference-checkpoint-sha256", default=REGISTERED["reference_checkpoint_sha256"])
    parser.add_argument("--candidate-run-identity", default=REGISTERED["candidate_run_identity"])
    parser.add_argument("--candidate-checkpoint-sha256", default=REGISTERED["candidate_checkpoint_sha256"])
    parser.add_argument("--allow-unregistered-identities", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    replicates = resolve_replicates(benchmark=args.benchmark, requested=args.replicates)
    run_diagnostics(split=args.split, label_dir=args.label_dir, image_dir=args.image_dir, reference_dir=args.reference_dir, candidate_dir=args.candidate_dir, output_dir=args.output_dir, mirror_dir=args.mirror_dir, replicates=replicates, seed=args.seed, checkpoint_interval=args.checkpoint_interval, mode="benchmark" if args.benchmark else "formal", resume=args.resume, stop_after=args.stop_after, reference_run_identity=args.reference_run_identity, reference_checkpoint_sha256=args.reference_checkpoint_sha256, candidate_run_identity=args.candidate_run_identity, candidate_checkpoint_sha256=args.candidate_checkpoint_sha256, strict_registered_identities=not args.allow_unregistered_identities)
    print(f"p3p5_p2_diagnostics=complete replicates={replicates}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
