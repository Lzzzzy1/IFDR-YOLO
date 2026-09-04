"""Resumable, paired image bootstrap for the leakage-free P2 factorial screen.

The four conditions are evaluated on the same image-cluster draw for every
replicate.  AP is always computed by the repository's exact KITTI AP_R40
evaluator; this module does not replace it with object-level approximations.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import socket
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from ifdr_yolo.data.kitti_types import Detection, Difficulty
from ifdr_yolo.data.splits import load_ids, sha256_file
from ifdr_yolo.eval.kitti_ap40 import evaluate_class
from ifdr_yolo.eval.prediction_io import (
    load_kitti_ground_truth,
    load_yolo_predictions,
)


CONDITIONS = ("C", "A", "B", "AB")
CLASSES = ("Pedestrian", "Cyclist")
EFFECTS = (
    "A_minus_C",
    "B_minus_C",
    "AB_minus_C",
    "AB_minus_max_A_B",
    "interaction",
)
CHECKPOINT_SCHEMA_VERSION = 1
FORMAL_REPLICATES = 10_000
DEFAULT_SEED = 17
DEFAULT_CHECKPOINT_INTERVAL = 25
CHECKPOINT_WALL_SECONDS = 240.0


def resolve_replicates(*, benchmark: bool, requested: int | None) -> int:
    """Resolve the explicit benchmark mode without weakening formal defaults."""

    if requested is not None and (
        isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0
    ):
        raise ValueError("replicates must be a positive integer")
    if benchmark:
        if requested is None:
            raise ValueError("benchmark mode requires an explicit replicate count")
        return requested
    if requested is not None and requested != FORMAL_REPLICATES:
        raise ValueError(
            f"formal mode is fixed at {FORMAL_REPLICATES} replicates; "
            "use --benchmark for a screen benchmark"
        )
    return FORMAL_REPLICATES


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _hash_files(directory: Path, suffix: str) -> str:
    if not directory.is_dir():
        raise FileNotFoundError(f"directory does not exist: {directory}")
    digest = hashlib.sha256()
    paths = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix == suffix)
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _hash_registered_labels(label_dir: Path, image_ids: tuple[str, ...]) -> str:
    if not label_dir.is_dir():
        raise FileNotFoundError(f"directory does not exist: {label_dir}")
    digest = hashlib.sha256()
    for image_id in image_ids:
        path = label_dir / f"{image_id}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"evaluation label does not exist: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _hash_image_dimensions(image_dir: Path, image_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for image_id in image_ids:
        image_path = image_dir / f"{image_id}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"evaluation image does not exist: {image_path}")
        with Image.open(image_path) as image:
            dimensions = f"{image_id}:{image.width}x{image.height}\n"
        digest.update(dimensions.encode("utf-8"))
    return digest.hexdigest()


def _validate_prediction_ids(prediction_dir: Path, image_ids: tuple[str, ...]) -> None:
    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"prediction directory does not exist: {prediction_dir}")
    expected = set(image_ids)
    actual = {path.stem for path in prediction_dir.glob("*.txt")}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "prediction IDs do not match split: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )


def _identity(
    *,
    split: Path,
    label_dir: Path,
    image_dir: Path,
    prediction_dirs: Mapping[str, Path],
    image_ids: tuple[str, ...],
) -> dict[str, object]:
    repository_root = Path(__file__).resolve().parents[1]
    implementation_paths = {
        "scripts/summarize_p2_factorial_screen.py": Path(__file__).resolve(),
        "ifdr_yolo/eval/kitti_ap40.py": repository_root / "ifdr_yolo" / "eval" / "kitti_ap40.py",
        "ifdr_yolo/eval/prediction_io.py": repository_root / "ifdr_yolo" / "eval" / "prediction_io.py",
        "ifdr_yolo/data/kitti_types.py": repository_root / "ifdr_yolo" / "data" / "kitti_types.py",
    }
    return {
        "split_sha256": sha256_file(split),
        "split_count": len(image_ids),
        "label_manifest_sha256": _hash_registered_labels(label_dir, image_ids),
        "image_dimensions_sha256": _hash_image_dimensions(image_dir, image_ids),
        "prediction_manifest_sha256": {
            condition: _hash_files(prediction_dirs[condition], ".txt")
            for condition in CONDITIONS
        },
        "prediction_dirs": {
            condition: str(prediction_dirs[condition].resolve())
            for condition in CONDITIONS
        },
        "implementation_sha256": {
            name: sha256_file(path) for name, path in implementation_paths.items()
        },
    }


def _tupleify(value: object) -> object:
    if isinstance(value, list):
        return tuple(_tupleify(item) for item in value)
    return value


def _rng_state_json(generator: random.Random) -> list[object]:
    return json.loads(json.dumps(generator.getstate()))


def _restore_rng_state(generator: random.Random, state: object) -> None:
    restored = _tupleify(state)
    if not isinstance(restored, tuple):
        raise ValueError("checkpoint RNG state is invalid")
    generator.setstate(restored)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile from no values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.  The remote benchmark is Linux.
    return value * 1024 if os.name != "nt" else value


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _resample_ground_truth(
    *,
    sampled_ids: Sequence[str],
    ground_truth: Mapping[str, tuple[object, ...]],
) -> dict[str, tuple[object, ...]]:
    sampled_ground_truth: dict[str, tuple[object, ...]] = {}
    for index, source_id in enumerate(sampled_ids):
        sampled_id = f"bootstrap_{index:06d}"
        sampled_ground_truth[sampled_id] = ground_truth[source_id]
    return sampled_ground_truth


def _resample_predictions(
    *,
    sampled_ids: Sequence[str],
    predictions: Mapping[str, tuple[Detection, ...]],
) -> dict[str, tuple[Detection, ...]]:
    sampled_predictions: dict[str, tuple[Detection, ...]] = {}
    for index, source_id in enumerate(sampled_ids):
        sampled_id = f"bootstrap_{index:06d}"
        sampled_predictions[sampled_id] = tuple(
            Detection(
                image_id=sampled_id,
                kind=detection.kind,
                score=detection.score,
                bbox=detection.bbox,
            )
            for detection in predictions[source_id]
        )
    return sampled_predictions


def _draw_digest(sampled_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sampled_ids).encode("utf-8")).hexdigest()


def _derive_effects(ap40: Mapping[str, Mapping[str, float]]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    macro = {condition: float(ap40[condition]["macro"]) for condition in CONDITIONS}
    effects = {
        "A_minus_C": macro["A"] - macro["C"],
        "B_minus_C": macro["B"] - macro["C"],
        "AB_minus_C": macro["AB"] - macro["C"],
        "AB_minus_max_A_B": macro["AB"] - max(macro["A"], macro["B"]),
        "interaction": macro["AB"] - macro["A"] - macro["B"] + macro["C"],
    }
    class_effects = {
        class_name: {
            "A_minus_C": ap40["A"][class_name] - ap40["C"][class_name],
            "B_minus_C": ap40["B"][class_name] - ap40["C"][class_name],
            "AB_minus_C": ap40["AB"][class_name] - ap40["C"][class_name],
            "AB_minus_max_A_B": ap40["AB"][class_name]
            - max(ap40["A"][class_name], ap40["B"][class_name]),
            "interaction": ap40["AB"][class_name]
            - ap40["A"][class_name]
            - ap40["B"][class_name]
            + ap40["C"][class_name],
        }
        for class_name in CLASSES
    }
    return effects, class_effects


def _replicate(
    *,
    replicate_index: int,
    image_ids: tuple[str, ...],
    ground_truth: Mapping[str, tuple[object, ...]],
    predictions: Mapping[str, Mapping[str, tuple[Detection, ...]]],
    generator: random.Random,
) -> dict[str, object]:
    sampled_ids = generator.choices(image_ids, k=len(image_ids))
    sampled_ground_truth = _resample_ground_truth(
        sampled_ids=sampled_ids,
        ground_truth=ground_truth,
    )
    ap40: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        sampled_condition = _resample_predictions(
            sampled_ids=sampled_ids,
            predictions=predictions[condition],
        )
        class_values = {
            class_name: evaluate_class(
                sampled_ground_truth,
                sampled_condition,
                class_name,
                Difficulty.MODERATE,
            ).ap40
            for class_name in CLASSES
        }
        class_values["macro"] = statistics.fmean(class_values.values())
        ap40[condition] = class_values

    effects, class_effects = _derive_effects(ap40)
    return {
        "replicate": replicate_index,
        "draw_sha256": _draw_digest(sampled_ids),
        "ap40": ap40,
        "effects": effects,
        "class_effects": class_effects,
    }


def _observed(
    *,
    ground_truth: Mapping[str, tuple[object, ...]],
    predictions: Mapping[str, Mapping[str, tuple[Detection, ...]]],
) -> dict[str, object]:
    ap40: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        class_values = {
            class_name: evaluate_class(
                ground_truth,
                predictions[condition],
                class_name,
                Difficulty.MODERATE,
            ).ap40
            for class_name in CLASSES
        }
        class_values["macro"] = statistics.fmean(class_values.values())
        ap40[condition] = class_values
    effects, class_effects = _derive_effects(ap40)
    return {"ap40": ap40, "effects": effects, "class_effects": class_effects}


def _summary(records: Sequence[Mapping[str, object]], confidence: float = 0.95) -> dict[str, object]:
    tail = (1.0 - confidence) / 2.0

    def summarize(values: Sequence[float]) -> dict[str, float | int]:
        return {
            "n": len(values),
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "ci_lower": _quantile(values, tail),
            "ci_upper": _quantile(values, 1.0 - tail),
        }

    ap_summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for condition in CONDITIONS:
        ap_summary[condition] = {}
        for key in (*CLASSES, "macro"):
            values = [float(record["ap40"][condition][key]) for record in records]  # type: ignore[index]
            ap_summary[condition][key] = summarize(values)
    effect_summary = {
        name: summarize([float(record["effects"][name]) for record in records])  # type: ignore[index]
        for name in EFFECTS
    }
    class_effect_summary = {
        class_name: {
            name: summarize(
                [float(record["class_effects"][class_name][name]) for record in records]  # type: ignore[index]
            )
            for name in EFFECTS
        }
        for class_name in CLASSES
    }
    return {
        "confidence": confidence,
        "ap40": ap_summary,
        "effects": effect_summary,
        "class_effects": class_effect_summary,
    }


def _checkpoint_payload(
    *,
    state: str,
    identity: Mapping[str, object],
    config: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    generator: random.Random,
    elapsed_seconds: float,
    started_at_unix: float,
    output_dir: Path,
    mirror_dir: Path,
    owner: Mapping[str, object],
) -> dict[str, object]:
    total_replicates = int(config["replicates"])
    remaining = total_replicates - len(records)
    rate = len(records) / elapsed_seconds if elapsed_seconds > 0 else 0.0
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "state": state,
        "identity": identity,
        "config": config,
        "owner": dict(owner),
        "completed": len(records),
        "completed_range": [0, len(records)],
        "next_replicate": len(records),
        "total_replicates": total_replicates,
        "remaining": remaining,
        "eta_seconds": remaining / rate if rate > 0 else None,
        "elapsed_seconds": elapsed_seconds,
        "started_at_unix": started_at_unix,
        "last_saved_at_unix": time.time(),
        "rng_state": _rng_state_json(generator),
        "records": list(records),
        "output_paths": {
            "checkpoint": str((output_dir / "checkpoint.json").resolve()),
            "mirror_checkpoint": str((mirror_dir / "checkpoint.json").resolve()),
        },
    }


def _write_checkpoint(
    *,
    payload: Mapping[str, object],
    output_dir: Path,
    mirror_dir: Path,
    progress_line: str,
) -> None:
    _atomic_write_json(output_dir / "checkpoint.json", payload)
    _atomic_write_json(mirror_dir / "checkpoint.json", payload)
    for directory in (output_dir, mirror_dir):
        with (directory / "progress.log").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(progress_line)
            stream.flush()
            os.fsync(stream.fileno())


def _load_checkpoint(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid checkpoint: {path}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported or malformed checkpoint schema")
    return payload


def _assert_resume_identity(
    checkpoint: Mapping[str, object],
    *,
    identity: Mapping[str, object],
    config: Mapping[str, object],
) -> None:
    if checkpoint.get("identity") != identity:
        raise ValueError("checkpoint identity mismatch")
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, Mapping):
        raise ValueError("checkpoint config is invalid")
    for key in ("replicates", "seed", "mode"):
        if saved_config.get(key) != config.get(key):
            raise ValueError(f"checkpoint config mismatch for {key}")
    records = checkpoint.get("records")
    completed = checkpoint.get("completed")
    if not isinstance(records, list) or completed != len(records):
        raise ValueError("checkpoint records are inconsistent")
    if checkpoint.get("next_replicate") != len(records):
        raise ValueError("checkpoint next replicate is inconsistent")
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("replicate") != index:
            raise ValueError("checkpoint replicate order is invalid")


def _csv_fields() -> tuple[str, ...]:
    fields = ["replicate", "draw_sha256"]
    fields.extend(f"{condition}_{name}_ap40" for condition in CONDITIONS for name in (*CLASSES, "macro"))
    fields.extend(EFFECTS)
    fields.extend(f"{class_name}_{name}" for class_name in CLASSES for name in EFFECTS)
    return tuple(fields)


def _record_csv_row(record: Mapping[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "replicate": record["replicate"],
        "draw_sha256": record["draw_sha256"],
    }
    for condition in CONDITIONS:
        for name in (*CLASSES, "macro"):
            row[f"{condition}_{name}_ap40"] = record["ap40"][condition][name]  # type: ignore[index]
    row.update(record["effects"])  # type: ignore[arg-type]
    for class_name in CLASSES:
        for name in EFFECTS:
            row[f"{class_name}_{name}"] = record["class_effects"][class_name][name]  # type: ignore[index]
    return row


def _write_final_csv(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    fields = _csv_fields()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_record_csv_row(record) for record in records)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _verify_complete_artifacts(
    *,
    output_dir: Path,
    mirror_dir: Path,
    identity: Mapping[str, object],
) -> dict[str, object]:
    names = ("factorial_bootstrap.json", "factorial_bootstrap.csv", "manifest.json")
    paths = {
        name: (output_dir / name, mirror_dir / name)
        for name in names
    }
    if any(not path.is_file() for pair in paths.values() for path in pair):
        raise ValueError("complete output is missing primary or mirror artifact")
    primary_manifest = json.loads(paths["manifest.json"][0].read_text(encoding="utf-8"))
    mirror_manifest = json.loads(paths["manifest.json"][1].read_text(encoding="utf-8"))
    if primary_manifest != mirror_manifest:
        raise ValueError("primary and mirror manifest differ")
    if primary_manifest.get("state") != "complete" or primary_manifest.get("identity") != identity:
        raise ValueError("complete manifest identity or state is invalid")
    for name in ("factorial_bootstrap.json", "factorial_bootstrap.csv"):
        primary_path, mirror_path = paths[name]
        primary_hash = sha256_file(primary_path)
        mirror_hash = sha256_file(mirror_path)
        record = primary_manifest.get("files", {}).get(name, {})
        if (
            primary_hash != mirror_hash
            or primary_hash != record.get("sha256")
            or mirror_hash != record.get("mirror_sha256")
        ):
            raise ValueError(f"primary/mirror {name} hash mismatch")
    payload = json.loads(paths["factorial_bootstrap.json"][0].read_text(encoding="utf-8"))
    if payload.get("identity") != identity or payload.get("state") != "complete":
        raise ValueError("complete JSON identity or state is invalid")
    return payload


def _publish_final(
    *,
    payload: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    identity: Mapping[str, object],
    output_dir: Path,
    mirror_dir: Path,
) -> None:
    for directory in (output_dir, mirror_dir):
        _atomic_write_json(directory / "factorial_bootstrap.json", payload)
        _write_final_csv(directory / "factorial_bootstrap.csv", records)
    files: dict[str, dict[str, object]] = {}
    for name in ("factorial_bootstrap.json", "factorial_bootstrap.csv"):
        files[name] = {
            "sha256": sha256_file(output_dir / name),
            "mirror_sha256": sha256_file(mirror_dir / name),
            "bytes": (output_dir / name).stat().st_size,
        }
    manifest = {
        "schema_version": 1,
        "state": "complete",
        "identity": identity,
        "files": files,
        "published_at_unix": time.time(),
    }
    _atomic_write_json(output_dir / "manifest.json", manifest)
    _atomic_write_json(mirror_dir / "manifest.json", manifest)


def run_bootstrap(
    *,
    split: Path,
    label_dir: Path,
    image_dir: Path,
    prediction_dirs: Mapping[str, Path],
    output_dir: Path,
    mirror_dir: Path,
    replicates: int,
    seed: int = DEFAULT_SEED,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    mode: str = "internal",
    resume: bool = False,
    stop_after: int | None = None,
) -> dict[str, object]:
    """Run or resume the paired four-condition image bootstrap."""

    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(checkpoint_interval, bool) or checkpoint_interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    if stop_after is not None and (stop_after <= 0 or stop_after > replicates):
        raise ValueError("stop_after must be within the replicate range")
    split = split.resolve()
    label_dir = label_dir.resolve()
    image_dir = image_dir.resolve()
    output_dir = output_dir.resolve()
    mirror_dir = mirror_dir.resolve()
    if _paths_overlap(output_dir, mirror_dir):
        raise ValueError("output and mirror directories must be distinct and non-overlapping")
    prediction_dirs = {condition: Path(prediction_dirs[condition]).resolve() for condition in CONDITIONS}

    image_ids = load_ids(split)
    image_sizes: dict[str, tuple[int, int]] = {}
    for image_id in image_ids:
        image_path = image_dir / f"{image_id}.png"
        with Image.open(image_path) as image:
            image_sizes[image_id] = image.size
    for condition in CONDITIONS:
        _validate_prediction_ids(prediction_dirs[condition], image_ids)
    ground_truth = load_kitti_ground_truth(label_dir, image_ids)
    predictions = {
        condition: load_yolo_predictions(prediction_dirs[condition], image_sizes)
        for condition in CONDITIONS
    }
    identity = _identity(
        split=split,
        label_dir=label_dir,
        image_dir=image_dir,
        prediction_dirs=prediction_dirs,
        image_ids=image_ids,
    )
    config = {
        "replicates": replicates,
        "seed": seed,
        "mode": mode,
        "difficulty": Difficulty.MODERATE.value,
        "classes": list(CLASSES),
        "conditions": list(CONDITIONS),
        "confidence": 0.95,
    }
    checkpoint_path = output_dir / "checkpoint.json"
    mirror_checkpoint_path = mirror_dir / "checkpoint.json"
    checkpoint: dict[str, object] | None = None
    reuse_payload: dict[str, object] | None = None
    current_owner = {"hostname": socket.gethostname(), "pid": os.getpid()}
    owner = current_owner
    if checkpoint_path.exists() or mirror_checkpoint_path.exists():
        if not resume:
            raise ValueError("existing checkpoint requires --resume")
        if not checkpoint_path.is_file() or not mirror_checkpoint_path.is_file():
            raise ValueError("primary and mirror checkpoints must both exist")
        checkpoint = _load_checkpoint(checkpoint_path)
        mirror_checkpoint = _load_checkpoint(mirror_checkpoint_path)
        if checkpoint != mirror_checkpoint:
            raise ValueError("primary and mirror checkpoints differ")
        _assert_resume_identity(checkpoint, identity=identity, config=config)
        saved_owner = checkpoint.get("owner")
        if not isinstance(saved_owner, Mapping):
            raise ValueError("checkpoint owner is invalid")
        saved_hostname = saved_owner.get("hostname")
        saved_pid = saved_owner.get("pid")
        if (
            saved_hostname == current_owner["hostname"]
            and isinstance(saved_pid, int)
            and saved_pid != current_owner["pid"]
            and _pid_alive(saved_pid)
        ):
            raise RuntimeError("another live owner is using this bootstrap output")
        records_object = checkpoint["records"]
        assert isinstance(records_object, list)
        records: list[dict[str, object]] = [dict(record) for record in records_object]  # type: ignore[arg-type]
        generator = random.Random(seed)
        _restore_rng_state(generator, checkpoint["rng_state"])
        elapsed_base = float(checkpoint["elapsed_seconds"])
        started_at_unix = float(checkpoint["started_at_unix"])
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        if mirror_dir.exists() and any(mirror_dir.iterdir()):
            raise FileExistsError(f"mirror directory is not empty: {mirror_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        mirror_dir.mkdir(parents=True, exist_ok=True)
        records = []
        generator = random.Random(seed)
        elapsed_base = 0.0
        started_at_unix = time.time()

    if len(records) > replicates:
        raise ValueError("checkpoint has more replicates than requested")
    if len(records) == replicates and resume:
        final_path = output_dir / "factorial_bootstrap.json"
        if final_path.is_file():
            try:
                return _verify_complete_artifacts(
                    output_dir=output_dir,
                    mirror_dir=mirror_dir,
                    identity=identity,
                )
            except ValueError:
                # Finalization may have been interrupted after the checkpoint.
                # Rebuild from the validated deterministic records below.
                try:
                    candidate = json.loads(final_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    candidate = None
                if (
                    isinstance(candidate, dict)
                    and candidate.get("identity") == identity
                    and candidate.get("state") == "complete"
                    and candidate.get("replicates") == records
                ):
                    reuse_payload = candidate

    run_started = time.monotonic()
    last_checkpoint = run_started

    def elapsed() -> float:
        return elapsed_base + (time.monotonic() - run_started)

    def save(state: str) -> None:
        current_elapsed = elapsed()
        now = time.time()
        rate = len(records) / current_elapsed if current_elapsed > 0 else 0.0
        remaining = replicates - len(records)
        line = json.dumps(
            {
                "state": state,
                "completed": len(records),
                "next_replicate": len(records),
                "total_replicates": replicates,
                "remaining": remaining,
                "elapsed_seconds": current_elapsed,
                "replicates_per_second": rate,
                "eta_seconds": remaining / rate if rate > 0 else None,
                "rss_bytes": _rss_bytes(),
                "saved_at_unix": now,
            },
            sort_keys=True,
        ) + "\n"
        payload = _checkpoint_payload(
            state=state,
            identity=identity,
            config=config,
            records=records,
            generator=generator,
            elapsed_seconds=current_elapsed,
            started_at_unix=started_at_unix,
            output_dir=output_dir,
            mirror_dir=mirror_dir,
            owner=owner,
        )
        _write_checkpoint(
            payload=payload,
            output_dir=output_dir,
            mirror_dir=mirror_dir,
            progress_line=line,
        )

    if checkpoint is None or checkpoint.get("owner") != owner:
        save("running")

    while len(records) < replicates:
        index = len(records)
        records.append(
            _replicate(
                replicate_index=index,
                image_ids=image_ids,
                ground_truth=ground_truth,
                predictions=predictions,
                generator=generator,
            )
        )
        if stop_after is not None and len(records) >= stop_after:
            save("running")
            raise RuntimeError("interrupted after requested replicate count")
        if len(records) % checkpoint_interval == 0 or time.monotonic() - last_checkpoint >= CHECKPOINT_WALL_SECONDS:
            save("running")
            last_checkpoint = time.monotonic()

    final_elapsed = elapsed()
    observed = _observed(ground_truth=ground_truth, predictions=predictions)
    payload = reuse_payload or {
        "schema_version": 1,
        "state": "complete",
        "metric": "KITTI_AP_R40_PAIRED_IMAGE_BOOTSTRAP_FACTORIAL",
        "evaluator": "ifdr_yolo.eval.kitti_ap40.evaluate_class",
        "difficulty": Difficulty.MODERATE.value,
        "classes": list(CLASSES),
        "conditions": list(CONDITIONS),
        "identity": identity,
        "config": config,
        "performance": {
            "elapsed_seconds": final_elapsed,
            "replicates_per_second": replicates / final_elapsed if final_elapsed > 0 else 0.0,
            "rss_bytes": _rss_bytes(),
            "eta_seconds_for_10000": (
                10_000 * final_elapsed / replicates if replicates and final_elapsed > 0 else None
            ),
        },
        "observed": observed,
        "summary": _summary(records),
        "replicates": records,
    }
    # Final transaction: checkpoint/progress first, then JSON/CSV, then manifests last.
    save("finalizing")
    _publish_final(
        payload=payload,
        records=records,
        identity=identity,
        output_dir=output_dir,
        mirror_dir=mirror_dir,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumable paired image bootstrap for C/A/B/AB P2 screen.")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    for condition in CONDITIONS:
        parser.add_argument(f"--prediction-{condition.lower()}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mirror-dir", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    replicates = resolve_replicates(benchmark=args.benchmark, requested=args.replicates)
    prediction_dirs = {condition: getattr(args, f"prediction_{condition.lower()}") for condition in CONDITIONS}
    run_bootstrap(
        split=args.split,
        label_dir=args.label_dir,
        image_dir=args.image_dir,
        prediction_dirs=prediction_dirs,
        output_dir=args.output_dir,
        mirror_dir=args.mirror_dir,
        replicates=replicates,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        mode="benchmark" if args.benchmark else "formal",
        resume=args.resume,
    )
    print(f"factorial_bootstrap=complete replicates={replicates}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
