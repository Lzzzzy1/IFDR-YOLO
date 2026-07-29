from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path

from ifdr_yolo.data.splits import discover_ids, sha256_file, validate_split


@dataclass(frozen=True)
class Phase1AuditSummary:
    image_count: int
    label_count: int
    yolo_row_count: int
    metadata_image_count: int
    metadata_object_count: int
    verified_source_hash_count: int


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_yolo_file(path: Path) -> int:
    row_count = 0
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"{path}:{line_number}: YOLO row must have 5 fields"
            )
        try:
            class_id = int(fields[0])
            values = tuple(float(value) for value in fields[1:])
        except ValueError as error:
            raise ValueError(
                f"{path}:{line_number}: invalid YOLO numeric field"
            ) from error
        if class_id not in (0, 1, 2):
            raise ValueError(f"{path}:{line_number}: invalid class ID {class_id}")
        if not all(isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError(
                f"{path}:{line_number}: normalized values must be finite in [0, 1]"
            )
        if values[2] <= 0.0 or values[3] <= 0.0:
            raise ValueError(
                f"{path}:{line_number}: box width and height must be positive"
            )
        row_count += 1
    return row_count


def audit_generated_dataset(
    *,
    source_image_dir: Path,
    source_label_dir: Path,
    train_ids: tuple[str, ...],
    val_ids: tuple[str, ...],
    generated_dir: Path,
    verify_all_source_hashes: bool,
) -> Phase1AuditSummary:
    available_ids = discover_ids(source_image_dir, source_label_dir)
    validate_split(train_ids, val_ids, available_ids)
    expected_ids = train_ids + val_ids

    generated_image_count = 0
    generated_label_count = 0
    yolo_row_count = 0
    for split_name, image_ids in (("train", train_ids), ("val", val_ids)):
        image_output = generated_dir / "images" / split_name
        label_output = generated_dir / "labels" / split_name
        generated_image_ids = {path.stem for path in image_output.glob("*.png")}
        generated_label_ids = {path.stem for path in label_output.glob("*.txt")}
        if generated_image_ids != set(image_ids):
            raise ValueError(f"generated {split_name} image IDs do not match split")
        if generated_label_ids != set(image_ids):
            raise ValueError(f"generated {split_name} label IDs do not match split")
        generated_image_count += len(generated_image_ids)
        generated_label_count += len(generated_label_ids)
        for image_id in image_ids:
            yolo_row_count += _validate_yolo_file(
                label_output / f"{image_id}.txt"
            )

    image_records = _load_jsonl(generated_dir / "metadata" / "images.jsonl")
    object_records = _load_jsonl(generated_dir / "metadata" / "objects.jsonl")
    record_by_id = {
        str(record["image_id"]): record for record in image_records
    }
    if set(record_by_id) != set(expected_ids):
        raise ValueError("image metadata IDs do not match split coverage")

    if verify_all_source_hashes:
        ids_to_verify = expected_ids
    else:
        sample_indexes = sorted(
            {0, len(expected_ids) // 4, len(expected_ids) // 2,
             (3 * len(expected_ids)) // 4, len(expected_ids) - 1}
        )
        ids_to_verify = tuple(expected_ids[index] for index in sample_indexes)

    for image_id in ids_to_verify:
        record = record_by_id[image_id]
        image_hash = sha256_file(source_image_dir / f"{image_id}.png")
        label_hash = sha256_file(source_label_dir / f"{image_id}.txt")
        if image_hash != record["source_sha256"]:
            raise ValueError(f"source image hash mismatch: {image_id}")
        if label_hash != record["source_label_sha256"]:
            raise ValueError(f"source label hash mismatch: {image_id}")

    manifest = json.loads(
        (generated_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest["image_count"] != len(expected_ids):
        raise ValueError("manifest image_count does not match split")
    if manifest["train_count"] != len(train_ids):
        raise ValueError("manifest train_count does not match split")
    if manifest["val_count"] != len(val_ids):
        raise ValueError("manifest val_count does not match split")
    if manifest["invalid_box_count"] != 0:
        raise ValueError("manifest reports invalid boxes")

    return Phase1AuditSummary(
        image_count=generated_image_count,
        label_count=generated_label_count,
        yolo_row_count=yolo_row_count,
        metadata_image_count=len(image_records),
        metadata_object_count=len(object_records),
        verified_source_hash_count=len(ids_to_verify),
    )
