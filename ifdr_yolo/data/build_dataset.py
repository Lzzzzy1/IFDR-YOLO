from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile

from PIL import Image

from ifdr_yolo.data.kitti_parser import parse_kitti_file
from ifdr_yolo.data.splits import discover_ids, sha256_file, validate_split
from ifdr_yolo.data.yolo_export import object_to_yolo


@dataclass(frozen=True)
class BuildSummary:
    image_count: int
    train_count: int
    val_count: int
    invalid_box_count: int
    clipped_box_count: int
    class_counts: dict[str, int]
    image_size_counts: dict[str, int]


def _ids_digest(image_ids: tuple[str, ...]) -> str:
    content = "".join(f"{image_id}\n" for image_id in image_ids).encode("utf-8")
    return sha256(content).hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def build_dataset(
    *,
    image_dir: Path,
    label_dir: Path,
    train_ids: tuple[str, ...],
    val_ids: tuple[str, ...],
    output_dir: Path,
    overwrite_generated: bool = False,
    git_commit: str | None = None,
) -> BuildSummary:
    image_dir = image_dir.resolve()
    label_dir = label_dir.resolve()
    output_dir = output_dir.resolve()
    if (
        output_dir in (image_dir, label_dir)
        or _is_relative_to(image_dir, output_dir)
        or _is_relative_to(label_dir, output_dir)
        or _is_relative_to(output_dir, image_dir)
        or _is_relative_to(output_dir, label_dir)
    ):
        raise ValueError("generated output must be separate from source data")

    available_ids = discover_ids(image_dir, label_dir)
    validate_split(train_ids, val_ids, available_ids)

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite_generated:
            raise FileExistsError(f"generated output is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    elif output_dir.exists():
        output_dir.rmdir()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}-",
            dir=output_dir.parent,
        )
    )

    invalid_box_count = 0
    clipped_box_count = 0
    class_counts: Counter[str] = Counter()
    image_size_counts: Counter[str] = Counter()
    object_records: list[dict[str, object]] = []
    image_records: list[dict[str, object]] = []

    try:
        for split_name, image_ids in (("train", train_ids), ("val", val_ids)):
            split_image_dir = staging / "images" / split_name
            split_label_dir = staging / "labels" / split_name
            split_image_dir.mkdir(parents=True)
            split_label_dir.mkdir(parents=True)

            for image_id in image_ids:
                source_image = image_dir / f"{image_id}.png"
                source_label = label_dir / f"{image_id}.txt"
                with Image.open(source_image) as image:
                    image_width, image_height = image.size
                    image.verify()

                image_size_counts[f"{image_width}x{image_height}"] += 1
                _link_or_copy(
                    source_image,
                    split_image_dir / source_image.name,
                )

                label_rows: list[str] = []
                objects = parse_kitti_file(source_label)
                for obj in objects:
                    class_counts[obj.kind] += 1
                    object_records.append(
                        {
                            "image_id": image_id,
                            "split": split_name,
                            **asdict(obj),
                        }
                    )
                    row = object_to_yolo(obj, image_width, image_height)
                    if row is None:
                        if obj.kind in {"Car", "Pedestrian", "Cyclist"}:
                            invalid_box_count += 1
                        continue
                    if (
                        obj.bbox.x1 < 0
                        or obj.bbox.y1 < 0
                        or obj.bbox.x2 > image_width
                        or obj.bbox.y2 > image_height
                    ):
                        clipped_box_count += 1
                    label_rows.append(row.serialize())

                (split_label_dir / f"{image_id}.txt").write_text(
                    "".join(f"{row}\n" for row in label_rows),
                    encoding="utf-8",
                    newline="\n",
                )
                image_records.append(
                    {
                        "image_id": image_id,
                        "split": split_name,
                        "width": image_width,
                        "height": image_height,
                        "source_sha256": sha256_file(source_image),
                    }
                )

        metadata_dir = staging / "metadata"
        metadata_dir.mkdir()
        _write_jsonl(metadata_dir / "objects.jsonl", object_records)
        _write_jsonl(metadata_dir / "images.jsonl", image_records)

        summary = BuildSummary(
            image_count=len(train_ids) + len(val_ids),
            train_count=len(train_ids),
            val_count=len(val_ids),
            invalid_box_count=invalid_box_count,
            clipped_box_count=clipped_box_count,
            class_counts=dict(sorted(class_counts.items())),
            image_size_counts=dict(sorted(image_size_counts.items())),
        )
        manifest = {
            **asdict(summary),
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "train_split_sha256": _ids_digest(train_ids),
            "val_split_sha256": _ids_digest(val_ids),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        staging.replace(output_dir)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
