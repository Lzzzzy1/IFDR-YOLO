from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

from PIL import Image

from ifdr_yolo.data.bdd100k import (
    BDD100K_IGNORED_CATEGORIES,
    parse_bdd100k_frame,
)
from ifdr_yolo.data.splits import sha256_file


@dataclass(frozen=True)
class BDD100KBuildSummary:
    split: str
    image_count: int
    object_count: int
    clipped_box_count: int
    class_counts: dict[str, int]
    ignored_category_counts: dict[str, int]
    weather_counts: dict[str, int]
    scene_counts: dict[str, int]
    timeofday_counts: dict[str, int]


@dataclass(frozen=True)
class BDD100KDatasetSummary:
    train: BDD100KBuildSummary
    val: BDD100KBuildSummary


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _load_frames(path: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid BDD100K annotation JSON: {path}") from error
    if not isinstance(payload, list) or not all(
        isinstance(frame, dict) for frame in payload
    ):
        raise ValueError("BDD100K annotations must be a list of frame mappings")
    return payload


def _frame_name(frame: dict[str, object]) -> str:
    name = frame.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("BDD100K frame name must be a non-empty string")
    path = Path(name)
    if path.name != name or name in {".", ".."}:
        raise ValueError("BDD100K frame name must be a plain filename")
    return name


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def _build_split_into(
    *,
    annotations_path: Path,
    image_dir: Path,
    staging: Path,
    split: str,
) -> BDD100KBuildSummary:
    frames = _load_frames(annotations_path)
    names = [_frame_name(frame) for frame in frames]
    duplicates = sorted(
        name for name, count in Counter(names).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate BDD100K frame name: {duplicates[0]}")

    split_image_dir = staging / "images" / split
    split_label_dir = staging / "labels" / split
    metadata_dir = staging / "metadata"
    split_image_dir.mkdir(parents=True, exist_ok=True)
    split_label_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    class_counts: Counter[str] = Counter()
    ignored_counts: Counter[str] = Counter()
    weather_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    timeofday_counts: Counter[str] = Counter()
    clipped_box_count = 0
    object_records: list[dict[str, object]] = []
    image_records: list[dict[str, object]] = []
    annotation_hash = sha256_file(annotations_path)

    for frame, name in zip(frames, names, strict=True):
        source_image = image_dir / name
        if not source_image.is_file():
            raise FileNotFoundError(f"missing BDD100K image: {source_image}")
        with Image.open(source_image) as image:
            image_width, image_height = image.size
            image.verify()

        parsed = parse_bdd100k_frame(
            frame,
            image_width=image_width,
            image_height=image_height,
        )
        attributes = frame["attributes"]
        assert isinstance(attributes, dict)
        weather_counts[parsed.weather] += 1
        scene_counts[parsed.scene] += 1
        timeofday_counts[parsed.timeofday] += 1
        labels = frame["labels"]
        assert isinstance(labels, list)
        for label in labels:
            if isinstance(label, dict):
                category = label.get("category")
                if category in BDD100K_IGNORED_CATEGORIES:
                    ignored_counts[str(category)] += 1

        _link_or_copy(source_image, split_image_dir / name)
        label_rows: list[str] = []
        for obj in parsed.objects:
            class_counts[obj.category] += 1
            if (
                obj.xyxy[0] <= 0.0
                or obj.xyxy[1] <= 0.0
                or obj.xyxy[2] >= image_width
                or obj.xyxy[3] >= image_height
            ):
                clipped_box_count += 1
            label_rows.append(obj.yolo_row.serialize())
            object_records.append(
                {
                    "image_name": name,
                    "split": split,
                    "category": obj.category,
                    "class_id": obj.class_id,
                    "xyxy": list(obj.xyxy),
                    "occluded": obj.occluded,
                    "truncated": obj.truncated,
                    "size_bin": obj.size_bin,
                    "yolo": list(obj.yolo_row.as_tuple()),
                }
            )

        label_path = split_label_dir / f"{Path(name).stem}.txt"
        label_path.write_text(
            "".join(f"{row}\n" for row in label_rows),
            encoding="utf-8",
            newline="\n",
        )
        image_records.append(
            {
                "image_name": name,
                "split": split,
                "width": image_width,
                "height": image_height,
                "weather": parsed.weather,
                "scene": parsed.scene,
                "timeofday": parsed.timeofday,
                "object_count": len(parsed.objects),
                "source_sha256": sha256_file(source_image),
                "annotations_sha256": annotation_hash,
            }
        )

    _write_jsonl(metadata_dir / f"objects_{split}.jsonl", object_records)
    _write_jsonl(metadata_dir / f"images_{split}.jsonl", image_records)
    return BDD100KBuildSummary(
        split=split,
        image_count=len(frames),
        object_count=len(object_records),
        clipped_box_count=clipped_box_count,
        class_counts=dict(sorted(class_counts.items())),
        ignored_category_counts=dict(sorted(ignored_counts.items())),
        weather_counts=dict(sorted(weather_counts.items())),
        scene_counts=dict(sorted(scene_counts.items())),
        timeofday_counts=dict(sorted(timeofday_counts.items())),
    )


def build_bdd100k_split(
    *,
    annotations_path: Path,
    image_dir: Path,
    output_dir: Path,
    split: str,
    overwrite_generated: bool = False,
    git_commit: str | None = None,
) -> BDD100KBuildSummary:
    """Convert one official BDD100K JSON split to an auditable YOLO tree."""
    if split not in {"train", "val"}:
        raise ValueError("BDD100K split must be train or val")
    annotations_path = annotations_path.resolve()
    image_dir = image_dir.resolve()
    output_dir = output_dir.resolve()
    if not annotations_path.is_file():
        raise FileNotFoundError(f"missing BDD100K annotations: {annotations_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"missing BDD100K image directory: {image_dir}")
    if (
        output_dir in (image_dir, annotations_path.parent)
        or _is_relative_to(image_dir, output_dir)
        or _is_relative_to(output_dir, image_dir)
        or _is_relative_to(annotations_path, output_dir)
    ):
        raise ValueError("generated BDD100K output must be separate from source data")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite_generated:
            raise FileExistsError(f"generated output is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        summary = _build_split_into(
            annotations_path=annotations_path,
            image_dir=image_dir,
            staging=staging,
            split=split,
        )
        manifest = {
            **asdict(summary),
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "annotations_sha256": sha256_file(annotations_path),
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


def build_bdd100k_dataset(
    *,
    train_annotations_path: Path,
    val_annotations_path: Path,
    image_dir: Path,
    output_dir: Path,
    overwrite_generated: bool = False,
    git_commit: str | None = None,
) -> BDD100KDatasetSummary:
    """Build train/val together and reject cross-split image leakage."""
    train_annotations_path = train_annotations_path.resolve()
    val_annotations_path = val_annotations_path.resolve()
    image_dir = image_dir.resolve()
    output_dir = output_dir.resolve()
    for annotations_path in (train_annotations_path, val_annotations_path):
        if not annotations_path.is_file():
            raise FileNotFoundError(f"missing BDD100K annotations: {annotations_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"missing BDD100K image directory: {image_dir}")
    if (
        output_dir == image_dir
        or _is_relative_to(image_dir, output_dir)
        or _is_relative_to(output_dir, image_dir)
        or _is_relative_to(train_annotations_path, output_dir)
        or _is_relative_to(val_annotations_path, output_dir)
    ):
        raise ValueError("generated BDD100K output must be separate from source data")

    train_names = {_frame_name(frame) for frame in _load_frames(train_annotations_path)}
    val_names = {_frame_name(frame) for frame in _load_frames(val_annotations_path)}
    overlap = sorted(train_names & val_names)
    if overlap:
        raise ValueError(f"BDD100K train/val overlap: {overlap[0]}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite_generated:
            raise FileExistsError(f"generated output is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        train_summary = _build_split_into(
            annotations_path=train_annotations_path,
            image_dir=image_dir,
            staging=staging,
            split="train",
        )
        val_summary = _build_split_into(
            annotations_path=val_annotations_path,
            image_dir=image_dir,
            staging=staging,
            split="val",
        )
        (staging / "dataset.yaml").write_text(
            "path: .\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: car\n"
            "  1: pedestrian\n"
            "  2: rider\n",
            encoding="utf-8",
            newline="\n",
        )
        manifest = {
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "class_names": ["car", "pedestrian", "rider"],
            "annotations_sha256": {
                "train": sha256_file(train_annotations_path),
                "val": sha256_file(val_annotations_path),
            },
            "splits": {
                "train": asdict(train_summary),
                "val": asdict(val_summary),
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        staging.replace(output_dir)
        return BDD100KDatasetSummary(train=train_summary, val=val_summary)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
