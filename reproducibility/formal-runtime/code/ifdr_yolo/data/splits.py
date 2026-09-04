from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def _validate_id(image_id: str) -> None:
    if len(image_id) != 6 or not image_id.isdigit():
        raise ValueError(f"invalid KITTI image ID: {image_id!r}")


def load_ids(path: Path) -> tuple[str, ...]:
    image_ids = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    for image_id in image_ids:
        _validate_id(image_id)
    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"duplicate image ID in split file: {path}")
    return image_ids


def discover_ids(image_dir: Path, label_dir: Path) -> set[str]:
    image_ids = {path.stem for path in image_dir.glob("*.png")}
    label_ids = {path.stem for path in label_dir.glob("*.txt")}
    if image_ids != label_ids:
        missing_images = sorted(label_ids - image_ids)
        missing_labels = sorted(image_ids - label_ids)
        raise ValueError(
            "image/label ID mismatch: "
            f"missing_images={missing_images[:5]}, "
            f"missing_labels={missing_labels[:5]}"
        )
    for image_id in image_ids:
        _validate_id(image_id)
    return image_ids


def validate_split(
    train_ids: tuple[str, ...],
    val_ids: tuple[str, ...],
    available_ids: set[str],
) -> None:
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise ValueError(f"train/val overlap: {sorted(overlap)[:5]}")
    covered_ids = set(train_ids) | set(val_ids)
    if covered_ids != available_ids:
        missing = sorted(available_ids - covered_ids)
        unknown = sorted(covered_ids - available_ids)
        raise ValueError(
            "split coverage mismatch: "
            f"missing={missing[:5]}, unknown={unknown[:5]}"
        )


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
