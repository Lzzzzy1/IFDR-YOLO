from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile

import yaml

from ifdr_yolo.data.splits import sha256_file


@dataclass(frozen=True)
class SmokeSelection:
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]


@dataclass(frozen=True)
class SmokeView:
    root: Path
    data_yaml: Path
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]


def select_smoke_ids(
    train_ids: tuple[str, ...],
    val_ids: tuple[str, ...],
    *,
    count: int = 16,
) -> SmokeSelection:
    if count <= 0:
        raise ValueError("smoke count must be positive")
    if len(train_ids) < count:
        raise ValueError(f"smoke train split requires at least {count} IDs")
    if len(val_ids) < count:
        raise ValueError(f"smoke val split requires at least {count} IDs")
    selected_train = train_ids[:count]
    selected_val = val_ids[:count]
    overlap = set(selected_train) & set(selected_val)
    if overlap:
        raise ValueError(f"smoke train/val overlap: {sorted(overlap)}")
    return SmokeSelection(
        train_ids=selected_train,
        val_ids=selected_val,
    )


def _paths_text(
    output_dir: Path,
    split: str,
    image_ids: tuple[str, ...],
) -> str:
    paths = [
        str((output_dir / "images" / split / f"{image_id}.png").resolve())
        for image_id in image_ids
    ]
    return "".join(f"{path}\n" for path in paths)


def _expected_files(
    *,
    output_dir: Path,
    generated_dir: Path,
    selection: SmokeSelection,
    train_source_sha256: str,
    val_source_sha256: str,
) -> dict[str, str]:
    output_dir = output_dir.resolve()
    generated_dir = generated_dir.resolve()
    train_list = output_dir / "train.txt"
    val_list = output_dir / "val.txt"
    data = {
        "path": str(output_dir),
        "train": str(train_list),
        "val": str(val_list),
        "names": {
            0: "Car",
            1: "Pedestrian",
            2: "Cyclist",
        },
    }
    manifest = {
        "generated_dir": str(generated_dir),
        "selection_count": len(selection.train_ids),
        "train_ids": list(selection.train_ids),
        "val_ids": list(selection.val_ids),
        "train_source_sha256": train_source_sha256,
        "val_source_sha256": val_source_sha256,
    }
    return {
        "train.txt": _paths_text(
            output_dir,
            "train",
            selection.train_ids,
        ),
        "val.txt": _paths_text(
            output_dir,
            "val",
            selection.val_ids,
        ),
        "data.yaml": yaml.safe_dump(data, sort_keys=False),
        "manifest.json": json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    }


def _copy_selected_files(
    *,
    destination_root: Path,
    generated_dir: Path,
    selection: SmokeSelection,
) -> None:
    for split, image_ids in (
        ("train", selection.train_ids),
        ("val", selection.val_ids),
    ):
        for image_id in image_ids:
            for kind, extension in (("images", ".png"), ("labels", ".txt")):
                source = generated_dir / kind / split / f"{image_id}{extension}"
                if not source.is_file():
                    raise FileNotFoundError(
                        f"smoke source file does not exist: {source}"
                    )
                destination = (
                    destination_root / kind / split / f"{image_id}{extension}"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)


def _verify_copied_files(
    *,
    output_dir: Path,
    generated_dir: Path,
    selection: SmokeSelection,
) -> None:
    for split, image_ids in (
        ("train", selection.train_ids),
        ("val", selection.val_ids),
    ):
        for image_id in image_ids:
            for kind, extension in (("images", ".png"), ("labels", ".txt")):
                source = generated_dir / kind / split / f"{image_id}{extension}"
                destination = (
                    output_dir / kind / split / f"{image_id}{extension}"
                )
                if not destination.is_file():
                    raise FileExistsError(
                        "smoke output already exists with different content: "
                        f"{destination}"
                    )
                if sha256_file(source) != sha256_file(destination):
                    raise FileExistsError(
                        "smoke output already exists with different content: "
                        f"{destination}"
                    )


def _verify_existing(
    *,
    output_dir: Path,
    generated_dir: Path,
    selection: SmokeSelection,
    expected: dict[str, str],
) -> None:
    if not output_dir.is_dir():
        raise FileExistsError(f"smoke output exists and is not a directory: {output_dir}")
    for name, content in expected.items():
        path = output_dir / name
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                f"smoke output already exists with different content: {path}"
            )
    _verify_copied_files(
        output_dir=output_dir,
        generated_dir=generated_dir,
        selection=selection,
    )


def build_smoke_view(
    *,
    output_dir: Path,
    generated_dir: Path,
    train_ids: tuple[str, ...],
    val_ids: tuple[str, ...],
    train_source_sha256: str,
    val_source_sha256: str,
    count: int = 16,
) -> SmokeView:
    selection = select_smoke_ids(train_ids, val_ids, count=count)
    output_dir = output_dir.resolve()
    expected = _expected_files(
        output_dir=output_dir,
        generated_dir=generated_dir,
        selection=selection,
        train_source_sha256=train_source_sha256,
        val_source_sha256=val_source_sha256,
    )
    if output_dir.exists():
        _verify_existing(
            output_dir=output_dir,
            generated_dir=generated_dir,
            selection=selection,
            expected=expected,
        )
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}-",
                dir=output_dir.parent,
            )
        )
        try:
            _copy_selected_files(
                destination_root=staging,
                generated_dir=generated_dir,
                selection=selection,
            )
            for name, content in expected.items():
                (staging / name).write_text(
                    content,
                    encoding="utf-8",
                    newline="\n",
                )
            staging.replace(output_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return SmokeView(
        root=output_dir,
        data_yaml=output_dir / "data.yaml",
        train_ids=selection.train_ids,
        val_ids=selection.val_ids,
    )
