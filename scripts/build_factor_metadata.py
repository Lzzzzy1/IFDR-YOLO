"""Build the immutable factor metadata development split and bundle."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.data.development_split import (
    REGISTERED_FRACTION,
    REGISTERED_SEED,
    build_development_split,
    write_split_outputs,
)
from ifdr_yolo.data.metadata_index import (
    KittiLabelCandidate,
    KittiMetadataObject,
    build_metadata_index,
)
from ifdr_yolo.data.kitti_parser import parse_kitti_file
from ifdr_yolo.data.kitti_types import TRAIN_CLASS_TO_ID
from ifdr_yolo.data.natural_degradation import load_natural_degradation_records


_FULL_ARTIFACT_NAMES = (
    "fit_ids.txt",
    "development_ids.txt",
    "metadata_index.json",
    "manifest.json",
)
_DEFAULT_OUTPUT_ROOT = "runs/factor-repair"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _load_jsonl(path: Path, *, name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on {name} line {line_number}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"{name} line {line_number} must be an object")
            rows.append(row)
    return rows


def _load_rows(path: Path) -> list[dict[str, object]]:
    """Compatibility loader for the Task1 split-only mode."""
    return _load_jsonl(path, name="input-jsonl")


def _require_regular_file(path: Path, *, name: str, nonempty: bool = False) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be an existing regular file: {path}")
    if nonempty and path.stat().st_size <= 0:
        raise ValueError(f"{name} must be non-empty: {path}")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_image_id(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} image_id must be exact non-empty text")
    if any(character.isspace() for character in value):
        raise ValueError(f"{name} image_id must not contain whitespace")
    return value


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character lowercase SHA256")
    return value


def _train_image_ids(
    path: Path,
) -> tuple[tuple[str, ...], set[str], dict[str, str]]:
    rows = _load_jsonl(path, name="images-jsonl")
    seen: set[str] = set()
    train_ids: list[str] = []
    train_label_hashes: dict[str, str] = {}
    for line_number, row in enumerate(rows, start=1):
        image_id = _validate_image_id(row.get("image_id"), name="images-jsonl")
        if image_id in seen:
            raise ValueError(f"duplicate image_id in images-jsonl: {image_id}")
        seen.add(image_id)
        split = row.get("split")
        if not isinstance(split, str) or not split or split.strip() != split:
            raise ValueError(f"images-jsonl line {line_number} split must be exact text")
        if split not in {"train", "val"}:
            raise ValueError(
                f"images-jsonl line {line_number} split must be train or val"
            )
        if split == "train":
            train_ids.append(image_id)
            if "source_label_sha256" not in row:
                raise ValueError(
                    f"images-jsonl line {line_number} train row requires "
                    "source_label_sha256"
                )
            train_label_hashes[image_id] = _validate_sha256(
                row["source_label_sha256"],
                name=f"images-jsonl line {line_number} source_label_sha256",
            )
    if not train_ids:
        raise ValueError("images-jsonl has no train images")
    return tuple(sorted(train_ids)), seen, train_label_hashes


def _raw_label_candidates(
    raw_label_dir: Path,
    *,
    train_ids: Sequence[str],
    allowed_image_ids: set[str],
    expected_hashes: Mapping[str, str],
) -> tuple[dict[str, list[KittiLabelCandidate]], str]:
    if raw_label_dir.is_symlink() or not raw_label_dir.is_dir():
        raise ValueError(
            f"raw-label-dir must be an existing regular directory: {raw_label_dir}"
        )

    labels: dict[str, list[KittiLabelCandidate]] = {}
    aggregate_entries: list[dict[str, str]] = []
    for image_id in sorted(train_ids):
        label_path = raw_label_dir / f"{image_id}.txt"
        _require_regular_file(label_path, name=f"raw label for {image_id}")
        label_hash = _sha256_file(label_path)
        expected_hash = expected_hashes.get(image_id)
        if expected_hash != label_hash:
            raise ValueError(
                f"raw label hash mismatch for {image_id}: "
                "source_label_sha256 does not match raw label"
            )
        aggregate_entries.append({"image_id": image_id, "sha256": label_hash})
        try:
            parsed_objects = parse_kitti_file(label_path)
        except (OSError, ValueError) as error:
            raise ValueError(f"unable to parse raw label {label_path}: {error}") from error
        candidates: list[KittiLabelCandidate] = []
        for object_index, parsed in enumerate(parsed_objects):
            class_id = TRAIN_CLASS_TO_ID.get(parsed.kind)
            if class_id is None:
                continue
            candidates.append(
                KittiLabelCandidate(
                    image_id=image_id,
                    object_index=object_index,
                    class_id=class_id,
                    class_name=parsed.kind,
                    bbox_xyxy=parsed.bbox.as_xyxy(),
                )
            )
        labels[image_id] = candidates

    for entry in raw_label_dir.iterdir():
        if entry.is_symlink():
            if entry.suffix.lower() == ".txt":
                raise ValueError(f"raw label must not be a symlink: {entry.name}")
            continue
        if entry.suffix.lower() != ".txt":
            continue
        if not entry.is_file():
            raise ValueError(f"raw label is not a regular file: {entry.name}")
        if entry.stem not in allowed_image_ids:
            raise ValueError(f"raw label has no images-jsonl row: {entry.name}")

    aggregate_payload = json.dumps(
        aggregate_entries,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return labels, hashlib.sha256(aggregate_payload).hexdigest()


def _metadata_records(
    path: Path,
    *,
    all_image_ids: set[str],
    train_ids: set[str],
) -> tuple[Any, ...]:
    raw_rows = _load_jsonl(path, name="metadata-jsonl")
    for row in raw_rows:
        image_id = _validate_image_id(row.get("image_id"), name="metadata-jsonl")
        if image_id not in all_image_ids:
            raise ValueError(f"metadata image_id is not present in images-jsonl: {image_id}")

    loaded = load_natural_degradation_records(path)
    return tuple(record for record in loaded.records if record.image_id in train_ids)


def _ids_bytes(ids: Sequence[str]) -> bytes:
    text = "\n".join(ids)
    return (text + "\n" if text else "").encode("utf-8")


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _full_bundle_bytes(
    *,
    metadata_jsonl: Path,
    images_jsonl: Path,
    raw_label_dir: Path,
    initialization_checkpoint: Path,
    output_root: str,
) -> dict[str, bytes]:
    _require_regular_file(metadata_jsonl, name="metadata-jsonl")
    _require_regular_file(images_jsonl, name="images-jsonl")
    _require_regular_file(
        initialization_checkpoint,
        name="initialization-checkpoint",
        nonempty=True,
    )

    train_id_tuple, all_image_ids, train_label_hashes = _train_image_ids(images_jsonl)
    train_ids = set(train_id_tuple)
    labels, raw_labels_sha256 = _raw_label_candidates(
        raw_label_dir,
        train_ids=train_id_tuple,
        allowed_image_ids=all_image_ids,
        expected_hashes=train_label_hashes,
    )
    records = _metadata_records(
        metadata_jsonl,
        all_image_ids=all_image_ids,
        train_ids=train_ids,
    )
    records_by_image: dict[str, list[Any]] = {image_id: [] for image_id in train_id_tuple}
    for record in records:
        records_by_image.setdefault(record.image_id, []).append(record)

    split_rows: list[dict[str, object]] = []
    for image_id in train_id_tuple:
        image_records = records_by_image[image_id]
        cyclist_records = [
            record for record in image_records if record.class_name == "Cyclist"
        ]
        joint = max(
            (
                1.0
                - (1.0 - record.sampling_score)
                * (1.0 - record.visibility_score)
                for record in cyclist_records
            ),
            default=0.0,
        )
        split_rows.append(
            {
                "image_id": image_id,
                "cyclist": bool(cyclist_records),
                "cyclist_joint": joint,
            }
        )
    split = build_development_split(
        split_rows,
        seed=REGISTERED_SEED,
        fraction=REGISTERED_FRACTION,
    )

    source_objects: list[KittiMetadataObject] = []
    for record in records:
        source = KittiMetadataObject(
            image_id=record.image_id,
            object_index=record.object_id,
            class_id=record.class_id,
            class_name=record.class_name,
            bbox_xyxy=record.bbox_xyxy,
            depth_m=record.depth_m,
            occlusion=record.occlusion_level,
            truncation=record.truncation,
        )
        source_objects.append(source)

    source_sha256 = _sha256_file(metadata_jsonl)
    images_sha256 = _sha256_file(images_jsonl)
    index = build_metadata_index(
        source_objects,
        labels=labels,
        split_sha256=split.sha256,
        source_sha256=source_sha256,
        label_source_sha256=raw_labels_sha256,
    )
    metadata_index_bytes = index.to_json_bytes()
    metadata_index_file_sha256 = hashlib.sha256(metadata_index_bytes).hexdigest()
    checkpoint_sha256 = _sha256_file(initialization_checkpoint)
    fit_bytes = _ids_bytes(split.fit_ids)
    development_bytes = _ids_bytes(split.development_ids)
    manifest = {
        "schema_version": 1,
        "source_metadata_sha256": source_sha256,
        "images_metadata_sha256": images_sha256,
        "raw_labels_sha256": raw_labels_sha256,
        "split_sha256": split.sha256,
        "metadata_index_sha256": index.sha256,
        "metadata_index_file_sha256": metadata_index_file_sha256,
        "initialization_checkpoint_sha256": checkpoint_sha256,
        "fit_ids_sha256": hashlib.sha256(fit_bytes).hexdigest(),
        "development_ids_sha256": hashlib.sha256(development_bytes).hexdigest(),
        "fit_count": len(split.fit_ids),
        "development_count": len(split.development_ids),
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    config = {
        "schema_version": 1,
        "identity": {
            "source_metadata_sha256": source_sha256,
            "images_metadata_sha256": images_sha256,
            "raw_labels_sha256": raw_labels_sha256,
            "split_sha256": split.sha256,
            "metadata_sha256": index.sha256,
            "initialization_checkpoint_sha256": checkpoint_sha256,
            "fit_ids_sha256": hashlib.sha256(fit_bytes).hexdigest(),
            "development_ids_sha256": hashlib.sha256(development_bytes).hexdigest(),
        },
        "development": {
            "seed": REGISTERED_SEED,
            "fraction": REGISTERED_FRACTION,
        },
        "conditions": {
            **{
                name: {"track": "metadata", "epochs": 60}
                for name in ("M1", "M2", "M3")
            },
            **{
                name: {"track": "factor", "epochs": 30}
                for name in ("F0", "F1", "F2", "F3")
            },
        },
        "task_adaptation_epochs": 60,
        "max_selected_factor_repairs": 1,
        "early_stopping": False,
        "training": {"imgsz": 640},
        "factor_loss": {
            "natural_gain": 1.0,
            "specificity_gain": 0.5,
            "specificity_margin": 0.05,
            "factor_weights": [1.0, 1.0],
        },
        "model": {
            "nodes": [11, 14, 17, 20, 23, 26],
            "primary_nodes": [17, 20, 23, 26],
        },
        "paths": {
            "metadata_jsonl": metadata_jsonl.resolve().as_posix(),
            "images_jsonl": images_jsonl.resolve().as_posix(),
            "raw_label_dir": raw_label_dir.resolve().as_posix(),
            "initialization_checkpoint": initialization_checkpoint.resolve().as_posix(),
            "output_root": output_root,
        },
        "schedule": {
            "replay": {
                "eta_peak": 0.30,
                "ramp_epochs": 5,
                "focus_end_epoch": 40,
                "recovery_start_epoch": 41,
                "total_epochs": 60,
                "priority_clip_quantile": 0.95,
                "eligible_floor": 0.05,
                "replacement": True,
                "draws_per_epoch": "fit_count",
            },
            "factor_calibration": {
                "epochs": 30,
                "views_per_sample": 3,
                "fusion_schedule": 0.0,
                "dcli_schedule": 0.0,
            },
            "task_adaptation": {"epochs": 60},
        },
        "checkpoint_policy": {
            "primary": "last.pt",
            "diagnostic": "best.pt",
            "early_stopping": False,
        },
        "metadata_replay": {
            "M1": "original",
            "M2": "cyclist_uniform",
            "M3": "joint_score",
        },
        "factor_gate": {
            "seed17_min_positive_primary_directions": 3,
            "formal_min_positive_seed_node_directions": 10,
            "formal_total_seed_node_directions": 12,
            "minimum_severity_ordering": 0.8,
            "diagnostic_reverse_abs_rho": 0.1,
            "selection_tie_tolerance": 1e-12,
            "require_paired_delta_ci_lower_positive": True,
            "require_zero_malformed": True,
        },
    }
    config_bytes = yaml.safe_dump(
        config,
        sort_keys=False,
        allow_unicode=True,
    ).encode("utf-8")
    return {
        "fit_ids.txt": fit_bytes,
        "development_ids.txt": development_bytes,
        "metadata_index.json": metadata_index_bytes,
        "manifest.json": manifest_bytes,
        "__config__": config_bytes,
    }


def _validate_existing_path(path: Path, expected: bytes, *, label: str) -> bool:
    if path.is_symlink():
        raise ValueError(f"existing {label} is a symlink: {path}")
    if not path.exists():
        return False
    if not path.is_file():
        raise ValueError(f"existing {label} is not a file: {path}")
    if path.read_bytes() != expected:
        raise ValueError(f"existing {label} is not identical: {path}")
    return True


def _validate_config_parent(path: Path) -> None:
    parent = path.parent
    while True:
        if parent.is_symlink():
            raise ValueError(f"config parent is a symlink: {parent}")
        if parent.exists() and not parent.is_dir():
            raise ValueError(f"config parent is not a directory: {parent}")
        if parent == parent.parent:
            break
        parent = parent.parent


def _canonical_path(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False)))


def _validate_output_path_distinctness(
    output_dir: Path,
    config_output: Path,
) -> None:
    config_key = _canonical_path(config_output)
    occupied = {
        _canonical_path(output_dir): "output directory",
        **{
            _canonical_path(output_dir / name): f"artifact {name}"
            for name in _FULL_ARTIFACT_NAMES
        },
    }
    collision = occupied.get(config_key)
    if collision is not None:
        raise ValueError(
            f"config-output path collision with {collision}; paths must be distinct: "
            f"{config_output}"
        )


def _write_full_bundle(
    *,
    output_dir: Path,
    config_output: Path,
    bundle: Mapping[str, bytes],
) -> None:
    _validate_output_path_distinctness(output_dir, config_output)
    if output_dir.is_symlink():
        raise ValueError(f"output directory is a symlink: {output_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output directory is not a directory: {output_dir}")
    _validate_config_parent(config_output)

    expected_existing = {
        name: _validate_existing_path(
            output_dir / name,
            bundle[name],
            label=name,
        )
        for name in _FULL_ARTIFACT_NAMES
    }
    config_existing = _validate_existing_path(
        config_output,
        bundle["__config__"],
        label="config-output",
    )
    if expected_existing["manifest.json"] and (
        not all(expected_existing.values()) or not config_existing
    ):
        raise ValueError("completed bundle is incomplete/corrupt")
    if all(expected_existing.values()) and config_existing:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".factor-metadata-", dir=str(output_dir.parent))
    )
    config_temp: Path | None = None
    try:
        for name in _FULL_ARTIFACT_NAMES:
            (staging_dir / name).write_bytes(bundle[name])
        for name in _FULL_ARTIFACT_NAMES[:-1]:
            if not expected_existing[name]:
                os.replace(staging_dir / name, output_dir / name)

        if not config_existing:
            config_output.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{config_output.name}.",
                dir=str(config_output.parent),
            )
            os.close(descriptor)
            config_temp = Path(temporary_name)
            config_temp.write_bytes(bundle["__config__"])
            os.replace(config_temp, config_output)
            config_temp = None

        # The manifest is the completion marker and is replaced last.
        if not expected_existing["manifest.json"]:
            os.replace(staging_dir / "manifest.json", output_dir / "manifest.json")
    finally:
        if config_temp is not None:
            config_temp.unlink(missing_ok=True)
        shutil.rmtree(staging_dir, ignore_errors=True)


def _build_full_mode(args: argparse.Namespace) -> int:
    output_root = (
        _DEFAULT_OUTPUT_ROOT
        if args.output_root is None
        else args.output_root.as_posix()
    )
    bundle = _full_bundle_bytes(
        metadata_jsonl=args.metadata_jsonl,
        images_jsonl=args.images_jsonl,
        raw_label_dir=args.raw_label_dir,
        initialization_checkpoint=args.initialization_checkpoint,
        output_root=output_root,
    )
    _write_full_bundle(
        output_dir=args.output_dir,
        config_output=args.config_output,
        bundle=bundle,
    )
    print(
        f"fit={len(bundle['fit_ids.txt'].decode('utf-8').splitlines())} "
        f"development={len(bundle['development_ids.txt'].decode('utf-8').splitlines())}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the deterministic KITTI factor metadata development split."
    )
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--metadata-jsonl", type=Path)
    parser.add_argument("--images-jsonl", type=Path)
    parser.add_argument("--raw-label-dir", type=Path)
    parser.add_argument("--initialization-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-output", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, default=REGISTERED_SEED)
    parser.add_argument("--fraction", type=float, default=REGISTERED_FRACTION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    full_values = (
        args.metadata_jsonl,
        args.images_jsonl,
        args.raw_label_dir,
        args.initialization_checkpoint,
        args.config_output,
    )
    if args.input_jsonl is not None and any(value is not None for value in full_values):
        parser.error("--input-jsonl cannot combine with full-mode arguments")
    if args.input_jsonl is None and args.metadata_jsonl is None:
        parser.error("one of --input-jsonl or --metadata-jsonl is required")
    if args.input_jsonl is not None:
        if args.output_root is not None:
            parser.error("--output-root is only valid in full mode")
        split = build_development_split(
            _load_rows(args.input_jsonl),
            seed=args.seed,
            fraction=args.fraction,
        )
        write_split_outputs(split, args.output_dir)
        print(
            f"fit={len(split.fit_ids)} "
            f"development={len(split.development_ids)} "
            f"sha256={split.sha256}"
        )
        return 0

    missing = [
        name
        for name, value in (
            ("--images-jsonl", args.images_jsonl),
            ("--raw-label-dir", args.raw_label_dir),
            ("--initialization-checkpoint", args.initialization_checkpoint),
            ("--config-output", args.config_output),
        )
        if value is None
    ]
    if missing:
        parser.error("full mode requires " + ", ".join(missing))
    if args.seed != REGISTERED_SEED or args.fraction != REGISTERED_FRACTION:
        parser.error("full mode requires registered seed=20260805 and fraction=0.10")
    return _build_full_mode(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
