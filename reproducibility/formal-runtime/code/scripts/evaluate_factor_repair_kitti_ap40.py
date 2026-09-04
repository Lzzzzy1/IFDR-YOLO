"""Evaluate a completed factor-repair ``last.pt`` on the fixed development split.

The factor-repair trainer's ``metrics_ap40_primary_last.json`` is an
Ultralytics validation record.  This entrypoint deliberately does not read or
rewrite that record.  It materializes the registered development images,
generates YOLO text predictions from the role-checked ``last.pt``, and invokes
the repository's KITTI AP40 evaluator.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.data.kitti_types import Difficulty, EVAL_CLASSES
from ifdr_yolo.data.splits import load_ids
from ifdr_yolo.eval.evaluate import evaluate_prediction_directory
from ifdr_yolo.experiments.ultralytics_runtime import UltralyticsAdapter


REGISTERED_DEVELOPMENT_COUNT = 371
REGISTERED_DEVELOPMENT_IDS_SHA256 = (
    "b1b6b6ee7e5398e93868fab407a2e8a86a53c753667002ef9b8381734ef2cda8"
)
CHECKPOINT_ROLE = "calibration_last"
PRIMARY_CHECKPOINT = "last.pt"
KITTI_EVALUATOR = "ifdr_yolo.kitti_ap40"

Predictor = Callable[..., Path]
Evaluator = Callable[..., Mapping[str, object]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _require_directory(path: Path, field: str) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise FileNotFoundError(f"{field} must not be a symlink: {candidate}")
    candidate = candidate.resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"{field} must be an existing directory: {candidate}")
    return candidate


def _require_regular_file(
    path: Path,
    field: str,
    *,
    allow_empty: bool = False,
) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise FileNotFoundError(f"{field} must not be a symlink: {candidate}")
    candidate = candidate.resolve()
    if not candidate.is_file() or (not allow_empty and candidate.stat().st_size <= 0):
        descriptor = "regular file" if allow_empty else "non-empty regular file"
        raise FileNotFoundError(f"{field} must be a {descriptor}: {candidate}")
    return candidate


def _resolve_last_checkpoint(run_dir: Path) -> Path:
    candidates = (run_dir / "weights" / PRIMARY_CHECKPOINT, run_dir / PRIMARY_CHECKPOINT)
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink() and candidate.stat().st_size > 0:
            return candidate.resolve()
    raise FileNotFoundError(f"primary {PRIMARY_CHECKPOINT} is missing or empty under {run_dir}")


def _resolve_role_path(run_dir: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("checkpoint role path is required")
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    direct = (run_dir / candidate).resolve()
    in_weights = (run_dir / "weights" / candidate).resolve()
    if in_weights.is_file():
        return in_weights
    return direct


def _checkpoint_record(run_dir: Path, *, expected_sha256: str | None) -> dict[str, object]:
    last = _resolve_last_checkpoint(run_dir)
    roles_path = _require_regular_file(run_dir / "checkpoint_roles.json", "checkpoint_roles.json")
    roles = _read_json(roles_path)
    role_value = roles.get("primary_checkpoint")
    if role_value is None:
        role_value = roles.get(CHECKPOINT_ROLE)
    if not isinstance(role_value, Mapping):
        raise ValueError("checkpoint_roles.json lacks primary_checkpoint/calibration_last")
    role = role_value.get("role")
    if role != "primary":
        raise ValueError("checkpoint_roles primary role must be primary")
    role_field = role_value.get("checkpoint_role")
    if role_field is not None and role_field != CHECKPOINT_ROLE:
        raise ValueError("checkpoint_roles checkpoint_role must be calibration_last")
    role_path = _resolve_role_path(run_dir, role_value.get("path", role_value.get("checkpoint_path")))
    if role_path != last:
        raise ValueError("checkpoint_roles primary path is not last.pt")
    role_hash = role_value.get("sha256", role_value.get("checkpoint_sha256"))
    if not isinstance(role_hash, str) or len(role_hash) != 64:
        raise ValueError("checkpoint_roles primary SHA256 is missing")
    role_hash = role_hash.lower()
    if any(character not in "0123456789abcdef" for character in role_hash):
        raise ValueError("checkpoint_roles primary SHA256 is invalid")
    actual_hash = _sha256_file(last)
    if actual_hash != role_hash:
        raise ValueError("checkpoint_roles primary SHA256 does not match last.pt")
    if expected_sha256 is not None and actual_hash != expected_sha256.lower():
        raise ValueError("last.pt does not match the supplied checkpoint SHA256")
    return {
        "path": str(last),
        "sha256": actual_hash,
        "role": "primary",
        "resolved_semantic_role": CHECKPOINT_ROLE,
        "role_field_missing": role_field is None,
        "checkpoint_roles_path": str(roles_path),
    }


def _development_split(
    path: Path,
    *,
    expected_count: int,
    expected_sha256: str,
) -> tuple[tuple[str, ...], str]:
    split = _require_regular_file(path, "development IDs")
    actual_sha256 = _sha256_file(split)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(
            "development IDs SHA256 mismatch: "
            f"expected={expected_sha256.lower()}, actual={actual_sha256}"
        )
    image_ids = load_ids(split)
    if len(image_ids) != expected_count:
        raise ValueError(
            f"development split must contain exactly {expected_count} IDs, got {len(image_ids)}"
        )
    return image_ids, actual_sha256


def _validate_source_files(
    image_dir: Path,
    label_dir: Path,
    image_ids: Sequence[str],
) -> tuple[Path, Path]:
    image_root = _require_directory(image_dir, "image directory")
    label_root = _require_directory(label_dir, "label directory")
    for image_id in image_ids:
        image = image_root / f"{image_id}.png"
        label = label_root / f"{image_id}.txt"
        _require_regular_file(image, f"development image {image_id}")
        _require_regular_file(label, f"development label {image_id}", allow_empty=True)
    return image_root, label_root


def _verify_staging(
    staging: Path,
    *,
    image_dir: Path,
    image_ids: Sequence[str],
) -> tuple[Path, ...]:
    if staging.is_symlink() or not staging.is_dir():
        raise ValueError(f"development staging directory is invalid: {staging}")
    expected_names = {f"{image_id}.png" for image_id in image_ids}
    entries = tuple(staging.iterdir())
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_names or any(entry.is_symlink() or not entry.is_file() for entry in entries):
        extra = sorted(actual_names - expected_names)
        missing = sorted(expected_names - actual_names)
        raise ValueError(
            f"development staging must contain exactly requested PNGs; extra={extra}, missing={missing}"
        )
    ordered: list[Path] = []
    for image_id in image_ids:
        source = image_dir / f"{image_id}.png"
        destination = staging / source.name
        try:
            same_file = os.path.samefile(source, destination)
        except OSError as error:
            raise ValueError(f"staging image cannot be compared with source: {destination}") from error
        if not same_file:
            raise ValueError(f"staging image is not a hardlink to source: {destination}")
        ordered.append(destination.resolve())
    return tuple(ordered)


def _create_staging(
    staging: Path,
    *,
    image_dir: Path,
    image_ids: Sequence[str],
) -> tuple[Path, ...]:
    if staging.exists():
        return _verify_staging(staging, image_dir=image_dir, image_ids=image_ids)
    staging.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{staging.name}.tmp-", dir=staging.parent)
    )
    try:
        for image_id in image_ids:
            source = image_dir / f"{image_id}.png"
            destination = temporary / source.name
            try:
                os.link(source, destination)
            except OSError as error:
                raise OSError(
                    f"cannot create required hardlink for development image {image_id}"
                ) from error
        _verify_staging(temporary, image_dir=image_dir, image_ids=image_ids)
        temporary.replace(staging)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _verify_staging(staging, image_dir=image_dir, image_ids=image_ids)


def _ensure_prediction_labels(labels: Path, image_ids: Sequence[str]) -> Path:
    if labels.is_symlink() or not labels.is_dir():
        raise ValueError(f"prediction labels directory is missing: {labels}")
    expected_names = {f"{image_id}.txt" for image_id in image_ids}
    entries = tuple(labels.iterdir())
    actual_names = {entry.name for entry in entries}
    if any(entry.is_symlink() or entry.is_dir() for entry in entries):
        raise ValueError(f"prediction labels contain non-regular artifacts: {labels}")
    extra = sorted(actual_names - expected_names)
    if extra:
        raise ValueError(f"prediction labels contain unexpected IDs: {extra[:5]}")
    for image_id in image_ids:
        path = labels / f"{image_id}.txt"
        if not path.exists():
            path.write_text("", encoding="utf-8", newline="\n")
    return labels.resolve()


def _predict_once(
    *,
    output_dir: Path,
    weights: Path,
    staging: Path,
    image_ids: Sequence[str],
    predictor: Predictor,
    prediction_args: Mapping[str, object],
) -> Path:
    final = output_dir / "prediction"
    if final.exists():
        return _ensure_prediction_labels(final / "labels", image_ids)
    temporary = output_dir / f".prediction.tmp-{uuid.uuid4().hex}"
    try:
        image_paths = tuple(staging / f"{image_id}.png" for image_id in image_ids)
        labels = Path(
            predictor(
                weights=weights,
                image_paths=image_paths,
                output_dir=temporary,
                args=dict(prediction_args),
            )
        )
        _ensure_prediction_labels(labels, image_ids)
        temporary.replace(final)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _ensure_prediction_labels(final / "labels", image_ids)


def _validate_kitti_payload(
    payload: Mapping[str, object],
    *,
    split_sha256: str,
    split_count: int,
) -> dict[str, object]:
    if payload.get("evaluator") != KITTI_EVALUATOR:
        raise ValueError("evaluation payload is not the repository KITTI AP40 evaluator")
    if payload.get("split_sha256") != split_sha256:
        raise ValueError("evaluation payload split SHA256 does not match development IDs")
    if payload.get("split_count") != split_count:
        raise ValueError("evaluation payload split count does not match development IDs")
    classes = payload.get("classes")
    if not isinstance(classes, Mapping):
        raise ValueError("KITTI AP40 payload classes are missing")
    if set(classes) != set(EVAL_CLASSES):
        raise ValueError("KITTI AP40 payload class set is incomplete")
    required_metrics = ("ap40", "num_valid_gt", "true_positives", "false_positives")
    for class_name in EVAL_CLASSES:
        difficulties = classes.get(class_name)
        if not isinstance(difficulties, Mapping) or set(difficulties) != {item.value for item in Difficulty}:
            raise ValueError(f"KITTI AP40 difficulty table is incomplete for {class_name}")
        for difficulty in Difficulty:
            metrics = difficulties.get(difficulty.value)
            if not isinstance(metrics, Mapping):
                raise ValueError(f"KITTI AP40 metrics are missing for {class_name}/{difficulty.value}")
            for field in required_metrics:
                if field not in metrics:
                    raise ValueError(f"KITTI AP40 metric {field} is missing for {class_name}/{difficulty.value}")
            try:
                ap40 = float(metrics["ap40"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"KITTI AP40 value is invalid for {class_name}/{difficulty.value}") from error
            if not math.isfinite(ap40):
                raise ValueError(f"KITTI AP40 value is not finite for {class_name}/{difficulty.value}")
            for field in ("num_valid_gt", "true_positives", "false_positives"):
                count = metrics[field]
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError(
                        f"KITTI AP40 count {field} is invalid for {class_name}/{difficulty.value}"
                    )
    return dict(payload)


def _existing_provenance_matches(
    provenance_path: Path,
    *,
    condition: str,
    checkpoint_sha256: str,
    development_sha256: str,
    image_dir: Path,
    label_dir: Path,
) -> None:
    if not provenance_path.is_file():
        return
    payload = _read_json(provenance_path)
    if payload.get("condition") != condition:
        raise ValueError("existing AP40 provenance condition mismatch")
    checkpoint = payload.get("primary_checkpoint")
    development = payload.get("development_split")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("sha256") != checkpoint_sha256:
        raise ValueError("existing AP40 provenance checkpoint mismatch")
    if not isinstance(development, Mapping) or development.get("sha256") != development_sha256:
        raise ValueError("existing AP40 provenance development split mismatch")
    if payload.get("image_dir") != str(image_dir) or payload.get("label_dir") != str(label_dir):
        raise ValueError("existing AP40 provenance dataset path mismatch")


def evaluate_factor_repair_kitti(
    *,
    run_dir: Path,
    condition: str,
    development_ids: Path,
    image_dir: Path,
    label_dir: Path,
    output_dir: Path,
    checkpoint_sha256: str | None = None,
    device: str = "0",
    conf: float = 0.001,
    iou: float = 0.7,
    max_det: int = 300,
    imgsz: int = 640,
    resume: bool = False,
    predictor: Predictor | None = None,
    evaluator: Evaluator | None = None,
) -> dict[str, object]:
    """Generate and evaluate one condition, with explicit resume semantics."""

    if condition not in {"F0", "F1", "F2", "F3"}:
        raise ValueError(f"unsupported factor-repair condition: {condition}")
    if conf < 0 or conf > 1 or iou < 0 or iou > 1 or max_det <= 0 or imgsz <= 0:
        raise ValueError("invalid prediction arguments")
    run_root = _require_directory(run_dir, "factor run directory")
    status_path = _require_regular_file(run_root / "status.json", "factor run status")
    status = _read_json(status_path)
    if status.get("state") != "complete":
        raise ValueError("factor run must be complete before KITTI AP40 evaluation")
    run_provenance = run_root / "provenance.json"
    if run_provenance.is_file():
        provenance_payload = _read_json(run_provenance)
        if provenance_payload.get("condition") not in {None, condition}:
            raise ValueError("factor run provenance condition mismatch")
    checkpoint = _checkpoint_record(run_root, expected_sha256=checkpoint_sha256)
    image_ids, split_sha256 = _development_split(
        development_ids,
        expected_count=REGISTERED_DEVELOPMENT_COUNT,
        expected_sha256=REGISTERED_DEVELOPMENT_IDS_SHA256,
    )
    image_root, label_root = _validate_source_files(image_dir, label_dir, image_ids)
    raw_output_root = output_dir.expanduser()
    if raw_output_root.is_symlink():
        raise FileExistsError(f"AP40 output path must not be a symlink: {raw_output_root}")
    output_root = raw_output_root.resolve()
    if output_root.exists() and not output_root.is_dir():
        raise FileExistsError(f"AP40 output path is not a directory: {output_root}")
    if output_root.exists() and not resume:
        raise FileExistsError(
            f"AP40 output already exists; pass --resume to reuse it: {output_root}"
        )
    if output_root.exists() and resume:
        existing_status = output_root / "status.json"
        if not existing_status.is_file():
            raise ValueError("--resume requires an existing AP40 status.json")
        existing_state = _read_json(existing_status).get("state")
        if existing_state not in {"running", "failed", "complete"}:
            raise ValueError("--resume requires a running, failed, or complete AP40 status")
        _existing_provenance_matches(
            output_root / "provenance.json",
            condition=condition,
            checkpoint_sha256=str(checkpoint["sha256"]),
            development_sha256=split_sha256,
            image_dir=image_root,
            label_dir=label_root,
        )
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output_root / "status.json",
        {
            "schema_version": 1,
            "state": "running",
            "condition": condition,
            "checkpoint_sha256": checkpoint["sha256"],
            "development_ids_sha256": split_sha256,
            "started_at_utc": _utc_now(),
        },
    )
    staging = output_root / "development-images"
    try:
        staged_paths = _create_staging(
            staging,
            image_dir=image_root,
            image_ids=image_ids,
        )
        prediction_args = {
            "conf": conf,
            "iou": iou,
            "max_det": max_det,
            "imgsz": imgsz,
            "device": device,
            "augment": False,
            "verbose": False,
        }
        labels = _predict_once(
            output_dir=output_root,
            weights=Path(str(checkpoint["path"])),
            staging=staging,
            image_ids=image_ids,
            predictor=predictor or UltralyticsAdapter().predict,
            prediction_args=prediction_args,
        )
        output_json = output_root / "kitti_ap40.json"
        if output_json.is_file():
            payload = _validate_kitti_payload(
                _read_json(output_json),
                split_sha256=split_sha256,
                split_count=len(image_ids),
            )
        else:
            raw_payload = (evaluator or evaluate_prediction_directory)(
                prediction_dir=labels,
                label_dir=label_root,
                image_dir=image_root,
                split_path=Path(development_ids).expanduser().resolve(),
            )
            payload = _validate_kitti_payload(
                raw_payload,
                split_sha256=split_sha256,
                split_count=len(image_ids),
            )
            _atomic_json(output_json, payload)
        provenance = {
            "schema_version": 1,
            "condition": condition,
            "evaluator": KITTI_EVALUATOR,
            "primary_checkpoint": checkpoint,
            "development_split": {
                "path": str(Path(development_ids).expanduser().resolve()),
                "sha256": split_sha256,
                "count": len(image_ids),
            },
            "image_dir": str(image_root),
            "label_dir": str(label_root),
            "staging_dir": str(staging),
            "prediction_dir": str(labels),
            "kitti_ap40_json": str(output_json),
            "prediction_args": prediction_args,
            "training_metric_artifact_ignored": str(run_root / "metrics_ap40_primary_last.json"),
            "training_metric_semantics": "Ultralytics validation metrics; not KITTI AP40",
            "staged_image_count": len(staged_paths),
            "completed_at_utc": _utc_now(),
        }
        _atomic_json(output_root / "provenance.json", provenance)
        _atomic_json(
            output_root / "status.json",
            {
                "schema_version": 1,
                "state": "complete",
                "condition": condition,
                "kitti_ap40_json": str(output_json),
                "provenance": str(output_root / "provenance.json"),
                "completed_at_utc": _utc_now(),
            },
        )
        return payload
    except BaseException as error:
        _atomic_json(
            output_root / "status.json",
            {
                "schema_version": 1,
                "state": "failed",
                "condition": condition,
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at_utc": _utc_now(),
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate development KITTI predictions from factor-repair last.pt and evaluate AP40."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=("F0", "F1", "F2", "F3"), required=True)
    parser.add_argument("--development-ids", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = evaluate_factor_repair_kitti(
        run_dir=args.run_dir,
        condition=args.condition,
        development_ids=args.development_ids,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        checkpoint_sha256=args.checkpoint_sha256,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        imgsz=args.imgsz,
        resume=args.resume,
    )
    print(f"KITTI AP40 {args.condition} COMPLETE")
    print(f"evaluator={payload['evaluator']}")
    print(f"output={args.output_dir.resolve() / 'kitti_ap40.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_ROLE",
    "KITTI_EVALUATOR",
    "PRIMARY_CHECKPOINT",
    "REGISTERED_DEVELOPMENT_COUNT",
    "REGISTERED_DEVELOPMENT_IDS_SHA256",
    "build_parser",
    "evaluate_factor_repair_kitti",
    "main",
]
