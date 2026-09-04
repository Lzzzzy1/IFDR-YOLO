"""Leakage-free plain-P2 reference preparation and recovery contract.

The oracle is only meaningful when the detector was trained on the registered
3,341-image fit split.  This module owns the small, deterministic preparation
layer used by the eventual runner; it deliberately does not alter the
baseline/IFDR trainers or start a remote job.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time
from typing import Any, Literal

import yaml

from ifdr_yolo.data.splits import load_ids, sha256_file
from ifdr_yolo.eval.evaluate import evaluate_prediction_directory, write_evaluation_json
from ifdr_yolo.experiments.baseline import _jsonable, ensure_prediction_files
from ifdr_yolo.experiments.config import BaselineConfig
from ifdr_yolo.experiments.ultralytics_runtime import (
    UltralyticsAdapter,
    bootstrap_ultralytics_config,
)


REGISTERED_FIT_COUNT = 3341
REGISTERED_DEVELOPMENT_COUNT = 371
REGISTERED_FULL_TRAIN_COUNT = REGISTERED_FIT_COUNT + REGISTERED_DEVELOPMENT_COUNT
REGISTERED_FIT_IDS_SHA256 = (
    "50a1f8d72b747d1e2f460a4e0a355469484d33dbf1e9928b64ba6594c1873362"
)
REGISTERED_DEVELOPMENT_IDS_SHA256 = (
    "b1b6b6ee7e5398e93868fab407a2e8a86a53c753667002ef9b8381734ef2cda8"
)
REGISTERED_P2_MODEL_SHA256 = (
    "0d2cbd9215dc62c4c9920127964ce0ab61b5d10581861d8e83c33fc168aa302a"
)
REGISTERED_PRETRAINED_SHA256 = (
    "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
)
REGISTERED_P3P5_MODEL_SHA256 = REGISTERED_PRETRAINED_SHA256
REGISTERED_P3P5_MODEL_ROLE = "plain_p3p5"
REFERENCE_EXECUTION_PURPOSE = "fit_reference"
STAGE9_REFERENCE_EXECUTION_PURPOSE = "stage9_seed0_candidate_selection_reference"
STAGE11_REFERENCE_EXECUTION_PURPOSE = "stage11_seeds1_4_matched_reference"
FORMAL_RERUN_REFERENCE_EXECUTION_PURPOSE = "stage11_formal_rerun_plain_reference"


class P2ReferenceIdentityError(ValueError):
    """Raised when a prepared job cannot be proven to be the same run."""


class ActiveReferenceRunError(RuntimeError):
    """Raised when a live owner still holds a fixed-output job."""


@dataclass(frozen=True)
class SplitIdentity:
    fit_ids: tuple[str, ...]
    development_ids: tuple[str, ...]
    fit_sha256: str
    development_sha256: str


@dataclass(frozen=True)
class P2ReferenceIdentity:
    model_sha256: str
    pretrained_sha256: str
    fit_ids_sha256: str
    development_ids_sha256: str
    config_sha256: str
    code_sha256: str
    model_role: str = "plain_p2"
    primary_checkpoint_role: str = "last.pt"
    run_mode: str = "legacy"
    benchmark_identity_sha256: str | None = None
    benchmark_execution_role: str | None = None
    benchmark_effective_epochs: int | None = None
    execution_purpose: str = REFERENCE_EXECUTION_PURPOSE

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        # Keep the historical P2 identity bytes stable; the mode field is
        # required only for the new P3--P5 smoke/full separation.
        if self.model_role == "plain_p2" and self.run_mode == "legacy":
            payload.pop("run_mode", None)
        if self.benchmark_identity_sha256 is None:
            payload.pop("benchmark_identity_sha256", None)
            payload.pop("benchmark_execution_role", None)
            payload.pop("benchmark_effective_epochs", None)
        if self.execution_purpose == REFERENCE_EXECUTION_PURPOSE:
            payload.pop("execution_purpose", None)
        return payload

    def digest(self) -> str:
        return _payload_sha256(self.as_dict())


@dataclass(frozen=True)
class P2ReferenceJob:
    output_dir: Path
    mirror_dir: Path
    resolved_data_yaml: Path
    identity_path: Path
    status_path: Path
    split_manifest_path: Path
    fit_ids: tuple[str, ...]
    development_ids: tuple[str, ...]
    identity: P2ReferenceIdentity
    identity_sha256: str
    state: str
    resumable: bool


@dataclass(frozen=True)
class P2ReferenceServices:
    """Injectable boundaries used by tests and the real runner."""

    adapter: Any
    evaluate: Callable[..., dict[str, object]] = evaluate_prediction_directory
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    pid_alive: Callable[[int], bool] = lambda pid: _pid_alive(pid)
    hostname: str = socket.gethostname()
    preflight: Callable[..., object] | None = None
    resume_training: Callable[..., None] | None = None


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise P2ReferenceIdentityError(f"existing artifact is not a regular file: {path}")
        if path.read_bytes() != content:
            raise P2ReferenceIdentityError(f"existing artifact differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _replace_bytes(path: Path, content: bytes) -> None:
    """Atomically replace a mutable progress/status artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _regular_file(path: Path, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {raw}")
    candidate = raw.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} must be a regular file: {candidate}")
    if candidate.stat().st_size <= 0:
        raise ValueError(f"{label} must be non-empty: {candidate}")
    return candidate


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _read_yaml(path: Path, label: str) -> dict[str, object]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"{label} is not valid YAML") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a mapping")
    return payload


def _hash_code(repository_root: Path) -> str:
    """Bind the runner to the current commit plus its source files."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        commit = completed.stdout.strip() if completed.returncode == 0 else ""
    except OSError:
        commit = ""
    files = [
        Path(__file__),
        repository_root / "ifdr_yolo" / "experiments" / "baseline.py",
        repository_root / "scripts" / "run_p2_fit_reference.py",
    ]
    digest = hashlib.sha256()
    digest.update(commit.encode("utf-8"))
    for path in files:
        if path.is_file():
            digest.update(path.resolve().as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _hash_code_p3p5(repository_root: Path) -> str:
    """Bind plain P3--P5 runs to the complete local evaluation contract."""

    root = Path(repository_root).resolve()
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        commit = completed.stdout.strip() if completed.returncode == 0 else ""
    except OSError:
        commit = ""
    files = (
        root / "ifdr_yolo" / "experiments" / "p2_fit_reference.py",
        root / "ifdr_yolo" / "experiments" / "baseline.py",
        root / "ifdr_yolo" / "experiments" / "config.py",
        root / "ifdr_yolo" / "experiments" / "ultralytics_runtime.py",
        root / "ifdr_yolo" / "eval" / "evaluate.py",
        root / "ifdr_yolo" / "eval" / "kitti_ap40.py",
        root / "scripts" / "run_p2_fit_reference.py",
        root / "scripts" / "run_p3p5_fit_reference.py",
    )
    digest = hashlib.sha256(commit.encode("utf-8"))
    for path in files:
        if path.is_file():
            digest.update(path.resolve().as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def build_reference_identity(
    *,
    model_sha256: str,
    pretrained_sha256: str,
    fit_ids_sha256: str,
    development_ids_sha256: str,
    config_sha256: str,
    code_sha256: str,
    model_role: str = "plain_p2",
    run_mode: str = "legacy",
    benchmark_identity_sha256: str | None = None,
    benchmark_execution_role: str | None = None,
    benchmark_effective_epochs: int | None = None,
    execution_purpose: str = REFERENCE_EXECUTION_PURPOSE,
) -> P2ReferenceIdentity:
    values = {
        "model_sha256": model_sha256,
        "pretrained_sha256": pretrained_sha256,
        "fit_ids_sha256": fit_ids_sha256,
        "development_ids_sha256": development_ids_sha256,
        "config_sha256": config_sha256,
        "code_sha256": code_sha256,
    }
    for field, value in values.items():
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
            raise ValueError(f"{field} must be a 64-hex SHA256")
    if model_role not in {"plain_p2", REGISTERED_P3P5_MODEL_ROLE}:
        raise ValueError(f"unsupported reference model role: {model_role}")
    if run_mode not in {"legacy", "dry-run", "smoke", "full"}:
        raise ValueError(f"unsupported reference run mode: {run_mode}")
    if execution_purpose not in {
        REFERENCE_EXECUTION_PURPOSE,
        STAGE9_REFERENCE_EXECUTION_PURPOSE,
        STAGE11_REFERENCE_EXECUTION_PURPOSE,
        FORMAL_RERUN_REFERENCE_EXECUTION_PURPOSE,
    }:
        raise ValueError(f"unsupported reference execution purpose: {execution_purpose}")
    benchmark_values = (benchmark_identity_sha256, benchmark_execution_role, benchmark_effective_epochs)
    if any(value is not None for value in benchmark_values):
        if not (isinstance(benchmark_identity_sha256, str) and len(benchmark_identity_sha256) == 64 and not set(benchmark_identity_sha256.lower()) - set("0123456789abcdef") and isinstance(benchmark_execution_role, str) and benchmark_effective_epochs in {1, 2}):
            raise ValueError("benchmark identity binding is invalid")
    return P2ReferenceIdentity(
        **{key: str(value).lower() for key, value in values.items()},
        model_role=model_role,
        run_mode=run_mode,
        benchmark_identity_sha256=benchmark_identity_sha256,
        benchmark_execution_role=benchmark_execution_role,
        benchmark_effective_epochs=benchmark_effective_epochs,
        execution_purpose=execution_purpose,
    )


def validate_fit_development_split(
    config: BaselineConfig,
    fit_ids_path: Path,
    development_ids_path: Path,
    *,
    expected_fit_count: int = REGISTERED_FIT_COUNT,
    expected_development_count: int = REGISTERED_DEVELOPMENT_COUNT,
    expected_fit_sha256: str = REGISTERED_FIT_IDS_SHA256,
    expected_development_sha256: str = REGISTERED_DEVELOPMENT_IDS_SHA256,
) -> SplitIdentity:
    fit_path = _regular_file(fit_ids_path, "fit split")
    development_path = _regular_file(development_ids_path, "development split")
    full_path = _regular_file(config.paths.train_ids, "full training split")
    fit_ids = load_ids(fit_path)
    development_ids = load_ids(development_path)
    full_ids = load_ids(full_path)
    if len(full_ids) != expected_fit_count + expected_development_count:
        raise ValueError(f"full training split count must be {expected_fit_count + expected_development_count}")
    if len(fit_ids) != expected_fit_count:
        raise ValueError(f"fit split count must be {expected_fit_count}")
    if len(development_ids) != expected_development_count:
        raise ValueError(f"development split count must be {expected_development_count}")
    overlap = sorted(set(fit_ids) & set(development_ids))
    if overlap:
        raise ValueError(f"fit/development overlap (leakage): {overlap[:5]}")
    if set(fit_ids) | set(development_ids) != set(full_ids):
        raise ValueError("fit/development split does not exactly cover full training split")
    fit_sha = sha256_file(fit_path)
    development_sha = sha256_file(development_path)
    if fit_sha != expected_fit_sha256.lower():
        raise ValueError(f"registered fit split hash mismatch: expected={expected_fit_sha256}, actual={fit_sha}")
    if development_sha != expected_development_sha256.lower():
        raise ValueError(f"registered development split hash mismatch: expected={expected_development_sha256}, actual={development_sha}")
    return SplitIdentity(fit_ids, development_ids, fit_sha, development_sha)


def validate_plain_p2_model(config: BaselineConfig) -> None:
    model_path = _regular_file(config.paths.model, "P2 model")
    if config.experiment.variant != "p2" or model_path.name != "kitti-p2-m.yaml":
        raise ValueError("reference requires the plain P2 model")
    actual = sha256_file(model_path)
    expected = config.paths.model_sha256.lower()
    if expected != REGISTERED_P2_MODEL_SHA256 or actual != REGISTERED_P2_MODEL_SHA256:
        raise ValueError("reference requires the registered plain P2 model")
    payload = _read_yaml(model_path, "P2 model")
    text = model_path.read_text(encoding="utf-8").lower()
    if "ifdr" in text or "fusion_gate" in text or "factor" in text:
        raise ValueError("reference requires the plain P2 model")
    if not payload:
        raise ValueError("reference requires the plain P2 model")
    if config.initialization is None:
        raise ValueError("plain P2 reference requires yolov8m initialization")
    pretrained = _regular_file(config.initialization.pretrained, "pretrained model")
    if sha256_file(pretrained) != REGISTERED_PRETRAINED_SHA256 or config.initialization.pretrained_sha256.lower() != REGISTERED_PRETRAINED_SHA256:
        raise ValueError("reference requires the registered yolov8m pretrained model")


def validate_plain_p3p5_model(
    config: BaselineConfig,
    *,
    benchmark_context: Any | None = None,
    execution_purpose: str = REFERENCE_EXECUTION_PURPOSE,
) -> None:
    """Validate the registered, unmodified YOLOv8m P3--P5 baseline."""

    if config.experiment.variant != "baseline":
        raise ValueError("P3-P5 reference requires variant=baseline")
    if config.experiment.model != "yolov8m":
        raise ValueError("P3-P5 reference requires experiment.model=yolov8m")
    if execution_purpose == STAGE9_REFERENCE_EXECUTION_PURPOSE:
        if benchmark_context is not None:
            raise ValueError("Stage9 reference forbids benchmark context")
        allowed_seeds, expected_epochs, label = {0}, 30, "Stage9 reference"
    elif execution_purpose == STAGE11_REFERENCE_EXECUTION_PURPOSE:
        if benchmark_context is not None:
            raise ValueError("Stage11 reference forbids benchmark context")
        allowed_seeds, expected_epochs, label = {1, 2, 3, 4}, 30, "Stage11 reference"
    elif execution_purpose == FORMAL_RERUN_REFERENCE_EXECUTION_PURPOSE:
        if benchmark_context is not None:
            raise ValueError("Stage11 formal rerun reference forbids benchmark context")
        allowed_seeds, expected_epochs, label = {0, 1, 2, 3, 4}, 30, "Stage11 formal rerun reference"
    elif execution_purpose != REFERENCE_EXECUTION_PURPOSE:
        raise ValueError(f"unsupported reference execution purpose: {execution_purpose}")
    elif benchmark_context is None:
        allowed_seeds, expected_epochs, label = {17}, 300, "P3-P5 reference"
    else:
        benchmark_identity = getattr(benchmark_context, "identity", None)
        if not isinstance(benchmark_identity, Mapping) or {
            "arm": benchmark_identity.get("arm"),
            "seed": benchmark_identity.get("seed"),
            "formal_epochs": benchmark_identity.get("formal_epochs"),
        } != {"arm": "P3P5_CONTROL", "seed": 0, "formal_epochs": 30}:
            raise ValueError("P3-P5 benchmark identity is not the registered seed-0 protocol")
        allowed_seeds, expected_epochs, label = {0}, 30, "P3-P5 reference"
    if config.experiment.seed not in allowed_seeds:
        expected = next(iter(allowed_seeds)) if len(allowed_seeds) == 1 else sorted(allowed_seeds)
        raise ValueError(f"{label} requires seed={expected}")
    if config.paths.model.name != "yolov8m.pt":
        raise ValueError("P3-P5 reference requires model basename yolov8m.pt")
    model_path = _regular_file(config.paths.model, "P3-P5 model")
    actual_sha = sha256_file(model_path)
    if (
        actual_sha != REGISTERED_P3P5_MODEL_SHA256
        or config.paths.model_sha256.lower() != REGISTERED_P3P5_MODEL_SHA256
    ):
        raise ValueError("P3-P5 reference requires the registered yolov8m.pt SHA256")
    if config.initialization is not None:
        raise ValueError("P3-P5 reference forbids extra initialization")
    if config.training.epochs != expected_epochs:
        raise ValueError(f"{label} requires epochs={expected_epochs}")
    if config.training.imgsz != 640:
        raise ValueError("P3-P5 reference requires imgsz=640")


def _validate_stage9_reference_contract(
    config: BaselineConfig,
    *,
    mode: str,
    benchmark_context: Any | None,
    execution_purpose: str,
) -> None:
    if execution_purpose == REFERENCE_EXECUTION_PURPOSE:
        return
    if execution_purpose not in {
        STAGE9_REFERENCE_EXECUTION_PURPOSE,
        STAGE11_REFERENCE_EXECUTION_PURPOSE,
        FORMAL_RERUN_REFERENCE_EXECUTION_PURPOSE,
    }:
        raise ValueError(f"unsupported reference execution purpose: {execution_purpose}")
    if benchmark_context is not None:
        label = "Stage9" if execution_purpose == STAGE9_REFERENCE_EXECUTION_PURPOSE else "Stage11"
        raise ValueError(f"{label} reference forbids benchmark context")
    if execution_purpose == STAGE9_REFERENCE_EXECUTION_PURPOSE:
        expected_seeds, label = {0}, "Stage9"
    elif execution_purpose == STAGE11_REFERENCE_EXECUTION_PURPOSE:
        expected_seeds, label = {1, 2, 3, 4}, "Stage11"
    else:
        expected_seeds, label = {0, 1, 2, 3, 4}, "Stage11 formal rerun"
    observed = {
        "seed": config.experiment.seed,
        "epochs": config.training.epochs,
        "imgsz": config.training.imgsz,
        "batch": config.training.batch,
        "workers": config.training.workers,
        "deterministic": config.training.deterministic,
        "mode": mode,
    }
    expected = {
        "seed": config.experiment.seed,
        "epochs": 30,
        "imgsz": 640,
        "batch": 16,
        "workers": 8,
        "deterministic": True,
        "mode": "full",
    }
    if config.experiment.seed not in expected_seeds:
        raise ValueError(f"{label} reference contract mismatch: seed must be one of {sorted(expected_seeds)}")
    if observed != expected:
        raise ValueError(f"{label} reference contract mismatch: expected={expected}, actual={observed}")


def extract_primary_metric(
    metrics: Mapping[str, object],
    *,
    split_sha256: str,
    checkpoint_sha256: str,
    split_count: int | None = None,
    identity_sha256: str | None = None,
) -> dict[str, object]:
    """Extract the frozen Pedestrian/Cyclist Moderate AP_R40 macro metric."""

    if metrics.get("evaluator") != "ifdr_yolo.kitti_ap40":
        raise ValueError("primary metric requires the exact kitti_ap40 evaluator")
    observed_split = metrics.get("split_sha256")
    if observed_split != split_sha256:
        raise ValueError("primary metric split SHA256 does not match the registered split")
    if split_count is not None and metrics.get("split_count") != split_count:
        raise ValueError("primary metric split count does not match the registered split")
    classes = metrics.get("classes")
    if not isinstance(classes, Mapping):
        raise ValueError("evaluator metrics have no classes mapping")

    values: dict[str, float] = {}
    for class_name in ("Pedestrian", "Cyclist"):
        class_payload = classes.get(class_name)
        if not isinstance(class_payload, Mapping):
            raise ValueError(f"evaluator metrics have no {class_name} class")
        moderate = class_payload.get("moderate")
        if not isinstance(moderate, Mapping):
            raise ValueError(f"evaluator metrics have no {class_name} Moderate metrics")
        ap40 = moderate.get("ap40")
        if isinstance(ap40, bool) or not isinstance(ap40, (int, float)) or not math.isfinite(float(ap40)):
            raise ValueError(f"{class_name} Moderate AP_R40 is not finite")
        values[class_name] = float(ap40)

    payload: dict[str, object] = {
        "schema_version": 1,
        "metric": "Pedestrian/Cyclist Moderate AP_R40 macro",
        "difficulty": "moderate",
        "classes": ["Pedestrian", "Cyclist"],
        "pedestrian_moderate_ap_r40": values["Pedestrian"],
        "cyclist_moderate_ap_r40": values["Cyclist"],
        "macro_ap_r40": (values["Pedestrian"] + values["Cyclist"]) / 2.0,
        "split_sha256": split_sha256,
        "split_count": metrics.get("split_count"),
        "last_pt_sha256": checkpoint_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    if identity_sha256 is not None:
        payload["identity_sha256"] = identity_sha256
    return payload


def validate_primary_checkpoint(path: Path) -> Path:
    checkpoint = _regular_file(path, "primary reference checkpoint")
    if checkpoint.name != "last.pt":
        raise ValueError("primary reference checkpoint must be last.pt, not best.pt")
    return checkpoint


def _source_file(generated: Path, kind: str, split: str, image_id: str) -> Path:
    base = generated / kind / split
    extension = ".png" if kind == "images" else ".txt"
    path = base / f"{image_id}{extension}"
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing generated {kind} for {image_id}: {path}")
    return path.resolve()


def _link_or_verify(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or not target.is_file() or not os.path.samefile(source, target):
            raise P2ReferenceIdentityError(f"existing hard-link view differs: {target}")
        return
    os.link(source, target)


def _materialize_views(config: BaselineConfig, output_dir: Path, split: SplitIdentity) -> dict[str, list[str]]:
    generated = config.paths.generated_data.resolve()
    paths: dict[str, list[str]] = {"fit": [], "development": []}
    for split_name, ids, source_split, view_split in (
        ("fit", split.fit_ids, "train", "train"),
        ("development", split.development_ids, "train", "val"),
    ):
        for image_id in ids:
            image_source = _source_file(generated, "images", source_split, image_id)
            label_source = _source_file(generated, "labels", source_split, image_id)
            image_target = output_dir / "view" / "images" / view_split / f"{image_id}.png"
            label_target = output_dir / "view" / "labels" / view_split / f"{image_id}.txt"
            _link_or_verify(image_source, image_target)
            _link_or_verify(label_source, label_target)
            paths[split_name].append(str(image_target.resolve()))
    return paths


def _mirror_artifacts(job: P2ReferenceJob, names: tuple[str, ...]) -> None:
    primary = job.output_dir.resolve()
    mirror = job.mirror_dir.resolve()
    if mirror == primary or mirror.is_relative_to(primary) or primary.is_relative_to(mirror):
        raise P2ReferenceIdentityError(
            "mirror directory must be a disjoint sibling of the primary job"
        )
    job.mirror_dir.mkdir(parents=True, exist_ok=True)
    file_records: list[dict[str, object]] = []
    for name in names:
        source = job.output_dir / name
        if not source.is_file():
            raise P2ReferenceIdentityError(f"cannot mirror missing artifact: {source}")
        target = job.mirror_dir / name
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(target)
        file_records.append(
            {
                "name": name,
                "size": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    generation = _payload_sha256(file_records)
    manifest = {
        "schema_version": 1,
        "generation": generation,
        "primary_output": str(primary),
        "files": file_records,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    # The manifest is deliberately published last. Readers can reject any
    # artifact set whose generation is not represented here.
    _replace_bytes(job.mirror_dir / "manifest.json", manifest_bytes)


def _status_payload(
    *,
    state: str,
    identity: P2ReferenceIdentity,
    identity_sha256: str,
    current_epoch: int,
    next_action: str,
    checkpoint: Path | None = None,
    started_at: str | None = None,
    elapsed_seconds: float = 0.0,
    error: BaseException | None = None,
    reference_label: str = "P2",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "reference": reference_label,
        "state": state,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "identity": identity.as_dict(),
        "identity_sha256": identity_sha256,
        "current_epoch": current_epoch,
        "next_action": next_action,
        "checkpoint_role": "last.pt",
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint and checkpoint.is_file() else None,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns if checkpoint and checkpoint.is_file() else None,
        "started_at_utc": started_at,
        "elapsed_seconds": elapsed_seconds,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = str(error)
    return payload


def prepare_p2_fit_reference(
    config: BaselineConfig,
    *,
    repository_root: Path,
    output_dir: Path,
    mirror_dir: Path,
    fit_ids: Path,
    development_ids: Path,
    expected_fit_count: int = REGISTERED_FIT_COUNT,
    expected_development_count: int = REGISTERED_DEVELOPMENT_COUNT,
    expected_fit_sha256: str = REGISTERED_FIT_IDS_SHA256,
    expected_development_sha256: str = REGISTERED_DEVELOPMENT_IDS_SHA256,
    identity: P2ReferenceIdentity | None = None,
    pid_alive: Callable[[int], bool] = _pid_alive,
    hostname: str = socket.gethostname(),
    benchmark_context: Any | None = None,
    reference_kind: Literal["p2", "p3p5"] = "p2",
    run_mode: str = "legacy",
    execution_purpose: str = REFERENCE_EXECUTION_PURPOSE,
) -> P2ReferenceJob:
    _validate_stage9_reference_contract(
        config,
        mode=run_mode,
        benchmark_context=benchmark_context,
        execution_purpose=execution_purpose,
    )
    if reference_kind == "p2":
        validate_plain_p2_model(config)
        model_role = "plain_p2"
        code_sha = _hash_code(Path(repository_root).resolve())
        pretrained_sha_override: str | None = None
        reference_label = "P2"
    elif reference_kind == "p3p5":
        validate_plain_p3p5_model(
            config,
            benchmark_context=benchmark_context,
            execution_purpose=execution_purpose,
        )
        model_role = REGISTERED_P3P5_MODEL_ROLE
        code_sha = _hash_code_p3p5(Path(repository_root).resolve())
        pretrained_sha_override = REGISTERED_P3P5_MODEL_SHA256
        reference_label = "P3-P5"
        if run_mode not in {"dry-run", "smoke", "full"}:
            raise ValueError(f"unsupported P3-P5 run mode: {run_mode}")
    else:
        raise ValueError(f"unknown reference kind: {reference_kind}")
    split = validate_fit_development_split(
        config,
        fit_ids,
        development_ids,
        expected_fit_count=expected_fit_count,
        expected_development_count=expected_development_count,
        expected_fit_sha256=expected_fit_sha256,
        expected_development_sha256=expected_development_sha256,
    )
    output = Path(output_dir).expanduser().resolve()
    mirror = Path(mirror_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = config.source_path
    if config_path is None:
        raise P2ReferenceIdentityError("baseline config source_path is required")
    config_path = _regular_file(config_path, "baseline config")
    model_sha = sha256_file(config.paths.model)
    pretrained_sha = (
        pretrained_sha_override
        if pretrained_sha_override is not None
        else sha256_file(config.initialization.pretrained) if config.initialization else ""
    )
    computed_identity = build_reference_identity(
        model_sha256=model_sha,
        pretrained_sha256=pretrained_sha,
        fit_ids_sha256=split.fit_sha256,
        development_ids_sha256=split.development_sha256,
        config_sha256=sha256_file(config_path),
        code_sha256=code_sha,
        model_role=model_role,
        run_mode=run_mode,
        benchmark_identity_sha256=(str(benchmark_context.identity["identity_sha256"]) if benchmark_context is not None else None),
        benchmark_execution_role=(str(benchmark_context.identity["execution_role"]) if benchmark_context is not None else None),
        benchmark_effective_epochs=(int(benchmark_context.identity["effective_epochs"]) if benchmark_context is not None else None),
        execution_purpose=execution_purpose,
    )
    if identity is not None and identity.as_dict() != computed_identity.as_dict():
        raise P2ReferenceIdentityError("supplied reference identity does not match current inputs")
    expected_identity = identity or computed_identity
    identity_path = output / "reference_identity.json"
    identity_sha = expected_identity.digest()
    if identity_path.is_file():
        try:
            existing = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise P2ReferenceIdentityError("existing reference identity is invalid") from error
        if existing.get("identity_sha256") != identity_sha or existing.get("identity") != expected_identity.as_dict():
            raise P2ReferenceIdentityError("existing reference identity does not match")
    status_path = output / "status.json"
    resumable = False
    state = "prepared"
    had_existing_content = output.exists() and any(output.iterdir())
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise P2ReferenceIdentityError("existing run status is invalid") from error
        if not isinstance(status, dict):
            raise P2ReferenceIdentityError("existing run status must be a JSON object")
        state_value = status.get("state")
        if not isinstance(state_value, str):
            raise P2ReferenceIdentityError("existing run status has no valid state")
        state = state_value
        if status.get("identity_sha256") != identity_sha:
            raise P2ReferenceIdentityError("existing run status identity does not match")
        try:
            owner_pid = int(status.get("pid", 0))
        except (TypeError, ValueError) as error:
            raise P2ReferenceIdentityError("existing run status has an invalid PID") from error
        owner_host = status.get("hostname")
        if state == "running" and owner_host != hostname:
            raise P2ReferenceIdentityError("running reference job belongs to another host")
        if state == "running" and pid_alive(owner_pid):
            raise ActiveReferenceRunError("a live owner holds the fixed-output reference job")
        if state in {"failed", "interrupted", "running"}:
            resumable = (output / "weights" / "last.pt").is_file() and (output / "results.csv").is_file()
            if resumable:
                _completed_epochs(output / "results.csv")
        elif state not in {"prepared", "trained", "evaluating", "complete"}:
            raise P2ReferenceIdentityError(f"unknown reference run state: {state}")
    elif had_existing_content:
        raise P2ReferenceIdentityError("existing reference job has no identity-bound status")
    _materialize_views(config, output, split)
    view_root = output / "view"
    data_payload = _read_yaml(config.paths.data, "data config")
    resolved_data = {
        **data_payload,
        "path": str(view_root),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "Car", 1: "Pedestrian", 2: "Cyclist"},
    }
    _atomic_bytes(output / "resolved_data.yaml", yaml.safe_dump(resolved_data, sort_keys=False).encode("utf-8"))
    resolved_config = _jsonable(asdict(config))
    if not isinstance(resolved_config, dict) or not isinstance(resolved_config.get("paths"), dict):
        raise P2ReferenceIdentityError("resolved baseline config paths are invalid")
    resolved_config["paths"]["data"] = str(output / "resolved_data.yaml")
    resolved_config["paths"]["train_ids"] = str(output / "fit_ids.txt")
    resolved_config["paths"]["val_ids"] = str(output / "development_ids.txt")
    _atomic_bytes(output / "config.resolved.yaml", yaml.safe_dump(resolved_config, sort_keys=False).encode("utf-8"))
    _atomic_bytes(output / "fit_ids.txt", "".join(f"{value}\n" for value in split.fit_ids).encode("utf-8"))
    _atomic_bytes(output / "development_ids.txt", "".join(f"{value}\n" for value in split.development_ids).encode("utf-8"))
    split_manifest = {
        "schema_version": 1,
        "primary_checkpoint_role": expected_identity.primary_checkpoint_role,
        "model_sha256": expected_identity.model_sha256,
        "pretrained_sha256": expected_identity.pretrained_sha256,
        "config_sha256": expected_identity.config_sha256,
        "code_sha256": expected_identity.code_sha256,
        "fit_count": len(split.fit_ids),
        "development_count": len(split.development_ids),
        "fit_ids_sha256": split.fit_sha256,
        "development_ids_sha256": split.development_sha256,
        "fit_ids": list(split.fit_ids),
        "development_ids": list(split.development_ids),
        "view_root": str(view_root),
    }
    split_manifest_path = output / "split_manifest.json"
    _atomic_json(split_manifest_path, split_manifest)
    _atomic_json(identity_path, {"identity": expected_identity.as_dict(), "identity_sha256": identity_sha})
    job = P2ReferenceJob(
        output_dir=output,
        mirror_dir=mirror,
        resolved_data_yaml=output / "resolved_data.yaml",
        identity_path=identity_path,
        status_path=status_path,
        split_manifest_path=split_manifest_path,
        fit_ids=split.fit_ids,
        development_ids=split.development_ids,
        identity=expected_identity,
        identity_sha256=identity_sha,
        state=state,
        resumable=resumable,
    )
    if not status_path.is_file():
        _atomic_json(
            status_path,
            _status_payload(
                state="prepared",
                identity=expected_identity,
                identity_sha256=identity_sha,
                current_epoch=0,
                next_action=f"start {reference_label} training with last.pt as the primary checkpoint",
                reference_label=reference_label,
            ),
        )
    _mirror_artifacts(job, ("reference_identity.json", "resolved_data.yaml", "config.resolved.yaml", "split_manifest.json", "status.json"))
    return job


def _default_resume_training(checkpoint: Path, job: P2ReferenceJob, *, device: str, workers: int, data_yaml: Path) -> None:
    from ultralytics import YOLO

    YOLO(str(checkpoint)).train(
        resume=True,
        data=str(data_yaml),
        project=str(job.output_dir.parent),
        name=job.output_dir.name,
        exist_ok=True,
        device=device,
        workers=workers,
    )


def _verify_trainer_output(result: object, job: P2ReferenceJob) -> Path:
    if result is None:
        raise RuntimeError("trainer did not return a checkpoint path")
    returned = Path(result).expanduser().resolve()
    if not returned.is_file() or returned.stat().st_size <= 0:
        raise FileNotFoundError(f"trainer checkpoint is missing or empty: {returned}")
    if returned.name not in {"best.pt", "last.pt"}:
        raise ValueError(f"trainer returned an unexpected checkpoint role: {returned.name}")
    primary = job.output_dir / "weights" / "last.pt"
    return validate_primary_checkpoint(primary)


def _published_artifact_names(job: P2ReferenceJob) -> tuple[str, ...]:
    """Return the small provenance set that is safe to mirror atomically."""

    names = [
        "reference_identity.json",
        "resolved_data.yaml",
        "config.resolved.yaml",
        "split_manifest.json",
        "status.json",
    ]
    for optional in (
        "observed_train_ids.txt",
        "post_training_leakage_audit.json",
        "checkpoint_provenance.json",
        "metrics_ap40.json",
        "primary_metric.json",
    ):
        if (job.output_dir / optional).is_file():
            names.append(optional)
    return tuple(names)


def _read_training_args(job: P2ReferenceJob) -> tuple[Path, str]:
    args_path = _regular_file(job.output_dir / "args.yaml", "Ultralytics args.yaml")
    payload = _read_yaml(args_path, "Ultralytics args.yaml")
    data_value = payload.get("data")
    if not isinstance(data_value, str) or not data_value.strip():
        raise ValueError("Ultralytics args.yaml must contain a data path")
    data_path = Path(data_value).expanduser()
    if not data_path.is_absolute():
        data_path = args_path.parent / data_path
    if data_path.resolve() != job.resolved_data_yaml.resolve():
        raise P2ReferenceIdentityError(
            "Ultralytics args.yaml data path does not resolve to this job's resolved_data.yaml"
        )
    return args_path, sha256_file(args_path)


def _find_train_cache(job: P2ReferenceJob) -> Path:
    """Locate only bounded Ultralytics train-cache locations under this run."""

    candidates = (
        job.output_dir / "train.cache",
        job.output_dir / "train.cache.npy",
        job.output_dir / "train.cache.npz",
        job.output_dir / "labels.cache",
        job.output_dir / "labels.cache.npy",
        job.output_dir / "view" / "labels" / "train.cache",
        job.output_dir / "view" / "labels" / "train.cache.npy",
        job.output_dir / "view" / "labels" / "train.cache.npz",
    )
    for candidate in candidates:
        if candidate.exists():
            return _regular_file(candidate, "train label cache")
    raise FileNotFoundError("train label cache was not produced by the trainer")


def _decode_train_cache(path: Path) -> object:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
        return payload
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    try:
        import numpy as np

        loaded = np.load(path, allow_pickle=True)
        if hasattr(loaded, "files"):
            return {name: loaded[name] for name in loaded.files}
        if hasattr(loaded, "item"):
            return loaded.item()
        return loaded
    except (ImportError, OSError, ValueError, EOFError, AttributeError):
        pass
    try:
        return pickle.loads(raw)
    except (pickle.UnpicklingError, EOFError, AttributeError, ValueError, TypeError) as error:
        raise ValueError(f"train label cache is not parseable: {path}") from error


def _cache_value_paths(value: object) -> list[str]:
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, (list, tuple)):
        paths: list[str] = []
        for item in value:
            paths.extend(_cache_value_paths(item))
        return paths
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _cache_value_paths(tolist())
    return []


def _extract_train_cache_ids(payload: object) -> tuple[str, ...]:
    if not isinstance(payload, Mapping):
        raise ValueError("train label cache must contain a mapping")
    records = payload.get("labels")
    if isinstance(records, Mapping):
        records = list(records.values())
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("train label cache has no non-empty labels records")
    ids: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("train label cache contains a malformed label record")
        values = record.get("im_file", record.get("im_files"))
        paths = _cache_value_paths(values)
        if not paths:
            raise ValueError("train label cache label record has no im_file")
        ids.extend(Path(value).stem for value in paths)
    if any(not image_id for image_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("train label cache contains duplicate or empty image IDs")
    return tuple(ids)


def _audit_training_inputs(job: P2ReferenceJob) -> Path:
    """Prove the trainer consumed exactly fit IDs before evaluation is allowed."""

    args_path, args_sha = _read_training_args(job)
    cache_path = _find_train_cache(job)
    cache_sha = sha256_file(cache_path)
    observed_ids = _extract_train_cache_ids(_decode_train_cache(cache_path))
    expected_ids = tuple(job.fit_ids)
    observed_set = set(observed_ids)
    expected_set = set(expected_ids)
    development_overlap = sorted(observed_set & set(job.development_ids))
    if len(observed_ids) != len(expected_ids) or observed_set != expected_set:
        missing = sorted(expected_set - observed_set)
        extra = sorted(observed_set - expected_set)
        raise P2ReferenceIdentityError(
            f"train cache IDs do not match fit manifest: missing={missing[:5]}, extra={extra[:5]}"
        )
    if development_overlap:
        raise P2ReferenceIdentityError(
            f"train cache contains development IDs: {development_overlap[:5]}"
        )
    fit_manifest = _regular_file(job.output_dir / "fit_ids.txt", "fit manifest")
    development_manifest = _regular_file(job.output_dir / "development_ids.txt", "development manifest")
    try:
        fit_manifest_ids = tuple(load_ids(fit_manifest))
        development_manifest_ids = tuple(load_ids(development_manifest))
    except (OSError, ValueError) as error:
        raise P2ReferenceIdentityError("split manifest is not readable") from error
    if fit_manifest_ids != expected_ids or development_manifest_ids != tuple(job.development_ids):
        raise P2ReferenceIdentityError("split manifest changed after preparation")
    # The identity binds the original registered split bytes.  Keep the
    # generated manifest's byte hash as an additional audit field because
    # Windows fixtures may normalize CRLF to LF while preserving IDs.
    fit_manifest_sha = job.identity.fit_ids_sha256
    development_manifest_sha = job.identity.development_ids_sha256
    observed_bytes = "".join(f"{image_id}\n" for image_id in expected_ids).encode("utf-8")
    observed_path = job.output_dir / "observed_train_ids.txt"
    _atomic_bytes(observed_path, observed_bytes)
    audit_payload: dict[str, object] = {
        "schema_version": 1,
        "args_path": str(args_path.resolve()),
        "args_sha256": args_sha,
        "train_cache_path": str(cache_path.resolve()),
        "train_cache_sha256": cache_sha,
        "observed_train_count": len(observed_ids),
        "observed_train_ids_sha256": sha256_file(observed_path),
        "fit_count": len(expected_ids),
        "fit_manifest_path": str(fit_manifest.resolve()),
        "fit_manifest_sha256": fit_manifest_sha,
        "fit_manifest_file_sha256": sha256_file(fit_manifest),
        "development_count": len(job.development_ids),
        "development_manifest_path": str(development_manifest.resolve()),
        "development_manifest_sha256": development_manifest_sha,
        "development_manifest_file_sha256": sha256_file(development_manifest),
        "intersection_count": len(development_overlap),
        "intersection_ids": development_overlap,
        "identity_sha256": job.identity_sha256,
    }
    audit_path = job.output_dir / "post_training_leakage_audit.json"
    _atomic_json(audit_path, audit_payload)
    _mirror_artifacts(job, _published_artifact_names(job))
    return audit_path


def _write_checkpoint_provenance(job: P2ReferenceJob, checkpoint: Path, audit_path: Path) -> Path:
    checkpoint = validate_primary_checkpoint(checkpoint)
    provenance = {
        "schema_version": 1,
        "checkpoint_role": "last.pt",
        "primary_checkpoint_role": "last.pt",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "last_pt_sha256": sha256_file(checkpoint),
        "model_sha256": job.identity.model_sha256,
        "pretrained_sha256": job.identity.pretrained_sha256,
        "fit_manifest_sha256": job.identity.fit_ids_sha256,
        "development_manifest_sha256": job.identity.development_ids_sha256,
        "config_sha256": job.identity.config_sha256,
        "code_sha256": job.identity.code_sha256,
        "identity_sha256": job.identity_sha256,
        "post_training_audit_path": str(audit_path.resolve()),
        "post_training_audit_sha256": sha256_file(audit_path),
    }
    provenance_path = job.output_dir / "checkpoint_provenance.json"
    _atomic_json(provenance_path, provenance)
    _mirror_artifacts(job, _published_artifact_names(job))
    return provenance_path


def _write_status(job: P2ReferenceJob, payload: dict[str, object]) -> None:
    _replace_bytes(
        job.status_path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    _mirror_artifacts(job, _published_artifact_names(job))


def _progress_monitor(
    job: P2ReferenceJob,
    *,
    identity: P2ReferenceIdentity,
    identity_sha256: str,
    started_at: str,
    start_clock: float,
    stop: threading.Event,
    errors: list[BaseException],
    reference_label: str = "P2",
) -> None:
    """Flush status and mirror after each complete trainer epoch."""

    observed = -1
    observed_checkpoint_mtime = 0
    results_path = job.output_dir / "results.csv"
    while not stop.wait(5.0):
        if not results_path.is_file():
            continue
        try:
            completed = _completed_epochs(results_path)
        except ValueError:
            # Ultralytics may be between two CSV writes; retry next poll.
            continue
        checkpoint = job.output_dir / "weights" / "last.pt"
        if completed <= observed or not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            continue
        checkpoint_mtime = checkpoint.stat().st_mtime_ns
        if checkpoint_mtime <= observed_checkpoint_mtime:
            continue
        observed = completed
        observed_checkpoint_mtime = checkpoint_mtime
        try:
            _write_status(
                job,
                _status_payload(
                    state="running",
                    identity=identity,
                    identity_sha256=identity_sha256,
                    current_epoch=completed,
                    next_action=f"continue fit-only {reference_label} training",
                    checkpoint=checkpoint,
                    started_at=started_at,
                    elapsed_seconds=time.monotonic() - start_clock,
                    reference_label=reference_label,
                ),
            )
        except BaseException as error:
            errors.append(error)
            stop.set()
            return


def run_p2_fit_reference(
    config: BaselineConfig,
    *,
    repository_root: Path,
    output_dir: Path,
    mirror_dir: Path,
    fit_ids: Path,
    development_ids: Path,
    mode: Literal["dry-run", "smoke", "full"] = "dry-run",
    device: str | None = None,
    resume: bool = False,
    services: P2ReferenceServices | None = None,
    expected_fit_count: int = REGISTERED_FIT_COUNT,
    expected_development_count: int = REGISTERED_DEVELOPMENT_COUNT,
    expected_fit_sha256: str = REGISTERED_FIT_IDS_SHA256,
    expected_development_sha256: str = REGISTERED_DEVELOPMENT_IDS_SHA256,
    identity: P2ReferenceIdentity | None = None,
    reference_kind: Literal["p2", "p3p5"] = "p2",
    smoke_imgsz: int = 320,
    benchmark_context: Any | None = None,
    execution_purpose: str = REFERENCE_EXECUTION_PURPOSE,
) -> P2ReferenceJob:
    if mode not in {"dry-run", "smoke", "full"}:
        raise ValueError(f"unknown reference mode: {mode}")
    root = Path(repository_root).resolve()
    _validate_stage9_reference_contract(
        config,
        mode=mode,
        benchmark_context=benchmark_context,
        execution_purpose=execution_purpose,
    )
    if benchmark_context is not None:
        from ifdr_yolo.experiments.kitti_seed0_training_benchmark import BenchmarkRunContext
        if not isinstance(benchmark_context, BenchmarkRunContext):
            raise ValueError("benchmark_context must be a BenchmarkRunContext")
        if config.experiment.seed != 0:
            raise ValueError("benchmark context requires seed 0")
    bootstrap_ultralytics_config(root)
    dependencies = services or P2ReferenceServices(adapter=UltralyticsAdapter())
    runtime = dependencies.adapter
    reference_label = "P2" if reference_kind == "p2" else "P3-P5"
    if mode == "full":
        if dependencies.preflight is not None:
            dependencies.preflight(
                config=config,
                adapter=runtime,
                repository_root=root,
                device_override=device,
            )
        else:
            # Reuse the baseline's authoritative dataset/model/runtime gate;
            # do not maintain a second, weaker copy here.
            from ifdr_yolo.experiments.baseline import _default_services, _preflight

            _preflight(
                config,
                mode="full",
                adapter=runtime,
                repository_root=root,
                services=_default_services(),
                device_override=device,
            )
    job = prepare_p2_fit_reference(
        config,
        repository_root=root,
        output_dir=output_dir,
        mirror_dir=mirror_dir,
        fit_ids=fit_ids,
        development_ids=development_ids,
        expected_fit_count=expected_fit_count,
        expected_development_count=expected_development_count,
        expected_fit_sha256=expected_fit_sha256,
        expected_development_sha256=expected_development_sha256,
        identity=identity,
        pid_alive=dependencies.pid_alive,
        hostname=dependencies.hostname,
        benchmark_context=benchmark_context,
        reference_kind=reference_kind,
        run_mode=(
            mode
            if reference_kind == "p3p5"
            or execution_purpose in {
                STAGE9_REFERENCE_EXECUTION_PURPOSE,
                STAGE11_REFERENCE_EXECUTION_PURPOSE,
                FORMAL_RERUN_REFERENCE_EXECUTION_PURPOSE,
            }
            else "legacy"
        ),
        execution_purpose=execution_purpose,
    )
    if mode == "dry-run":
        return job
    if job.state == "prepared":
        if resume:
            raise ValueError("--resume is only valid for a stale interrupted reference job")
    elif job.state in {"failed", "interrupted", "running"}:
        if not resume:
            raise ValueError("existing reference job requires --resume; fresh training is refused")
        if not job.resumable:
            raise ValueError("existing reference job is not resumable: last.pt/results.csv are incomplete")
    elif job.state in {"trained", "evaluating", "complete"}:
        raise ValueError(f"reference job is already {job.state}; refusing duplicate training")
    else:
        raise ValueError(f"unsupported reference job state: {job.state}")
    stage11_recovery_root: Path | None = None
    if execution_purpose in {
        STAGE11_REFERENCE_EXECUTION_PURPOSE,
        FORMAL_RERUN_REFERENCE_EXECUTION_PURPOSE,
    }:
        from ifdr_yolo.experiments.stage11_full_recovery import validate_stage11_local_recovery_root
        stage11_recovery_root = job.output_dir.with_name(f"{job.output_dir.name}.stage11-recovery")
        validate_stage11_local_recovery_root(stage11_recovery_root, resume=resume)
    started = dependencies.now().astimezone(timezone.utc).isoformat()
    start_clock = time.monotonic()
    last = job.output_dir / "weights" / "last.pt"
    before_epochs = _completed_epochs(job.output_dir / "results.csv") if job.resumable else 0
    _write_status(
        job,
        _status_payload(
            state="running",
            identity=job.identity,
            identity_sha256=job.identity_sha256,
            current_epoch=before_epochs,
            next_action=(
                f"resume fit-only {reference_label} training"
                if job.resumable
                else f"train fit-only {reference_label}"
            ),
            checkpoint=last if last.is_file() else None,
            started_at=started,
            reference_label=reference_label,
        ),
    )
    progress_stop = threading.Event()
    progress_errors: list[BaseException] = []
    progress_thread = threading.Thread(
        target=_progress_monitor,
        kwargs={
            "job": job,
            "identity": job.identity,
            "identity_sha256": job.identity_sha256,
            "started_at": started,
            "start_clock": start_clock,
            "stop": progress_stop,
            "errors": progress_errors,
            "reference_label": reference_label,
        },
        name="p2-reference-progress",
        daemon=True,
    )
    progress_thread.start()

    def stop_progress() -> None:
        progress_stop.set()
        progress_thread.join(timeout=10.0)

    try:
        if job.resumable:
            if benchmark_context is not None:
                from ultralytics import YOLO
                from ifdr_yolo.experiments.kitti_seed0_training_benchmark import configure_benchmark_callbacks, prepare_resume_checkpoint
                resumed_checkpoint = prepare_resume_checkpoint(
                    benchmark_context.primary_root, benchmark_context.mirror_root, benchmark_context.identity, last, ambient_seed=999,
                )
                model = YOLO(str(resumed_checkpoint))
                configure_benchmark_callbacks(model, benchmark_context, resume=True)
                def resume_fn(checkpoint: Path, job: P2ReferenceJob, *, device: str, workers: int, data_yaml: Path) -> None:
                    model.train(resume=True, data=str(data_yaml), project=str(job.output_dir.parent), name=job.output_dir.name, exist_ok=True, device=device, workers=workers)
            elif stage11_recovery_root is not None:
                from ultralytics import YOLO
                from ifdr_yolo.experiments.stage11_full_recovery import configure_stage11_local_recovery
                model = YOLO(str(last))
                configure_stage11_local_recovery(
                    model,
                    job.output_dir,
                    stage11_recovery_root,
                    execution_identity_sha256=job.identity_sha256,
                    resume=True,
                )

                def resume_fn(checkpoint: Path, job: P2ReferenceJob, *, device: str, workers: int, data_yaml: Path) -> None:
                    model.train(resume=True, data=str(data_yaml), project=str(job.output_dir.parent), name=job.output_dir.name, exist_ok=True, device=device, workers=workers)
            else:
                resume_fn = dependencies.resume_training
                if resume_fn is None:
                    resume_fn = _default_resume_training
            resume_result = resume_fn(
                last,
                job,
                device=device or config.training.device,
                workers=config.training.workers,
                data_yaml=job.resolved_data_yaml,
            )
            if resume_result is not None:
                _verify_trainer_output(resume_result, job)
        else:
            prepared = runtime.prepare_model(
                model_path=config.paths.model,
                model_sha256=job.identity.model_sha256,
                initialization=config.initialization,
                seed=config.experiment.seed,
                deterministic=config.training.deterministic,
            )
            args = _jsonable(asdict(config.training))
            assert isinstance(args, dict)
            args.update({"device": device or config.training.device, "seed": config.experiment.seed, "val": True, "save": True, "plots": True, "pretrained": False, "save_period": -1})
            if benchmark_context is not None:
                args["epochs"] = benchmark_context.identity["effective_epochs"]
                if reference_kind == "p3p5":
                    args.update({"imgsz": 640, "batch": 16, "workers": 8, "amp": True})
                configure = __import__("ifdr_yolo.experiments.kitti_seed0_training_benchmark", fromlist=["configure_benchmark_callbacks"]).configure_benchmark_callbacks
                configure(prepared.handle, benchmark_context, resume=False)
            elif stage11_recovery_root is not None:
                from ifdr_yolo.experiments.stage11_full_recovery import configure_stage11_local_recovery
                configure_stage11_local_recovery(
                    prepared.handle,
                    job.output_dir,
                    stage11_recovery_root,
                    execution_identity_sha256=job.identity_sha256,
                    resume=False,
                )
            elif mode == "smoke":
                args.update(
                    {
                        "epochs": 1,
                        "imgsz": smoke_imgsz,
                        "batch": 2,
                        "workers": 0,
                        "amp": config.training.amp if reference_kind == "p3p5" else False,
                    }
                )
            train_result = runtime.train(
                prepared_model=prepared,
                data_path=job.resolved_data_yaml,
                run_dir=job.output_dir,
                args=args,
            )
            if progress_errors:
                raise RuntimeError("progress mirror failed") from progress_errors[0]
            _verify_trainer_output(train_result, job)
        if progress_errors:
            raise RuntimeError("progress mirror failed") from progress_errors[0]
        stop_progress()
        last = validate_primary_checkpoint(last)
        completed = _completed_epochs(job.output_dir / "results.csv")
        audit_path = _audit_training_inputs(job)
        _write_checkpoint_provenance(job, last, audit_path)
        _write_status(
            job,
            _status_payload(
                state="trained",
                identity=job.identity,
                identity_sha256=job.identity_sha256,
                current_epoch=completed,
                next_action=f"evaluate {reference_label} development split",
                checkpoint=last,
                started_at=started,
                elapsed_seconds=time.monotonic() - start_clock,
                reference_label=reference_label,
            ),
        )
        image_dir = job.output_dir / "view" / "images" / "val"
        labels_dir = runtime.predict(
            weights=last,
            image_paths=tuple(image_dir / f"{image_id}.png" for image_id in job.development_ids),
            output_dir=job.output_dir / "predictions",
            args={
                "device": device or config.training.device,
                "imgsz": config.training.imgsz,
                "conf": config.prediction.conf,
                "iou": config.prediction.iou,
                "max_det": config.prediction.max_det,
                "augment": False,
                "verbose": False,
            },
        )
        ensure_prediction_files(labels_dir, job.development_ids)
        metrics = dependencies.evaluate(prediction_dir=labels_dir, label_dir=config.paths.raw_labels, image_dir=config.paths.raw_images, split_path=job.output_dir / "development_ids.txt")
        metrics_path = job.output_dir / "metrics_ap40.json"
        if reference_kind == "p3p5":
            _atomic_json(metrics_path, metrics)
            split_sha = sha256_file(job.output_dir / "development_ids.txt")
            primary = extract_primary_metric(
                metrics,
                split_sha256=split_sha,
                checkpoint_sha256=sha256_file(last),
                split_count=len(job.development_ids),
                identity_sha256=job.identity_sha256,
            )
            _atomic_json(job.output_dir / "primary_metric.json", primary)
        else:
            write_evaluation_json(metrics_path, metrics)
        _write_status(
            job,
            _status_payload(
                state="complete",
                identity=job.identity,
                identity_sha256=job.identity_sha256,
                current_epoch=completed,
                next_action=(
                    "use last.pt for the resolution oracle"
                    if reference_kind == "p2"
                    else "use last.pt for the plain P3-P5 baseline"
                ),
                checkpoint=last,
                started_at=started,
                elapsed_seconds=time.monotonic() - start_clock,
                reference_label=reference_label,
            ),
        )
        return job
    except BaseException as error:
        stop_progress()
        try:
            current = _completed_epochs(job.output_dir / "results.csv")
        except ValueError:
            current = 0
        _write_status(
            job,
            _status_payload(
                state="failed",
                identity=job.identity,
                identity_sha256=job.identity_sha256,
                current_epoch=current,
                next_action=f"resume {reference_label} with the same identity",
                checkpoint=last if last.is_file() else None,
                started_at=started,
                elapsed_seconds=time.monotonic() - start_clock,
                error=error,
                reference_label=reference_label,
            ),
        )
        raise
    finally:
        stop_progress()


def prepare_p3p5_fit_reference(
    config: BaselineConfig,
    *,
    repository_root: Path,
    output_dir: Path,
    mirror_dir: Path,
    fit_ids: Path,
    development_ids: Path,
    run_mode: Literal["dry-run", "smoke", "full"] = "dry-run",
    expected_fit_count: int = REGISTERED_FIT_COUNT,
    expected_development_count: int = REGISTERED_DEVELOPMENT_COUNT,
    expected_fit_sha256: str = REGISTERED_FIT_IDS_SHA256,
    expected_development_sha256: str = REGISTERED_DEVELOPMENT_IDS_SHA256,
    identity: P2ReferenceIdentity | None = None,
    pid_alive: Callable[[int], bool] = _pid_alive,
    hostname: str = socket.gethostname(),
    execution_purpose: str = REFERENCE_EXECUTION_PURPOSE,
) -> P2ReferenceJob:
    """Prepare a strict plain P3--P5 baseline without starting training."""

    return prepare_p2_fit_reference(
        config,
        repository_root=repository_root,
        output_dir=output_dir,
        mirror_dir=mirror_dir,
        fit_ids=fit_ids,
        development_ids=development_ids,
        expected_fit_count=expected_fit_count,
        expected_development_count=expected_development_count,
        expected_fit_sha256=expected_fit_sha256,
        expected_development_sha256=expected_development_sha256,
        identity=identity,
        pid_alive=pid_alive,
        hostname=hostname,
        reference_kind="p3p5",
        run_mode=run_mode,
        execution_purpose=execution_purpose,
    )


def run_p3p5_fit_reference(
    config: BaselineConfig,
    *,
    repository_root: Path,
    output_dir: Path,
    mirror_dir: Path,
    fit_ids: Path,
    development_ids: Path,
    mode: Literal["dry-run", "smoke", "full"] = "dry-run",
    device: str | None = None,
    resume: bool = False,
    services: P2ReferenceServices | None = None,
    expected_fit_count: int = REGISTERED_FIT_COUNT,
    expected_development_count: int = REGISTERED_DEVELOPMENT_COUNT,
    expected_fit_sha256: str = REGISTERED_FIT_IDS_SHA256,
    expected_development_sha256: str = REGISTERED_DEVELOPMENT_IDS_SHA256,
    identity: P2ReferenceIdentity | None = None,
    benchmark_context: Any | None = None,
    execution_purpose: str = REFERENCE_EXECUTION_PURPOSE,
) -> P2ReferenceJob:
    """Run the registered plain P3--P5 baseline through the shared runner."""

    return run_p2_fit_reference(
        config,
        repository_root=repository_root,
        output_dir=output_dir,
        mirror_dir=mirror_dir,
        fit_ids=fit_ids,
        development_ids=development_ids,
        mode=mode,
        device=device,
        resume=resume,
        services=services,
        expected_fit_count=expected_fit_count,
        expected_development_count=expected_development_count,
        expected_fit_sha256=expected_fit_sha256,
        expected_development_sha256=expected_development_sha256,
        identity=identity,
        reference_kind="p3p5",
        smoke_imgsz=640,
        benchmark_context=benchmark_context,
        execution_purpose=execution_purpose,
    )


P3P5ReferenceIdentity = P2ReferenceIdentity
P3P5ReferenceJob = P2ReferenceJob
P3P5ReferenceServices = P2ReferenceServices


def _completed_epochs(path: Path) -> int:
    if not path.is_file():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= 1:
        return 0
    epochs: list[int] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        try:
            epochs.append(int(float(line.split(",", 1)[0])))
        except (ValueError, IndexError) as error:
            raise ValueError("results.csv contains invalid epoch rows") from error
    if not epochs:
        return 0
    start = epochs[0]
    if epochs != list(range(start, start + len(epochs))):
        raise ValueError("results.csv epochs must be contiguous")
    return len(epochs)


__all__ = [
    "ActiveReferenceRunError",
    "P2ReferenceIdentity",
    "P2ReferenceIdentityError",
    "P2ReferenceJob",
    "P2ReferenceServices",
    "P3P5ReferenceIdentity",
    "P3P5ReferenceJob",
    "P3P5ReferenceServices",
    "REGISTERED_DEVELOPMENT_COUNT",
    "REGISTERED_DEVELOPMENT_IDS_SHA256",
    "REGISTERED_FIT_COUNT",
    "REGISTERED_FIT_IDS_SHA256",
    "REGISTERED_FULL_TRAIN_COUNT",
    "REGISTERED_P2_MODEL_SHA256",
    "REGISTERED_PRETRAINED_SHA256",
    "REGISTERED_P3P5_MODEL_ROLE",
    "REGISTERED_P3P5_MODEL_SHA256",
    "build_reference_identity",
    "extract_primary_metric",
    "prepare_p3p5_fit_reference",
    "prepare_p2_fit_reference",
    "run_p3p5_fit_reference",
    "run_p2_fit_reference",
    "validate_fit_development_split",
    "validate_plain_p2_model",
    "validate_plain_p3p5_model",
    "validate_primary_checkpoint",
]
