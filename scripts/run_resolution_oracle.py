"""Run the staged, leakage-free value-of-resolution oracle.

The runner is intentionally conservative: every image is a recovery unit,
ground truth is read only by O1/O2 selection, and O2's candidate pool is
atomically frozen before that read is possible.  The module is also usable in
tests with a small fake adapter; no Ultralytics-specific mAP is used here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import time
from typing import Any

import yaml
from PIL import Image

from ifdr_yolo.data.kitti_types import BoundingBox, Detection, EVAL_CLASSES, TRAIN_CLASS_TO_ID
from ifdr_yolo.data.splits import load_ids, sha256_file
from ifdr_yolo.eval.evaluate import evaluate_prediction_directory
from ifdr_yolo.eval.prediction_io import load_kitti_ground_truth, load_yolo_predictions
from ifdr_yolo.eval.resolution_oracle import (
    CropWindow,
    ImageSize,
    OracleCandidate,
    UtilityComponents,
    box_to_normalized_yolo,
    build_o1_candidates,
    build_o2_candidate_pool,
    crop_to_full_box,
    fuse_detections,
    moderate_utility,
)
from ifdr_yolo.experiments.baseline import ensure_prediction_files
from ifdr_yolo.experiments.ultralytics_runtime import UltralyticsAdapter, bootstrap_ultralytics_config


REGISTERED_DEVELOPMENT_COUNT = 371
REGISTERED_DEVELOPMENT_IDS_SHA256 = (
    "b1b6b6ee7e5398e93868fab407a2e8a86a53c753667002ef9b8381734ef2cda8"
)


class OracleIdentityError(ValueError):
    """Raised when a fixed-output oracle cannot be proven reproducible."""


class ActiveOracleRunError(RuntimeError):
    """Raised when a live process owns a fixed-output oracle directory."""


@dataclass(frozen=True)
class OraclePaths:
    model: Path
    checkpoint: Path
    development_ids: Path
    raw_images: Path
    raw_labels: Path
    model_sha256: str
    checkpoint_sha256: str | None
    fit_ids: Path | None = None
    fit_manifest_sha256: str | None = None


@dataclass(frozen=True)
class OracleRules:
    image_size: int = 640
    crop_fraction: float = 0.5
    small_height_px: float = 40.0
    proposal_confidence: tuple[float, float] = (0.001, 0.25)
    proposal_limit: int = 18
    max_crops_per_image: int = 1
    nms_iou: float = 0.70
    max_det: int = 300
    o1_min_delta_ap40: float = 3.0
    o2_min_delta_ap40: float = 5.0
    no_harm_near_large: float = 0.5
    no_harm_class: float = 1.0


@dataclass(frozen=True)
class OracleConfig:
    schema_version: int
    dataset: str
    model_role: str
    seed: int
    checkpoint_role: str
    paths: OraclePaths
    rules: OracleRules
    source_path: Path


@dataclass(frozen=True)
class ResolutionOracleServices:
    """Injectable boundaries for deterministic tests and the real runner."""

    adapter: Any
    evaluate: Callable[..., dict[str, object]] = evaluate_prediction_directory
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    pid_alive: Callable[[int], bool] = lambda pid: _pid_alive(pid)
    hostname: str = socket.gethostname()


@dataclass(frozen=True)
class ResolutionOracleResult:
    output_dir: Path
    mirror_dir: Path
    state: str
    decision_path: Path
    identity_sha256: str


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _oracle_code_hash(repository_root: Path) -> str:
    """Hash the commit and every source file that can change oracle semantics."""

    digest = hashlib.sha256()
    try:
        import subprocess

        commit = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        commit = ""
    digest.update(commit.encode("utf-8"))
    for path in (
        Path(__file__),
        repository_root / "ifdr_yolo" / "eval" / "resolution_oracle.py",
        repository_root / "ifdr_yolo" / "eval" / "evaluate.py",
        repository_root / "ifdr_yolo" / "eval" / "prediction_io.py",
    ):
        if path.is_file():
            digest.update(path.resolve().as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, CropWindow):
        return list(value.as_xyxy())
    if isinstance(value, OracleCandidate):
        return _candidate_payload(value)
    if isinstance(value, UtilityComponents):
        return asdict(value) | {"utility": value.utility}
    if isinstance(value, Detection):
        return {
            "image_id": value.image_id,
            "kind": value.kind,
            "score": value.score,
            "bbox": list(value.bbox.as_xyxy()),
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # Windows treats os.kill(pid, 0) inconsistently (and some runners
        # interpret it as termination).  Query the process handle instead.
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return False
            kernel32.CloseHandle(handle)
            return True
        except (AttributeError, OSError):
            return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _required_file(path: Path, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"{label} must be a non-empty regular file: {path}")
    return path


def _resolve_path(value: object, field: str, root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _sha(value: object, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError(f"{field} must be a 64-hex SHA256")
    return value.lower()


def load_oracle_config(path: Path, *, repository_root: Path | None = None) -> OracleConfig:
    """Load the registered oracle protocol without accepting threshold overrides."""

    path = Path(path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("oracle config must have schema_version=1")
    # configs/experiments/oracles/<name>.yaml -> repository root is parents[3]
    root = (repository_root or path.parents[3]).resolve()
    experiment = payload.get("experiment")
    paths = payload.get("paths")
    raw_rules = payload.get("oracle")
    if not isinstance(experiment, dict) or not isinstance(paths, dict) or not isinstance(raw_rules, dict):
        raise ValueError("oracle config requires experiment, paths and oracle mappings")
    dataset = experiment.get("dataset")
    model_role = experiment.get("model_role")
    checkpoint_role = experiment.get("checkpoint_role")
    seed = experiment.get("seed")
    if dataset != "kitti" or model_role != "plain_p2" or checkpoint_role != "last.pt" or not isinstance(seed, int):
        raise ValueError("oracle config must identify the plain-P2 last.pt KITTI reference")
    model_sha = _sha(paths.get("model_sha256"), "paths.model_sha256")
    checkpoint_sha = _sha(paths.get("checkpoint_sha256"), "paths.checkpoint_sha256", nullable=True)
    fit_sha = _sha(paths.get("fit_manifest_sha256"), "paths.fit_manifest_sha256", nullable=True)
    assert isinstance(model_sha, str)
    confidence = raw_rules.get("proposal_confidence", [0.001, 0.25])
    if not isinstance(confidence, (list, tuple)) or len(confidence) != 2:
        raise ValueError("oracle.proposal_confidence must contain two values")
    low, high = float(confidence[0]), float(confidence[1])
    rules = OracleRules(
        image_size=int(raw_rules.get("image_size", 640)),
        crop_fraction=float(raw_rules.get("crop_fraction", 0.5)),
        small_height_px=float(raw_rules.get("small_height_px", 40.0)),
        proposal_confidence=(low, high),
        proposal_limit=int(raw_rules.get("proposal_limit", 18)),
        max_crops_per_image=int(raw_rules.get("max_crops_per_image", 1)),
        nms_iou=float(raw_rules.get("nms_iou", 0.70)),
        max_det=int(raw_rules.get("max_det", 300)),
        o1_min_delta_ap40=float(raw_rules.get("o1_min_delta_ap40", 3.0)),
        o2_min_delta_ap40=float(raw_rules.get("o2_min_delta_ap40", 5.0)),
        no_harm_near_large=float(raw_rules.get("no_harm_near_large", 0.5)),
        no_harm_class=float(raw_rules.get("no_harm_class", 1.0)),
    )
    if rules.crop_fraction != 0.5 or rules.max_crops_per_image != 1:
        raise ValueError("oracle crop fraction and one-crop budget are registered constants")
    if rules.proposal_confidence != (0.001, 0.25) or rules.proposal_limit != 18:
        raise ValueError("oracle proposal policy is registered and cannot be overridden")
    if rules.nms_iou != 0.70 or rules.max_det != 300:
        raise ValueError("oracle NMS policy is registered and cannot be overridden")
    if (
        rules.image_size != 640
        or rules.small_height_px != 40.0
        or rules.o1_min_delta_ap40 != 3.0
        or rules.o2_min_delta_ap40 != 5.0
        or rules.no_harm_near_large != 0.5
        or rules.no_harm_class != 1.0
    ):
        raise ValueError("oracle score and no-harm gates are registered constants")
    return OracleConfig(
        schema_version=1,
        dataset="kitti",
        model_role="plain_p2",
        seed=seed,
        checkpoint_role="last.pt",
        paths=OraclePaths(
            model=_resolve_path(paths.get("model"), "paths.model", root),
            checkpoint=_resolve_path(paths.get("checkpoint"), "paths.checkpoint", root),
            development_ids=_resolve_path(paths.get("development_ids"), "paths.development_ids", root),
            raw_images=_resolve_path(paths.get("raw_images"), "paths.raw_images", root),
            raw_labels=_resolve_path(paths.get("raw_labels"), "paths.raw_labels", root),
            model_sha256=model_sha,
            checkpoint_sha256=checkpoint_sha,
            fit_ids=(None if paths.get("fit_ids") is None else _resolve_path(paths.get("fit_ids"), "paths.fit_ids", root)),
            fit_manifest_sha256=fit_sha,
        ),
        rules=rules,
        source_path=path,
    )


def _candidate_payload(candidate: OracleCandidate) -> dict[str, object]:
    return {
        "source": candidate.source,
        "rank": candidate.rank,
        "window": list(candidate.window.as_xyxy()),
        "proposal_score": candidate.proposal_score,
    }


def _candidate_from_payload(payload: Mapping[str, object]) -> OracleCandidate:
    window = payload.get("window")
    if not isinstance(window, (list, tuple)) or len(window) != 4:
        raise ValueError("candidate window must contain four coordinates")
    return OracleCandidate(
        window=CropWindow(*(float(value) for value in window)),
        source=str(payload["source"]),
        rank=int(payload["rank"]),
        proposal_score=(None if payload.get("proposal_score") is None else float(payload["proposal_score"])),
    )


def _image_sizes(image_dir: Path, image_ids: Sequence[str]) -> dict[str, ImageSize]:
    result: dict[str, ImageSize] = {}
    for image_id in image_ids:
        path = image_dir / f"{image_id}.png"
        if not path.is_file():
            raise FileNotFoundError(f"development image missing: {path}")
        with Image.open(path) as image:
            result[image_id] = ImageSize(*image.size)
    return result


def _read_predictions(directory: Path, image_sizes: Mapping[str, ImageSize]) -> dict[str, tuple[Detection, ...]]:
    return load_yolo_predictions(directory, {key: (value.width, value.height) for key, value in image_sizes.items()})


def _write_detection_file(path: Path, image_size: ImageSize, detections: Sequence[Detection]) -> None:
    lines: list[str] = []
    for detection in detections:
        class_id = TRAIN_CLASS_TO_ID.get(detection.kind)
        if class_id is None:
            continue
        values = box_to_normalized_yolo(detection.bbox, image_size)
        lines.append(f"{class_id} " + " ".join(f"{value:.8f}" for value in (*values, detection.score)))
    _atomic_write(path, ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))


def _write_prediction_directory(
    root: Path,
    predictions: Mapping[str, Sequence[Detection]],
    image_sizes: Mapping[str, ImageSize],
) -> Path:
    labels = root / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    for image_id in sorted(image_sizes):
        _write_detection_file(labels / f"{image_id}.txt", image_sizes[image_id], predictions.get(image_id, ()))
    ensure_prediction_files(labels, tuple(sorted(image_sizes)))
    return labels


def _map_crop_predictions(
    image_id: str,
    crop_predictions: Sequence[Detection],
    crop: CropWindow,
    image_size: ImageSize,
) -> tuple[Detection, ...]:
    return tuple(
        Detection(
            image_id=image_id,
            kind=detection.kind,
            score=detection.score,
            bbox=crop_to_full_box(detection.bbox, crop, image_size=image_size),
        )
        for detection in crop_predictions
    )


def _capture_crop(source: Path, crop: CropWindow, destination: Path) -> ImageSize:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        coordinates = crop.as_xyxy()
        if any(value != int(value) for value in coordinates):
            raise ValueError("crop geometry must provide integer pixel bounds")
        box = tuple(int(value) for value in coordinates)
        cropped = image.crop(box)
        cropped.save(destination, format="PNG")
        return ImageSize(*cropped.size)


def _copy_atomic(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    _atomic_write(destination, source.read_bytes())


def _reject_symlink_ancestors(path: Path) -> None:
    current = path
    while True:
        if current.exists() and current.is_symlink():
            raise OracleIdentityError(f"oracle path may not traverse a symlink: {current}")
        if current.parent == current:
            break
        current = current.parent


class _OracleRunner:
    def __init__(self, config: OracleConfig, *, repository_root: Path, output_dir: Path, mirror_dir: Path, services: ResolutionOracleServices, resume: bool) -> None:
        self.config = config
        self.root = repository_root.resolve()
        _reject_symlink_ancestors(Path(output_dir))
        _reject_symlink_ancestors(Path(mirror_dir))
        self.output = output_dir.resolve()
        self.mirror = mirror_dir.resolve()
        self.services = services
        self.resume = resume
        if self.output == self.mirror or self.output.is_relative_to(self.mirror) or self.mirror.is_relative_to(self.output):
            raise OracleIdentityError("primary and mirror directories must be separate")
        self.output.mkdir(parents=True, exist_ok=True)
        self.mirror.mkdir(parents=True, exist_ok=True)
        self.image_ids = load_ids(config.paths.development_ids)
        if len(self.image_ids) != REGISTERED_DEVELOPMENT_COUNT:
            raise OracleIdentityError(f"development split must contain {REGISTERED_DEVELOPMENT_COUNT} images")
        actual_split_sha = sha256_file(config.paths.development_ids)
        if actual_split_sha != REGISTERED_DEVELOPMENT_IDS_SHA256:
            raise OracleIdentityError("development manifest SHA256 does not match the registered split")
        self.image_sizes = _image_sizes(config.paths.raw_images, self.image_ids)
        self.ground_truth = load_kitti_ground_truth(config.paths.raw_labels, self.image_ids)
        self.identity = self._build_identity()
        self.identity_sha = _payload_sha256(self.identity)
        self.status_path = self.output / "status.json"
        self.identity_path = self.output / "run_identity.json"
        self.prior_elapsed = 0.0
        self.resume_phase = "base"
        self.resume_completed: list[str] = []
        self._load_or_initialize_identity()

    def _build_identity(self) -> dict[str, object]:
        checkpoint = _required_file(self.config.paths.checkpoint, "plain-P2 checkpoint")
        if checkpoint.name != self.config.checkpoint_role or checkpoint.name != "last.pt":
            raise OracleIdentityError("oracle checkpoint role must be last.pt")
        checkpoint_sha = sha256_file(checkpoint)
        expected = self.config.paths.checkpoint_sha256
        if expected is None:
            raise OracleIdentityError("formal oracle run requires checkpoint_sha256 in the frozen config")
        if checkpoint_sha != expected:
            raise OracleIdentityError("checkpoint SHA256 does not match oracle config")
        model = _required_file(self.config.paths.model, "plain-P2 model")
        model_sha = sha256_file(model)
        if model_sha != self.config.paths.model_sha256:
            raise OracleIdentityError("plain-P2 model SHA256 does not match oracle config")
        provenance_root = checkpoint.parent.parent if checkpoint.parent.name == "weights" else checkpoint.parent
        provenance_path = provenance_root / "checkpoint_provenance.json"
        audit_path = provenance_root / "post_training_leakage_audit.json"
        if not provenance_path.is_file() or not audit_path.is_file():
            raise OracleIdentityError(
                "plain-P2 checkpoint requires adjacent checkpoint_provenance.json "
                "and post_training_leakage_audit.json"
            )
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise OracleIdentityError("reference provenance is not valid JSON") from error
        if provenance.get("checkpoint_role") != "last.pt" or provenance.get("primary_checkpoint_role") != "last.pt":
            raise OracleIdentityError("checkpoint provenance role must be last.pt")
        if provenance.get("checkpoint_sha256") != checkpoint_sha or provenance.get("last_pt_sha256") != checkpoint_sha:
            raise OracleIdentityError("checkpoint provenance SHA does not match last.pt")
        if provenance.get("model_sha256") != model_sha:
            raise OracleIdentityError("checkpoint provenance model SHA mismatch")
        if provenance.get("development_manifest_sha256") != sha256_file(self.config.paths.development_ids):
            raise OracleIdentityError("checkpoint provenance development manifest mismatch")
        expected_fit_sha = self.config.paths.fit_manifest_sha256
        if expected_fit_sha is None or provenance.get("fit_manifest_sha256") != expected_fit_sha:
            raise OracleIdentityError("checkpoint provenance fit manifest mismatch")
        if self.config.paths.fit_ids is None or not self.config.paths.fit_ids.is_file() or sha256_file(self.config.paths.fit_ids) != expected_fit_sha:
            raise OracleIdentityError("registered fit manifest is missing or changed")
        if audit.get("intersection_count") != 0:
            raise OracleIdentityError("fit/development intersection_count must equal zero")
        if audit.get("fit_manifest_sha256") != expected_fit_sha or audit.get("development_manifest_sha256") != sha256_file(self.config.paths.development_ids):
            raise OracleIdentityError("post-training leakage audit split identity mismatch")
        if provenance.get("post_training_audit_sha256") != sha256_file(audit_path):
            raise OracleIdentityError("checkpoint provenance audit SHA mismatch")
        audit_identity = audit.get("identity_sha256")
        provenance_identity = provenance.get("identity_sha256")
        if not isinstance(audit_identity, str) or len(audit_identity) != 64 or audit_identity != provenance_identity:
            raise OracleIdentityError("reference provenance identity is missing or inconsistent")
        config_sha = sha256_file(self.config.source_path)
        return {
            "dataset": self.config.dataset,
            "model_role": self.config.model_role,
            "model_path": str(model),
            "model_sha256": model_sha,
            "checkpoint_path": str(checkpoint),
            "checkpoint_role": self.config.checkpoint_role,
            "checkpoint_sha256": checkpoint_sha,
            "code_sha256": _oracle_code_hash(self.root),
            "config_sha256": config_sha,
            "development_ids_path": str(self.config.paths.development_ids),
            "development_ids_sha256": sha256_file(self.config.paths.development_ids),
            "development_count": len(self.image_ids),
            "seed": self.config.seed,
            "reference_provenance_sha256": sha256_file(provenance_path),
            "post_training_audit_sha256": sha256_file(audit_path),
            "reference_identity_sha256": audit_identity,
        }

    def _load_or_initialize_identity(self) -> None:
        if self.identity_path.exists():
            payload = json.loads(self.identity_path.read_text(encoding="utf-8"))
            if payload.get("identity") != self.identity or payload.get("identity_sha256") != self.identity_sha:
                raise OracleIdentityError("existing oracle identity differs; resume is fail-closed")
            if not self.status_path.exists():
                raise OracleIdentityError("oracle identity exists without status; refusing ambiguous resume")
            status = json.loads(self.status_path.read_text(encoding="utf-8"))
            state = status.get("state")
            if isinstance(status.get("phase"), str):
                self.resume_phase = str(status["phase"])
            if isinstance(status.get("completed_image_ids"), list):
                self.resume_completed = [str(value) for value in status["completed_image_ids"]]
            raw_elapsed = status.get("elapsed_seconds")
            if isinstance(raw_elapsed, (int, float)) and math.isfinite(float(raw_elapsed)):
                self.prior_elapsed = max(0.0, float(raw_elapsed))
            if state == "complete":
                raise ValueError("oracle job is already complete; refusing duplicate run")
            if state == "running":
                owner = status.get("hostname")
                if isinstance(owner, str) and owner and owner != self.services.hostname:
                    raise OracleIdentityError("running oracle job belongs to a different host")
                raw_pid = status.get("pid")
                pid = int(raw_pid) if isinstance(raw_pid, int) else 0
                if self.services.pid_alive(pid):
                    raise ActiveOracleRunError("an active process owns this oracle job")
            elif state not in {"failed", "interrupted", "prepared"}:
                raise OracleIdentityError(f"unknown oracle job state: {state!r}")
            if not self.resume:
                raise ValueError("existing oracle job requires --resume")
            return
        if self.resume:
            raise OracleIdentityError("--resume requires an existing identity")
        _atomic_json(self.identity_path, {"identity": self.identity, "identity_sha256": self.identity_sha})
        self._mirror_files(("run_identity.json",))

    def _status(self, *, phase: str, state: str, completed: Sequence[str], next_id: str | None, started: float, **extra: object) -> None:
        elapsed = self.prior_elapsed + max(0.0, time.monotonic() - started)
        rate = len(completed) / elapsed if elapsed > 0 else 0.0
        remaining = max(0, len(self.image_ids) - len(completed))
        payload: dict[str, object] = {
            "state": state,
            "phase": phase,
            "identity_sha256": self.identity_sha,
            "completed_image_ids": sorted(set(completed)),
            "next_image_id": next_id,
            "elapsed_seconds": elapsed,
            "rate_images_per_second": rate,
            "eta_seconds": remaining / rate if rate > 0 else None,
            "last_saved_at": self.services.now().astimezone(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "hostname": self.services.hostname,
            **extra,
        }
        _atomic_json(self.status_path, payload)
        self._mirror_files(("run_identity.json", "status.json"))

    def _mirror_files(self, names: Sequence[str]) -> None:
        self.mirror.mkdir(parents=True, exist_ok=True)
        existing: dict[str, dict[str, object]] = {}
        manifest_path = self.mirror / "manifest.json"
        if manifest_path.is_file():
            try:
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
                existing = {
                    str(item["name"]): dict(item)
                    for item in previous.get("files", [])
                    if isinstance(item, Mapping) and isinstance(item.get("name"), str)
                }
                generation = int(previous.get("generation", 0)) + 1
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                raise OracleIdentityError("mirror manifest is malformed") from error
        else:
            generation = 1
        for name in names:
            source = self.output / name
            if not source.is_file():
                continue
            destination = self.mirror / name
            _copy_atomic(source, destination)
            existing[name] = {"name": name, "sha256": sha256_file(source), "size": source.stat().st_size}
        manifest = {"generation": generation, "identity_sha256": self.identity_sha, "files": [existing[name] for name in sorted(existing)]}
        # The manifest is deliberately written last, after all mirror files.
        _atomic_json(self.mirror / "manifest.json", manifest)

    def _journal_path(self, phase: str, image_id: str) -> Path:
        return self.output / "journals" / phase / f"{image_id}.json"

    def _load_journal(self, phase: str, image_id: str) -> dict[str, object] | None:
        path = self._journal_path(phase, image_id)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("identity_sha256") != self.identity_sha:
            raise OracleIdentityError(f"{phase}/{image_id} journal identity mismatch")
        for field in ("raw_prediction", "fused_prediction"):
            value = payload.get(field)
            if value is None:
                continue
            path = Path(str(value))
            if not path.is_absolute():
                path = self.output / path
            expected_sha = payload.get(f"{field}_sha256")
            if not path.is_file() or path.stat().st_size <= 0 or not isinstance(expected_sha, str):
                raise OracleIdentityError(f"{phase}/{image_id} prediction artifact is missing or empty")
            if sha256_file(path) != expected_sha:
                raise OracleIdentityError(f"{phase}/{image_id} prediction artifact SHA mismatch")
        records = payload.get("candidate_records")
        if isinstance(records, list):
            for record in records:
                if not isinstance(record, Mapping):
                    raise OracleIdentityError(f"{phase}/{image_id} candidate record is malformed")
                raw = record.get("raw_prediction")
                raw_sha = record.get("raw_prediction_sha256")
                if not isinstance(raw, str) or not isinstance(raw_sha, str):
                    raise OracleIdentityError(f"{phase}/{image_id} candidate provenance is incomplete")
                raw_path = Path(raw)
                if not raw_path.is_absolute():
                    raw_path = self.output / raw_path
                if not raw_path.is_file() or raw_path.stat().st_size <= 0 or sha256_file(raw_path) != raw_sha:
                    raise OracleIdentityError(f"{phase}/{image_id} candidate prediction SHA mismatch")
        return payload

    def _write_journal(self, phase: str, image_id: str, payload: Mapping[str, object]) -> None:
        body = {"image_id": image_id, "phase": phase, "identity_sha256": self.identity_sha, **dict(payload)}
        journal = self._journal_path(phase, image_id)
        _atomic_json(journal, body)
        names = ["run_identity.json", journal.relative_to(self.output).as_posix(), "status.json"]
        for field in ("raw_prediction", "fused_prediction"):
            value = body.get(field)
            if isinstance(value, str):
                path = Path(value)
                if path.is_absolute():
                    path = path.resolve().relative_to(self.output)
                names.append(path.as_posix())
        self._mirror_files(tuple(dict.fromkeys(names)))

    def _stage_image(self, image_id: str, source: Path, stage: str) -> Path:
        directory = self.output / "staging" / stage / image_id
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{image_id}.png"
        if not target.exists():
            shutil.copy2(source, target)
        return target

    def _predict_image(self, image_id: str, image_path: Path, stage: str) -> dict[str, tuple[Detection, ...]]:
        output = self.output / "predictions" / stage / image_id
        labels = self.services.adapter.predict(
            weights=self.config.paths.checkpoint,
            image_paths=(image_path,),
            output_dir=output,
            args={"device": "0", "imgsz": self.config.rules.image_size, "conf": 0.001, "iou": 0.7, "max_det": 300, "verbose": False},
        )
        ensure_prediction_files(Path(labels), (image_id,))
        with Image.open(image_path) as image:
            size = ImageSize(*image.size)
        return _read_predictions(Path(labels), {image_id: size})

    def _base_phase(self, started: float) -> dict[str, tuple[Detection, ...]]:
        phase = "base"
        raw_root = self.output / "raw" / phase
        raw_root.mkdir(parents=True, exist_ok=True)
        predictions: dict[str, tuple[Detection, ...]] = {}
        done: list[str] = []
        for image_id in self.image_ids:
            journal = self._load_journal(phase, image_id)
            raw = raw_root / f"{image_id}.txt"
            if journal is not None and raw.is_file():
                predictions[image_id] = _read_predictions(raw_root, {image_id: self.image_sizes[image_id]})[image_id]
                done.append(image_id)
                continue
            source = self._stage_image(image_id, self.config.paths.raw_images / f"{image_id}.png", phase)
            result = self._predict_image(image_id, source, phase)
            _copy_atomic(self.output / "predictions" / phase / image_id / "labels" / f"{image_id}.txt", raw)
            predictions[image_id] = result[image_id]
            self._write_journal(
                phase,
                image_id,
                {
                    "raw_prediction": raw.relative_to(self.output).as_posix(),
                    "raw_prediction_sha256": sha256_file(raw),
                    "prediction_complete": True,
                },
            )
            done.append(image_id)
            self._status(phase=phase, state="running", completed=done, next_id=self.image_ids[len(done)] if len(done) < len(self.image_ids) else None, started=started)
        return predictions

    def _candidate_phase(self, phase: str, candidates: Mapping[str, Sequence[OracleCandidate]], started: float) -> tuple[dict[str, tuple[Detection, ...]], dict[str, OracleCandidate | None], dict[str, UtilityComponents]]:
        crop_root = self.output / "raw" / phase
        crop_root.mkdir(parents=True, exist_ok=True)
        mapped: dict[str, tuple[Detection, ...]] = {}
        selected: dict[str, OracleCandidate | None] = {}
        utilities: dict[str, UtilityComponents] = {}
        done: list[str] = []
        base_predictions = _read_predictions(self.output / "raw" / "base", self.image_sizes)
        for image_id in self.image_ids:
            journal = self._load_journal(phase, image_id)
            if journal is not None:
                chosen = journal.get("selected")
                selected[image_id] = None if chosen is None else _candidate_from_payload(chosen)
                utility = journal.get("utility")
                if isinstance(utility, dict):
                    utilities[image_id] = UtilityComponents(float(utility["delta_tp"]), float(utility["delta_mean_iou"]), float(utility["delta_fp"]), float(utility["delta_duplicates"]))
                mapped[image_id] = _read_predictions(self.output / "predictions" / phase / "labels", self.image_sizes).get(image_id, ())
                done.append(image_id)
                continue
            image_candidates = tuple(candidates.get(image_id, ()))
            best: OracleCandidate | None = None
            best_utility = UtilityComponents(0.0, 0.0, 0.0, 0.0)
            best_mapped: tuple[Detection, ...] = ()
            candidate_records: list[dict[str, object]] = []
            for index, candidate in enumerate(image_candidates):
                crop_path = self.output / "crops" / phase / f"{image_id}-{index:03d}.png"
                crop_size = _capture_crop(self.config.paths.raw_images / f"{image_id}.png", candidate.window, crop_path)
                source = self._stage_image(image_id, crop_path, f"{phase}-{index:03d}")
                result = self._predict_image(image_id, source, f"{phase}-{index:03d}")[image_id]
                raw = crop_root / f"{image_id}-{index:03d}.txt"
                _copy_atomic(self.output / "predictions" / f"{phase}-{index:03d}" / image_id / "labels" / f"{image_id}.txt", raw)
                mapped_candidate = _map_crop_predictions(image_id, result, candidate.window, self.image_sizes[image_id])
                fused = fuse_detections(base_predictions[image_id], mapped_candidate, iou_threshold=self.config.rules.nms_iou, max_det=self.config.rules.max_det)
                utility = moderate_utility({image_id: base_predictions[image_id]}, {image_id: fused}, {image_id: self.ground_truth[image_id]})
                candidate_records.append(
                    {
                        "candidate": _candidate_payload(candidate),
                        "raw_prediction": raw.relative_to(self.output).as_posix(),
                        "raw_prediction_sha256": sha256_file(raw),
                        "mapped_prediction": [_jsonable(value) for value in mapped_candidate],
                        "utility": asdict(utility) | {"utility": utility.utility},
                    }
                )
                if utility.utility > best_utility.utility:
                    best, best_utility, best_mapped = candidate, utility, mapped_candidate
            selected[image_id] = best
            utilities[image_id] = best_utility
            fused = fuse_detections(base_predictions[image_id], best_mapped, iou_threshold=self.config.rules.nms_iou, max_det=self.config.rules.max_det) if best is not None else base_predictions[image_id]
            mapped[image_id] = fused
            labels = _write_prediction_directory(self.output / "predictions" / phase, mapped, self.image_sizes)
            chosen_payload = None if best is None else _candidate_payload(best)
            fused_path = labels / f"{image_id}.txt"
            self._write_journal(
                phase,
                image_id,
                {
                    "selected": chosen_payload,
                    "utility": asdict(best_utility) | {"utility": best_utility.utility},
                    "candidate_records": candidate_records,
                    "fused_prediction": fused_path.relative_to(self.output).as_posix(),
                    "fused_prediction_sha256": sha256_file(fused_path),
                    "prediction_complete": True,
                },
            )
            done.append(image_id)
            self._status(phase=phase, state="running", completed=done, next_id=self.image_ids[len(done)] if len(done) < len(self.image_ids) else None, started=started)
        return mapped, selected, utilities

    def _load_frozen_candidate_pool(self, path: Path, expected_sha: str) -> dict[str, tuple[OracleCandidate, ...]]:
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise OracleIdentityError("O2 candidate pool is missing or its SHA changed")
        pools: dict[str, tuple[OracleCandidate, ...]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            image_id = payload.get("image_id")
            raw = payload.get("candidates")
            if not isinstance(image_id, str) or not isinstance(raw, list):
                raise OracleIdentityError("O2 candidate pool line is malformed")
            pools[image_id] = tuple(_candidate_from_payload(item) for item in raw if isinstance(item, Mapping))
        if tuple(sorted(pools)) != tuple(sorted(self.image_ids)):
            raise OracleIdentityError("O2 candidate pool does not cover the development manifest")
        return pools

    def _run_impl(self) -> ResolutionOracleResult:
        started = time.monotonic()
        self.current_started = started
        initial_completed = self.resume_completed if self.resume_phase == "base" else []
        _atomic_json(self.status_path, {"state": "running", "phase": self.resume_phase if self.resume_phase != "evaluate" else "base", "identity_sha256": self.identity_sha, "completed_image_ids": initial_completed, "next_image_id": self.image_ids[len(initial_completed)] if len(initial_completed) < len(self.image_ids) else None, "elapsed_seconds": self.prior_elapsed, "eta_seconds": None, "pid": os.getpid(), "hostname": self.services.hostname})
        base = self._base_phase(started)
        base_labels = _write_prediction_directory(self.output / "predictions" / "base", base, self.image_sizes)
        base_metrics = self.services.evaluate(prediction_dir=base_labels, label_dir=self.config.paths.raw_labels, image_dir=self.config.paths.raw_images, split_path=self.config.paths.development_ids)
        _atomic_json(self.output / "base_metrics_ap40.json", base_metrics)
        o1_candidates = build_o1_candidates(self.ground_truth, self.image_sizes)
        for image_id, values in o1_candidates.items():
            self._write_journal("o1_candidates", image_id, {"candidates": [_candidate_payload(value) for value in values]})
        o1, _, _ = self._candidate_phase("o1_select", o1_candidates, started)
        o1_labels = _write_prediction_directory(self.output / "predictions" / "o1", o1, self.image_sizes)
        o1_metrics = self.services.evaluate(prediction_dir=o1_labels, label_dir=self.config.paths.raw_labels, image_dir=self.config.paths.raw_images, split_path=self.config.paths.development_ids)
        _atomic_json(self.output / "o1_metrics_ap40.json", o1_metrics)
        base_ap = _macro_moderate(base_metrics)
        o1_ap = _macro_moderate(o1_metrics)
        if o1_ap - base_ap < self.config.rules.o1_min_delta_ap40:
            decision = {"route": "A", "o1": {"delta_moderate_macro_ap40": o1_ap - base_ap}, "decision": "STOP", "reason": "O1 below registered +3.0 gate"}
            _atomic_json(self.output / "route_a_decision.json", decision)
            _atomic_json(self.output / "stratified_no_harm.json", {"status": "NOT_RUN", "decision": "STOP"})
            _atomic_json(self.output / "latency_compute.json", {"images": len(self.image_ids), "one_crop_budget": self.config.rules.max_crops_per_image})
            self._mirror_files(("base_metrics_ap40.json", "o1_metrics_ap40.json", "route_a_decision.json", "stratified_no_harm.json", "latency_compute.json"))
            self._status(phase="evaluate", state="complete", completed=self.image_ids, next_id=None, started=started, decision="STOP", candidate_pool_sha256=None)
            return ResolutionOracleResult(self.output, self.mirror, "complete", self.output / "route_a_decision.json", self.identity_sha)
        pools = build_o2_candidate_pool(self.image_sizes, base, confidence_min=self.config.rules.proposal_confidence[0], confidence_max=self.config.rules.proposal_confidence[1], proposal_limit=self.config.rules.proposal_limit)
        pool_lines = [json.dumps({"image_id": image_id, "candidates": [_candidate_payload(value) for value in pools[image_id]]}, sort_keys=True) for image_id in self.image_ids]
        pool_path = self.output / "candidate_pool.jsonl"
        pool_bytes = ("\n".join(pool_lines) + "\n").encode("utf-8")
        if pool_path.is_file() and pool_path.read_bytes() != pool_bytes:
            raise OracleIdentityError("existing O2 candidate pool differs; refusing overwrite")
        _atomic_write(pool_path, pool_bytes)
        pool_sha = sha256_file(pool_path)
        pool_sha_path = self.output / "candidate_pool.sha256"
        if pool_sha_path.is_file() and pool_sha_path.read_text(encoding="utf-8").strip() != pool_sha:
            raise OracleIdentityError("existing O2 candidate pool digest differs")
        _atomic_write(pool_sha_path, (pool_sha + "\n").encode("utf-8"))
        self._mirror_files(("candidate_pool.jsonl", "candidate_pool.sha256"))
        frozen_pools = self._load_frozen_candidate_pool(pool_path, pool_sha)
        o2, _, _ = self._candidate_phase("o2_select", frozen_pools, started)
        o2_labels = _write_prediction_directory(self.output / "predictions" / "o2", o2, self.image_sizes)
        o2_metrics = self.services.evaluate(prediction_dir=o2_labels, label_dir=self.config.paths.raw_labels, image_dir=self.config.paths.raw_images, split_path=self.config.paths.development_ids)
        _atomic_json(self.output / "o2_metrics_ap40.json", o2_metrics)
        o2_ap = _macro_moderate(o2_metrics)
        # The exact stratified evaluator/no-harm slices are a separate frozen
        # gate.  Until they are actually computed, a score delta cannot be
        # promoted to an ``ADVANCE`` claim.
        score_decision = "PENDING_NO_HARM" if o2_ap - base_ap >= self.config.rules.o2_min_delta_ap40 else "STOP"
        decision = {"route": "A", "o1": {"delta_moderate_macro_ap40": o1_ap - base_ap}, "o2": {"delta_moderate_macro_ap40": o2_ap - base_ap}, "decision": score_decision, "candidate_pool_sha256": pool_sha, "gates": {"o1_min": self.config.rules.o1_min_delta_ap40, "o2_min": self.config.rules.o2_min_delta_ap40}}
        _atomic_json(self.output / "route_a_decision.json", decision)
        _atomic_json(self.output / "stratified_no_harm.json", {"status": "PENDING_NO_HARM", "decision": decision["decision"], "required": {"pedestrian_cyclist_small_distant_positive": True, "class_drop_max": self.config.rules.no_harm_class, "near_large_drop_max": self.config.rules.no_harm_near_large}})
        _atomic_json(self.output / "latency_compute.json", {"images": len(self.image_ids), "one_crop_budget": self.config.rules.max_crops_per_image})
        self._mirror_files(("base_metrics_ap40.json", "o1_metrics_ap40.json", "o2_metrics_ap40.json", "candidate_pool.jsonl", "candidate_pool.sha256", "route_a_decision.json", "stratified_no_harm.json", "latency_compute.json"))
        self._status(phase="evaluate", state="complete", completed=self.image_ids, next_id=None, started=started, candidate_pool_sha256=pool_sha, decision=decision["decision"])
        return ResolutionOracleResult(self.output, self.mirror, "complete", self.output / "route_a_decision.json", self.identity_sha)

    def run(self) -> ResolutionOracleResult:
        try:
            return self._run_impl()
        except BaseException as error:
            try:
                previous = json.loads(self.status_path.read_text(encoding="utf-8")) if self.status_path.is_file() else {}
                completed = previous.get("completed_image_ids", [])
                if not isinstance(completed, list):
                    completed = []
                phase = str(previous.get("phase", "unknown"))
                self._status(
                    phase=phase,
                    state="failed",
                    completed=[str(value) for value in completed],
                    next_id=previous.get("next_image_id") if isinstance(previous.get("next_image_id"), str) else None,
                    started=getattr(self, "current_started", time.monotonic()),
                    error=repr(error),
                )
            except BaseException:
                # Preserve the original failure; a mirror failure is itself
                # visible through the missing/unchanged status artifact.
                pass
            raise


def _macro_moderate(metrics: Mapping[str, object]) -> float:
    classes = metrics.get("classes", {})
    values: list[float] = []
    if not isinstance(classes, Mapping):
        raise ValueError("metrics classes must be a mapping")
    for class_name in EVAL_CLASSES:
        payload = classes.get(class_name, {})
        moderate = payload.get("moderate", {}) if isinstance(payload, Mapping) else {}
        ap = moderate.get("ap40", moderate.get("ap", 0.0)) if isinstance(moderate, Mapping) else 0.0
        values.append(float(ap))
    return sum(values) / len(values)


def run_resolution_oracle(
    config: OracleConfig,
    *,
    repository_root: Path,
    output_dir: Path,
    mirror_dir: Path,
    services: ResolutionOracleServices | None = None,
    resume: bool = False,
) -> ResolutionOracleResult:
    bootstrap_ultralytics_config(Path(repository_root).resolve())
    dependencies = services or ResolutionOracleServices(adapter=UltralyticsAdapter())
    return _OracleRunner(config, repository_root=repository_root, output_dir=output_dir, mirror_dir=mirror_dir, services=dependencies, resume=resume).run()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mirror-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_oracle_config(args.config, repository_root=args.repository_root)
    result = run_resolution_oracle(config, repository_root=args.repository_root, output_dir=args.output_dir, mirror_dir=args.mirror_dir, resume=args.resume)
    print(json.dumps({"state": result.state, "decision": str(result.decision_path), "identity_sha256": result.identity_sha256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ActiveOracleRunError",
    "OracleConfig",
    "OracleIdentityError",
    "OraclePaths",
    "OracleRules",
    "ResolutionOracleResult",
    "ResolutionOracleServices",
    "load_oracle_config",
    "run_resolution_oracle",
]
