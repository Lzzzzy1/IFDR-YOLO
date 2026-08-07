"""Immutable runtime contract for the formal F0--F3 factor calibration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from ifdr_yolo.data.metadata_index import FactorMetadataIndex, load_metadata_index
from ifdr_yolo.experiments.run_store import atomic_write_json


REGISTERED_FACTOR_CONDITIONS = ("F0", "F1", "F2", "F3")
REGISTERED_EPOCHS = 30
_HASH_RE = set("0123456789abcdef")

# These values are copied from the registered
# ``configs/experiments/kitti_ifdr_yolov8m_s17.yaml``.  They are deliberately
# constants here so a factor-repair YAML cannot become an accidental CLI
# override for the formal entrypoint.
_FIXED_TRAINING: dict[str, object] = {
    "imgsz": 640,
    "batch": 16,
    "workers": 8,
    "device": "0",
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "seed": 17,
    "amp": True,
    "deterministic": True,
    "cache": False,
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _HASH_RE for char in value.lower()):
        raise ValueError(f"{field} must be a 64-hex SHA256")
    return value.lower()


def _value(source: object, name: str, default: object = None) -> object:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _identity(config: object, name: str) -> object:
    identity = _value(config, "identity")
    if identity is None:
        raise ValueError("factor config identity is required")
    result = _value(identity, name)
    if result is None:
        raise ValueError(f"config.identity.{name} is required")
    return result


def _regular_file(path: object, field: str) -> Path:
    if path is None:
        raise ValueError(f"{field} is required")
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{field} must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{field} must be an existing regular file: {resolved}")
    if resolved.stat().st_size <= 0:
        raise ValueError(f"{field} must be non-empty: {resolved}")
    return resolved


def _read_ids(path: Path, field: str) -> tuple[str, ...]:
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"{field} is empty")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{field} must be UTF-8 text") from error
    values = tuple(line.strip() for line in text.splitlines() if line.strip())
    if not values or any(not value or any(char.isspace() for char in value) for value in values):
        raise ValueError(f"{field} contains invalid image IDs")
    if len(values) != len(set(values)):
        raise ValueError(f"{field} contains duplicate image IDs")
    return values


def _find_repository_root(source_path: Path) -> Path:
    for candidate in (source_path.parent, *source_path.parents):
        if (candidate / "configs").is_dir() and (candidate / "models").is_dir():
            return candidate
    return source_path.parent


def _fixed_config_paths(source_path: Path) -> tuple[Path | None, Path | None, str | None]:
    root = _find_repository_root(source_path)
    fixed = root / "configs" / "experiments" / "kitti_ifdr_yolov8m_s17.yaml"
    if not fixed.is_file():
        return None, None, None
    try:
        payload = yaml.safe_load(fixed.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError("registered IFDR config cannot be read") from error
    if not isinstance(payload, Mapping) or not isinstance(payload.get("paths"), Mapping):
        raise ValueError("registered IFDR config paths are invalid")
    paths = payload["paths"]
    model = paths.get("model")
    data = paths.get("data")
    expected_model_hash = paths.get("model_sha256")
    model_path = Path(model) if isinstance(model, str) else None
    data_path = Path(data) if isinstance(data, str) else None
    if model_path is not None and not model_path.is_absolute():
        model_path = root / model_path
    if data_path is not None and not data_path.is_absolute():
        data_path = root / data_path
    return model_path, data_path, expected_model_hash if isinstance(expected_model_hash, str) else None


def _validate_fixed_training_yaml(source_path: Path) -> None:
    root = _find_repository_root(source_path)
    fixed = root / "configs" / "experiments" / "kitti_ifdr_yolov8m_s17.yaml"
    if not fixed.is_file():
        return
    payload = _check_yaml(fixed, "registered IFDR config")
    training = payload.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("registered IFDR config training is invalid")
    for name, expected in _FIXED_TRAINING.items():
        if name in training and training[name] != expected:
            raise ValueError(f"registered IFDR training.{name} drifted from fixed value {expected!r}")


def _check_yaml(path: Path, field: str) -> Mapping[str, object]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"{field} is not valid YAML") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field} must contain a YAML mapping")
    return payload


def _training_value(config: object, name: str) -> object:
    training = _value(config, "training")
    configured = _value(training, name) if training is not None else None
    expected = _FIXED_TRAINING[name]
    if configured is not None and configured != expected:
        raise ValueError(f"registered training.{name} is fixed at {expected!r}")
    return expected


def _image_dir(data_payload: Mapping[str, object], data_yaml: Path, split: str) -> Path:
    root_value = data_payload.get("path", data_yaml.parent)
    root = Path(root_value) if isinstance(root_value, str) else data_yaml.parent
    if not root.is_absolute():
        repository_root = _find_repository_root(data_yaml)
        candidate = repository_root / root
        root = candidate if candidate.exists() else data_yaml.parent / root
    split_value = data_payload.get(split)
    if not isinstance(split_value, str) or not split_value.strip():
        raise ValueError(f"data YAML {split} path is required")
    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = root / split_path
    return split_path.resolve()


def _resolve_images(data_payload: Mapping[str, object], data_yaml: Path, split: str, ids: tuple[str, ...]) -> tuple[Path, ...]:
    directory = _image_dir(data_payload, data_yaml, split)
    if not directory.is_dir():
        raise FileNotFoundError(f"{split} image directory does not exist: {directory}")
    extensions = (".png", ".jpg", ".jpeg", ".bmp")
    resolved: list[Path] = []
    for image_id in ids:
        candidates = tuple(directory / f"{image_id}{extension}" for extension in extensions)
        found = next((candidate for candidate in candidates if candidate.is_file() and not candidate.is_symlink()), None)
        if found is None:
            raise FileNotFoundError(f"{split} image is missing for ID {image_id}: {directory}")
        resolved.append(found.resolve())
    return tuple(resolved)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    atomic_write_json(path, payload)


def _write_if_identical(path: Path, content: bytes, field: str) -> None:
    """Publish a resolved artifact atomically and refuse divergent resumes."""

    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{field} is not a regular file: {path}")
        if path.read_bytes() != content:
            raise ValueError(f"existing {field} is not identical: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


@dataclass(frozen=True)
class FactorRepairRuntime:
    """All paths, scientific values, and split identities for one run."""

    config: object
    condition: str
    run_dir: Path
    model_yaml: Path
    data_yaml: Path
    resolved_data_yaml: Path
    initialization_checkpoint: Path
    metadata_index_path: Path
    metadata_index: FactorMetadataIndex
    fit_ids: tuple[str, ...]
    development_ids: tuple[str, ...]
    fit_manifest: Path
    development_manifest: Path
    model_sha256: str
    data_sha256: str
    initialization_checkpoint_sha256: str
    metadata_index_sha256: str
    metadata_index_file_sha256: str
    fit_ids_sha256: str
    development_ids_sha256: str
    epochs: int
    registered_epochs: int
    imgsz: int
    batch: int
    workers: int
    device: str
    optimizer: str
    lr0: float
    lrf: float
    momentum: float
    weight_decay: float
    warmup_epochs: float
    seed: int
    amp: bool
    deterministic: bool
    cache: bool
    smoke_mode: bool = False
    run_mode: str = "formal"

    def __post_init__(self) -> None:
        if self.condition not in REGISTERED_FACTOR_CONDITIONS:
            raise ValueError("condition must be one of F0, F1, F2, or F3")
        if self.registered_epochs != REGISTERED_EPOCHS:
            raise ValueError("registered factor calibration budget is 30 epochs")
        expected_epochs = 1 if self.smoke_mode else REGISTERED_EPOCHS
        if self.epochs != expected_epochs:
            raise ValueError("only smoke_mode=True may use a one-epoch budget")
        if self.run_mode != ("nonformal" if self.smoke_mode else "formal"):
            raise ValueError("runtime run_mode does not match smoke_mode")
        object.__setattr__(self, "run_dir", Path(self.run_dir).resolve())
        object.__setattr__(self, "model_yaml", Path(self.model_yaml).resolve())
        object.__setattr__(self, "data_yaml", Path(self.data_yaml).resolve())
        object.__setattr__(self, "resolved_data_yaml", Path(self.resolved_data_yaml).resolve())
        object.__setattr__(self, "initialization_checkpoint", Path(self.initialization_checkpoint).resolve())
        object.__setattr__(self, "metadata_index_path", Path(self.metadata_index_path).resolve())
        object.__setattr__(self, "fit_manifest", Path(self.fit_manifest).resolve())
        object.__setattr__(self, "development_manifest", Path(self.development_manifest).resolve())
        object.__setattr__(self, "fit_ids", tuple(self.fit_ids))
        object.__setattr__(self, "development_ids", tuple(self.development_ids))

    def static_provenance(self, *, trainable: tuple[str, ...] = (), frozen: tuple[str, ...] = ()) -> dict[str, object]:
        return {
            "schema_version": 1,
            "condition": self.condition,
            "run_mode": self.run_mode,
            "smoke_mode": self.smoke_mode,
            "registered_epochs": self.registered_epochs,
            "actual_epochs": self.epochs,
            "model_yaml": self.model_yaml.as_posix(),
            "data_yaml": self.data_yaml.as_posix(),
            "resolved_data_yaml": self.resolved_data_yaml.as_posix(),
            "initialization_checkpoint": self.initialization_checkpoint.as_posix(),
            "model_sha256": self.model_sha256,
            "data_sha256": self.data_sha256,
            "initialization_checkpoint_sha256": self.initialization_checkpoint_sha256,
            "fit_ids_sha256": self.fit_ids_sha256,
            "development_ids_sha256": self.development_ids_sha256,
            "metadata_index_sha256": self.metadata_index_sha256,
            "metadata_index_file_sha256": self.metadata_index_file_sha256,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "workers": self.workers,
            "device": self.device,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "seed": self.seed,
            "amp": self.amp,
            "deterministic": self.deterministic,
            "cache": self.cache,
            "fit_ids": self.fit_ids,
            "development_ids": self.development_ids,
            "trainable_parameter_names": tuple(trainable),
            "frozen_parameter_names": tuple(frozen),
        }

    def write_provenance(self, *, trainable: tuple[str, ...], frozen: tuple[str, ...]) -> None:
        payload = self.static_provenance(trainable=tuple(trainable), frozen=tuple(frozen))
        _write_json(self.run_dir / "resolved_runtime.json", payload)
        provenance_path = self.run_dir / "provenance.json"
        existing: dict[str, object] = {}
        if provenance_path.is_file():
            try:
                raw = json.loads(provenance_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("existing factor provenance is invalid") from error
            if not isinstance(raw, Mapping):
                raise ValueError("existing factor provenance must be a mapping")
            existing = dict(raw)
        if existing:
            # The runner's top-level identity/git/checkpoint fields are an
            # independent audit record; never replace them with runtime's
            # similarly named path/hash fields.
            merged = {**existing, "runtime": payload}
        else:
            merged = {**payload, "runtime": payload}
        _write_json(provenance_path, merged)


def build_factor_repair_runtime(
    config: object,
    *,
    condition: str = "F0",
    model_yaml: str | Path | None = None,
    data_yaml: str | Path | None = None,
    initialization_checkpoint: str | Path | None = None,
    run_dir: str | Path | None = None,
    smoke_mode: bool = False,
) -> FactorRepairRuntime:
    """Validate the protocol bundle and prepare immutable resolved manifests."""

    if condition not in REGISTERED_FACTOR_CONDITIONS:
        raise ValueError("condition must be one of F0, F1, F2, or F3")
    configured_condition = _value(config, "condition")
    if configured_condition is not None and configured_condition != condition:
        raise ValueError("factor config condition does not match registered condition")
    conditions = _value(config, "conditions")
    if conditions is not None:
        condition_config = _value(conditions, condition)
        if condition_config is None:
            raise ValueError(f"factor config does not register {condition}")
        condition_track = _value(condition_config, "track")
        condition_epochs = _value(condition_config, "epochs")
        if condition_track != "factor" or condition_epochs != REGISTERED_EPOCHS:
            raise ValueError(f"{condition} condition budget must be the registered 30 factor epochs")
    schedule = _value(config, "schedule")
    calibration_schedule = _value(schedule, "factor_calibration") if schedule is not None else None
    if calibration_schedule is not None:
        if _value(calibration_schedule, "epochs") != REGISTERED_EPOCHS:
            raise ValueError("factor calibration schedule must be fixed at 30 epochs")
        if _value(calibration_schedule, "fusion_schedule") != 0.0 or _value(calibration_schedule, "dcli_schedule") != 0.0:
            raise ValueError("factor calibration fusion and dcli schedules must be zero")
    if not isinstance(smoke_mode, bool):
        raise ValueError("smoke_mode must be boolean")
    if _value(config, "early_stopping", False):
        raise ValueError("factor calibration disables early stopping")
    # Passing an arbitrary epoch override is deliberately unsupported; callers
    # must opt into the single registered smoke budget instead.
    source_value = _value(config, "source_path")
    if source_value is None:
        raise ValueError("factor config source_path is required")
    source_path = Path(source_value).expanduser().resolve()
    _validate_fixed_training_yaml(source_path)
    bundle_root = source_path.parent
    metadata_path = _regular_file(bundle_root / "metadata_index.json", "metadata_index.json")
    fit_path = _regular_file(bundle_root / "fit_ids.txt", "fit_ids.txt")
    development_path = _regular_file(bundle_root / "development_ids.txt", "development_ids.txt")

    fit_ids = _read_ids(fit_path, "fit_ids.txt")
    development_ids = _read_ids(development_path, "development_ids.txt")
    overlap = sorted(set(fit_ids) & set(development_ids))
    if overlap:
        raise ValueError(f"fit/development ID overlap (leakage): {overlap}")
    fit_ids_sha256 = file_sha256(fit_path)
    development_ids_sha256 = file_sha256(development_path)
    if fit_ids_sha256 != _sha(_identity(config, "fit_ids_sha256"), "identity.fit_ids_sha256"):
        raise ValueError("fit IDs SHA256 mismatch")
    if development_ids_sha256 != _sha(_identity(config, "development_ids_sha256"), "identity.development_ids_sha256"):
        raise ValueError("development IDs SHA256 mismatch")

    metadata_index_sha256 = _sha(_identity(config, "metadata_sha256"), "identity.metadata_sha256")
    metadata_index_file_hash = file_sha256(metadata_path)
    metadata_index = load_metadata_index(metadata_path, expected_sha256=metadata_index_sha256)
    if metadata_index.sha256 != metadata_index_sha256:
        raise ValueError("metadata index SHA256 mismatch")
    if not metadata_index.by_image:
        raise ValueError("metadata index must be non-empty")
    # Metadata may legitimately omit an image with no eligible objects, but a
    # calibration split must have at least one bound record.
    if not any(image_id in metadata_index.by_image for image_id in fit_ids):
        raise ValueError("fit IDs have no metadata index records")

    model_candidate = model_yaml
    data_candidate = data_yaml
    expected_model_hash: str | None = None
    if model_candidate is None or data_candidate is None:
        fixed_model, fixed_data, expected_model_hash = _fixed_config_paths(source_path)
        model_candidate = model_candidate or fixed_model
        data_candidate = data_candidate or fixed_data
    model_path = _regular_file(model_candidate, "model YAML")
    data_path = _regular_file(data_candidate, "data YAML")
    model_payload = _check_yaml(model_path, "model YAML")
    data_payload = _check_yaml(data_path, "data YAML")
    if "names" not in data_payload:
        raise ValueError("data YAML names are required")
    model_hash = file_sha256(model_path)
    if expected_model_hash is not None and model_hash != _sha(expected_model_hash, "paths.model_sha256"):
        raise ValueError("model YAML SHA256 mismatch")
    data_hash = file_sha256(data_path)

    checkpoint_candidate = initialization_checkpoint
    if checkpoint_candidate is None:
        paths = _value(config, "paths")
        checkpoint_candidate = _value(paths, "initialization_checkpoint") if paths is not None else None
    checkpoint_path = _regular_file(checkpoint_candidate, "initialization checkpoint")
    checkpoint_hash = file_sha256(checkpoint_path)
    expected_checkpoint_hash = _sha(_identity(config, "initialization_checkpoint_sha256"), "identity.initialization_checkpoint_sha256")
    if checkpoint_hash != expected_checkpoint_hash:
        raise ValueError("initialization checkpoint SHA256 mismatch")

    training = {name: _training_value(config, name) for name in _FIXED_TRAINING}
    fit_images = _resolve_images(data_payload, data_path, "train", fit_ids)
    # Development IDs are a held-out subset of the original training pool;
    # public ``val`` is reserved for the later external evaluation.
    development_images = _resolve_images(data_payload, data_path, "train", development_ids)
    target_run_dir = Path(run_dir) if run_dir is not None else bundle_root / "runs" / "factor-repair" / condition
    target_run_dir = target_run_dir.expanduser().resolve()
    target_run_dir.mkdir(parents=True, exist_ok=True)
    fit_manifest = target_run_dir / "fit_images.txt"
    development_manifest = target_run_dir / "development_images.txt"
    fit_manifest_bytes = "".join(f"{path.as_posix()}\n" for path in fit_images).encode("utf-8")
    development_manifest_bytes = "".join(f"{path.as_posix()}\n" for path in development_images).encode("utf-8")
    _write_if_identical(fit_manifest, fit_manifest_bytes, "fit image manifest")
    _write_if_identical(development_manifest, development_manifest_bytes, "development image manifest")
    resolved_data_yaml = target_run_dir / "resolved_data.yaml"
    resolved_data = dict(data_payload)
    resolved_data["path"] = "/"
    resolved_data["train"] = fit_manifest.as_posix()
    resolved_data["val"] = development_manifest.as_posix()
    if "nc" not in resolved_data:
        names = resolved_data.get("names")
        resolved_data["nc"] = len(names) if isinstance(names, Mapping) else len(names or ())
    resolved_data_bytes = yaml.safe_dump(resolved_data, sort_keys=False).encode("utf-8")
    _write_if_identical(resolved_data_yaml, resolved_data_bytes, "resolved data YAML")

    runtime = FactorRepairRuntime(
        config=config,
        condition=condition,
        run_dir=target_run_dir,
        model_yaml=model_path,
        data_yaml=data_path,
        resolved_data_yaml=resolved_data_yaml,
        initialization_checkpoint=checkpoint_path,
        metadata_index_path=metadata_path,
        metadata_index=metadata_index,
        fit_ids=fit_ids,
        development_ids=development_ids,
        fit_manifest=fit_manifest,
        development_manifest=development_manifest,
        model_sha256=model_hash,
        data_sha256=data_hash,
        initialization_checkpoint_sha256=checkpoint_hash,
        metadata_index_sha256=metadata_index_sha256,
        metadata_index_file_sha256=metadata_index_file_hash,
        fit_ids_sha256=fit_ids_sha256,
        development_ids_sha256=development_ids_sha256,
        epochs=1 if smoke_mode else REGISTERED_EPOCHS,
        registered_epochs=REGISTERED_EPOCHS,
        imgsz=int(training["imgsz"]),
        batch=int(training["batch"]),
        workers=int(training["workers"]),
        device=str(training["device"]),
        optimizer=str(training["optimizer"]),
        lr0=float(training["lr0"]),
        lrf=float(training["lrf"]),
        momentum=float(training["momentum"]),
        weight_decay=float(training["weight_decay"]),
        warmup_epochs=float(training["warmup_epochs"]),
        seed=int(training["seed"]),
        amp=bool(training["amp"]),
        deterministic=bool(training["deterministic"]),
        cache=bool(training["cache"]),
        smoke_mode=smoke_mode,
        run_mode="nonformal" if smoke_mode else "formal",
    )
    # Parameter names are filled once the model's semantic phase exists; the
    # keys are present from the first byte so provenance consumers fail closed
    # rather than infer missing state.
    runtime.write_provenance(trainable=(), frozen=())
    return runtime


load_factor_repair_runtime = build_factor_repair_runtime
FactorRuntime = FactorRepairRuntime


__all__ = [
    "FactorRepairRuntime",
    "FactorRuntime",
    "REGISTERED_FACTOR_CONDITIONS",
    "REGISTERED_EPOCHS",
    "build_factor_repair_runtime",
    "load_factor_repair_runtime",
    "file_sha256",
]
