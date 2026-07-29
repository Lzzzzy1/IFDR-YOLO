from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    model: str
    variant: str
    seed: int


@dataclass(frozen=True)
class PathsConfig:
    model: Path
    model_sha256: str
    data: Path
    generated_data: Path
    raw_images: Path
    raw_labels: Path
    train_ids: Path
    val_ids: Path


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
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
    patience: int
    amp: bool
    deterministic: bool
    cache: bool


@dataclass(frozen=True)
class PredictionConfig:
    conf: float
    iou: float
    max_det: int
    half: bool


@dataclass(frozen=True)
class BaselineConfig:
    schema_version: int
    experiment: ExperimentConfig
    paths: PathsConfig
    training: TrainingConfig
    prediction: PredictionConfig


def _require_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"{field} must be a mapping with string keys")
    return value


def _require_fields(
    mapping: dict[str, Any],
    *,
    field: str,
    expected: set[str],
) -> None:
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing:
        raise ValueError(f"missing {field} fields: {missing}")
    if unknown:
        raise ValueError(f"unknown {field} fields: {unknown}")


def _require_int(
    value: object,
    field: str,
    *,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return value


def _require_float(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if minimum is not None:
        invalid = result < minimum if minimum_inclusive else result <= minimum
        if invalid:
            operator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{field} must be {operator} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be <= {maximum}")
    return result


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _resolve_path(value: object, field: str, root: Path) -> Path:
    text = _require_text(value, field)
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _parse_experiment(value: object) -> ExperimentConfig:
    mapping = _require_mapping(value, "experiment")
    expected = {"dataset", "model", "variant", "seed"}
    _require_fields(mapping, field="experiment", expected=expected)
    return ExperimentConfig(
        dataset=_require_text(mapping["dataset"], "experiment.dataset"),
        model=_require_text(mapping["model"], "experiment.model"),
        variant=_require_text(mapping["variant"], "experiment.variant"),
        seed=_require_int(mapping["seed"], "experiment.seed", minimum=0),
    )


def _parse_paths(value: object, root: Path) -> PathsConfig:
    mapping = _require_mapping(value, "paths")
    expected = {
        "model",
        "model_sha256",
        "data",
        "generated_data",
        "raw_images",
        "raw_labels",
        "train_ids",
        "val_ids",
    }
    _require_fields(mapping, field="paths", expected=expected)
    model_sha256 = _require_text(
        mapping["model_sha256"],
        "paths.model_sha256",
    ).lower()
    if len(model_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in model_sha256
    ):
        raise ValueError("paths.model_sha256 must be 64 hexadecimal characters")
    return PathsConfig(
        model=_resolve_path(mapping["model"], "paths.model", root),
        model_sha256=model_sha256,
        data=_resolve_path(mapping["data"], "paths.data", root),
        generated_data=_resolve_path(
            mapping["generated_data"],
            "paths.generated_data",
            root,
        ),
        raw_images=_resolve_path(
            mapping["raw_images"],
            "paths.raw_images",
            root,
        ),
        raw_labels=_resolve_path(
            mapping["raw_labels"],
            "paths.raw_labels",
            root,
        ),
        train_ids=_resolve_path(
            mapping["train_ids"],
            "paths.train_ids",
            root,
        ),
        val_ids=_resolve_path(mapping["val_ids"], "paths.val_ids", root),
    )


def _parse_training(value: object) -> TrainingConfig:
    mapping = _require_mapping(value, "training")
    expected = {
        "epochs",
        "imgsz",
        "batch",
        "workers",
        "device",
        "optimizer",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "patience",
        "amp",
        "deterministic",
        "cache",
    }
    _require_fields(mapping, field="training", expected=expected)
    return TrainingConfig(
        epochs=_require_int(mapping["epochs"], "training.epochs", minimum=1),
        imgsz=_require_int(mapping["imgsz"], "training.imgsz", minimum=1),
        batch=_require_int(mapping["batch"], "training.batch", minimum=1),
        workers=_require_int(mapping["workers"], "training.workers", minimum=0),
        device=_require_text(mapping["device"], "training.device"),
        optimizer=_require_text(mapping["optimizer"], "training.optimizer"),
        lr0=_require_float(mapping["lr0"], "training.lr0", minimum=0.0),
        lrf=_require_float(mapping["lrf"], "training.lrf", minimum=0.0),
        momentum=_require_float(
            mapping["momentum"],
            "training.momentum",
            minimum=0.0,
        ),
        weight_decay=_require_float(
            mapping["weight_decay"],
            "training.weight_decay",
            minimum=0.0,
        ),
        warmup_epochs=_require_float(
            mapping["warmup_epochs"],
            "training.warmup_epochs",
            minimum=0.0,
        ),
        patience=_require_int(
            mapping["patience"],
            "training.patience",
            minimum=0,
        ),
        amp=_require_bool(mapping["amp"], "training.amp"),
        deterministic=_require_bool(
            mapping["deterministic"],
            "training.deterministic",
        ),
        cache=_require_bool(mapping["cache"], "training.cache"),
    )


def _parse_prediction(value: object) -> PredictionConfig:
    mapping = _require_mapping(value, "prediction")
    expected = {"conf", "iou", "max_det", "half"}
    _require_fields(mapping, field="prediction", expected=expected)
    return PredictionConfig(
        conf=_require_float(
            mapping["conf"],
            "prediction.conf",
            minimum=0.0,
            maximum=1.0,
        ),
        iou=_require_float(
            mapping["iou"],
            "prediction.iou",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        ),
        max_det=_require_int(
            mapping["max_det"],
            "prediction.max_det",
            minimum=1,
        ),
        half=_require_bool(mapping["half"], "prediction.half"),
    )


def load_baseline_config(
    path: Path,
    *,
    repository_root: Path,
) -> BaselineConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    mapping = _require_mapping(payload, "baseline config root")
    expected = {
        "schema_version",
        "experiment",
        "paths",
        "training",
        "prediction",
    }
    _require_fields(mapping, field="top-level", expected=expected)
    schema_version = _require_int(
        mapping["schema_version"],
        "schema_version",
        minimum=1,
    )
    if schema_version != 1:
        raise ValueError(f"unsupported schema_version: {schema_version}")
    root = repository_root.resolve()
    return BaselineConfig(
        schema_version=schema_version,
        experiment=_parse_experiment(mapping["experiment"]),
        paths=_parse_paths(mapping["paths"], root),
        training=_parse_training(mapping["training"]),
        prediction=_parse_prediction(mapping["prediction"]),
    )
