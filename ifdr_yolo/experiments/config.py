from __future__ import annotations

from dataclasses import dataclass
import math
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
class InitializationConfig:
    pretrained: Path
    pretrained_sha256: str
    strategy: str
    max_layer: int
    expected_items: int


@dataclass(frozen=True)
class BaselineConfig:
    schema_version: int
    experiment: ExperimentConfig
    paths: PathsConfig
    training: TrainingConfig
    prediction: PredictionConfig
    initialization: InitializationConfig | None = None
    source_path: Path | None = None


@dataclass(frozen=True)
class IFDRScheduleConfig:
    frozen_epochs: int
    ramp_epochs: int


@dataclass(frozen=True)
class IFDRComponentsConfig:
    fusion_gate: bool
    dcli: bool
    factor_supervision: bool
    interventions: bool
    semantic_protection: bool = False
    counterfactual_consistency: bool = False


@dataclass(frozen=True)
class IFDRInterventionConfig:
    base_seed: int
    identity_probability: float
    sampling_probability: float
    visibility_probability: float
    minimum_strength: float
    maximum_strength: float


@dataclass(frozen=True)
class IFDRLossConfig:
    dcli_beta: float
    uncertainty_calibration_gain: float
    factor_supervision_gain: float
    factor_weights: tuple[float, float]
    dfl_entropy_weight: float
    counterfactual_gain: float = 0.0


@dataclass(frozen=True)
class IFDRMethodConfig:
    reliability_channels: int
    components: IFDRComponentsConfig
    schedule: IFDRScheduleConfig
    intervention: IFDRInterventionConfig
    loss: IFDRLossConfig


@dataclass(frozen=True)
class IFDRConfig:
    schema_version: int
    experiment: ExperimentConfig
    paths: PathsConfig
    initialization: InitializationConfig
    method: IFDRMethodConfig
    training: TrainingConfig
    prediction: PredictionConfig
    source_path: Path | None = None


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
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected - optional)
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
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
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


def _require_sha256(value: object, field: str) -> str:
    result = _require_text(value, field).lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{field} must be 64 hexadecimal characters")
    return result


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
    model_sha256 = _require_sha256(
        mapping["model_sha256"],
        "paths.model_sha256",
    )
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


def _parse_initialization(
    value: object,
    root: Path,
) -> InitializationConfig:
    mapping = _require_mapping(value, "initialization")
    expected = {
        "pretrained",
        "pretrained_sha256",
        "strategy",
        "max_layer",
        "expected_items",
    }
    _require_fields(mapping, field="initialization", expected=expected)
    strategy = _require_text(mapping["strategy"], "initialization.strategy")
    if strategy != "semantic_prefix":
        raise ValueError(
            "initialization.strategy must be 'semantic_prefix'"
        )
    return InitializationConfig(
        pretrained=_resolve_path(
            mapping["pretrained"],
            "initialization.pretrained",
            root,
        ),
        pretrained_sha256=_require_sha256(
            mapping["pretrained_sha256"],
            "initialization.pretrained_sha256",
        ),
        strategy=strategy,
        max_layer=_require_int(
            mapping["max_layer"],
            "initialization.max_layer",
            minimum=0,
        ),
        expected_items=_require_int(
            mapping["expected_items"],
            "initialization.expected_items",
            minimum=1,
        ),
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
    _require_fields(
        mapping,
        field="top-level",
        expected=expected,
        optional={"initialization"},
    )
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
        initialization=(
            _parse_initialization(mapping["initialization"], root)
            if "initialization" in mapping
            else None
        ),
        source_path=path.resolve(),
    )


def _parse_ifdr_schedule(value: object) -> IFDRScheduleConfig:
    mapping = _require_mapping(value, "ifdr.schedule")
    _require_fields(
        mapping,
        field="ifdr.schedule",
        expected={"frozen_epochs", "ramp_epochs"},
    )
    return IFDRScheduleConfig(
        frozen_epochs=_require_int(
            mapping["frozen_epochs"],
            "ifdr.schedule.frozen_epochs",
            minimum=0,
        ),
        ramp_epochs=_require_int(
            mapping["ramp_epochs"],
            "ifdr.schedule.ramp_epochs",
            minimum=1,
        ),
    )


def _parse_ifdr_components(value: object) -> IFDRComponentsConfig:
    mapping = _require_mapping(value, "ifdr.components")
    fields = {
        "fusion_gate",
        "dcli",
        "factor_supervision",
        "interventions",
    }
    optional = {
        "semantic_protection",
        "counterfactual_consistency",
    }
    _require_fields(
        mapping,
        field="ifdr.components",
        expected=fields,
        optional=optional,
    )
    return IFDRComponentsConfig(
        fusion_gate=_require_bool(
            mapping["fusion_gate"],
            "ifdr.components.fusion_gate",
        ),
        dcli=_require_bool(mapping["dcli"], "ifdr.components.dcli"),
        factor_supervision=_require_bool(
            mapping["factor_supervision"],
            "ifdr.components.factor_supervision",
        ),
        interventions=_require_bool(
            mapping["interventions"],
            "ifdr.components.interventions",
        ),
        semantic_protection=_require_bool(
            mapping.get("semantic_protection", False),
            "ifdr.components.semantic_protection",
        ),
        counterfactual_consistency=_require_bool(
            mapping.get("counterfactual_consistency", False),
            "ifdr.components.counterfactual_consistency",
        ),
    )


def _parse_ifdr_intervention(value: object) -> IFDRInterventionConfig:
    mapping = _require_mapping(value, "ifdr.intervention")
    fields = {
        "base_seed",
        "identity_probability",
        "sampling_probability",
        "visibility_probability",
        "minimum_strength",
        "maximum_strength",
    }
    _require_fields(mapping, field="ifdr.intervention", expected=fields)
    probabilities = tuple(
        _require_float(
            mapping[field],
            f"ifdr.intervention.{field}",
            minimum=0.0,
            maximum=1.0,
        )
        for field in (
            "identity_probability",
            "sampling_probability",
            "visibility_probability",
        )
    )
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-12):
        raise ValueError("IFDR intervention probabilities must sum to one")
    minimum_strength = _require_float(
        mapping["minimum_strength"],
        "ifdr.intervention.minimum_strength",
        minimum=0.0,
        maximum=1.0,
    )
    maximum_strength = _require_float(
        mapping["maximum_strength"],
        "ifdr.intervention.maximum_strength",
        minimum=0.0,
        maximum=1.0,
    )
    if minimum_strength > maximum_strength:
        raise ValueError("IFDR intervention strength bounds are inverted")
    return IFDRInterventionConfig(
        base_seed=_require_int(
            mapping["base_seed"],
            "ifdr.intervention.base_seed",
            minimum=0,
        ),
        identity_probability=probabilities[0],
        sampling_probability=probabilities[1],
        visibility_probability=probabilities[2],
        minimum_strength=minimum_strength,
        maximum_strength=maximum_strength,
    )


def _parse_ifdr_loss(value: object) -> IFDRLossConfig:
    mapping = _require_mapping(value, "ifdr.loss")
    fields = {
        "dcli_beta",
        "uncertainty_calibration_gain",
        "factor_supervision_gain",
        "factor_weights",
        "dfl_entropy_weight",
    }
    _require_fields(
        mapping,
        field="ifdr.loss",
        expected=fields,
        optional={"counterfactual_gain"},
    )
    raw_weights = mapping["factor_weights"]
    if not isinstance(raw_weights, (list, tuple)) or len(raw_weights) != 2:
        raise ValueError("ifdr.loss.factor_weights must contain two values")
    factor_weights = tuple(
        _require_float(
            value,
            f"ifdr.loss.factor_weights[{index}]",
            minimum=0.0,
        )
        for index, value in enumerate(raw_weights)
    )
    entropy_weight = _require_float(
        mapping["dfl_entropy_weight"],
        "ifdr.loss.dfl_entropy_weight",
        minimum=0.0,
    )
    if sum(factor_weights) + entropy_weight <= 0.0:
        raise ValueError("at least one IFDR uncertainty weight must be positive")
    return IFDRLossConfig(
        dcli_beta=_require_float(
            mapping["dcli_beta"],
            "ifdr.loss.dcli_beta",
            minimum=0.0,
            maximum=1.0,
        ),
        uncertainty_calibration_gain=_require_float(
            mapping["uncertainty_calibration_gain"],
            "ifdr.loss.uncertainty_calibration_gain",
            minimum=0.0,
            maximum=1.0,
        ),
        factor_supervision_gain=_require_float(
            mapping["factor_supervision_gain"],
            "ifdr.loss.factor_supervision_gain",
            minimum=0.0,
            maximum=1.0,
        ),
        factor_weights=(factor_weights[0], factor_weights[1]),
        dfl_entropy_weight=entropy_weight,
        counterfactual_gain=_require_float(
            mapping.get("counterfactual_gain", 0.0),
            "ifdr.loss.counterfactual_gain",
            minimum=0.0,
            maximum=1.0,
        ),
    )


def _parse_ifdr_method(value: object) -> IFDRMethodConfig:
    mapping = _require_mapping(value, "ifdr")
    _require_fields(
        mapping,
        field="ifdr",
        expected={
            "reliability_channels",
            "components",
            "schedule",
            "intervention",
            "loss",
        },
    )
    components = _parse_ifdr_components(mapping["components"])
    loss = _parse_ifdr_loss(mapping["loss"])
    if components.counterfactual_consistency:
        if not components.interventions:
            raise ValueError(
                "counterfactual consistency requires interventions"
            )
        if not components.factor_supervision:
            raise ValueError(
                "counterfactual consistency requires factor supervision"
            )
        if loss.counterfactual_gain <= 0.0:
            raise ValueError(
                "counterfactual consistency requires a positive gain"
            )
    elif loss.counterfactual_gain != 0.0:
        raise ValueError(
            "counterfactual_gain must be zero when consistency is disabled"
        )
    return IFDRMethodConfig(
        reliability_channels=_require_int(
            mapping["reliability_channels"],
            "ifdr.reliability_channels",
            minimum=1,
        ),
        components=components,
        schedule=_parse_ifdr_schedule(mapping["schedule"]),
        intervention=_parse_ifdr_intervention(mapping["intervention"]),
        loss=loss,
    )


def load_ifdr_config(
    path: Path,
    *,
    repository_root: Path,
) -> IFDRConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    mapping = _require_mapping(payload, "IFDR config root")
    expected = {
        "schema_version",
        "experiment",
        "paths",
        "initialization",
        "ifdr",
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
    experiment = _parse_experiment(mapping["experiment"])
    if not (
        experiment.variant == "ifdr"
        or experiment.variant.startswith("ifdr-")
    ):
        raise ValueError(
            "IFDR experiment.variant must be 'ifdr' or start with 'ifdr-'"
        )
    return IFDRConfig(
        schema_version=schema_version,
        experiment=experiment,
        paths=_parse_paths(mapping["paths"], root),
        initialization=_parse_initialization(
            mapping["initialization"],
            root,
        ),
        method=_parse_ifdr_method(mapping["ifdr"]),
        training=_parse_training(mapping["training"]),
        prediction=_parse_prediction(mapping["prediction"]),
        source_path=path.resolve(),
    )
