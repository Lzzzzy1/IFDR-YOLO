from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

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
    gradient_diagnostic_interval: int = 0


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
        optional={"gradient_diagnostic_interval"},
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
        gradient_diagnostic_interval=_require_int(
            mapping.get("gradient_diagnostic_interval", 0),
            "ifdr.gradient_diagnostic_interval",
            minimum=0,
        ),
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


# ---------------------------------------------------------------------------
# Factor-repair development protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DevelopmentProtocolConfig:
    """The immutable development split identity."""

    seed: int
    fraction: float


@dataclass(frozen=True)
class MetadataReplayConfig:
    """The three registered metadata replay recipes."""

    m1: str
    m2: str
    m3: str

    def __getitem__(self, condition: str) -> str:
        try:
            return {"M1": self.m1, "M2": self.m2, "M3": self.m3}[condition]
        except KeyError as error:
            raise KeyError(condition) from error

    @property
    def M1(self) -> str:  # noqa: N802 - matches the registered condition name
        return self.m1

    @property
    def M2(self) -> str:  # noqa: N802 - matches the registered condition name
        return self.m2

    @property
    def M3(self) -> str:  # noqa: N802 - matches the registered condition name
        return self.m3


@dataclass(frozen=True)
class FactorAlignmentConfig:
    """Registered weights for the target-conditioned alignment objective."""

    natural_gain: float
    specificity_gain: float
    specificity_margin: float
    factor_weights: tuple[float, float]


@dataclass(frozen=True)
class FactorGateConfig:
    """Pre-registered mechanism-gate thresholds.

    Bootstrap seed and replicate count intentionally do not appear here.  The
    gate implementation owns those constants so a YAML file cannot silently
    alter the statistical test after a checkpoint has been observed.
    """

    seed17_min_positive_primary_directions: int
    formal_min_positive_seed_node_directions: int
    formal_total_seed_node_directions: int
    minimum_severity_ordering: float
    diagnostic_reverse_abs_rho: float
    selection_tie_tolerance: float
    require_paired_delta_ci_lower_positive: bool
    require_zero_malformed: bool


@dataclass(frozen=True)
class RepairConditionConfig:
    """One registered M/F condition and its fixed calibration budget."""

    track: str
    epochs: int


@dataclass(frozen=True)
class FactorIdentityConfig:
    source_metadata_sha256: str
    images_metadata_sha256: str
    raw_labels_sha256: str
    split_sha256: str
    metadata_sha256: str
    initialization_checkpoint_sha256: str
    fit_ids_sha256: str
    development_ids_sha256: str


@dataclass(frozen=True)
class FactorTrainingConfig:
    imgsz: int


@dataclass(frozen=True)
class FactorModelConfig:
    nodes: tuple[int, ...]
    primary_nodes: tuple[int, ...]


@dataclass(frozen=True)
class FactorPathsConfig:
    metadata_jsonl: Path
    images_jsonl: Path
    raw_label_dir: Path
    initialization_checkpoint: Path
    output_root: Path


@dataclass(frozen=True)
class ReplayScheduleConfig:
    eta_peak: float
    ramp_epochs: int
    focus_end_epoch: int
    recovery_start_epoch: int
    total_epochs: int
    priority_clip_quantile: float
    eligible_floor: float
    replacement: bool
    draws_per_epoch: str


@dataclass(frozen=True)
class FactorCalibrationScheduleConfig:
    epochs: int
    views_per_sample: int
    fusion_schedule: float
    dcli_schedule: float


@dataclass(frozen=True)
class TaskAdaptationScheduleConfig:
    epochs: int


@dataclass(frozen=True)
class FactorScheduleConfig:
    replay: ReplayScheduleConfig
    factor_calibration: FactorCalibrationScheduleConfig
    task_adaptation: TaskAdaptationScheduleConfig


@dataclass(frozen=True)
class CheckpointPolicyConfig:
    primary: str
    diagnostic: str
    early_stopping: bool


@dataclass(frozen=True)
class FactorRepairConfig:
    schema_version: int
    identity: FactorIdentityConfig
    development: DevelopmentProtocolConfig
    conditions: Mapping[str, RepairConditionConfig]
    task_adaptation_epochs: int
    max_selected_factor_repairs: int
    early_stopping: bool
    training: FactorTrainingConfig
    factor_loss: FactorAlignmentConfig
    model: FactorModelConfig
    paths: FactorPathsConfig
    schedule: FactorScheduleConfig
    checkpoint_policy: CheckpointPolicyConfig
    metadata_replay: MetadataReplayConfig
    factor_gate: FactorGateConfig
    source_path: Path | None = None

    def require_condition(self, name: str) -> RepairConditionConfig:
        if not isinstance(name, str) or name not in self.conditions:
            raise ValueError(f"unregistered factor-repair condition: {name!r}")
        return self.conditions[name]


_FACTOR_IDENTITY_FIELDS = {
    "source_metadata_sha256",
    "images_metadata_sha256",
    "raw_labels_sha256",
    "split_sha256",
    "metadata_sha256",
    "initialization_checkpoint_sha256",
    "fit_ids_sha256",
    "development_ids_sha256",
}
_FACTOR_CONDITION_NAMES = ("M1", "M2", "M3", "F0", "F1", "F2", "F3")
_FACTOR_NODES = (11, 14, 17, 20, 23, 26)
_FACTOR_PRIMARY_NODES = (17, 20, 23, 26)


def _freeze_factor_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a read-only shallow mapping for parsed protocol metadata."""

    return MappingProxyType(dict(mapping))


def _parse_factor_identity(value: object) -> FactorIdentityConfig:
    mapping = _require_mapping(value, "identity")
    _require_fields(mapping, field="identity", expected=_FACTOR_IDENTITY_FIELDS)
    values = {
        field: _require_sha256(mapping[field], f"identity.{field}")
        for field in sorted(_FACTOR_IDENTITY_FIELDS)
    }
    return FactorIdentityConfig(**values)


def _parse_development_protocol(value: object) -> DevelopmentProtocolConfig:
    mapping = _require_mapping(value, "development")
    _require_fields(
        mapping,
        field="development",
        expected={"seed", "fraction"},
    )
    seed = _require_int(mapping["seed"], "development.seed", minimum=0)
    fraction = _require_float(
        mapping["fraction"],
        "development.fraction",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    if seed != 20260805:
        raise ValueError("development.seed must equal registered seed 20260805")
    if fraction != 0.10:
        raise ValueError(
            "development.fraction must equal registered fraction 0.10"
        )
    return DevelopmentProtocolConfig(seed=seed, fraction=fraction)


def _parse_factor_conditions(value: object) -> Mapping[str, RepairConditionConfig]:
    mapping = _require_mapping(value, "conditions")
    expected = set(_FACTOR_CONDITION_NAMES)
    _require_fields(mapping, field="conditions", expected=expected)
    conditions: dict[str, RepairConditionConfig] = {}
    for name in _FACTOR_CONDITION_NAMES:
        condition_mapping = _require_mapping(
            mapping[name],
            f"conditions.{name}",
        )
        _require_fields(
            condition_mapping,
            field=f"conditions.{name}",
            expected={"track", "epochs"},
        )
        track = _require_text(
            condition_mapping["track"],
            f"conditions.{name}.track",
        )
        expected_track = "metadata" if name.startswith("M") else "factor"
        if track != expected_track:
            raise ValueError(
                f"conditions.{name}.track must equal registered track "
                f"{expected_track!r}"
            )
        epochs = _require_int(
            condition_mapping["epochs"],
            f"conditions.{name}.epochs",
            minimum=1,
        )
        expected_epochs = 60 if name.startswith("M") else 30
        if epochs != expected_epochs:
            raise ValueError(
                f"conditions.{name}.epochs must equal registered budget {expected_epochs}"
            )
        conditions[name] = RepairConditionConfig(track=track, epochs=epochs)
    return _freeze_factor_mapping(conditions)


def _parse_factor_training(value: object) -> FactorTrainingConfig:
    mapping = _require_mapping(value, "training")
    _require_fields(mapping, field="training", expected={"imgsz"})
    imgsz = _require_int(mapping["imgsz"], "training.imgsz", minimum=1)
    if imgsz != 640:
        raise ValueError("training.imgsz must equal registered image size 640")
    return FactorTrainingConfig(imgsz=imgsz)


def _parse_factor_alignment(value: object) -> FactorAlignmentConfig:
    mapping = _require_mapping(value, "factor_loss")
    _require_fields(
        mapping,
        field="factor_loss",
        expected={"natural_gain", "specificity_gain", "specificity_margin", "factor_weights"},
    )
    natural_gain = _require_float(
        mapping["natural_gain"], "factor_loss.natural_gain", minimum=0.0
    )
    specificity_gain = _require_float(
        mapping["specificity_gain"],
        "factor_loss.specificity_gain",
        minimum=0.0,
    )
    specificity_margin = _require_float(
        mapping["specificity_margin"],
        "factor_loss.specificity_margin",
        minimum=0.0,
    )
    raw_weights = mapping["factor_weights"]
    if not isinstance(raw_weights, (list, tuple)) or len(raw_weights) != 2:
        raise ValueError("factor_loss.factor_weights must contain two values")
    factor_weights = tuple(
        _require_float(value, f"factor_loss.factor_weights[{index}]", minimum=0.0)
        for index, value in enumerate(raw_weights)
    )
    registered = (1.0, 0.5, 0.05, (1.0, 1.0))
    if natural_gain != registered[0]:
        raise ValueError("factor_loss.natural_gain must equal registered value 1.0")
    if specificity_gain != registered[1]:
        raise ValueError("factor_loss.specificity_gain must equal registered value 0.5")
    if specificity_margin != registered[2]:
        raise ValueError("factor_loss.specificity_margin must equal registered value 0.05")
    if factor_weights != registered[3]:
        raise ValueError("factor_loss.factor_weights must equal registered values (1.0, 1.0)")
    return FactorAlignmentConfig(
        natural_gain=natural_gain,
        specificity_gain=specificity_gain,
        specificity_margin=specificity_margin,
        factor_weights=(factor_weights[0], factor_weights[1]),
    )


def _parse_factor_model(value: object) -> FactorModelConfig:
    mapping = _require_mapping(value, "model")
    _require_fields(
        mapping,
        field="model",
        expected={"nodes", "primary_nodes"},
    )

    def parse_nodes(raw: object, field: str) -> tuple[int, ...]:
        if not isinstance(raw, (list, tuple)):
            raise ValueError(f"{field} must be a sequence")
        nodes = tuple(
            _require_int(node, f"{field}[]", minimum=0)
            for node in raw
        )
        if len(set(nodes)) != len(nodes):
            raise ValueError(f"{field} must not contain duplicates")
        return nodes

    nodes = parse_nodes(mapping["nodes"], "model.nodes")
    primary_nodes = parse_nodes(mapping["primary_nodes"], "model.primary_nodes")
    if nodes != _FACTOR_NODES:
        raise ValueError(f"model.nodes must equal registered nodes {_FACTOR_NODES}")
    if primary_nodes != _FACTOR_PRIMARY_NODES:
        raise ValueError(
            "model.primary_nodes must equal registered nodes "
            f"{_FACTOR_PRIMARY_NODES}"
        )
    if not set(primary_nodes).issubset(nodes):
        raise ValueError("model.primary_nodes must be a subset of model.nodes")
    return FactorModelConfig(nodes=nodes, primary_nodes=primary_nodes)


def _parse_factor_paths(value: object, root: Path) -> FactorPathsConfig:
    mapping = _require_mapping(value, "paths")
    expected = {
        "metadata_jsonl",
        "images_jsonl",
        "raw_label_dir",
        "initialization_checkpoint",
        "output_root",
    }
    _require_fields(mapping, field="paths", expected=expected)
    return FactorPathsConfig(
        metadata_jsonl=_resolve_path(
            mapping["metadata_jsonl"],
            "paths.metadata_jsonl",
            root,
        ),
        images_jsonl=_resolve_path(
            mapping["images_jsonl"],
            "paths.images_jsonl",
            root,
        ),
        raw_label_dir=_resolve_path(
            mapping["raw_label_dir"],
            "paths.raw_label_dir",
            root,
        ),
        initialization_checkpoint=_resolve_path(
            mapping["initialization_checkpoint"],
            "paths.initialization_checkpoint",
            root,
        ),
        output_root=_resolve_path(
            mapping["output_root"],
            "paths.output_root",
            root,
        ),
    )


def _parse_factor_schedule(value: object) -> FactorScheduleConfig:
    mapping = _require_mapping(value, "schedule")
    _require_fields(
        mapping,
        field="schedule",
        expected={"replay", "factor_calibration", "task_adaptation"},
    )

    replay_mapping = _require_mapping(mapping["replay"], "schedule.replay")
    _require_fields(
        replay_mapping,
        field="schedule.replay",
        expected={
            "eta_peak",
            "ramp_epochs",
            "focus_end_epoch",
            "recovery_start_epoch",
            "total_epochs",
            "priority_clip_quantile",
            "eligible_floor",
            "replacement",
            "draws_per_epoch",
        },
    )
    replay = ReplayScheduleConfig(
        eta_peak=_require_float(
            replay_mapping["eta_peak"],
            "schedule.replay.eta_peak",
            minimum=0.0,
        ),
        ramp_epochs=_require_int(
            replay_mapping["ramp_epochs"],
            "schedule.replay.ramp_epochs",
            minimum=1,
        ),
        focus_end_epoch=_require_int(
            replay_mapping["focus_end_epoch"],
            "schedule.replay.focus_end_epoch",
            minimum=1,
        ),
        recovery_start_epoch=_require_int(
            replay_mapping["recovery_start_epoch"],
            "schedule.replay.recovery_start_epoch",
            minimum=1,
        ),
        total_epochs=_require_int(
            replay_mapping["total_epochs"],
            "schedule.replay.total_epochs",
            minimum=1,
        ),
        priority_clip_quantile=_require_float(
            replay_mapping["priority_clip_quantile"],
            "schedule.replay.priority_clip_quantile",
            minimum=0.0,
            maximum=1.0,
        ),
        eligible_floor=_require_float(
            replay_mapping["eligible_floor"],
            "schedule.replay.eligible_floor",
            minimum=0.0,
            maximum=1.0,
        ),
        replacement=_require_bool(replay_mapping["replacement"], "schedule.replay.replacement"),
        draws_per_epoch=_require_text(
            replay_mapping["draws_per_epoch"],
            "schedule.replay.draws_per_epoch",
        ),
    )
    if replay != ReplayScheduleConfig(
        0.3,
        5,
        40,
        41,
        60,
        0.95,
        0.05,
        True,
        "fit_count",
    ):
        raise ValueError("schedule.replay does not match registered schedule")

    calibration_mapping = _require_mapping(
        mapping["factor_calibration"], "schedule.factor_calibration"
    )
    _require_fields(
        calibration_mapping,
        field="schedule.factor_calibration",
        expected={"epochs", "views_per_sample", "fusion_schedule", "dcli_schedule"},
    )
    calibration = FactorCalibrationScheduleConfig(
        epochs=_require_int(
            calibration_mapping["epochs"],
            "schedule.factor_calibration.epochs",
            minimum=1,
        ),
        views_per_sample=_require_int(
            calibration_mapping["views_per_sample"],
            "schedule.factor_calibration.views_per_sample",
            minimum=1,
        ),
        fusion_schedule=_require_float(
            calibration_mapping["fusion_schedule"],
            "schedule.factor_calibration.fusion_schedule",
            minimum=0.0,
        ),
        dcli_schedule=_require_float(
            calibration_mapping["dcli_schedule"],
            "schedule.factor_calibration.dcli_schedule",
            minimum=0.0,
        ),
    )
    if calibration != FactorCalibrationScheduleConfig(30, 3, 0.0, 0.0):
        raise ValueError("schedule.factor_calibration does not match registered schedule")

    adaptation_mapping = _require_mapping(
        mapping["task_adaptation"], "schedule.task_adaptation"
    )
    _require_fields(
        adaptation_mapping,
        field="schedule.task_adaptation",
        expected={"epochs"},
    )
    adaptation = TaskAdaptationScheduleConfig(
        epochs=_require_int(
            adaptation_mapping["epochs"],
            "schedule.task_adaptation.epochs",
            minimum=1,
        )
    )
    if adaptation.epochs != 60:
        raise ValueError("schedule.task_adaptation.epochs must equal registered budget 60")
    return FactorScheduleConfig(
        replay=replay,
        factor_calibration=calibration,
        task_adaptation=adaptation,
    )


def _parse_checkpoint_policy(value: object) -> CheckpointPolicyConfig:
    mapping = _require_mapping(value, "checkpoint_policy")
    _require_fields(
        mapping,
        field="checkpoint_policy",
        expected={"primary", "diagnostic", "early_stopping"},
    )
    primary = _require_text(mapping["primary"], "checkpoint_policy.primary")
    diagnostic = _require_text(mapping["diagnostic"], "checkpoint_policy.diagnostic")
    early_stopping = _require_bool(
        mapping["early_stopping"], "checkpoint_policy.early_stopping"
    )
    if (primary, diagnostic, early_stopping) != ("last.pt", "best.pt", False):
        raise ValueError("checkpoint_policy does not match registered policy")
    return CheckpointPolicyConfig(
        primary=primary,
        diagnostic=diagnostic,
        early_stopping=early_stopping,
    )


def _parse_metadata_replay(value: object) -> MetadataReplayConfig:
    mapping = _require_mapping(value, "metadata_replay")
    _require_fields(mapping, field="metadata_replay", expected={"M1", "M2", "M3"})
    recipes = tuple(
        _require_text(mapping[name], f"metadata_replay.{name}")
        for name in ("M1", "M2", "M3")
    )
    if recipes != ("original", "cyclist_uniform", "joint_score"):
        raise ValueError("metadata_replay does not match registered recipes")
    return MetadataReplayConfig(m1=recipes[0], m2=recipes[1], m3=recipes[2])


def _parse_factor_gate(value: object) -> FactorGateConfig:
    mapping = _require_mapping(value, "factor_gate")
    expected = {
        "seed17_min_positive_primary_directions",
        "formal_min_positive_seed_node_directions",
        "formal_total_seed_node_directions",
        "minimum_severity_ordering",
        "diagnostic_reverse_abs_rho",
        "selection_tie_tolerance",
        "require_paired_delta_ci_lower_positive",
        "require_zero_malformed",
    }
    _require_fields(mapping, field="factor_gate", expected=expected)
    gate = FactorGateConfig(
        seed17_min_positive_primary_directions=_require_int(
            mapping["seed17_min_positive_primary_directions"],
            "factor_gate.seed17_min_positive_primary_directions",
            minimum=1,
        ),
        formal_min_positive_seed_node_directions=_require_int(
            mapping["formal_min_positive_seed_node_directions"],
            "factor_gate.formal_min_positive_seed_node_directions",
            minimum=1,
        ),
        formal_total_seed_node_directions=_require_int(
            mapping["formal_total_seed_node_directions"],
            "factor_gate.formal_total_seed_node_directions",
            minimum=1,
        ),
        minimum_severity_ordering=_require_float(
            mapping["minimum_severity_ordering"],
            "factor_gate.minimum_severity_ordering",
            minimum=0.0,
            maximum=1.0,
        ),
        diagnostic_reverse_abs_rho=_require_float(
            mapping["diagnostic_reverse_abs_rho"],
            "factor_gate.diagnostic_reverse_abs_rho",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        ),
        selection_tie_tolerance=_require_float(
            mapping["selection_tie_tolerance"],
            "factor_gate.selection_tie_tolerance",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        require_paired_delta_ci_lower_positive=_require_bool(
            mapping["require_paired_delta_ci_lower_positive"],
            "factor_gate.require_paired_delta_ci_lower_positive",
        ),
        require_zero_malformed=_require_bool(
            mapping["require_zero_malformed"],
            "factor_gate.require_zero_malformed",
        ),
    )
    registered = FactorGateConfig(3, 10, 12, 0.8, 0.1, 1.0e-12, True, True)
    if gate != registered:
        raise ValueError("factor_gate does not match registered thresholds")
    return gate


def load_factor_repair_config(
    path: Path,
    *,
    repository_root: Path | None = None,
) -> FactorRepairConfig:
    """Load the single, pre-registered factor-repair development protocol.

    This loader deliberately accepts no command-line overrides: every budget,
    threshold, seed, and factor weight is validated against the approved
    protocol before a runner can consume it.
    """

    path = Path(path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    mapping = _require_mapping(payload, "factor-repair config root")
    expected = {
        "schema_version",
        "identity",
        "development",
        "conditions",
        "task_adaptation_epochs",
        "max_selected_factor_repairs",
        "early_stopping",
        "training",
        "factor_loss",
        "model",
        "paths",
        "schedule",
        "checkpoint_policy",
        "metadata_replay",
        "factor_gate",
    }
    _require_fields(mapping, field="top-level", expected=expected)
    schema_version = _require_int(mapping["schema_version"], "schema_version", minimum=1)
    if schema_version != 1:
        raise ValueError(f"unsupported schema_version: {schema_version}")

    task_adaptation_epochs = _require_int(
        mapping["task_adaptation_epochs"],
        "task_adaptation_epochs",
        minimum=1,
    )
    if task_adaptation_epochs != 60:
        raise ValueError("task_adaptation_epochs must equal registered budget 60")
    max_selected = _require_int(
        mapping["max_selected_factor_repairs"],
        "max_selected_factor_repairs",
        minimum=0,
    )
    if max_selected != 1:
        raise ValueError("max_selected_factor_repairs must equal registered value 1")
    early_stopping = _require_bool(mapping["early_stopping"], "early_stopping")
    if early_stopping:
        raise ValueError("early_stopping must be false for registered protocol")

    root = (repository_root or path.parent).resolve()
    return FactorRepairConfig(
        schema_version=schema_version,
        identity=_parse_factor_identity(mapping["identity"]),
        development=_parse_development_protocol(mapping["development"]),
        conditions=_parse_factor_conditions(mapping["conditions"]),
        task_adaptation_epochs=task_adaptation_epochs,
        max_selected_factor_repairs=max_selected,
        early_stopping=early_stopping,
        training=_parse_factor_training(mapping["training"]),
        factor_loss=_parse_factor_alignment(mapping["factor_loss"]),
        model=_parse_factor_model(mapping["model"]),
        paths=_parse_factor_paths(mapping["paths"], root),
        schedule=_parse_factor_schedule(mapping["schedule"]),
        checkpoint_policy=_parse_checkpoint_policy(mapping["checkpoint_policy"]),
        metadata_replay=_parse_metadata_replay(mapping["metadata_replay"]),
        factor_gate=_parse_factor_gate(mapping["factor_gate"]),
        source_path=path,
    )
