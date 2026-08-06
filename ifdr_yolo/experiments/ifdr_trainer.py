from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, RANK

from ifdr_yolo.data.ifdr_dataset import (
    COUNTERFACTUAL_IMAGE_KEY,
    build_ifdr_dataset,
)
from ifdr_yolo.data.interventions.sampler import SamplingPolicy
from ifdr_yolo.models.ifdr_model import IFDRDetectionModel


def _non_negative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class FusionSchedule:
    """Epoch schedule that preserves the pretrained graph before gating."""

    frozen_epochs: int = 5
    ramp_epochs: int = 10

    def __post_init__(self) -> None:
        _non_negative_integer(self.frozen_epochs, "frozen_epochs")
        ramp_epochs = _non_negative_integer(self.ramp_epochs, "ramp_epochs")
        if ramp_epochs == 0:
            raise ValueError("ramp_epochs must be positive")

    def value_at(self, epoch: int) -> float:
        epoch = _non_negative_integer(epoch, "epoch")
        if epoch < self.frozen_epochs:
            return 0.0
        progress = (epoch - self.frozen_epochs + 1) / self.ramp_epochs
        return min(1.0, progress)


@dataclass(frozen=True)
class IFDRComponentSwitches:
    fusion_gate: bool = True
    dcli: bool = True
    factor_supervision: bool = True
    interventions: bool = True
    semantic_protection: bool = False
    counterfactual_consistency: bool = False

    def __post_init__(self) -> None:
        for field in (
            "fusion_gate",
            "dcli",
            "factor_supervision",
            "interventions",
            "semantic_protection",
            "counterfactual_consistency",
        ):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"{field} must be a boolean")


def _unwrap_training_model(model: object) -> object:
    while hasattr(model, "module"):
        model = getattr(model, "module")
    return model


def apply_fusion_schedule(trainer: object) -> float:
    """Set the schedule from the absolute epoch, including resumed runs."""

    epoch = getattr(trainer, "epoch", None)
    schedule = getattr(trainer, "fusion_schedule", None)
    if not isinstance(schedule, FusionSchedule):
        raise RuntimeError("trainer does not define a valid fusion schedule")
    value = schedule.value_at(epoch)
    model = _unwrap_training_model(getattr(trainer, "model", None))
    switches = getattr(
        trainer,
        "component_switches",
        IFDRComponentSwitches(),
    )
    if not isinstance(switches, IFDRComponentSwitches):
        raise RuntimeError("trainer component switches are invalid")
    component_setter = getattr(model, "set_component_schedules", None)
    legacy_setter = getattr(model, "set_reliability_schedule", None)
    if callable(component_setter):
        component_setter(
            fusion=value if switches.fusion_gate else 0.0,
            dcli=value if switches.dcli else 0.0,
            factor_supervision=(
                value if switches.factor_supervision else 0.0
            ),
        )
    elif callable(legacy_setter):
        legacy_setter(value)
    else:
        raise RuntimeError("training model does not support reliability gating")
    train_loader = getattr(trainer, "train_loader", None)
    dataset = getattr(train_loader, "dataset", None)
    epoch_setter = getattr(dataset, "set_epoch", None)
    if callable(epoch_setter):
        epoch_setter(epoch)
    setattr(trainer, "fusion_schedule_value", value)
    return value


def flush_gradient_diagnostics(trainer: object) -> None:
    model = _unwrap_training_model(getattr(trainer, "model", None))
    drain = getattr(model, "drain_gradient_diagnostics", None)
    if not callable(drain):
        return
    records = drain()
    if not records:
        return
    save_dir = Path(getattr(trainer, "save_dir"))
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "gradient_diagnostics.jsonl"
    with path.open("a", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def apply_semantic_calibration_phase(
    trainer: object,
    phase: object | None = None,
    *,
    variant: str | None = None,
    epochs: int = 30,
    optimizer: object | None = None,
) -> object:
    from ifdr_yolo.experiments.factor_repair import (
        SemanticCalibrationPhase,
        semantic_calibration_phase,
    )

    if phase is not None:
        if not isinstance(phase, SemanticCalibrationPhase):
            raise TypeError("phase must be a SemanticCalibrationPhase")
        variant = phase.variant
        epochs = phase.epochs
    elif variant is None:
        raise ValueError("variant is required when phase is not provided")
    model = _unwrap_training_model(getattr(trainer, "model", None))
    if optimizer is None:
        optimizer = getattr(trainer, "optimizer", None)
    applied = semantic_calibration_phase(
        model,
        variant=variant,
        epochs=epochs,
        optimizer=optimizer,
    )
    setattr(trainer, "semantic_calibration_phase", applied)
    return applied


def run_validation_without_optimizer_step(trainer: object, batch: object) -> object:
    from ifdr_yolo.experiments.factor_repair import run_calibration_validation

    model = _unwrap_training_model(getattr(trainer, "model", None))
    return run_calibration_validation(
        model,
        batch,
        optimizer=getattr(trainer, "optimizer", None),
    )


class IFDRDetectionTrainer(DetectionTrainer):
    """Ultralytics-compatible trainer that owns the IFDR model lifecycle."""

    IFDR_LOSS_NAMES = (
        "box_loss",
        "cls_loss",
        "dfl_loss",
        "factor_loss",
        "counterfactual_loss",
    )

    def __init__(
        self,
        cfg: dict | str = DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
        *,
        fusion_schedule: FusionSchedule | None = None,
        component_switches: IFDRComponentSwitches | None = None,
        intervention_seed: int | None = None,
        intervention_policy: SamplingPolicy | None = None,
    ) -> None:
        self.fusion_schedule = fusion_schedule or FusionSchedule()
        self.component_switches = (
            component_switches or IFDRComponentSwitches()
        )
        self.fusion_schedule_value = 0.0
        self.intervention_seed = intervention_seed
        self.intervention_policy = intervention_policy or SamplingPolicy()
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.loss_names = self.IFDR_LOSS_NAMES
        if self.intervention_seed is None:
            self.intervention_seed = int(self.args.seed)
        if (
            isinstance(self.intervention_seed, bool)
            or not isinstance(self.intervention_seed, int)
            or self.intervention_seed < 0
        ):
            raise ValueError("intervention_seed must be a non-negative integer")
        self.add_callback("on_train_epoch_start", apply_fusion_schedule)
        self.add_callback(
            "on_train_batch_end",
            flush_gradient_diagnostics,
        )

    def get_model(
        self,
        cfg: str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> IFDRDetectionModel:
        model = self.set_model_names_for_load(
            IFDRDetectionModel(
                cfg=cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
            )
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = self.IFDR_LOSS_NAMES
        return validator

    def build_dataset(
        self,
        img_path: str,
        mode: str = "train",
        batch: int | None = None,
    ):
        model = _unwrap_training_model(self.model)
        stride = max(int(model.stride.max()), 32)
        component_switches = getattr(
            self,
            "component_switches",
            IFDRComponentSwitches(),
        )
        return build_ifdr_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=stride,
            intervention_seed=self.intervention_seed,
            interventions_enabled=(
                mode == "train"
                and component_switches.interventions
            ),
            counterfactual_enabled=(
                mode == "train"
                and component_switches.counterfactual_consistency
            ),
            intervention_policy=getattr(
                self,
                "intervention_policy",
                None,
            ),
        )

    def preprocess_batch(self, batch: dict) -> dict:
        batch = super().preprocess_batch(batch)
        counterfactual = batch.get(COUNTERFACTUAL_IMAGE_KEY)
        if counterfactual is None:
            return batch
        if not isinstance(counterfactual, torch.Tensor):
            raise RuntimeError("counterfactual images must be a tensor")
        counterfactual = counterfactual.float() / 255
        if counterfactual.shape[-2:] != batch["img"].shape[-2:]:
            counterfactual = F.interpolate(
                counterfactual,
                size=batch["img"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        batch[COUNTERFACTUAL_IMAGE_KEY] = counterfactual
        return batch
