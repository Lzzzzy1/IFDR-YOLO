from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, RANK

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
    setter = getattr(model, "set_reliability_schedule", None)
    if not callable(setter):
        raise RuntimeError("training model does not support reliability gating")
    setter(value)
    setattr(trainer, "fusion_schedule_value", value)
    return value


class IFDRDetectionTrainer(DetectionTrainer):
    """Ultralytics-compatible trainer that owns the IFDR model lifecycle."""

    def __init__(
        self,
        cfg: dict | str = DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
        *,
        fusion_schedule: FusionSchedule | None = None,
    ) -> None:
        self.fusion_schedule = fusion_schedule or FusionSchedule()
        self.fusion_schedule_value = 0.0
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.add_callback("on_train_epoch_start", apply_fusion_schedule)

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
