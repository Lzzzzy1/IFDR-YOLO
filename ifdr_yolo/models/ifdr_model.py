from __future__ import annotations

from dataclasses import dataclass

import torch

from ultralytics.nn.modules import Concat
from ultralytics.nn.tasks import DetectionModel

from ifdr_yolo.models.gated_fusion import (
    ReliabilityContext,
    ReliabilityGatedConcat,
)


@dataclass(frozen=True)
class FusionNodeSpec:
    index: int
    input_channels: tuple[int, int]

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise ValueError("fusion node index must be an integer")
        if self.index < 0:
            raise ValueError("fusion node index must be non-negative")
        if (
            not isinstance(self.input_channels, tuple)
            or len(self.input_channels) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in self.input_channels
            )
        ):
            raise ValueError(
                "fusion node input_channels must contain two positive integers"
            )


DEFAULT_P2_FUSION_SPECS = (
    FusionNodeSpec(11, (576, 384)),
    FusionNodeSpec(14, (384, 192)),
    FusionNodeSpec(17, (192, 96)),
    FusionNodeSpec(20, (96, 192)),
    FusionNodeSpec(23, (192, 384)),
    FusionNodeSpec(26, (384, 576)),
)


def _copy_graph_attributes(source: object, target: object) -> None:
    for name in ("i", "f", "type"):
        if hasattr(source, name):
            setattr(target, name, getattr(source, name))
    target.np = sum(parameter.numel() for parameter in target.parameters())


def install_reliability_fusion(
    model: DetectionModel,
    *,
    specs: tuple[FusionNodeSpec, ...] = DEFAULT_P2_FUSION_SPECS,
    reliability_channels: int = 32,
) -> tuple[int, ...]:
    if not specs:
        raise ValueError("at least one fusion node is required")
    indexes = tuple(spec.index for spec in specs)
    if len(set(indexes)) != len(indexes):
        raise ValueError("fusion node indices must be unique")
    layer_count = len(model.model)
    for spec in specs:
        if spec.index >= layer_count:
            raise ValueError(
                f"fusion node index {spec.index} exceeds model layers"
            )
        layer = model.model[spec.index]
        if not isinstance(layer, Concat):
            raise ValueError(
                f"fusion node {spec.index} must replace an Ultralytics Concat"
            )
        if getattr(layer, "d", None) != 1:
            raise ValueError(
                f"fusion node {spec.index} must concatenate channels"
            )

    for spec in specs:
        original = model.model[spec.index]
        replacement = ReliabilityGatedConcat(
            input_channels=spec.input_channels,
            reliability_channels=reliability_channels,
        )
        _copy_graph_attributes(original, replacement)
        model.model[spec.index] = replacement
    return indexes


class IFDRDetectionModel(DetectionModel):
    def __init__(
        self,
        cfg: str,
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        reliability_channels: int = 32,
        dcli_beta: float = 0.5,
        uncertainty_calibration_gain: float = 0.1,
        uncertainty_factor_weights: tuple[float, float] = (1.0, 1.0),
        dfl_entropy_weight: float = 1.0,
        fusion_specs: tuple[
            FusionNodeSpec,
            ...,
        ] = DEFAULT_P2_FUSION_SPECS,
    ) -> None:
        self.dcli_beta = dcli_beta
        self.uncertainty_calibration_gain = uncertainty_calibration_gain
        self.uncertainty_factor_weights = uncertainty_factor_weights
        self.dfl_entropy_weight = dfl_entropy_weight
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.register_buffer(
            "_ifdr_schedule",
            torch.tensor(0.0),
            persistent=False,
        )
        self._fusion_node_indices = install_reliability_fusion(
            self,
            specs=fusion_specs,
            reliability_channels=reliability_channels,
        )

    @property
    def fusion_node_indices(self) -> tuple[int, ...]:
        return self._fusion_node_indices

    def set_reliability_schedule(self, value: float) -> None:
        for index in self._fusion_node_indices:
            layer = self.model[index]
            assert isinstance(layer, ReliabilityGatedConcat)
            layer.set_schedule(value)
        self._ifdr_schedule.fill_(float(value))

    @property
    def ifdr_schedule(self) -> float:
        return float(self._ifdr_schedule)

    def consume_reliability_context(
        self,
    ) -> dict[int, ReliabilityContext]:
        contexts: dict[int, ReliabilityContext] = {}
        for index in self._fusion_node_indices:
            layer = self.model[index]
            assert isinstance(layer, ReliabilityGatedConcat)
            contexts[index] = layer.consume_context()
        return contexts

    def init_criterion(self):
        from ifdr_yolo.losses.ifdr_detection import IFDRDetectionLoss

        return IFDRDetectionLoss(
            self,
            beta=self.dcli_beta,
            calibration_gain=self.uncertainty_calibration_gain,
            factor_weights=self.uncertainty_factor_weights,
            entropy_weight=self.dfl_entropy_weight,
        )
