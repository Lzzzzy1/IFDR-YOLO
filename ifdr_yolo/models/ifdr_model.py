from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ultralytics.nn.modules import Concat
from ultralytics.nn.tasks import DetectionModel

from ifdr_yolo.models.gated_fusion import (
    ReliabilityContext,
    ReliabilityEstimator,
    ReliabilityGatedConcat,
    ResidualFactorAdapter,
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
    semantic_protection: bool = False,
) -> tuple[int, ...]:
    if not specs:
        raise ValueError("at least one fusion node is required")
    indexes = tuple(spec.index for spec in specs)
    if len(set(indexes)) != len(indexes):
        raise ValueError("fusion node indices must be unique")
    layer_count = len(model.model)
    shared_estimator = ReliabilityEstimator(reliability_channels)
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
            reliability_estimator=shared_estimator,
            semantic_protection=semantic_protection,
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
        factor_supervision_gain: float = 0.2,
        semantic_protection: bool = False,
        counterfactual_gain: float = 0.0,
        fusion_specs: tuple[
            FusionNodeSpec,
            ...,
        ] = DEFAULT_P2_FUSION_SPECS,
    ) -> None:
        self.dcli_beta = dcli_beta
        self.uncertainty_calibration_gain = uncertainty_calibration_gain
        self.uncertainty_factor_weights = uncertainty_factor_weights
        self.dfl_entropy_weight = dfl_entropy_weight
        self.factor_supervision_gain = factor_supervision_gain
        self.counterfactual_gain = counterfactual_gain
        if not isinstance(semantic_protection, bool):
            raise ValueError("semantic_protection must be a boolean")
        self.semantic_protection = semantic_protection
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.register_buffer(
            "_fusion_schedule",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "_dcli_schedule",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "_factor_supervision_schedule",
            torch.tensor(0.0),
            persistent=False,
        )
        self._fusion_node_indices = install_reliability_fusion(
            self,
            specs=fusion_specs,
            reliability_channels=reliability_channels,
            semantic_protection=semantic_protection,
        )
        self.localization_adapter = (
            ResidualFactorAdapter()
            if semantic_protection
            else None
        )

    @property
    def fusion_node_indices(self) -> tuple[int, ...]:
        return self._fusion_node_indices

    def set_reliability_schedule(self, value: float) -> None:
        self.set_component_schedules(
            fusion=value,
            dcli=value,
            factor_supervision=value,
        )

    def set_component_schedules(
        self,
        *,
        fusion: float,
        dcli: float,
        factor_supervision: float,
    ) -> None:
        values = {
            "fusion": fusion,
            "dcli": dcli,
            "factor_supervision": factor_supervision,
        }
        for name, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(
                    f"{name} schedule must be finite and within [0, 1]"
                )
        for index in self._fusion_node_indices:
            layer = self.model[index]
            assert isinstance(layer, ReliabilityGatedConcat)
            layer.set_schedule(fusion)
        self._fusion_schedule.fill_(float(fusion))
        self._dcli_schedule.fill_(float(dcli))
        self._factor_supervision_schedule.fill_(
            float(factor_supervision)
        )

    @property
    def ifdr_schedule(self) -> float:
        return self.dcli_schedule

    @property
    def fusion_schedule(self) -> float:
        return float(self._fusion_schedule)

    @property
    def dcli_schedule(self) -> float:
        return float(self._dcli_schedule)

    @property
    def factor_supervision_schedule(self) -> float:
        return float(self._factor_supervision_schedule)

    def consume_reliability_context(
        self,
    ) -> dict[int, ReliabilityContext]:
        contexts: dict[int, ReliabilityContext] = {}
        for index in self._fusion_node_indices:
            layer = self.model[index]
            assert isinstance(layer, ReliabilityGatedConcat)
            contexts[index] = layer.consume_context()
        return contexts

    def adapt_localization_factors(
        self,
        factors: torch.Tensor,
    ) -> torch.Tensor:
        if self.localization_adapter is None:
            return factors
        return self.localization_adapter(factors.detach())

    def init_criterion(self):
        from ifdr_yolo.losses.ifdr_detection import IFDRDetectionLoss

        return IFDRDetectionLoss(
            self,
            beta=self.dcli_beta,
            calibration_gain=self.uncertainty_calibration_gain,
            factor_weights=self.uncertainty_factor_weights,
            entropy_weight=self.dfl_entropy_weight,
            factor_supervision_gain=self.factor_supervision_gain,
            counterfactual_gain=self.counterfactual_gain,
        )
