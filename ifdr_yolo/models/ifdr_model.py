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
from ifdr_yolo.data.ifdr_dataset import (
    COUNTERFACTUAL_IMAGE_KEY,
    COUNTERFACTUAL_WEIGHT_KEY,
)
from ifdr_yolo.experiments.gradient_diagnostics import (
    ScheduledGradientDiagnostics,
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


def _split_paired_tensor(
    value: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if value.ndim == 0 or value.shape[0] != 2 * batch_size:
        raise RuntimeError(
            "paired prediction tensors must have leading dimension 2B"
        )
    return value[:batch_size]


def _main_paired_predictions(
    predictions: dict[str, object],
    batch_size: int,
) -> dict[str, object]:
    if not isinstance(predictions, dict):
        raise RuntimeError("paired IFDR predictions must be a mapping")
    result: dict[str, object] = {}
    for name, value in predictions.items():
        if isinstance(value, torch.Tensor):
            result[name] = _split_paired_tensor(value, batch_size)
        elif isinstance(value, list) and all(
            isinstance(item, torch.Tensor) for item in value
        ):
            result[name] = [
                _split_paired_tensor(item, batch_size) for item in value
            ]
        else:
            raise RuntimeError(
                f"unsupported paired prediction field: {name}"
            )
    return result


def _split_paired_contexts(
    contexts: dict[int, ReliabilityContext],
    batch_size: int,
) -> tuple[
    dict[int, ReliabilityContext],
    dict[int, ReliabilityContext],
]:
    main: dict[int, ReliabilityContext] = {}
    clean: dict[int, ReliabilityContext] = {}
    for index, context in contexts.items():
        if (
            context.factors.shape[0] != 2 * batch_size
            or context.branch_weights.shape[0] != 2 * batch_size
        ):
            raise RuntimeError(
                "paired reliability contexts must have leading dimension 2B"
            )
        main[index] = ReliabilityContext(
            factors=context.factors[:batch_size],
            branch_weights=context.branch_weights[:batch_size],
            gate_strength=context.gate_strength,
        )
        clean[index] = ReliabilityContext(
            factors=context.factors[batch_size:],
            branch_weights=context.branch_weights[batch_size:],
            gate_strength=context.gate_strength,
        )
    return main, clean


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
        gradient_diagnostic_interval: int = 0,
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
        self._loss_reliability_contexts: tuple[
            dict[int, ReliabilityContext],
            dict[int, ReliabilityContext] | None,
        ] | None = None
        self._gradient_diagnostics = ScheduledGradientDiagnostics(
            interval=gradient_diagnostic_interval,
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

    def consume_loss_reliability_contexts(
        self,
    ) -> tuple[
        dict[int, ReliabilityContext],
        dict[int, ReliabilityContext] | None,
    ]:
        contexts = self._loss_reliability_contexts
        if contexts is not None:
            self._loss_reliability_contexts = None
            return contexts
        return self.consume_reliability_context(), None

    def _counterfactual_pair_is_active(
        self,
        batch: dict[str, object],
    ) -> bool:
        weight = batch.get(COUNTERFACTUAL_WEIGHT_KEY)
        return (
            self.counterfactual_gain > 0.0
            and self.factor_supervision_schedule > 0.0
            and isinstance(weight, torch.Tensor)
            and bool(torch.any(weight > 0.0))
        )

    def loss(self, batch, preds=None):
        if preds is not None or not self._counterfactual_pair_is_active(batch):
            return super().loss(batch, preds)
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        image = batch.get("img")
        clean_image = batch.get(COUNTERFACTUAL_IMAGE_KEY)
        if not isinstance(image, torch.Tensor) or not isinstance(
            clean_image,
            torch.Tensor,
        ):
            raise RuntimeError(
                "counterfactual training requires tensor image pairs"
            )
        if image.shape != clean_image.shape or image.ndim != 4:
            raise RuntimeError(
                "counterfactual image pairs must have matching BCHW shapes"
            )
        batch_size = image.shape[0]
        if batch_size <= 0:
            raise RuntimeError("counterfactual image batch must not be empty")

        paired_predictions = self.forward(
            torch.cat((image, clean_image), dim=0)
        )
        paired_contexts = self.consume_reliability_context()
        self._loss_reliability_contexts = _split_paired_contexts(
            paired_contexts,
            batch_size,
        )
        main_predictions = _main_paired_predictions(
            paired_predictions,
            batch_size,
        )
        try:
            return self.criterion(main_predictions, batch)
        finally:
            self._loss_reliability_contexts = None

    def adapt_localization_factors(
        self,
        factors: torch.Tensor,
    ) -> torch.Tensor:
        if self.localization_adapter is None:
            return factors
        return self.localization_adapter(factors.detach())

    def shared_reliability_parameters(
        self,
    ) -> tuple[torch.nn.Parameter, ...]:
        first_layer = self.model[self._fusion_node_indices[0]]
        assert isinstance(first_layer, ReliabilityGatedConcat)
        return tuple(first_layer.reliability_estimator.parameters())

    def observe_gradient_diagnostics(
        self,
        losses: dict[str, torch.Tensor],
    ) -> dict[str, object] | None:
        return self._gradient_diagnostics.observe(
            losses,
            self.shared_reliability_parameters(),
        )

    def drain_gradient_diagnostics(
        self,
    ) -> tuple[dict[str, object], ...]:
        return self._gradient_diagnostics.drain()

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
