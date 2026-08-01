from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import weakref

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist

from ifdr_yolo.losses.dcli import (
    dcli_localization_error,
    derive_localization_uncertainty,
    normalized_dfl_entropy,
)
from ifdr_yolo.data.ifdr_dataset import (
    COUNTERFACTUAL_DELTA_KEY,
    COUNTERFACTUAL_IMAGE_KEY,
    COUNTERFACTUAL_WEIGHT_KEY,
)
from ifdr_yolo.models.gated_fusion import ReliabilityContext


FINAL_PYRAMID_CONTEXT_NODES = (17, 20, 23, 26)
ALL_FUSION_CONTEXT_NODES = (11, 14, 17, 20, 23, 26)


def stable_ciou(
    predicted_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
) -> torch.Tensor:
    """Compute CIoU in at least FP32 to prevent P2 geometry overflow."""

    if (
        not isinstance(predicted_boxes, torch.Tensor)
        or not isinstance(target_boxes, torch.Tensor)
        or predicted_boxes.shape != target_boxes.shape
        or predicted_boxes.ndim < 2
        or predicted_boxes.shape[-1] != 4
        or not predicted_boxes.is_floating_point()
        or not target_boxes.is_floating_point()
    ):
        raise ValueError(
            "predicted_boxes and target_boxes must be matching [..., 4] "
            "floating tensors"
        )
    if not (
        torch.isfinite(predicted_boxes).all()
        and torch.isfinite(target_boxes).all()
    ):
        raise ValueError("CIoU box coordinates must contain finite values")
    dtype = (
        torch.float64
        if torch.float64 in (predicted_boxes.dtype, target_boxes.dtype)
        else torch.float32
    )
    overlap = bbox_iou(
        predicted_boxes.to(dtype=dtype),
        target_boxes.to(dtype=dtype),
        xywh=False,
        CIoU=True,
    )
    if not torch.isfinite(overlap).all():
        raise FloatingPointError(
            "CIoU remained non-finite after FP32 geometry promotion"
        )
    return overlap


def _bounded_scalar(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{field} must be finite and within [0, 1]")
    return float(value)


def flatten_pyramid_factors(
    contexts: Mapping[int, ReliabilityContext],
    feature_maps: Sequence[torch.Tensor],
    *,
    node_indices: tuple[int, ...] = FINAL_PYRAMID_CONTEXT_NODES,
) -> torch.Tensor:
    """Align P2-P5 factor maps with Ultralytics' flattened anchor order."""

    if len(feature_maps) != len(node_indices):
        raise ValueError("feature maps and factor nodes must have equal length")
    flattened: list[torch.Tensor] = []
    batch_size: int | None = None
    for node_index, feature_map in zip(node_indices, feature_maps):
        if node_index not in contexts:
            raise ValueError(f"missing reliability context for node {node_index}")
        if not isinstance(feature_map, torch.Tensor) or feature_map.ndim != 4:
            raise ValueError("feature maps must be BCHW tensors")
        factors = contexts[node_index].factors
        if (
            factors.ndim != 4
            or factors.shape[1] != 2
            or factors.shape[0] != feature_map.shape[0]
            or factors.shape[2:] != feature_map.shape[2:]
        ):
            raise ValueError(
                f"factor map at node {node_index} does not match detect feature"
            )
        if batch_size is None:
            batch_size = factors.shape[0]
        elif factors.shape[0] != batch_size:
            raise ValueError("all factor maps must share batch size")
        flattened.append(
            factors.permute(0, 2, 3, 1).reshape(factors.shape[0], -1, 2)
        )
    return torch.cat(flattened, dim=1)


def multiscale_factor_supervision(
    contexts: Mapping[int, ReliabilityContext],
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    node_indices: tuple[int, ...] = ALL_FUSION_CONTEXT_NODES,
) -> torch.Tensor:
    """Supervise factor semantics at every bidirectional fusion node."""

    if (
        not isinstance(target, torch.Tensor)
        or not isinstance(weight, torch.Tensor)
        or target.ndim != 4
        or target.shape[1] != 2
        or target.shape != weight.shape
        or not target.is_floating_point()
        or not weight.is_floating_point()
    ):
        raise ValueError(
            "factor target and weight must be matching [batch, 2, h, w] tensors"
        )
    if torch.any(weight < 0.0) or not torch.isfinite(weight).all():
        raise ValueError("factor weights must be finite and non-negative")
    node_losses: list[torch.Tensor] = []
    for node_index in node_indices:
        if node_index not in contexts:
            raise ValueError(f"missing reliability context for node {node_index}")
        factors = contexts[node_index].factors
        if factors.shape[:2] != target.shape[:2]:
            raise ValueError(
                f"factor batch/channels at node {node_index} do not match target"
            )
        size = factors.shape[-2:]
        scaled_target = F.interpolate(target, size=size, mode="area")
        scaled_weight = F.interpolate(weight, size=size, mode="area")
        numerator = (
            F.smooth_l1_loss(
                factors,
                scaled_target,
                reduction="none",
            )
            * scaled_weight
        ).sum()
        denominator = scaled_weight.sum()
        node_loss = torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(1e-12),
            factors.sum() * 0.0,
        )
        node_losses.append(node_loss)
    return torch.stack(node_losses).mean()


def multiscale_counterfactual_consistency(
    intervention_contexts: Mapping[int, ReliabilityContext],
    clean_contexts: Mapping[int, ReliabilityContext],
    delta_target: torch.Tensor,
    weight: torch.Tensor,
    *,
    node_indices: tuple[int, ...] = ALL_FUSION_CONTEXT_NODES,
) -> torch.Tensor:
    """Match intervention-induced factor deltas at every fusion node."""

    if (
        not isinstance(delta_target, torch.Tensor)
        or not isinstance(weight, torch.Tensor)
        or delta_target.ndim != 4
        or delta_target.shape[1] != 2
        or delta_target.shape != weight.shape
        or not delta_target.is_floating_point()
        or not weight.is_floating_point()
    ):
        raise ValueError(
            "counterfactual delta and weight must be matching "
            "[batch, 2, h, w] tensors"
        )
    if (
        not torch.isfinite(delta_target).all()
        or not torch.isfinite(weight).all()
        or torch.any(weight < 0.0)
    ):
        raise ValueError(
            "counterfactual targets must be finite with non-negative weights"
        )
    node_losses: list[torch.Tensor] = []
    for node_index in node_indices:
        if node_index not in intervention_contexts:
            raise ValueError(
                f"intervention contexts missing node {node_index}"
            )
        if node_index not in clean_contexts:
            raise ValueError(f"clean contexts missing node {node_index}")
        intervention = intervention_contexts[node_index].factors
        clean = clean_contexts[node_index].factors
        if intervention.shape != clean.shape:
            raise ValueError(
                f"counterfactual factor shapes differ at node {node_index}"
            )
        if intervention.shape[:2] != delta_target.shape[:2]:
            raise ValueError(
                f"counterfactual factors do not match target at node {node_index}"
            )
        size = intervention.shape[-2:]
        scaled_target = F.interpolate(delta_target, size=size, mode="area")
        scaled_weight = F.interpolate(weight, size=size, mode="area")
        numerator = (
            F.smooth_l1_loss(
                intervention - clean,
                scaled_target,
                reduction="none",
            )
            * scaled_weight
        ).sum()
        denominator = scaled_weight.sum()
        node_loss = torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(1e-12),
            (intervention.sum() + clean.sum()) * 0.0,
        )
        node_losses.append(node_loss)
    return torch.stack(node_losses).mean()


class DCLIBboxLoss(BboxLoss):
    """Ultralytics bbox loss with anchor-aligned, bounded DCLI weighting."""

    def __init__(
        self,
        reg_max: int = 16,
        *,
        beta: float = 0.5,
        calibration_gain: float = 0.1,
    ) -> None:
        super().__init__(reg_max)
        self.beta = _bounded_scalar(beta, "beta")
        self.calibration_gain = _bounded_scalar(
            calibration_gain,
            "calibration_gain",
        )
        self.schedule = 1.0
        self._uncertainty: torch.Tensor | None = None

    def set_schedule(self, value: float) -> None:
        self.schedule = _bounded_scalar(value, "schedule")

    def set_uncertainty(self, uncertainty: torch.Tensor) -> None:
        if (
            not isinstance(uncertainty, torch.Tensor)
            or uncertainty.ndim != 2
            or not uncertainty.is_floating_point()
        ):
            raise ValueError("uncertainty must have shape [batch, anchors]")
        if self._uncertainty is not None:
            raise RuntimeError("previous uncertainty has not been consumed")
        self._uncertainty = uncertainty

    def discard_uncertainty(self) -> None:
        self._uncertainty = None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        uncertainty = self._uncertainty
        self._uncertainty = None
        if uncertainty is None:
            raise RuntimeError("DCLI requires fresh localization uncertainty")
        if uncertainty.shape != fg_mask.shape:
            raise ValueError("uncertainty must align with foreground anchors")

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = stable_ciou(
            pred_bboxes[fg_mask],
            target_bboxes[fg_mask],
        )
        localization_error = 1.0 - iou
        foreground_uncertainty = uncertainty[fg_mask].unsqueeze(-1)
        conditioned_error = dcli_localization_error(
            localization_error,
            foreground_uncertainty,
            beta=self.beta * self.schedule,
        )
        loss_iou = (conditioned_error * weight).sum() / target_scores_sum
        effective_calibration_gain = self.calibration_gain * self.schedule
        if effective_calibration_gain > 0.0:
            calibration = F.smooth_l1_loss(
                foreground_uncertainty,
                localization_error.detach().clamp(0.0, 1.0),
                reduction="none",
            )
            loss_iou = loss_iou + effective_calibration_gain * (
                (calibration * weight).sum() / target_scores_sum
            )

        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points,
                target_bboxes,
                self.dfl_loss.reg_max - 1,
            )
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask],
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(
                    pred_dist[fg_mask],
                    target_ltrb[fg_mask],
                    reduction="none",
                ).mean(-1, keepdim=True)
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        return loss_iou, loss_dfl


class IFDRDetectionLoss(v8DetectionLoss):
    """Detection criterion closing factor prediction, DFL entropy and DCLI."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        beta: float = 0.5,
        calibration_gain: float = 0.1,
        factor_weights: tuple[float, float] = (1.0, 1.0),
        entropy_weight: float = 1.0,
        factor_supervision_gain: float = 0.2,
        counterfactual_gain: float = 0.0,
    ) -> None:
        super().__init__(model)
        self._model_ref = weakref.ref(model)
        self.factor_weights = factor_weights
        self.entropy_weight = entropy_weight
        self.factor_supervision_gain = _bounded_scalar(
            factor_supervision_gain,
            "factor_supervision_gain",
        )
        self.counterfactual_gain = _bounded_scalar(
            counterfactual_gain,
            "counterfactual_gain",
        )
        self.bbox_loss = DCLIBboxLoss(
            self.reg_max,
            beta=beta,
            calibration_gain=calibration_gain,
        ).to(self.device)

    def get_assigned_targets_and_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, object],
    ) -> tuple:
        model = self._model_ref()
        if model is None:
            raise RuntimeError("IFDR model is no longer available")
        contexts = model.consume_reliability_context()
        factor_target = batch.get("ifdr_factor_target")
        factor_weight = batch.get("ifdr_factor_weight")
        if not isinstance(factor_target, torch.Tensor) or not isinstance(
            factor_weight,
            torch.Tensor,
        ):
            raise RuntimeError("batch is missing IFDR factor supervision")
        factor_loss = multiscale_factor_supervision(
            contexts,
            factor_target,
            factor_weight,
        )
        counterfactual_loss = factor_loss * 0.0
        counterfactual_weight = batch.get(COUNTERFACTUAL_WEIGHT_KEY)
        if (
            self.counterfactual_gain > 0.0
            and model.factor_supervision_schedule > 0.0
            and isinstance(counterfactual_weight, torch.Tensor)
            and torch.any(counterfactual_weight > 0.0)
        ):
            counterfactual_image = batch.get(COUNTERFACTUAL_IMAGE_KEY)
            counterfactual_delta = batch.get(COUNTERFACTUAL_DELTA_KEY)
            if (
                not isinstance(counterfactual_image, torch.Tensor)
                or counterfactual_image.ndim != 4
            ):
                raise RuntimeError(
                    "batch is missing BCHW counterfactual images"
                )
            if not isinstance(counterfactual_delta, torch.Tensor):
                raise RuntimeError(
                    "batch is missing counterfactual delta targets"
                )
            model(counterfactual_image)
            clean_contexts = model.consume_reliability_context()
            counterfactual_loss = multiscale_counterfactual_consistency(
                contexts,
                clean_contexts,
                counterfactual_delta,
                counterfactual_weight,
            )
        factors = flatten_pyramid_factors(contexts, preds["feats"])
        factors = model.adapt_localization_factors(factors)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        dfl_entropy = normalized_dfl_entropy(
            pred_distri,
            reg_max=self.reg_max,
        )
        uncertainty = derive_localization_uncertainty(
            factors,
            dfl_entropy,
            factor_weights=self.factor_weights,
            entropy_weight=self.entropy_weight,
        )
        self.bbox_loss.set_schedule(model.dcli_schedule)
        self.bbox_loss.set_uncertainty(uncertainty)
        try:
            result = super().get_assigned_targets_and_loss(
                preds,
                batch,
            )
            assignments, detection_loss, _ = result
            factor_component = (
                factor_loss
                * self.factor_supervision_gain
                * model.factor_supervision_schedule
            )
            counterfactual_component = (
                counterfactual_loss
                * self.counterfactual_gain
                * model.factor_supervision_schedule
            )
            auxiliary = torch.stack(
                (
                    factor_component + counterfactual_component,
                    factor_loss * 0.0,
                    factor_loss * 0.0,
                )
            )
            loss = detection_loss + auxiliary
            reported = torch.cat(
                (
                    detection_loss.detach(),
                    factor_component.detach().reshape(1),
                    counterfactual_component.detach().reshape(1),
                )
            )
            return assignments, loss, reported
        finally:
            self.bbox_loss.discard_uncertainty()
