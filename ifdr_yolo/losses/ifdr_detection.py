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
from ifdr_yolo.models.gated_fusion import ReliabilityContext


FINAL_PYRAMID_CONTEXT_NODES = (17, 20, 23, 26)


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
        iou = bbox_iou(
            pred_bboxes[fg_mask],
            target_bboxes[fg_mask],
            xywh=False,
            CIoU=True,
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
    ) -> None:
        super().__init__(model)
        self._model_ref = weakref.ref(model)
        self.factor_weights = factor_weights
        self.entropy_weight = entropy_weight
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
        factors = flatten_pyramid_factors(contexts, preds["feats"])
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
        self.bbox_loss.set_schedule(model.ifdr_schedule)
        self.bbox_loss.set_uncertainty(uncertainty)
        try:
            return super().get_assigned_targets_and_loss(preds, batch)
        finally:
            self.bbox_loss.discard_uncertainty()
