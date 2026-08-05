"""Object-balanced natural factor supervision."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import numbers

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ObjectFactorTarget:
    """Natural factor targets attached to one annotated object."""

    batch_index: int
    class_id: int
    box_xyxy_normalized: tuple[float, float, float, float]
    target: tuple[float, float]
    valid: tuple[bool, bool]


def _sequence_values(value: object, field: str, length: int) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence of length {length}")
    values = tuple(value)
    if len(values) != length:
        raise ValueError(f"{field} must contain exactly {length} values")
    return values


def _finite_real(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _validate_box(box_xyxy_normalized: object) -> tuple[float, float, float, float]:
    values = _sequence_values(box_xyxy_normalized, "normalized ROI", 4)
    x1, y1, x2, y2 = tuple(
        _finite_real(value, "normalized ROI coordinate") for value in values
    )
    if x1 >= x2 or y1 >= y2:
        raise ValueError("normalized ROI must be ordered and non-empty")
    return x1, y1, x2, y2


def map_normalized_box_to_feature_roi(
    box_xyxy_normalized: Sequence[float],
    height: int,
    width: int,
) -> tuple[int, int, int, int] | None:
    """Map one normalized box independently to one feature-map size."""

    if (
        isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
        or isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
    ):
        raise ValueError("feature-map height and width must be positive integers")

    x1, y1, x2, y2 = _validate_box(box_xyxy_normalized)
    x1, x2 = max(0.0, x1), min(1.0, x2)
    y1, y2 = max(0.0, y1), min(1.0, y2)

    left = math.floor(x1 * width)
    top = math.floor(y1 * height)
    right = math.ceil(x2 * width)
    bottom = math.ceil(y2 * height)
    left, right = max(0, min(width, left)), max(0, min(width, right))
    top, bottom = max(0, min(height, top)), max(0, min(height, bottom))
    if left >= right or top >= bottom:
        return None
    return left, top, right, bottom


def pool_object_roi(
    factor_map: torch.Tensor,
    roi: tuple[int, int, int, int],
) -> torch.Tensor:
    """Average a ``[channels, height, width]`` object ROI spatially."""

    if not isinstance(factor_map, torch.Tensor) or factor_map.ndim != 3:
        raise ValueError("object factor map must be a CHW tensor")
    left, top, right, bottom = roi
    if not (
        isinstance(left, int)
        and isinstance(top, int)
        and isinstance(right, int)
        and isinstance(bottom, int)
        and 0 <= left < right <= factor_map.shape[-1]
        and 0 <= top < bottom <= factor_map.shape[-2]
    ):
        raise ValueError("ROI must be a non-empty in-bounds integer rectangle")
    return factor_map[:, top:bottom, left:right].mean(dim=(-2, -1))


def _validate_node_maps(
    node_maps: object,
    *,
    check_finite: bool,
) -> tuple[torch.Tensor, ...]:
    if isinstance(node_maps, (str, bytes)) or not isinstance(node_maps, Sequence):
        raise ValueError("node_maps must be a sequence of tensors")
    maps = tuple(node_maps)
    if not maps:
        raise ValueError("node_maps must contain at least one node map")

    batch_size: int | None = None
    device: torch.device | None = None
    for factor_map in maps:
        if (
            not isinstance(factor_map, torch.Tensor)
            or factor_map.ndim != 4
            or factor_map.shape[1] != 2
            or not factor_map.is_floating_point()
            or factor_map.shape[0] <= 0
            or factor_map.shape[-2] <= 0
            or factor_map.shape[-1] <= 0
        ):
            raise ValueError(
                "each node map must be a non-empty floating BCHW tensor with 2 channels"
            )
        if check_finite and not torch.isfinite(factor_map).all():
            raise ValueError("node factor maps must contain only finite values")
        if batch_size is None:
            batch_size = factor_map.shape[0]
            device = factor_map.device
        elif factor_map.shape[0] != batch_size:
            raise ValueError("all node maps must share the same batch size")
        elif factor_map.device != device:
            raise ValueError("all node maps must share the same device")
    return maps


def _validate_targets(
    targets: object,
    batch_size: int | None,
) -> tuple[ObjectFactorTarget, ...]:
    if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence):
        raise ValueError("targets must be a sequence of ObjectFactorTarget values")

    validated: list[ObjectFactorTarget] = []
    for item in targets:
        if not isinstance(item, ObjectFactorTarget):
            raise ValueError("targets must contain only ObjectFactorTarget values")
        if (
            isinstance(item.batch_index, bool)
            or not isinstance(item.batch_index, int)
            or item.batch_index < 0
            or (batch_size is not None and item.batch_index >= batch_size)
        ):
            raise ValueError("target batch_index must be an in-bounds integer")
        if (
            isinstance(item.class_id, bool)
            or not isinstance(item.class_id, int)
            or item.class_id < 0
        ):
            raise ValueError("target class_id must be a non-negative integer")
        _validate_box(item.box_xyxy_normalized)
        for value in _sequence_values(item.target, "target", 2):
            target_value = _finite_real(value, "target value")
            if not 0.0 <= target_value <= 1.0:
                raise ValueError("target value must be within [0, 1]")
        valid = _sequence_values(item.valid, "valid", 2)
        if any(type(value) is not bool for value in valid):
            raise ValueError("valid must contain exactly two boolean values")
        validated.append(item)
    return tuple(validated)


def _validate_empty_counter(empty_roi_counter: object) -> None:
    if (
        not isinstance(empty_roi_counter, list)
        or not empty_roi_counter
        or isinstance(empty_roi_counter[0], bool)
        or not isinstance(empty_roi_counter[0], int)
        or empty_roi_counter[0] < 0
    ):
        raise ValueError("empty_roi_counter must be a non-empty list of integers")


def _differentiable_zero(node_maps: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return node_maps[0].sum() * 0.0


def object_balanced_factor_loss(
    node_maps: Sequence[torch.Tensor],
    targets: Sequence[ObjectFactorTarget],
    *,
    empty_roi_counter: list[int] | None = None,
    check_finite: bool = True,
) -> torch.Tensor:
    """Compute object -> class -> node balanced natural factor loss."""

    if type(check_finite) is not bool:
        raise ValueError("check_finite must be a boolean")
    maps = _validate_node_maps(node_maps, check_finite=check_finite)
    batch_size = maps[0].shape[0]
    validated_targets = _validate_targets(targets, batch_size)
    if empty_roi_counter is not None:
        _validate_empty_counter(empty_roi_counter)
    if not validated_targets:
        return _differentiable_zero(maps)

    node_losses: list[torch.Tensor] = []
    for factor_map in maps:
        class_losses: dict[int, list[torch.Tensor]] = {}
        for item in validated_targets:
            roi = map_normalized_box_to_feature_roi(
                item.box_xyxy_normalized,
                factor_map.shape[-2],
                factor_map.shape[-1],
            )
            if roi is None:
                if empty_roi_counter is not None:
                    empty_roi_counter[0] += 1
                continue
            pooled = pool_object_roi(factor_map[item.batch_index], roi)
            if not any(item.valid):
                continue
            mask = torch.as_tensor(item.valid, device=pooled.device, dtype=torch.bool)
            truth = pooled.new_tensor(item.target)
            object_loss = F.smooth_l1_loss(
                pooled[mask],
                truth[mask],
                reduction="mean",
            )
            class_losses.setdefault(item.class_id, []).append(object_loss)
        if class_losses:
            class_means = [
                torch.stack(class_losses[class_id]).mean()
                for class_id in sorted(class_losses)
            ]
            node_losses.append(torch.stack(class_means).mean())

    if not node_losses:
        return _differentiable_zero(maps)
    return torch.stack(node_losses).mean()


__all__ = [
    "ObjectFactorTarget",
    "map_normalized_box_to_feature_roi",
    "object_balanced_factor_loss",
    "pool_object_roi",
]
