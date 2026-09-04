"""Object-balanced natural factor supervision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


def factor_specificity_loss(
    clean: torch.Tensor,
    target: torch.Tensor,
    background: torch.Tensor,
    *,
    margin: float = 0.05,
    clean_background: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rank intervention response over a matched background response.

    The margin is part of the registered experiment and is intentionally not
    configurable.  All three tensors are kept in the computation graph so
    the caller can backpropagate to each view's factor path.
    """

    if margin != 0.05:
        raise ValueError("registered specificity margin is 0.05")
    if clean_background is None:
        clean_background = clean
    if not all(
        isinstance(value, torch.Tensor)
        and value.is_floating_point()
        for value in (clean, target, background, clean_background)
    ):
        raise ValueError("specificity inputs must be floating tensors")
    if (
        clean.shape != target.shape
        or clean.shape != background.shape
        or clean.shape != clean_background.shape
    ):
        raise ValueError("specificity inputs must share shape")
    if (
        clean.device != target.device
        or clean.device != background.device
        or clean.device != clean_background.device
    ):
        raise ValueError("specificity inputs must share device")
    if clean.numel() == 0:
        return clean.sum() * 0.0
    return torch.relu(
        (background - clean_background) + 0.05 - (target - clean)
    ).mean()


def _pair_field(pair: object, *names: str, default: object = None) -> object:
    """Read a frozen pair attribute without importing the dataset type."""

    for name in names:
        if isinstance(pair, Mapping) and name in pair:
            return pair[name]
        if hasattr(pair, name):
            return getattr(pair, name)
    return default


def _pair_float(pair: object, names: tuple[str, ...], field: str) -> float:
    value = _pair_field(pair, *names)
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"specificity pair {field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"specificity pair {field} must be finite")
    return result


def _pair_box(pair: object, names: tuple[str, ...], field: str) -> tuple[float, float, float, float]:
    value = _pair_field(pair, *names)
    if value is None:
        raise ValueError(f"specificity pair is missing {field}")
    return _validate_box(value)


def _pair_batch_index(pair: object) -> int:
    value = _pair_field(
        pair,
        "batch_index",
        "batch_idx",
        "target_batch_index",
        default=0,
    )
    if value is None:
        value = 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("specificity pair batch index must be non-negative")
    return value


def _pair_channels(pair: object) -> tuple[int, ...]:
    """Resolve only the channels explicitly selected by the pair metadata."""

    channel = _pair_field(
        pair,
        "factor_channel",
        "channel",
        "factor_index",
        "channel_index",
    )
    if channel is not None:
        if isinstance(channel, bool) or not isinstance(channel, int) or channel not in (0, 1):
            raise ValueError("specificity pair factor channel must be 0 or 1")
        return (channel,)
    factor_kind = _pair_field(pair, "factor_kind", "factor", "kind")
    if factor_kind is None:
        valid = _pair_field(pair, "valid", "valid_channels")
        if valid is not None:
            values = _sequence_values(valid, "specificity pair valid", 2)
            if any(type(value) is not bool for value in values):
                raise ValueError("specificity pair valid channels must be boolean")
            selected = tuple(index for index, value in enumerate(values) if value)
            if len(selected) != 1:
                raise ValueError("specificity pair must select exactly one factor channel")
            return selected
        raise ValueError("specificity pair is missing factor channel")
    factor_kind = getattr(factor_kind, "value", factor_kind)
    if not isinstance(factor_kind, str):
        raise ValueError("specificity pair factor kind must be a string")
    normalized = factor_kind.strip().lower()
    if normalized in {"sampling", "sample", "0"}:
        return (0,)
    if normalized in {"visibility", "visible", "1"}:
        return (1,)
    raise ValueError("specificity pair factor kind must select sampling or visibility")


def factor_specificity_from_contexts(
    clean_contexts: Mapping[int, object],
    target_contexts: Mapping[int, object],
    background_contexts: Mapping[int, object],
    intervention: object,
    *,
    node_indices: tuple[int, ...] = (17, 20, 23, 26),
) -> torch.Tensor:
    """Pool matched target/background ROIs and rank their factor deltas."""

    if isinstance(intervention, Mapping) or hasattr(intervention, "severity"):
        pairs = (intervention,)
    elif isinstance(intervention, Sequence) and not isinstance(intervention, (str, bytes)):
        pairs = tuple(intervention)
    else:
        raise ValueError("intervention must be a sequence of specificity pairs")
    maps_by_view: list[tuple[torch.Tensor, ...]] = []
    for name, contexts in (
        ("clean", clean_contexts),
        ("target", target_contexts),
        ("background", background_contexts),
    ):
        if not isinstance(contexts, Mapping):
            raise ValueError(f"{name} contexts must be a mapping")
        maps: list[torch.Tensor] = []
        for node in node_indices:
            if node not in contexts:
                raise ValueError(f"{name} contexts missing node {node}")
            context = contexts[node]
            factors = getattr(context, "factors", None)
            if (
                not isinstance(factors, torch.Tensor)
                or factors.ndim != 4
                or factors.shape[1] != 2
                or not factors.is_floating_point()
            ):
                raise ValueError(f"{name} context node {node} has invalid factors")
            maps.append(factors)
        maps_by_view.append(tuple(maps))
    clean_maps, target_maps, background_maps = maps_by_view
    if not pairs:
        return (
            clean_maps[0].sum()
            + target_maps[0].sum()
            + background_maps[0].sum()
        ) * 0.0

    node_losses: list[torch.Tensor] = []
    for clean_map, target_map, background_map in zip(
        clean_maps,
        target_maps,
        background_maps,
    ):
        if clean_map.shape != target_map.shape or clean_map.shape != background_map.shape:
            raise ValueError("specificity context factor maps must share shape")
        weighted_losses: list[torch.Tensor] = []
        weights: list[float] = []
        for pair in pairs:
            severity = _pair_float(pair, ("severity", "intervention_severity"), "severity")
            weight_value = _pair_field(pair, "weight", "specificity_weight", default=1.0)
            if isinstance(weight_value, bool) or not isinstance(weight_value, numbers.Real):
                raise ValueError("specificity pair weight must be a real number")
            pair_weight = float(weight_value)
            if not math.isfinite(pair_weight) or pair_weight < 0.0:
                raise ValueError("specificity pair weight must be finite and non-negative")
            if severity < 0.25 or pair_weight == 0.0:
                continue
            target_box = _pair_box(
                pair,
                ("target_box_xyxy_normalized", "target_box", "object_box", "box_xyxy_normalized"),
                "target ROI",
            )
            background_box = _pair_box(
                pair,
                ("background_box_xyxy_normalized", "background_box", "background_region"),
                "background ROI",
            )
            batch_index = _pair_batch_index(pair)
            if batch_index >= clean_map.shape[0]:
                raise ValueError("specificity pair batch index is out of bounds")
            target_roi = map_normalized_box_to_feature_roi(
                target_box, clean_map.shape[-2], clean_map.shape[-1]
            )
            background_roi = map_normalized_box_to_feature_roi(
                background_box, clean_map.shape[-2], clean_map.shape[-1]
            )
            if target_roi is None or background_roi is None:
                continue
            channels = _pair_channels(pair)
            clean_target = pool_object_roi(clean_map[batch_index], target_roi)[list(channels)]
            clean_background = pool_object_roi(clean_map[batch_index], background_roi)[list(channels)]
            target_target = pool_object_roi(target_map[batch_index], target_roi)[list(channels)]
            background_target = pool_object_roi(background_map[batch_index], background_roi)[list(channels)]
            weighted_losses.append(
                factor_specificity_loss(
                    clean_target,
                    target_target,
                    background_target,
                    clean_background=clean_background,
                )
            )
            weights.append(pair_weight)
        if weighted_losses:
            weight_tensor = weighted_losses[0].new_tensor(weights)
            node_losses.append(torch.stack(weighted_losses).mul(weight_tensor).sum() / weight_tensor.sum())
        else:
            node_losses.append(
                (clean_map.sum() + target_map.sum() + background_map.sum()) * 0.0
            )
    return torch.stack(node_losses).mean()


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
    "factor_specificity_from_contexts",
    "factor_specificity_loss",
    "map_normalized_box_to_feature_roi",
    "object_balanced_factor_loss",
    "pool_object_roi",
]
