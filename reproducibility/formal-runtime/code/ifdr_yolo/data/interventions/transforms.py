from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np

from ifdr_yolo.data.interventions.schema import (
    InterventionKind,
    InterventionRole,
    InterventionSpec,
)
from ifdr_yolo.data.interventions.targets import FactorTarget


FEATHER_FRACTION = 0.15
MINIMUM_DOWNSAMPLE_SCALE = 0.125


@dataclass(frozen=True)
class AppliedIntervention:
    image: np.ndarray
    sampling_target: np.ndarray
    visibility_target: np.ndarray
    sampling_weight: np.ndarray
    visibility_weight: np.ndarray
    parameters: dict[str, Any]


def _validate_image(image: object) -> np.ndarray:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] <= 0
        or image.shape[1] <= 0
    ):
        raise ValueError("image must be a non-empty uint8 HWC array")
    return image


def _region_pixels(
    spec: InterventionSpec,
    *,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    if spec.role is InterventionRole.GLOBAL:
        return 0, 0, width, height
    assert spec.region_xyxy is not None
    x1, y1, x2, y2 = spec.region_xyxy
    left = min(int(math.floor(x1 * width)), width - 1)
    top = min(int(math.floor(y1 * height)), height - 1)
    right = max(min(int(math.ceil(x2 * width)), width), left + 1)
    bottom = max(min(int(math.ceil(y2 * height)), height), top + 1)
    return left, top, right, bottom


def _soft_region_mask(
    *,
    height: int,
    width: int,
    region: tuple[int, int, int, int],
) -> np.ndarray:
    left, top, right, bottom = region
    if region == (0, 0, width, height):
        return np.ones((height, width), dtype=np.float32)
    region_width = right - left
    region_height = bottom - top
    feather = max(
        1.0,
        min(region_width, region_height) * FEATHER_FRACTION,
    )
    xs = np.arange(left, right, dtype=np.float32) + 0.5
    ys = np.arange(top, bottom, dtype=np.float32) + 0.5
    x_weight = np.clip(
        np.minimum(xs - left, right - xs) / feather,
        0.0,
        1.0,
    )
    y_weight = np.clip(
        np.minimum(ys - top, bottom - ys) / feather,
        0.0,
        1.0,
    )
    local = np.minimum(y_weight[:, None], x_weight[None, :])
    mask = np.zeros((height, width), dtype=np.float32)
    mask[top:bottom, left:right] = local
    return mask


def _sampling_view(
    crop: np.ndarray,
    *,
    strength: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    scale = max(
        MINIMUM_DOWNSAMPLE_SCALE,
        1.0 - (1.0 - MINIMUM_DOWNSAMPLE_SCALE) * strength,
    )
    height, width = crop.shape[:2]
    generator = np.random.default_rng(seed)
    phase_x, phase_y = generator.uniform(-0.5, 0.5, size=2)
    shifted = cv2.warpAffine(
        crop,
        np.array(
            [[1.0, 0.0, phase_x], [0.0, 1.0, phase_y]],
            dtype=np.float32,
        ),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    reduced_width = max(1, int(round(width * scale)))
    reduced_height = max(1, int(round(height * scale)))
    reduced = cv2.resize(
        shifted,
        (reduced_width, reduced_height),
        interpolation=cv2.INTER_AREA,
    )
    restored = cv2.resize(
        reduced,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    return restored, {
        "downsample_scale": float(scale),
        "reduced_width": reduced_width,
        "reduced_height": reduced_height,
        "phase_x": float(phase_x),
        "phase_y": float(phase_y),
        "down_interpolation": "area",
        "up_interpolation": "linear",
    }


def _visibility_view(
    crop: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    height, width = crop.shape[:2]
    generator = np.random.default_rng(seed)
    grid_height = min(4, height)
    grid_width = min(4, width)
    base = crop.reshape(-1, 3).mean(axis=0)
    field = generator.normal(
        loc=base,
        scale=24.0,
        size=(grid_height, grid_width, 3),
    )
    field = np.clip(field, 0.0, 255.0).astype(np.float32)
    occluder = cv2.resize(
        field,
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    return np.clip(occluder, 0.0, 255.0), {
        "occluder": "low_frequency_local_color",
        "grid_width": grid_width,
        "grid_height": grid_height,
        "noise_scale": 24.0,
    }


def _factor_maps(
    target: FactorTarget,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support = mask > 0.0
    sampling_target = np.zeros_like(mask)
    visibility_target = np.zeros_like(mask)
    sampling_weight = (
        mask.copy()
        if target.sampling_valid
        else np.zeros_like(mask)
    )
    visibility_weight = (
        mask.copy()
        if target.visibility_valid
        else np.zeros_like(mask)
    )
    if target.sampling_valid:
        sampling_target[support] = target.sampling
    if target.visibility_valid:
        visibility_target[support] = target.visibility
    return (
        sampling_target,
        visibility_target,
        sampling_weight,
        visibility_weight,
    )


def apply_intervention(
    image: np.ndarray,
    spec: InterventionSpec,
    target: FactorTarget,
) -> AppliedIntervention:
    source = _validate_image(image)
    height, width = source.shape[:2]
    region = _region_pixels(
        spec,
        height=height,
        width=width,
    )
    mask = _soft_region_mask(
        height=height,
        width=width,
        region=region,
    )
    result = source.copy()
    parameters: dict[str, Any] = {
        "operator": spec.kind.value,
        "region_pixels": list(region),
        "feather_fraction": FEATHER_FRACTION,
        "seed": spec.seed,
        "strength": spec.strength,
    }
    left, top, right, bottom = region
    crop = source[top:bottom, left:right]
    local_mask = mask[top:bottom, left:right, None]
    if spec.kind is InterventionKind.SAMPLING and spec.strength > 0.0:
        degraded, details = _sampling_view(
            crop,
            strength=spec.strength,
            seed=spec.seed,
        )
        alpha = local_mask
        mixed = crop.astype(np.float32) * (1.0 - alpha)
        mixed += degraded.astype(np.float32) * alpha
        result[top:bottom, left:right] = np.clip(
            np.rint(mixed),
            0,
            255,
        ).astype(np.uint8)
        parameters.update(details)
    elif (
        spec.kind is InterventionKind.VISIBILITY
        and spec.strength > 0.0
    ):
        occluder, details = _visibility_view(crop, seed=spec.seed)
        alpha = local_mask * spec.strength
        mixed = crop.astype(np.float32) * (1.0 - alpha)
        mixed += occluder * alpha
        result[top:bottom, left:right] = np.clip(
            np.rint(mixed),
            0,
            255,
        ).astype(np.uint8)
        parameters.update(details)
    factor_maps = _factor_maps(target, mask)
    return AppliedIntervention(
        image=result,
        sampling_target=factor_maps[0],
        visibility_target=factor_maps[1],
        sampling_weight=factor_maps[2],
        visibility_weight=factor_maps[3],
        parameters=parameters,
    )
