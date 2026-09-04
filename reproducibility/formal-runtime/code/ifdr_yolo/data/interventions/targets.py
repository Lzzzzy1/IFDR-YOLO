from __future__ import annotations

from dataclasses import dataclass
import math

from ifdr_yolo.data.interventions.schema import (
    InterventionKind,
    InterventionRole,
    InterventionSpec,
)


def _optional_unit_interval(
    value: float | None,
    field: str,
) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{field} must be finite and within [0, 1]")
    return float(value)


def _combine_degradation(base: float, added: float) -> float:
    if added == 0.0:
        return base
    return 1.0 - (1.0 - base) * (1.0 - added)


@dataclass(frozen=True)
class FactorTarget:
    sampling: float
    visibility: float
    sampling_valid: bool
    visibility_valid: bool

    def __post_init__(self) -> None:
        for field in ("sampling", "visibility"):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{field} must be finite and within [0, 1]")
            object.__setattr__(self, field, float(value))
        if not isinstance(self.sampling_valid, bool):
            raise ValueError("sampling_valid must be a boolean")
        if not isinstance(self.visibility_valid, bool):
            raise ValueError("visibility_valid must be a boolean")


def factor_target_for_spec(
    spec: InterventionSpec,
    *,
    natural_sampling: float | None = None,
    natural_occlusion: float | None = None,
) -> FactorTarget:
    sampling_base = _optional_unit_interval(
        natural_sampling,
        "natural_sampling",
    )
    visibility_base = _optional_unit_interval(
        natural_occlusion,
        "natural_occlusion",
    )
    if spec.role is InterventionRole.BACKGROUND:
        sampling_base = 0.0
        visibility_base = 0.0
    elif spec.role is InterventionRole.GLOBAL:
        if sampling_base is None and spec.kind is InterventionKind.SAMPLING:
            sampling_base = 0.0
        if (
            visibility_base is None
            and spec.kind is InterventionKind.VISIBILITY
        ):
            visibility_base = 0.0

    sampling_added = (
        spec.strength
        if spec.kind is InterventionKind.SAMPLING
        else 0.0
    )
    visibility_added = (
        spec.strength
        if spec.kind is InterventionKind.VISIBILITY
        else 0.0
    )
    sampling_valid = sampling_base is not None
    visibility_valid = visibility_base is not None
    sampling = (
        _combine_degradation(sampling_base, sampling_added)
        if sampling_valid
        else 0.0
    )
    visibility = (
        _combine_degradation(visibility_base, visibility_added)
        if visibility_valid
        else 0.0
    )
    return FactorTarget(
        sampling=sampling,
        visibility=visibility,
        sampling_valid=sampling_valid,
        visibility_valid=visibility_valid,
    )
