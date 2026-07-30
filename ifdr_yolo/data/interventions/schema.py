from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any


class InterventionKind(str, Enum):
    IDENTITY = "identity"
    SAMPLING = "sampling"
    VISIBILITY = "visibility"


class InterventionRole(str, Enum):
    GLOBAL = "global"
    OBJECT = "object"
    BACKGROUND = "background"


def _normalized_region(
    value: object,
) -> tuple[float, float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("region_xyxy must contain four normalized values")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in value
    ):
        raise ValueError("region_xyxy values must be numeric")
    region = tuple(float(item) for item in value)
    if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in region):
        raise ValueError("region_xyxy values must be finite and within [0, 1]")
    x1, y1, x2, y2 = region
    if x1 >= x2 or y1 >= y2:
        raise ValueError("region_xyxy must have positive area")
    return region


@dataclass(frozen=True)
class InterventionSpec:
    image_id: str
    kind: InterventionKind
    role: InterventionRole
    strength: float
    seed: int
    object_id: int | None = None
    region_xyxy: tuple[float, float, float, float] | None = None
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.image_id, str) or not self.image_id.strip():
            raise ValueError("image_id must be non-empty text")
        if not isinstance(self.kind, InterventionKind):
            raise ValueError("kind must be an InterventionKind")
        if not isinstance(self.role, InterventionRole):
            raise ValueError("role must be an InterventionRole")
        if (
            isinstance(self.strength, bool)
            or not isinstance(self.strength, (int, float))
            or not math.isfinite(float(self.strength))
            or not 0.0 <= float(self.strength) <= 1.0
        ):
            raise ValueError("strength must be finite and within [0, 1]")
        object.__setattr__(self, "strength", float(self.strength))
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if self.object_id is not None and (
            isinstance(self.object_id, bool)
            or not isinstance(self.object_id, int)
            or self.object_id < 0
        ):
            raise ValueError("object_id must be a non-negative integer")
        if self.kind is InterventionKind.IDENTITY and self.strength != 0.0:
            raise ValueError("identity strength must be zero")

        region = (
            None
            if self.region_xyxy is None
            else _normalized_region(self.region_xyxy)
        )
        object.__setattr__(self, "region_xyxy", region)
        if self.role is InterventionRole.GLOBAL:
            if self.object_id is not None or region is not None:
                raise ValueError(
                    "global intervention forbids object_id and region_xyxy"
                )
        elif self.role is InterventionRole.OBJECT:
            if self.object_id is None or region is None:
                raise ValueError(
                    "object intervention requires object_id and region_xyxy"
                )
        elif self.role is InterventionRole.BACKGROUND:
            if self.object_id is not None:
                raise ValueError(
                    "background intervention forbids object_id"
                )
            if region is None:
                raise ValueError(
                    "background intervention requires region_xyxy"
                )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "image_id": self.image_id,
            "kind": self.kind.value,
            "role": self.role.value,
            "strength": self.strength,
            "seed": self.seed,
            "object_id": self.object_id,
            "region_xyxy": (
                None
                if self.region_xyxy is None
                else list(self.region_xyxy)
            ),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "InterventionSpec":
        if not isinstance(payload, dict) or not all(
            isinstance(key, str) for key in payload
        ):
            raise ValueError(
                "intervention payload must be a mapping with string keys"
            )
        expected = {
            "schema_version",
            "image_id",
            "kind",
            "role",
            "strength",
            "seed",
            "object_id",
            "region_xyxy",
        }
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        if missing:
            raise ValueError(f"missing intervention fields: {missing}")
        if unknown:
            raise ValueError(f"unknown intervention fields: {unknown}")
        if payload["schema_version"] != 1:
            raise ValueError("unsupported intervention schema_version")
        try:
            kind = InterventionKind(payload["kind"])
        except (TypeError, ValueError) as error:
            raise ValueError("invalid intervention kind") from error
        try:
            role = InterventionRole(payload["role"])
        except (TypeError, ValueError) as error:
            raise ValueError("invalid intervention role") from error
        return cls(
            image_id=payload["image_id"],
            kind=kind,
            role=role,
            strength=payload["strength"],
            seed=payload["seed"],
            object_id=payload["object_id"],
            region_xyxy=payload["region_xyxy"],
        )
