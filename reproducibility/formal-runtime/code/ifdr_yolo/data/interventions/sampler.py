from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random

from ifdr_yolo.data.interventions.schema import (
    InterventionKind,
    InterventionRole,
    InterventionSpec,
)


def _probability(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{field} must be finite and within [0, 1]")
    return float(value)


@dataclass(frozen=True)
class SamplingPolicy:
    identity_probability: float = 0.2
    sampling_probability: float = 0.4
    visibility_probability: float = 0.4
    minimum_strength: float = 0.1
    maximum_strength: float = 0.8

    def __post_init__(self) -> None:
        probability_fields = (
            "identity_probability",
            "sampling_probability",
            "visibility_probability",
        )
        for field in probability_fields:
            object.__setattr__(
                self,
                field,
                _probability(getattr(self, field), field),
            )
        total = sum(getattr(self, field) for field in probability_fields)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("intervention probabilities must sum to one")
        minimum = _probability(
            self.minimum_strength,
            "minimum_strength",
        )
        maximum = _probability(
            self.maximum_strength,
            "maximum_strength",
        )
        if minimum > maximum:
            raise ValueError(
                "minimum_strength must not exceed maximum_strength"
            )
        object.__setattr__(self, "minimum_strength", minimum)
        object.__setattr__(self, "maximum_strength", maximum)


def _stable_seed(*parts: object) -> int:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


class DeterministicInterventionSampler:
    def __init__(
        self,
        *,
        base_seed: int,
        policy: SamplingPolicy | None = None,
    ) -> None:
        if (
            isinstance(base_seed, bool)
            or not isinstance(base_seed, int)
            or base_seed < 0
        ):
            raise ValueError("base_seed must be a non-negative integer")
        self.base_seed = base_seed
        self.policy = policy or SamplingPolicy()

    def _factor(
        self,
        *,
        image_id: str,
        object_id: int,
        epoch: int,
        slot: int,
    ) -> tuple[InterventionKind, float]:
        selection_seed = _stable_seed(
            "ifdr-intervention-v1",
            self.base_seed,
            image_id,
            object_id,
            epoch,
            slot,
            "factor",
        )
        generator = random.Random(selection_seed)
        draw = generator.random()
        if draw < self.policy.identity_probability:
            return InterventionKind.IDENTITY, 0.0
        if draw < (
            self.policy.identity_probability
            + self.policy.sampling_probability
        ):
            kind = InterventionKind.SAMPLING
        else:
            kind = InterventionKind.VISIBILITY
        strength = generator.uniform(
            self.policy.minimum_strength,
            self.policy.maximum_strength,
        )
        return kind, strength

    def sample_matched_pair(
        self,
        *,
        image_id: str,
        object_id: int,
        epoch: int,
        slot: int,
        object_region: tuple[float, float, float, float],
        background_region: tuple[float, float, float, float],
    ) -> tuple[InterventionSpec, InterventionSpec]:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            raise ValueError("slot must be a non-negative integer")
        kind, strength = self._factor(
            image_id=image_id,
            object_id=object_id,
            epoch=epoch,
            slot=slot,
        )
        common = (
            "ifdr-intervention-v1",
            self.base_seed,
            image_id,
            object_id,
            epoch,
            slot,
        )
        object_spec = InterventionSpec(
            image_id=image_id,
            kind=kind,
            role=InterventionRole.OBJECT,
            strength=strength,
            seed=_stable_seed(*common, "object"),
            object_id=object_id,
            region_xyxy=object_region,
        )
        background_spec = InterventionSpec(
            image_id=image_id,
            kind=kind,
            role=InterventionRole.BACKGROUND,
            strength=strength,
            seed=_stable_seed(*common, "background"),
            region_xyxy=background_region,
        )
        return object_spec, background_spec
