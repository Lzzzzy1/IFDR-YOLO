"""Natural-transfer evidence for IFDR reliability factors.

The audit deliberately has no dependency on a statistics package.  Values are
pooled at the image level for uncertainty estimates, while all of the
registered controls (height and class) are handled with NumPy only.  The
public functions return JSON-friendly dictionaries; the two records exposed
by this module are frozen dataclasses so an observation or gate decision
cannot be changed after it has been recorded.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Callable, Iterable, Sequence

import numpy as np


_FACTORS = ("sampling", "visibility")
_REGION_ROLES = frozenset(("target", "background"))
_INTERVENTION_KINDS = frozenset(("natural", "clean", "sampling", "visibility"))
_DEFAULT_SEEDS = (17, 29, 41)
_DEFAULT_NODES = (11, 14, 17, 20, 23, 26)
_DEFAULT_BOOTSTRAP_REPLICATES = 2000
_DEFAULT_BOOTSTRAP_SEED = 20260804
_DEFAULT_MONOTONIC_THRESHOLD = 0.80
DEFAULT_INTERVENTION_SEVERITIES = (0.25, 0.50, 0.75, 1.0)
_MAX_EVIDENCE_EXAMPLES = 100


def _is_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _unit(value: object, name: str) -> float:
    number = _finite(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return number


@dataclass(frozen=True)
class NaturalFactorObservation:
    """One pooled ROI observation.

    ``natural`` rows are object targets and have no pair.  Controlled rows
    carry a pair id and are either a clean baseline or one of the two factor
    interventions.  The validation here is intentionally strict so an audit
    cannot silently mix a target and an unmatched background.
    """

    seed: int
    node_id: int
    image_id: str
    object_id: int
    class_id: int
    box_height: float
    region_role: str
    intervention_kind: str
    intervention_severity: float
    pair_id: str | None
    natural_sampling: float
    natural_visibility: float
    predicted_sampling: float
    predicted_visibility: float
    branch_weights: tuple[float, float]
    # The audit only needs the numeric class id.  Keeping an optional name is
    # useful when JSONL producers already carry it and does not change the
    # registered observation schema.
    class_name: str | None = None
    intervention_factor: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.seed, "seed"),
            (self.node_id, "node_id"),
            (self.object_id, "object_id"),
            (self.class_id, "class_id"),
        ):
            if not _is_integer(value) or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.class_id not in (0, 1, 2):
            raise ValueError("class_id must be one of 0, 1, or 2")
        if not isinstance(self.image_id, str) or not self.image_id.strip():
            raise ValueError("image_id must be a non-empty string")
        if self.class_name is not None and (
            not isinstance(self.class_name, str) or not self.class_name.strip()
        ):
            raise ValueError("class_name must be a non-empty string when provided")
        if self.class_name is not None:
            expected_class_name = ("Car", "Pedestrian", "Cyclist")[self.class_id]
            if self.class_name != expected_class_name:
                raise ValueError(
                    f"class_name {self.class_name!r} conflicts with class_id {self.class_id}"
                )
        if not isinstance(self.region_role, str) or self.region_role not in _REGION_ROLES:
            raise ValueError("region_role must be target or background")
        if (
            not isinstance(self.intervention_kind, str)
            or self.intervention_kind not in _INTERVENTION_KINDS
        ):
            raise ValueError(
                "intervention_kind must be natural, clean, sampling, or visibility"
            )
        height = _finite(self.box_height, "box_height")
        if height <= 0.0:
            raise ValueError("box_height must be positive")
        severity = _unit(self.intervention_severity, "intervention_severity")
        _unit(self.natural_sampling, "natural_sampling")
        _unit(self.natural_visibility, "natural_visibility")
        _unit(self.predicted_sampling, "predicted_sampling")
        _unit(self.predicted_visibility, "predicted_visibility")
        if not isinstance(self.branch_weights, tuple) or len(self.branch_weights) != 2:
            raise ValueError("branch_weights must be a two-element tuple")
        for index, weight in enumerate(self.branch_weights):
            _unit(weight, f"branch_weights[{index}]")
        if abs(float(self.branch_weights[0]) + float(self.branch_weights[1]) - 1.0) > 1e-6:
            raise ValueError("branch_weights must sum to 1 within 1e-6")

        if self.intervention_kind == "natural":
            if self.region_role != "target":
                raise ValueError("natural observations must be target rows")
            if severity != 0.0:
                raise ValueError("natural observations must have severity 0")
            if self.pair_id is not None:
                raise ValueError("natural observations must not have a pair_id")
            if self.intervention_factor is not None:
                raise ValueError("natural observations must have intervention_factor=None")
        else:
            if self.pair_id is None or not isinstance(self.pair_id, str) or not self.pair_id.strip():
                raise ValueError("intervention rows require a non-empty pair_id")
            if self.intervention_factor not in {"sampling", "visibility"}:
                raise ValueError(
                    "intervention rows require intervention_factor sampling or visibility"
                )
            if (
                self.intervention_kind in {"sampling", "visibility"}
                and self.intervention_factor != self.intervention_kind
            ):
                raise ValueError(
                    "intervention_factor must match intervention_kind for factor rows"
                )
            if self.intervention_kind == "clean" and severity != 0.0:
                raise ValueError("clean intervention rows must have severity 0")
            if self.intervention_kind in {"sampling", "visibility"} and severity <= 0.0:
                raise ValueError(
                    f"{self.intervention_kind} intervention severity must be greater than 0 "
                    "(severity 0 is reserved for clean/natural)"
                )

        # Dataclasses do not coerce NumPy scalar values.  Normalize the
        # validated record once so every downstream result is JSON-native and
        # callers cannot accidentally leak ``np.int64``/``np.float64`` values.
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "node_id", int(self.node_id))
        object.__setattr__(self, "object_id", int(self.object_id))
        object.__setattr__(self, "class_id", int(self.class_id))
        object.__setattr__(self, "box_height", float(height))
        object.__setattr__(self, "intervention_severity", float(severity))
        object.__setattr__(self, "natural_sampling", float(self.natural_sampling))
        object.__setattr__(self, "natural_visibility", float(self.natural_visibility))
        object.__setattr__(self, "predicted_sampling", float(self.predicted_sampling))
        object.__setattr__(self, "predicted_visibility", float(self.predicted_visibility))
        object.__setattr__(
            self,
            "branch_weights",
            (float(self.branch_weights[0]), float(self.branch_weights[1])),
        )


def _sort_key(row: NaturalFactorObservation) -> tuple[Any, ...]:
    return (
        row.image_id,
        row.seed,
        row.node_id,
        row.object_id,
        row.pair_id or "",
        row.intervention_kind,
        row.intervention_severity,
        row.region_role,
    )


def _validated_observations(
    observations: Iterable[NaturalFactorObservation],
) -> tuple[NaturalFactorObservation, ...]:
    rows = tuple(observations)
    if not rows:
        return ()
    if any(not isinstance(row, NaturalFactorObservation) for row in rows):
        raise ValueError("observations must contain NaturalFactorObservation records")
    ordered = tuple(sorted(rows, key=_sort_key))
    natural_identity: dict[tuple[int, int, str, int], NaturalFactorObservation] = {}
    for row in ordered:
        if row.intervention_kind != "natural":
            continue
        identity = (row.seed, row.node_id, row.image_id, row.object_id)
        previous = natural_identity.get(identity)
        if previous is not None:
            if previous == row:
                detail = "duplicate natural observation"
            else:
                detail = "conflicting natural observation"
            raise ValueError(f"{detail} for identity {identity}")
        natural_identity[identity] = row
    return ordered


def _validate_factor(factor: str) -> str:
    if not isinstance(factor, str) or factor not in _FACTORS:
        raise ValueError("factor must be sampling or visibility")
    return factor


def _validate_expected_severities(
    severities: Sequence[float],
) -> tuple[float, ...]:
    values = tuple(_unit(value, "expected intervention severity") for value in severities)
    if not values:
        raise ValueError("expected intervention severities must not be empty")
    if any(value <= 0.0 for value in values):
        raise ValueError("expected intervention severities must be within (0, 1]")
    if any(
        abs(values[left] - values[right]) <= 1e-9
        for left in range(len(values))
        for right in range(left + 1, len(values))
    ) or any(values[index + 1] <= values[index] for index in range(len(values) - 1)):
        raise ValueError("expected intervention severities must be unique and increasing")
    return values


def _canonical_severity(value: float, expected: Sequence[float]) -> float:
    """Map a measured severity to its registered value within float tolerance."""

    value = float(value)
    matches = [candidate for candidate in expected if abs(value - candidate) <= 1e-9]
    if len(matches) == 1:
        return float(matches[0])
    return value


def _validate_bootstrap(replicates: int, seed: int, confidence: float) -> None:
    if not _is_integer(replicates) or int(replicates) < 2:
        raise ValueError("bootstrap replicates must be an integer greater than or equal to 2")
    if not _is_integer(seed) or int(seed) < 0:
        raise ValueError("bootstrap seed must be a non-negative integer")
    confidence_value = _finite(confidence, "bootstrap confidence")
    if not 0.0 < confidence_value < 1.0:
        raise ValueError("bootstrap confidence must be within (0, 1)")


def average_tie_rank(values: Sequence[float]) -> tuple[float, ...]:
    """Return one-based average-tie ranks without SciPy."""

    values_tuple = tuple(values)
    if not values_tuple:
        return ()
    numbers = [_finite(value, "rank value") for value in values_tuple]
    ordered = sorted(enumerate(numbers), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(numbers)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][1] == ordered[position][1]:
            end += 1
        mean_rank = (position + 1 + end) / 2.0
        for index in range(position, end):
            ranks[ordered[index][0]] = mean_rank
        position = end
    return tuple(ranks)

def _result(
    *,
    rho: float | None,
    n: int,
    status: str,
    reason: str = "",
) -> dict[str, object]:
    success = status == "ok" and rho is not None and math.isfinite(rho)
    if success:
        rho = float(max(-1.0, min(1.0, rho)))
    else:
        rho = None
    return {
        "rho": rho,
        "n": int(n),
        "status": status,
        "success": success,
        "finite": status in {"ok", "constant", "insufficient"},
        "reason": reason,
    }


def _pearson(x: Sequence[float], y: Sequence[float]) -> dict[str, object]:
    if len(x) != len(y):
        raise ValueError("correlation vectors must have equal length")
    n = len(x)
    if n < 2:
        return _result(rho=None, n=n, status="insufficient", reason="at least two samples are required")
    left = np.asarray(tuple(x), dtype=np.float64)
    right = np.asarray(tuple(y), dtype=np.float64)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("correlation vectors must be finite")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    left_norm = float(np.linalg.norm(left_centered))
    right_norm = float(np.linalg.norm(right_centered))
    if left_norm == 0.0 or right_norm == 0.0:
        return _result(rho=None, n=n, status="constant", reason="a correlation vector is constant")
    rho = float(np.dot(left_centered, right_centered) / (left_norm * right_norm))
    if not math.isfinite(rho):
        return _result(rho=None, n=n, status="nonfinite", reason="correlation was non-finite")
    return _result(rho=rho, n=n, status="ok")


def spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, object]:
    """Average-tie Spearman correlation with explicit failure status."""

    if len(x) != len(y):
        raise ValueError("correlation vectors must have equal length")
    if len(x) < 2:
        return _result(
            rho=None,
            n=len(x),
            status="insufficient",
            reason="at least two samples are required",
        )
    return _pearson(average_tie_rank(x), average_tie_rank(y))


def partial_spearman(
    target: Sequence[float],
    prediction: Sequence[float],
    box_height: Sequence[float],
    class_ids: Sequence[int],
) -> dict[str, object]:
    """Partial Spearman controlling ranked height and reference-coded class."""

    lengths = {len(target), len(prediction), len(box_height), len(class_ids)}
    if len(lengths) != 1:
        raise ValueError("partial Spearman vectors must have equal length")
    n = len(target)
    if n < 3:
        return _result(
            rho=None,
            n=n,
            status="insufficient",
            reason="at least three samples are required for residual Spearman",
        )
    height = np.asarray(tuple(_finite(value, "box_height") for value in box_height), dtype=np.float64)
    classes: list[int] = []
    for value in class_ids:
        if not _is_integer(value) or int(value) < 0:
            raise ValueError("class_ids must be non-negative integers")
        classes.append(int(value))
    target_rank = np.asarray(average_tie_rank(target), dtype=np.float64)
    prediction_rank = np.asarray(average_tie_rank(prediction), dtype=np.float64)
    if not np.isfinite(target_rank).all() or not np.isfinite(prediction_rank).all():
        raise ValueError("partial Spearman vectors must be finite")
    if np.linalg.norm(target_rank - target_rank.mean()) == 0.0 or np.linalg.norm(
        prediction_rank - prediction_rank.mean()
    ) == 0.0:
        return _result(rho=None, n=n, status="constant", reason="target or prediction is constant")
    height_rank = np.asarray(average_tie_rank(height), dtype=np.float64)
    unique_classes = sorted(set(classes))
    # Intercept + all class indicators is rank-deficient.  Use the first
    # registered class as the reference level so the residual estimand is
    # stable and the bootstrap can reuse fixed cross-products.
    class_dummies = unique_classes[1:]
    design = np.column_stack(
        [
            np.ones(n, dtype=np.float64),
            height_rank,
            *[np.asarray([float(value == klass) for value in classes]) for klass in class_dummies],
        ]
    )
    try:
        target_fit = design @ np.linalg.lstsq(design, target_rank, rcond=None)[0]
        prediction_fit = design @ np.linalg.lstsq(design, prediction_rank, rcond=None)[0]
    except np.linalg.LinAlgError:
        return _result(rho=None, n=n, status="nonfinite", reason="control regression failed")
    target_residual = target_rank - target_fit
    prediction_residual = prediction_rank - prediction_fit
    # ``lstsq`` leaves machine-scale residue when a ranked response is fully
    # explained by the registered controls.  Treat that residue as constant,
    # using a scale-relative tolerance so a genuine (even small) rank signal
    # is not rounded away.
    target_centered_residual = target_residual - target_residual.mean()
    prediction_centered_residual = prediction_residual - prediction_residual.mean()
    target_tolerance = 1e-12 * max(1.0, float(np.linalg.norm(target_rank)))
    prediction_tolerance = 1e-12 * max(1.0, float(np.linalg.norm(prediction_rank)))
    if (
        float(np.linalg.norm(target_centered_residual)) <= target_tolerance
        or float(np.linalg.norm(prediction_centered_residual)) <= prediction_tolerance
    ):
        return _result(
            rho=None,
            n=n,
            status="constant",
            reason="residual target or prediction is constant",
        )
    result = _pearson(target_residual, prediction_residual)
    if result["status"] == "constant":
        result["reason"] = "residual target or prediction is constant"
    return result


def _factor_natural(row: NaturalFactorObservation, factor: str) -> float:
    return row.natural_sampling if factor == "sampling" else row.natural_visibility


def _factor_prediction(row: NaturalFactorObservation, factor: str) -> float:
    return row.predicted_sampling if factor == "sampling" else row.predicted_visibility


def _natural_rows(observations: Sequence[NaturalFactorObservation]) -> tuple[NaturalFactorObservation, ...]:
    return tuple(
        row
        for row in observations
        if row.intervention_kind == "natural" and row.region_role == "target"
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_image_ids(
    image_ids: Sequence[str],
    *,
    replicates: int = _DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = _DEFAULT_BOOTSTRAP_SEED,
) -> tuple[tuple[str, ...], ...]:
    """Draw image IDs with replacement; every box in an image follows its ID."""

    _validate_bootstrap(replicates, seed, 0.95)
    unique = tuple(sorted(set(image_ids)))
    if not unique:
        return ()
    if int(replicates) * len(unique) > 100_000:
        raise ValueError("bootstrap image-id samples are limited to 100000 draws")
    generator = np.random.default_rng(int(seed))
    samples: list[tuple[str, ...]] = []
    for _ in range(int(replicates)):
        indices = generator.integers(0, len(unique), size=len(unique))
        samples.append(tuple(unique[int(index)] for index in indices))
    return tuple(samples)


def _cluster_rows(
    observations: Sequence[NaturalFactorObservation],
    sampled_images: Sequence[str],
) -> tuple[NaturalFactorObservation, ...]:
    by_image: dict[str, list[NaturalFactorObservation]] = defaultdict(list)
    for row in observations:
        by_image[row.image_id].append(row)
    rows: list[NaturalFactorObservation] = []
    for image_id in sampled_images:
        rows.extend(sorted(by_image[image_id], key=_sort_key))
    return tuple(rows)


@dataclass(frozen=True)
class _RankGroups:
    """Sorted value groups and row-to-group inverse for weighted midranks."""

    values: np.ndarray
    inverse: np.ndarray


@dataclass(frozen=True)
class _BootstrapData:
    """Natural rows and all reusable value/image grouping metadata."""

    images: tuple[str, ...]
    image_inverse: np.ndarray
    row_counts: np.ndarray
    target: np.ndarray
    prediction: np.ndarray
    height: np.ndarray
    classes: np.ndarray
    target_groups: _RankGroups
    prediction_groups: _RankGroups
    height_groups: _RankGroups
    class_dummies: tuple[int, ...]


def _value_groups(values: np.ndarray) -> _RankGroups:
    unique, inverse = np.unique(values, return_inverse=True)
    return _RankGroups(
        values=np.asarray(unique, dtype=np.float64),
        inverse=np.asarray(inverse, dtype=np.intp),
    )


def _precompute_bootstrap_data(
    observations: Sequence[NaturalFactorObservation],
    factor: str,
) -> _BootstrapData:
    """Precompute image and value groups once; ranks remain replicate-specific."""

    images = tuple(sorted({row.image_id for row in observations}))
    image_index = {image_id: index for index, image_id in enumerate(images)}
    image_inverse = np.asarray(
        [image_index[row.image_id] for row in observations], dtype=np.intp
    )
    target = np.asarray(tuple(_factor_natural(row, factor) for row in observations), dtype=np.float64)
    prediction = np.asarray(
        tuple(_factor_prediction(row, factor) for row in observations), dtype=np.float64
    )
    height = np.asarray(tuple(row.box_height for row in observations), dtype=np.float64)
    classes = np.asarray(tuple(int(row.class_id) for row in observations), dtype=np.intp)
    class_values = tuple(sorted(set(int(value) for value in classes)))
    return _BootstrapData(
        images=images,
        image_inverse=image_inverse,
        row_counts=np.bincount(image_inverse, minlength=len(images)).astype(np.float64),
        target=target,
        prediction=prediction,
        height=height,
        classes=classes,
        target_groups=_value_groups(target),
        prediction_groups=_value_groups(prediction),
        height_groups=_value_groups(height),
        class_dummies=class_values[1:],
    )


def _rank_from_groups(groups: _RankGroups, observation_weights: np.ndarray) -> np.ndarray | None:
    total = float(np.sum(observation_weights))
    if total < 2.0:
        return None
    group_weights = np.bincount(
        groups.inverse,
        weights=observation_weights,
        minlength=len(groups.values),
    ).astype(np.float64)
    group_rank = np.cumsum(group_weights) - 0.5 * group_weights + 0.5
    # Affine rank normalization preserves all correlations and keeps the
    # residual least-squares scale stable across multinomial draws.
    return (group_rank[groups.inverse] - 1.0) / (total - 1.0)


def _observation_weights(data: _BootstrapData, image_weights: np.ndarray) -> np.ndarray:
    return np.asarray(image_weights[data.image_inverse], dtype=np.float64)


def _weighted_pearson(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
) -> float | None:
    total = float(np.sum(weights))
    if total < 2.0:
        return None
    left_mean = float(np.dot(weights, left) / total)
    right_mean = float(np.dot(weights, right) / total)
    left_centered = left - left_mean
    right_centered = right - right_mean
    left_ss = float(np.dot(weights, left_centered * left_centered))
    right_ss = float(np.dot(weights, right_centered * right_centered))
    if left_ss <= 0.0 or right_ss <= 0.0:
        return None
    covariance = float(np.dot(weights, left_centered * right_centered))
    rho = covariance / math.sqrt(left_ss * right_ss)
    return float(max(-1.0, min(1.0, rho))) if math.isfinite(rho) else None


def _exact_raw_statistic(
    data: _BootstrapData,
    image_weights: np.ndarray,
) -> float | None:
    observation_weights = _observation_weights(data, image_weights)
    target_rank = _rank_from_groups(data.target_groups, observation_weights)
    prediction_rank = _rank_from_groups(data.prediction_groups, observation_weights)
    if target_rank is None or prediction_rank is None:
        return None
    return _weighted_pearson(target_rank, prediction_rank, observation_weights)


def _exact_residual_statistic(
    data: _BootstrapData,
    image_weights: np.ndarray,
) -> float | None:
    observation_weights = _observation_weights(data, image_weights)
    target_rank = _rank_from_groups(data.target_groups, observation_weights)
    prediction_rank = _rank_from_groups(data.prediction_groups, observation_weights)
    height_rank = _rank_from_groups(data.height_groups, observation_weights)
    if target_rank is None or prediction_rank is None or height_rank is None:
        return None
    total = float(np.sum(observation_weights))
    if total < 3.0:
        return None
    design = np.column_stack(
        [
            np.ones(len(data.classes), dtype=np.float64),
            height_rank,
            *[
                np.asarray(data.classes == klass, dtype=np.float64)
                for klass in data.class_dummies
            ],
        ]
    )
    weighted_design = design * observation_weights[:, None]
    design_cross = design.T @ weighted_design
    target_cross = design.T @ (target_rank * observation_weights)
    prediction_cross = design.T @ (prediction_rank * observation_weights)
    try:
        design_pinv = np.linalg.pinv(design_cross, rcond=1e-12)
    except np.linalg.LinAlgError:
        return None
    target_beta = design_pinv @ target_cross
    prediction_beta = design_pinv @ prediction_cross
    target_ss = float(
        np.dot(observation_weights, target_rank * target_rank)
        - np.dot(target_beta, target_cross)
    )
    prediction_ss = float(
        np.dot(observation_weights, prediction_rank * prediction_rank)
        - np.dot(prediction_beta, prediction_cross)
    )
    residual_cross = float(
        np.dot(observation_weights, target_rank * prediction_rank)
        - np.dot(target_beta, prediction_cross)
    )
    target_mean = float(np.dot(observation_weights, target_rank) / total)
    prediction_mean = float(np.dot(observation_weights, prediction_rank) / total)
    target_raw_ss = float(
        np.dot(observation_weights, (target_rank - target_mean) ** 2)
    )
    prediction_raw_ss = float(
        np.dot(observation_weights, (prediction_rank - prediction_mean) ** 2)
    )
    tolerance = np.finfo(np.float64).eps * 1024.0 * max(
        1.0, target_raw_ss, prediction_raw_ss
    )
    if -tolerance <= target_ss < 0.0:
        target_ss = 0.0
    if -tolerance <= prediction_ss < 0.0:
        prediction_ss = 0.0
    if abs(residual_cross) <= tolerance:
        residual_cross = 0.0
    if target_ss <= tolerance or prediction_ss <= tolerance:
        return None
    rho = residual_cross / math.sqrt(target_ss * prediction_ss)
    return float(max(-1.0, min(1.0, rho))) if math.isfinite(rho) else None


def _exact_bootstrap(
    *,
    data: _BootstrapData,
    statistic: Callable[[_BootstrapData, np.ndarray], float | None],
    estimate: float | None,
    replicates: int,
    seed: int,
    confidence: float,
    return_samples: bool = False,
) -> dict[str, object]:
    _validate_bootstrap(replicates, seed, confidence)
    image_count = len(data.images)
    if image_count == 0:
        return {
            "estimate": estimate,
            "ci_lower": None,
            "ci_upper": None,
            "status": "insufficient",
            "reason": "no image clusters",
            "replicates": int(replicates),
            "valid_replicates": 0,
            "sampling_unit": "image_id",
            "unique_image_count": 0,
        }
    if return_samples and int(replicates) * image_count > 100_000:
        raise ValueError("return_samples is limited to 100000 sampled image draws")
    generator = np.random.default_rng(int(seed))
    values: list[float] = []
    sampled_ids: list[list[str]] = []
    sampled_sizes: list[list[int]] = []
    for _ in range(int(replicates)):
        sampled_indices = generator.integers(0, image_count, size=image_count)
        image_weights = np.bincount(sampled_indices, minlength=image_count).astype(np.float64)
        value = statistic(data, image_weights)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
        if return_samples:
            sampled_ids.append([data.images[int(index)] for index in sampled_indices])
            sampled_sizes.append(
                [int(data.row_counts[int(index)]) for index in sampled_indices]
            )
    required_valid = math.ceil(0.95 * int(replicates))
    if len(values) < required_valid:
        return {
            "estimate": estimate,
            "ci_lower": None,
            "ci_upper": None,
            "status": "insufficient",
            "reason": "fewer than 95% of bootstrap replicates were finite",
            "replicates": int(replicates),
            "valid_replicates": len(values),
            "sampling_unit": "image_id",
            "unique_image_count": image_count,
            **({"sampled_image_ids": sampled_ids, "sampled_cluster_sizes": sampled_sizes} if return_samples else {}),
        }
    tail = (1.0 - float(confidence)) / 2.0
    return {
        "estimate": estimate,
        "ci_lower": _quantile(values, tail),
        "ci_upper": _quantile(values, 1.0 - tail),
        "status": "ok",
        "reason": "",
        "replicates": int(replicates),
        "valid_replicates": len(values),
        "sampling_unit": "image_id",
        "unique_image_count": image_count,
        **({"sampled_image_ids": sampled_ids, "sampled_cluster_sizes": sampled_sizes} if return_samples else {}),
    }


def image_cluster_bootstrap(
    observations: Iterable[NaturalFactorObservation],
    factor: str = "sampling",
    *,
    replicates: int = _DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = _DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
    return_samples: bool = False,
) -> dict[str, object]:
    """Bootstrap raw natural Spearman by unique image ID."""

    factor = _validate_factor(factor)
    rows = _validated_observations(observations)
    natural = _natural_rows(rows)
    data = _precompute_bootstrap_data(natural, factor)
    estimate_result = spearman(
        tuple(_factor_natural(row, factor) for row in natural),
        tuple(_factor_prediction(row, factor) for row in natural),
    )
    estimate = estimate_result.get("rho")
    estimate_value = float(estimate) if isinstance(estimate, Real) and math.isfinite(float(estimate)) else None
    result = _exact_bootstrap(
        data=data,
        statistic=_exact_raw_statistic,
        estimate=estimate_value,
        replicates=replicates,
        seed=seed,
        confidence=confidence,
        return_samples=return_samples,
    )
    # ``image_cluster_bootstrap`` historically reads naturally as a raw
    # estimate, so preserve that label alongside the generic estimate key.
    result["factor"] = factor
    result["metric"] = "raw_spearman"
    return result


def controlled_monotonicity(
    observations: Iterable[NaturalFactorObservation],
    *,
    factor: str,
    control_factor: str | None = None,
) -> dict[str, object]:
    """Compare factor tertiles within average-rank quartiles of its control."""

    factor = _validate_factor(factor)
    if control_factor is None:
        control_factor = "visibility" if factor == "sampling" else "sampling"
    control_factor = _validate_factor(control_factor)
    if control_factor == factor:
        raise ValueError("control_factor must be the other natural factor")
    rows = _natural_rows(_validated_observations(observations))
    n = len(rows)
    if n < 6:
        empty_bins = tuple(
            {
                "bin": bin_index,
                "count": 0,
                "lower_count": 0,
                "upper_count": 0,
                "lower_prediction_mean": None,
                "upper_prediction_mean": None,
                "success": None,
            }
            for bin_index in range(4)
        )
        return {
            "success": False,
            "status": "insufficient",
            "reason": "at least six natural observations are required",
            "eligible": 0,
            "eligible_bins": 0,
            "successful": 0,
            "rate": None,
            "bins": empty_bins,
            "factor": factor,
            "control_factor": control_factor,
        }
    target_values = tuple(_factor_natural(row, factor) for row in rows)
    control_values = tuple(_factor_natural(row, control_factor) for row in rows)
    if len(set(control_values)) < 4:
        empty_bins = tuple(
            {
                "bin": bin_index,
                "count": 0,
                "lower_count": 0,
                "upper_count": 0,
                "lower_prediction_mean": None,
                "upper_prediction_mean": None,
                "success": None,
            }
            for bin_index in range(4)
        )
        return {
            "success": False,
            "status": "insufficient",
            "reason": "control factor must have at least four distinct values",
            "eligible": 0,
            "eligible_bins": 0,
            "successful": 0,
            "rate": None,
            "bins": empty_bins,
            "factor": factor,
            "control_factor": control_factor,
        }
    target_ranks = np.asarray(average_tie_rank(target_values), dtype=np.float64)
    control_ranks = np.asarray(average_tie_rank(control_values), dtype=np.float64)
    target_predictions = np.asarray(
        tuple(_factor_prediction(row, factor) for row in rows), dtype=np.float64
    )
    bins: list[dict[str, object]] = []
    successful = 0
    eligible = 0
    for bin_index in range(4):
        # Rank bins are assigned from average ranks, making equal controls
        # deterministic and preventing input order from splitting ties.
        selected = [
            index
            for index, rank in enumerate(control_ranks)
            if min(3, int((float(rank) - 1.0) * 4.0 / n)) == bin_index
        ]
        lower = [index for index in selected if target_ranks[index] <= n / 3.0]
        upper = [index for index in selected if target_ranks[index] > 2.0 * n / 3.0]
        bin_result: dict[str, object] = {
            "bin": bin_index,
            "count": len(selected),
            "lower_count": len(lower),
            "upper_count": len(upper),
            "lower_prediction_mean": None,
            "upper_prediction_mean": None,
            "success": None,
        }
        if lower and upper:
            eligible += 1
            lower_mean = float(target_predictions[lower].mean())
            upper_mean = float(target_predictions[upper].mean())
            passed = upper_mean > lower_mean
            if passed:
                successful += 1
            bin_result.update(
                {
                    "lower_prediction_mean": lower_mean,
                    "upper_prediction_mean": upper_mean,
                    "success": passed,
                }
            )
        bins.append(bin_result)
    rate = successful / eligible if eligible else None
    status = "ok" if eligible >= 2 else "insufficient"
    return {
        "success": bool(status == "ok" and rate is not None and rate >= _DEFAULT_MONOTONIC_THRESHOLD),
        "status": status,
        "reason": "" if status == "ok" else (
            "no control quartile had both target tertiles"
            if eligible == 0
            else "fewer than two eligible control quartiles"
        ),
        "eligible": eligible,
        "eligible_bins": eligible,
        "successful": successful,
        "rate": rate,
        "bins": tuple(bins),
        "factor": factor,
        "control_factor": control_factor,
    }


def intervention_statistics(
    observations: Iterable[NaturalFactorObservation],
    *,
    factor: str,
    expected_intervention_severities: Sequence[float] = DEFAULT_INTERVENTION_SEVERITIES,
) -> dict[str, object]:
    """Compute paired responses using clean rows as the internal manifest.

    A pair absent together with its clean manifest is unobservable here; the
    upstream observer/progress manifest is responsible for detecting that
    source-level omission.
    """

    factor = _validate_factor(factor)
    expected_severities = _validate_expected_severities(expected_intervention_severities)
    rows = _validated_observations(observations)
    # Index once by physical pair and by factor group.  In particular, do not
    # scan ``rows`` for a clean baseline for every group: real audits contain
    # all six nodes, three seeds, and many objects per image.
    clean_by_base: dict[tuple[Any, ...], list[NaturalFactorObservation]] = defaultdict(list)
    grouped: dict[tuple[Any, ...], list[NaturalFactorObservation]] = defaultdict(list)
    for row in rows:
        if row.pair_id is None:
            continue
        base = (row.seed, row.node_id, row.image_id, row.object_id, row.pair_id)
        if row.intervention_kind == "clean" and row.intervention_factor == factor:
            clean_by_base[base].append(row)
        elif row.intervention_kind == factor:
            grouped[base].append(row)

    target_response_sum = 0.0
    background_response_sum = 0.0
    paired_effect_sum = 0.0
    response_count = 0
    malformed_details: list[dict[str, object]] = []
    unordered_details: list[dict[str, object]] = []
    eligible_by_seed_node: dict[str, int] = defaultdict(int)
    eligible = 0
    ordered = 0
    malformed = 0
    all_bases = sorted(set(clean_by_base) | set(grouped))
    for base in all_bases:
        key = base + (factor,)
        candidates = grouped.get(base, [])
        clean_candidates = clean_by_base.get(base, [])
        clean_target = [row for row in clean_candidates if row.region_role == "target"]
        clean_background = [row for row in clean_candidates if row.region_role == "background"]
        by_severity: dict[float, dict[str, list[NaturalFactorObservation]]] = defaultdict(
            lambda: {"target": [], "background": []}
        )
        for candidate in candidates:
            severity = _canonical_severity(candidate.intervention_severity, expected_severities)
            by_severity[severity][candidate.region_role].append(candidate)

        reasons: list[str] = []
        if len(clean_target) != 1:
            reasons.append(f"clean_target_count={len(clean_target)} expected=1")
        if len(clean_background) != 1:
            reasons.append(f"clean_background_count={len(clean_background)} expected=1")
        if len(expected_severities) < 2:
            reasons.append("expected_severity_count must be at least 2")
        if not candidates:
            reasons.append("factor_intervention_rows_missing")
        observed_severities = set(by_severity)
        missing_severities = [
            severity for severity in expected_severities if severity not in observed_severities
        ]
        extra_severities = sorted(observed_severities - set(expected_severities))
        if missing_severities:
            reasons.append(
                "missing_expected_severities="
                + ",".join(f"{severity:.12g}" for severity in missing_severities)
            )
        if extra_severities:
            reasons.append(
                "unexpected_severities="
                + ",".join(f"{severity:.12g}" for severity in extra_severities)
            )
        if len(observed_severities) < 2:
            reasons.append(
                f"positive_severity_count={len(observed_severities)} expected_at_least=2"
            )
        severity_counts: dict[str, dict[str, int]] = {}
        for severity in sorted(observed_severities | set(expected_severities)):
            target_count = len(by_severity.get(severity, {}).get("target", []))
            background_count = len(by_severity.get(severity, {}).get("background", []))
            severity_key = f"{severity:.12g}"
            severity_counts[severity_key] = {
                "target": target_count,
                "background": background_count,
            }
            if severity in set(expected_severities) and target_count != 1:
                reasons.append(
                    f"severity={severity:.12g}_target_count={target_count} expected=1"
                )
            if severity in set(expected_severities) and background_count != 1:
                reasons.append(
                    f"severity={severity:.12g}_background_count={background_count} expected=1"
                )
        if reasons:
            malformed += 1
            if len(malformed_details) < _MAX_EVIDENCE_EXAMPLES:
                severity_items = sorted(severity_counts.items())
                malformed_details.append(
                    {
                        "seed": key[0],
                        "node_id": key[1],
                        "image_id": key[2],
                        "object_id": key[3],
                        "pair_id": key[4],
                        "factor": factor,
                        "expected_severities": tuple(expected_severities),
                        "reasons": tuple(reasons),
                        "severity_counts": dict(severity_items[:_MAX_EVIDENCE_EXAMPLES]),
                        "severity_counts_truncated": len(severity_items) > _MAX_EVIDENCE_EXAMPLES,
                    }
                )
            # A malformed group contributes no response or ordering evidence.
            continue

        clean_target_value = _factor_prediction(clean_target[0], factor)
        clean_background_value = _factor_prediction(clean_background[0], factor)
        effects: list[float] = []
        pair_target: list[float] = []
        pair_background: list[float] = []
        for severity in sorted(by_severity):
            target = by_severity[severity]["target"]
            background = by_severity[severity]["background"]
            target_response = _factor_prediction(target[0], factor) - clean_target_value
            background_response = _factor_prediction(background[0], factor) - clean_background_value
            pair_target.append(target_response)
            pair_background.append(background_response)
            effects.append(target_response - background_response)
        eligible += 1
        eligible_by_seed_node[f"{key[0]}:{key[1]}"] += 1
        is_ordered = all(
            effects[index + 1] >= effects[index] - 1e-12
            for index in range(len(effects) - 1)
        ) and any(effects[index + 1] > effects[index] + 1e-12 for index in range(len(effects) - 1))
        if is_ordered:
            ordered += 1
        target_response_sum += math.fsum(pair_target)
        background_response_sum += math.fsum(pair_background)
        paired_effect_sum += math.fsum(effects)
        response_count += len(pair_target)
        if not is_ordered and len(unordered_details) < _MAX_EVIDENCE_EXAMPLES:
            unordered_details.append(
                {
                    "seed": key[0],
                    "node_id": key[1],
                    "image_id": key[2],
                    "object_id": key[3],
                    "pair_id": key[4],
                    "effects": tuple(effects),
                }
            )
    malformed_evidence = {
        "items": tuple(malformed_details),
        "total": malformed,
        "truncated": malformed > len(malformed_details),
    }
    unordered_evidence = {
        "items": tuple(unordered_details),
        "total": eligible - ordered,
        "truncated": (eligible - ordered) > len(unordered_details),
    }
    if not eligible:
        return {
            "factor": factor,
            "expected_severities": tuple(expected_severities),
            "status": "malformed" if malformed else "insufficient",
            "reason": (
                f"{malformed} malformed intervention pairs"
                if malformed
                else "no complete pairs with at least two severities"
            ),
            "eligible": 0,
            "ordered": 0,
            "ordered_pair_rate": None,
            "target_mean_response": None,
            "background_mean_response": None,
            "paired_mean": None,
            "malformed": malformed,
            "malformed_examples": malformed_evidence,
            "unordered_examples": unordered_evidence,
            "eligible_by_seed_node": dict(eligible_by_seed_node),
        }
    return {
        "factor": factor,
        "expected_severities": tuple(expected_severities),
        "status": "malformed" if malformed else "ok",
        "reason": "" if not malformed else f"{malformed} malformed intervention pairs",
        "eligible": eligible,
        "ordered": ordered,
        "ordered_pair_rate": ordered / eligible,
        "target_mean_response": target_response_sum / response_count,
        "background_mean_response": background_response_sum / response_count,
        "paired_mean": paired_effect_sum / response_count,
        "malformed": malformed,
        "malformed_examples": malformed_evidence,
        "unordered_examples": unordered_evidence,
        "eligible_by_seed_node": dict(eligible_by_seed_node),
    }


def natural_factor_alignment(
    observations: Iterable[NaturalFactorObservation],
    *,
    factor: str,
    bootstrap_replicates: int = _DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = _DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Return pooled and seed/node natural alignment statistics."""

    factor = _validate_factor(factor)
    rows = _validated_observations(observations)
    natural = _natural_rows(rows)

    def raw_statistic(sample: tuple[NaturalFactorObservation, ...]) -> dict[str, object]:
        return spearman(
            tuple(_factor_natural(row, factor) for row in _natural_rows(sample)),
            tuple(_factor_prediction(row, factor) for row in _natural_rows(sample)),
        )

    def residual_statistic(sample: tuple[NaturalFactorObservation, ...]) -> dict[str, object]:
        sample_natural = _natural_rows(sample)
        return partial_spearman(
            tuple(_factor_natural(row, factor) for row in sample_natural),
            tuple(_factor_prediction(row, factor) for row in sample_natural),
            tuple(row.box_height for row in sample_natural),
            tuple(row.class_id for row in sample_natural),
        )

    pooled_raw = raw_statistic(natural)
    pooled_residual = residual_statistic(natural)
    bootstrap_data = _precompute_bootstrap_data(natural, factor)
    raw_estimate = pooled_raw.get("rho")
    raw_estimate = (
        float(raw_estimate)
        if isinstance(raw_estimate, Real) and math.isfinite(float(raw_estimate))
        else None
    )
    residual_estimate = pooled_residual.get("rho")
    residual_estimate = (
        float(residual_estimate)
        if isinstance(residual_estimate, Real) and math.isfinite(float(residual_estimate))
        else None
    )
    pooled_raw_ci = _exact_bootstrap(
        data=bootstrap_data,
        statistic=_exact_raw_statistic,
        estimate=raw_estimate,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    pooled_residual_ci = _exact_bootstrap(
        data=bootstrap_data,
        statistic=_exact_residual_statistic,
        estimate=residual_estimate,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    seed_node: dict[str, dict[str, object]] = {}
    by_seed_node: dict[tuple[int, int], list[NaturalFactorObservation]] = defaultdict(list)
    for row in natural:
        by_seed_node[(row.seed, row.node_id)].append(row)
    for key in sorted(by_seed_node):
        subset = tuple(by_seed_node[key])
        raw = raw_statistic(subset)
        residual = residual_statistic(subset)
        seed_node[f"{key[0]}:{key[1]}"] = {
            "raw": raw,
            "residual": residual,
            "direction": raw.get("rho"),
        }
    return {
        "factor": factor,
        "natural_count": len(natural),
        "pooled_raw": pooled_raw,
        "pooled_residual": pooled_residual,
        "pooled_raw_ci": pooled_raw_ci,
        "pooled_residual_ci": pooled_residual_ci,
        "seed_node": seed_node,
    }

def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class NaturalFactorGateDecision:
    """Auditable result of the two-factor natural-transfer gate."""

    passed: bool
    factor_results: Mapping[str, Mapping[str, object]]
    reasons: tuple[str, ...]
    required_seeds: tuple[int, ...]
    required_nodes: tuple[int, ...]
    monotonic_threshold: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor_results", _freeze(self.factor_results))

    @property
    def gate_passed(self) -> bool:
        return self.passed

    @property
    def sampling_pass(self) -> bool:
        return bool(self.factor_results.get("sampling", {}).get("passed", False))

    @property
    def visibility_pass(self) -> bool:
        return bool(self.factor_results.get("visibility", {}).get("passed", False))

    @property
    def sampling(self) -> Mapping[str, object]:
        return self.factor_results.get("sampling", {})

    @property
    def visibility(self) -> Mapping[str, object]:
        return self.factor_results.get("visibility", {})

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "required_seeds": list(self.required_seeds),
            "required_nodes": list(self.required_nodes),
            "monotonic_threshold": self.monotonic_threshold,
            "factors": _json_safe(_thaw(self.factor_results)),
        }


def audit_natural_factors(
    observations: Iterable[NaturalFactorObservation],
    *,
    required_seeds: Sequence[int] = _DEFAULT_SEEDS,
    required_nodes: Sequence[int] = _DEFAULT_NODES,
    monotonic_threshold: float = _DEFAULT_MONOTONIC_THRESHOLD,
    bootstrap_replicates: int = _DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = _DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
    expected_intervention_severities: Sequence[float] = DEFAULT_INTERVENTION_SEVERITIES,
) -> NaturalFactorGateDecision:
    """Run alignment, monotonicity, intervention, and stability checks."""

    rows = _validated_observations(observations)
    seeds = tuple(required_seeds)
    nodes = tuple(required_nodes)
    if not seeds or any(not _is_integer(value) or int(value) < 0 for value in seeds):
        raise ValueError("required_seeds must contain non-negative integers")
    if not nodes or any(not _is_integer(value) or int(value) < 0 for value in nodes):
        raise ValueError("required_nodes must contain non-negative integers")
    if len(set(int(value) for value in seeds)) != len(seeds):
        raise ValueError("required_seeds must not contain duplicates")
    if len(set(int(value) for value in nodes)) != len(nodes):
        raise ValueError("required_nodes must not contain duplicates")
    threshold = _unit(monotonic_threshold, "monotonic_threshold")
    if threshold <= 0.0:
        raise ValueError("monotonic_threshold must be greater than zero")
    # Validate bootstrap arguments once even when both factors have no rows.
    _validate_bootstrap(bootstrap_replicates, bootstrap_seed, confidence)
    expected_severities = _validate_expected_severities(expected_intervention_severities)

    factor_results: dict[str, dict[str, object]] = {}
    all_reasons: list[str] = []
    expected_pairs = tuple((int(seed), int(node)) for seed in seeds for node in nodes)
    for factor in _FACTORS:
        alignment = natural_factor_alignment(
            rows,
            factor=factor,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            confidence=confidence,
        )
        monotonic = controlled_monotonicity(
            rows,
            factor=factor,
            control_factor="visibility" if factor == "sampling" else "sampling",
        )
        intervention = intervention_statistics(
            rows,
            factor=factor,
            expected_intervention_severities=expected_severities,
        )
        reasons: list[str] = []
        seed_node_results = alignment["seed_node"]
        assert isinstance(seed_node_results, dict)
        for seed, node in expected_pairs:
            result = seed_node_results.get(f"{seed}:{node}")
            if result is None:
                reasons.append(f"{factor}_missing_seed_{seed}_node_{node}")
                continue
            raw = result["raw"]
            residual = result["residual"]
            if not isinstance(raw, dict) or not raw.get("success") or not isinstance(raw.get("rho"), Real) or float(raw["rho"]) <= 0.0:
                reasons.append(f"{factor}_direction_seed_{seed}_node_{node}")
            if not isinstance(residual, dict) or not residual.get("success") or not isinstance(residual.get("rho"), Real) or float(residual["rho"]) <= 0.0:
                reasons.append(f"{factor}_residual_direction_seed_{seed}_node_{node}")

        raw_ci = alignment["pooled_raw_ci"]
        residual_ci = alignment["pooled_residual_ci"]
        if not isinstance(raw_ci, dict) or raw_ci.get("status") != "ok" or not isinstance(raw_ci.get("ci_lower"), Real) or float(raw_ci["ci_lower"]) <= 0.0:
            reasons.append(f"{factor}_pooled_raw_ci_crosses_zero")
        if not isinstance(residual_ci, dict) or residual_ci.get("status") != "ok" or not isinstance(residual_ci.get("ci_lower"), Real) or float(residual_ci["ci_lower"]) <= 0.0:
            reasons.append(f"{factor}_pooled_residual_ci_crosses_zero")
        mono_rate = monotonic.get("rate")
        if monotonic.get("status") != "ok" or not isinstance(mono_rate, Real) or float(mono_rate) < threshold:
            reasons.append(f"{factor}_controlled_monotonicity_below_threshold")
        malformed_count = intervention.get("malformed", 0)
        if isinstance(malformed_count, Integral) and int(malformed_count) > 0:
            reasons.append(f"{factor}_malformed_intervention_pairs")
        eligible_by_seed_node = intervention.get("eligible_by_seed_node", {})
        if not isinstance(eligible_by_seed_node, dict):
            eligible_by_seed_node = {}
        for seed, node in expected_pairs:
            if int(eligible_by_seed_node.get(f"{seed}:{node}", 0)) < 1:
                reasons.append(f"{factor}_missing_intervention_seed_{seed}_node_{node}")
        target_mean = intervention.get("target_mean_response")
        background_mean = intervention.get("background_mean_response")
        paired_mean = intervention.get("paired_mean")
        ordered_rate = intervention.get("ordered_pair_rate")
        if (
            intervention.get("status") != "ok"
            or not isinstance(target_mean, Real)
            or not isinstance(background_mean, Real)
            or float(target_mean) <= float(background_mean) + 1e-12
        ):
            reasons.append(f"{factor}_target_response_not_stronger_than_background")
        if not isinstance(paired_mean, Real) or float(paired_mean) <= 1e-12:
            reasons.append(f"{factor}_paired_response_not_positive")
        if not isinstance(ordered_rate, Real) or float(ordered_rate) < threshold:
            reasons.append(f"{factor}_intervention_order_below_threshold")
        passed = not reasons
        factor_results[factor] = {
            "passed": passed,
            "reasons": tuple(reasons),
            "alignment": alignment,
            "pooled_raw": alignment["pooled_raw"],
            "pooled_residual": alignment["pooled_residual"],
            "pooled_raw_ci": raw_ci,
            "pooled_residual_ci": residual_ci,
            "seed_node": seed_node_results,
            "controlled_monotonicity": monotonic,
            "intervention": intervention,
        }
        all_reasons.extend(reasons)

    return NaturalFactorGateDecision(
        passed=all(result["passed"] for result in factor_results.values()),
        factor_results=factor_results,
        reasons=tuple(all_reasons),
        required_seeds=tuple(int(seed) for seed in seeds),
        required_nodes=tuple(int(node) for node in nodes),
        monotonic_threshold=threshold,
    )

__all__ = [
    "DEFAULT_INTERVENTION_SEVERITIES",
    "NaturalFactorObservation",
    "NaturalFactorGateDecision",
    "average_tie_rank",
    "spearman",
    "partial_spearman",
    "bootstrap_image_ids",
    "image_cluster_bootstrap",
    "controlled_monotonicity",
    "intervention_statistics",
    "natural_factor_alignment",
    "audit_natural_factors",
]
