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
from dataclasses import dataclass
import math
from numbers import Integral, Real
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

    def __post_init__(self) -> None:
        for value, name in (
            (self.seed, "seed"),
            (self.node_id, "node_id"),
            (self.object_id, "object_id"),
            (self.class_id, "class_id"),
        ):
            if not _is_integer(value) or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.image_id, str) or not self.image_id.strip():
            raise ValueError("image_id must be a non-empty string")
        if self.class_name is not None and (
            not isinstance(self.class_name, str) or not self.class_name.strip()
        ):
            raise ValueError("class_name must be a non-empty string when provided")
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

        if self.intervention_kind == "natural":
            if self.region_role != "target":
                raise ValueError("natural observations must be target rows")
            if severity != 0.0:
                raise ValueError("natural observations must have severity 0")
            if self.pair_id is not None:
                raise ValueError("natural observations must not have a pair_id")
        else:
            if self.pair_id is None or not isinstance(self.pair_id, str) or not self.pair_id.strip():
                raise ValueError("intervention rows require a non-empty pair_id")
            if self.intervention_kind == "clean" and severity != 0.0:
                raise ValueError("clean intervention rows must have severity 0")


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
    return tuple(sorted(rows, key=_sort_key))


def _validate_factor(factor: str) -> str:
    if factor not in _FACTORS:
        raise ValueError("factor must be sampling or visibility")
    return factor


def _validate_bootstrap(replicates: int, seed: int, confidence: float) -> None:
    if not _is_integer(replicates) or int(replicates) <= 0:
        raise ValueError("bootstrap replicates must be a positive integer")
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


# A plural spelling is convenient for callers and keeps the rank primitive
# discoverable without introducing a second implementation.
average_tie_ranks = average_tie_rank


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
    """Partial Spearman controlling ranked height and class one-hot columns."""

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
    design = np.column_stack(
        [
            np.ones(n, dtype=np.float64),
            height_rank,
            *[np.asarray([float(value == klass) for value in classes]) for klass in unique_classes],
        ]
    )
    if n <= design.shape[1]:
        return _result(
            rho=None,
            n=n,
            status="insufficient",
            reason="not enough observations for registered controls",
        )
    try:
        target_fit = design @ np.linalg.lstsq(design, target_rank, rcond=None)[0]
        prediction_fit = design @ np.linalg.lstsq(design, prediction_rank, rcond=None)[0]
    except np.linalg.LinAlgError:
        return _result(rho=None, n=n, status="nonfinite", reason="control regression failed")
    target_residual = target_rank - target_fit
    prediction_residual = prediction_rank - prediction_fit
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


def _cluster_bootstrap(
    observations: Sequence[NaturalFactorObservation],
    statistic: Callable[[tuple[NaturalFactorObservation, ...]], dict[str, object]],
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, object]:
    _validate_bootstrap(replicates, seed, confidence)
    unique_images = tuple(sorted({row.image_id for row in observations}))
    if not unique_images:
        return {
            "estimate": None,
            "ci_lower": None,
            "ci_upper": None,
            "status": "insufficient",
            "reason": "no image clusters",
            "replicates": 0,
            "valid_replicates": 0,
            "sampling_unit": "image_id",
            "unique_image_count": 0,
        }
    estimate_result = statistic(tuple(observations))
    estimate = estimate_result.get("rho")
    if estimate is None or not isinstance(estimate, Real) or not math.isfinite(float(estimate)):
        estimate = None
    generator = np.random.default_rng(int(seed))
    values: list[float] = []
    for _ in range(int(replicates)):
        sampled_indices = generator.integers(0, len(unique_images), size=len(unique_images))
        sampled = tuple(unique_images[int(index)] for index in sampled_indices)
        result = statistic(_cluster_rows(observations, sampled))
        value = result.get("rho")
        if isinstance(value, Real) and math.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return {
            "estimate": estimate,
            "ci_lower": None,
            "ci_upper": None,
            "status": "insufficient",
            "reason": "bootstrap produced no finite replicates",
            "replicates": int(replicates),
            "valid_replicates": 0,
            "sampling_unit": "image_id",
            "unique_image_count": len(unique_images),
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
        "unique_image_count": len(unique_images),
    }


def image_cluster_bootstrap(
    observations: Iterable[NaturalFactorObservation],
    factor: str = "sampling",
    *,
    replicates: int = _DEFAULT_BOOTSTRAP_REPLICATES,
    reps: int | None = None,
    seed: int = _DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
    return_samples: bool = False,
) -> dict[str, object]:
    """Bootstrap raw natural Spearman by unique image ID."""

    factor = _validate_factor(factor)
    if reps is not None:
        replicates = reps
    rows = _validated_observations(observations)
    natural = _natural_rows(rows)

    def statistic(sample: tuple[NaturalFactorObservation, ...]) -> dict[str, object]:
        sample_natural = _natural_rows(sample)
        return spearman(
            tuple(_factor_natural(row, factor) for row in sample_natural),
            tuple(_factor_prediction(row, factor) for row in sample_natural),
        )

    result = _cluster_bootstrap(
        natural,
        statistic,
        replicates=replicates,
        seed=seed,
        confidence=confidence,
    )
    # ``image_cluster_bootstrap`` historically reads naturally as a raw
    # estimate, so preserve that label alongside the generic estimate key.
    result["factor"] = factor
    result["metric"] = "raw_spearman"
    if return_samples:
        result["sampled_image_ids"] = [
            list(sample)
            for sample in bootstrap_image_ids(
                tuple(sorted({row.image_id for row in natural})),
                replicates=replicates,
                seed=seed,
            )
        ]
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
        return {
            "success": False,
            "status": "insufficient",
            "reason": "at least six natural observations are required",
            "eligible": 0,
            "successful": 0,
            "rate": None,
            "bins": (),
        }
    target_values = tuple(_factor_natural(row, factor) for row in rows)
    control_values = tuple(_factor_natural(row, control_factor) for row in rows)
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
        if lower and upper:
            eligible += 1
            lower_mean = float(target_predictions[lower].mean())
            upper_mean = float(target_predictions[upper].mean())
            passed = upper_mean > lower_mean
            if passed:
                successful += 1
            bins.append(
                {
                    "bin": bin_index,
                    "count": len(selected),
                    "lower_count": len(lower),
                    "upper_count": len(upper),
                    "lower_prediction_mean": lower_mean,
                    "upper_prediction_mean": upper_mean,
                    "success": passed,
                }
            )
    rate = successful / eligible if eligible else None
    return {
        "success": bool(eligible and rate is not None and rate >= _DEFAULT_MONOTONIC_THRESHOLD),
        "status": "ok" if eligible else "insufficient",
        "reason": "" if eligible else "no control quartile had both target tertiles",
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
) -> dict[str, object]:
    """Compute paired target/background responses for one intervention factor."""

    factor = _validate_factor(factor)
    rows = _validated_observations(observations)
    grouped: dict[tuple[Any, ...], list[NaturalFactorObservation]] = defaultdict(list)
    for row in rows:
        if row.intervention_kind != factor or row.pair_id is None:
            continue
        grouped[
            (
                row.seed,
                row.node_id,
                row.image_id,
                row.object_id,
                row.pair_id,
                row.intervention_kind,
            )
        ].append(row)

    target_responses: list[float] = []
    background_responses: list[float] = []
    paired_effects: list[float] = []
    pair_details: list[dict[str, object]] = []
    eligible = 0
    ordered = 0
    malformed = 0
    for key in sorted(grouped):
        candidates = grouped[key]
        clean_candidates = [
            candidate
            for candidate in rows
            if (
                candidate.seed,
                candidate.node_id,
                candidate.image_id,
                candidate.object_id,
                candidate.pair_id,
            )
            == key[:5]
            and candidate.intervention_kind == "clean"
        ]
        clean_target = [row for row in clean_candidates if row.region_role == "target"]
        clean_background = [row for row in clean_candidates if row.region_role == "background"]
        # A manifest may intentionally share one clean target/background pair
        # between the sampling and visibility interventions.  Repeated clean
        # rows are therefore accepted only when the channel being measured is
        # identical; conflicting baselines remain an auditable malformed pair.
        if not clean_target or not clean_background:
            malformed += 1
            continue
        clean_target_values = {_factor_prediction(row, factor) for row in clean_target}
        clean_background_values = {_factor_prediction(row, factor) for row in clean_background}
        if len(clean_target_values) != 1 or len(clean_background_values) != 1:
            malformed += 1
            continue
        clean_target_value = next(iter(clean_target_values))
        clean_background_value = next(iter(clean_background_values))
        by_severity: dict[float, dict[str, list[NaturalFactorObservation]]] = defaultdict(
            lambda: {"target": [], "background": []}
        )
        for candidate in candidates:
            by_severity[candidate.intervention_severity][candidate.region_role].append(candidate)
        effects: list[float] = []
        pair_target: list[float] = []
        pair_background: list[float] = []
        severities: list[float] = []
        for severity in sorted(by_severity):
            target = by_severity[severity]["target"]
            background = by_severity[severity]["background"]
            if len(target) != 1 or len(background) != 1:
                continue
            target_response = _factor_prediction(target[0], factor) - clean_target_value
            background_response = _factor_prediction(background[0], factor) - clean_background_value
            pair_target.append(target_response)
            pair_background.append(background_response)
            effects.append(target_response - background_response)
            severities.append(float(severity))
        if len(effects) < 2:
            malformed += 1
            continue
        eligible += 1
        is_ordered = all(
            effects[index + 1] >= effects[index] - 1e-12
            for index in range(len(effects) - 1)
        ) and any(effects[index + 1] > effects[index] + 1e-12 for index in range(len(effects) - 1))
        if is_ordered:
            ordered += 1
        target_responses.extend(pair_target)
        background_responses.extend(pair_background)
        paired_effects.extend(effects)
        pair_details.append(
            {
                "seed": key[0],
                "node_id": key[1],
                "image_id": key[2],
                "object_id": key[3],
                "pair_id": key[4],
                "severities": tuple(severities),
                "paired_effects": tuple(effects),
                "ordered": is_ordered,
            }
        )
    if not eligible:
        return {
            "factor": factor,
            "status": "insufficient",
            "reason": "no complete pairs with at least two severities",
            "eligible": 0,
            "ordered": 0,
            "ordered_pair_rate": None,
            "ordered_rate": None,
            "target_mean_response": None,
            "target_mean": None,
            "background_mean_response": None,
            "background_mean": None,
            "paired_mean": None,
            "malformed": malformed,
            "pairs": tuple(pair_details),
        }
    return {
        "factor": factor,
        "status": "ok",
        "reason": "",
        "eligible": eligible,
        "ordered": ordered,
        "ordered_pair_rate": ordered / eligible,
        "ordered_rate": ordered / eligible,
        "target_mean_response": float(np.mean(target_responses)),
        "target_mean": float(np.mean(target_responses)),
        "background_mean_response": float(np.mean(background_responses)),
        "background_mean": float(np.mean(background_responses)),
        "paired_mean": float(np.mean(paired_effects)),
        "malformed": malformed,
        "pairs": tuple(pair_details),
    }


def natural_factor_alignment(
    observations: Iterable[NaturalFactorObservation],
    *,
    factor: str,
    bootstrap_replicates: int = _DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_reps: int | None = None,
    bootstrap_seed: int = _DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Return pooled and seed/node natural alignment statistics."""

    factor = _validate_factor(factor)
    if bootstrap_reps is not None:
        bootstrap_replicates = bootstrap_reps
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
    pooled_raw_ci = _cluster_bootstrap(
        natural,
        raw_statistic,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    pooled_residual_ci = _cluster_bootstrap(
        natural,
        residual_statistic,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    seed_node: dict[tuple[int, int], dict[str, object]] = {}
    by_seed_node: dict[tuple[int, int], list[NaturalFactorObservation]] = defaultdict(list)
    for row in natural:
        by_seed_node[(row.seed, row.node_id)].append(row)
    for key in sorted(by_seed_node):
        subset = tuple(by_seed_node[key])
        raw = raw_statistic(subset)
        residual = residual_statistic(subset)
        seed_node[key] = {"raw": raw, "residual": residual, "direction": raw.get("rho")}
    return {
        "factor": factor,
        "natural_count": len(natural),
        "raw": pooled_raw,
        "residual": pooled_residual,
        "pooled_raw": pooled_raw,
        "pooled_residual": pooled_residual,
        "raw_ci": pooled_raw_ci,
        "residual_ci": pooled_residual_ci,
        "pooled_raw_ci": pooled_raw_ci,
        "pooled_residual_ci": pooled_residual_ci,
        "raw_ci_lower": pooled_raw_ci.get("ci_lower"),
        "raw_ci_upper": pooled_raw_ci.get("ci_upper"),
        "residual_ci_lower": pooled_residual_ci.get("ci_lower"),
        "residual_ci_upper": pooled_residual_ci.get("ci_upper"),
        "seed_node": seed_node,
    }


# Names used by the CLI and by downstream callers are intentionally explicit.
compute_natural_alignment = natural_factor_alignment
compute_controlled_monotonicity = controlled_monotonicity
compute_intervention_statistics = intervention_statistics
spearman_correlation = spearman
partial_spearman_correlation = partial_spearman
bootstrap_spearman = image_cluster_bootstrap
compute_intervention_effects = intervention_statistics


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class NaturalFactorGateDecision:
    """Auditable result of the two-factor natural-transfer gate."""

    passed: bool
    factor_results: dict[str, dict[str, object]]
    reasons: tuple[str, ...]
    required_seeds: tuple[int, ...]
    required_nodes: tuple[int, ...]
    monotonic_threshold: float

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
    def sampling(self) -> dict[str, object]:
        return self.factor_results.get("sampling", {})

    @property
    def visibility(self) -> dict[str, object]:
        return self.factor_results.get("visibility", {})

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "required_seeds": list(self.required_seeds),
            "required_nodes": list(self.required_nodes),
            "monotonic_threshold": self.monotonic_threshold,
            "factors": _json_safe(self.factor_results),
        }

    as_dict = to_dict


def audit_natural_factors(
    observations: Iterable[NaturalFactorObservation],
    *,
    required_seeds: Sequence[int] = _DEFAULT_SEEDS,
    required_nodes: Sequence[int] = _DEFAULT_NODES,
    monotonic_threshold: float = _DEFAULT_MONOTONIC_THRESHOLD,
    bootstrap_replicates: int = _DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_reps: int | None = None,
    bootstrap_seed: int = _DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> NaturalFactorGateDecision:
    """Run alignment, monotonicity, intervention, and stability checks."""

    rows = _validated_observations(observations)
    if bootstrap_reps is not None:
        bootstrap_replicates = bootstrap_reps
    seeds = tuple(required_seeds)
    nodes = tuple(required_nodes)
    if not seeds or any(not _is_integer(value) or int(value) < 0 for value in seeds):
        raise ValueError("required_seeds must contain non-negative integers")
    if not nodes or any(not _is_integer(value) or int(value) < 0 for value in nodes):
        raise ValueError("required_nodes must contain non-negative integers")
    threshold = _unit(monotonic_threshold, "monotonic_threshold")
    if threshold <= 0.0:
        raise ValueError("monotonic_threshold must be greater than zero")
    # Validate bootstrap arguments once even when both factors have no rows.
    _validate_bootstrap(bootstrap_replicates, bootstrap_seed, confidence)

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
        intervention = intervention_statistics(rows, factor=factor)
        reasons: list[str] = []
        seed_node_results = alignment["seed_node"]
        assert isinstance(seed_node_results, dict)
        for seed, node in expected_pairs:
            result = seed_node_results.get((seed, node))
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
        target_mean = intervention.get("target_mean_response")
        background_mean = intervention.get("background_mean_response")
        paired_mean = intervention.get("paired_mean")
        ordered_rate = intervention.get("ordered_pair_rate")
        if (
            intervention.get("status") != "ok"
            or not isinstance(target_mean, Real)
            or not isinstance(background_mean, Real)
            or float(target_mean) <= float(background_mean)
        ):
            reasons.append(f"{factor}_target_response_not_stronger_than_background")
        if not isinstance(paired_mean, Real) or float(paired_mean) <= 0.0:
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


evaluate_natural_factor_gate = audit_natural_factors
run_natural_factor_audit = audit_natural_factors
evaluate_gate = audit_natural_factors


__all__ = [
    "NaturalFactorObservation",
    "NaturalFactorGateDecision",
    "average_tie_rank",
    "average_tie_ranks",
    "spearman",
    "spearman_correlation",
    "partial_spearman",
    "partial_spearman_correlation",
    "bootstrap_image_ids",
    "image_cluster_bootstrap",
    "bootstrap_spearman",
    "controlled_monotonicity",
    "compute_controlled_monotonicity",
    "intervention_statistics",
    "compute_intervention_statistics",
    "compute_intervention_effects",
    "natural_factor_alignment",
    "compute_natural_alignment",
    "audit_natural_factors",
    "evaluate_natural_factor_gate",
    "run_natural_factor_audit",
    "evaluate_gate",
]
