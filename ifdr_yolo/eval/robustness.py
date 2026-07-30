from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RobustnessCondition:
    kind: str
    strength: float
    seed: int
    metrics: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("kind must be non-empty text")
        if (
            isinstance(self.strength, bool)
            or not isinstance(self.strength, (int, float))
            or not math.isfinite(float(self.strength))
            or not 0.0 <= float(self.strength) <= 1.0
        ):
            raise ValueError("strength must be finite and within [0, 1]")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.metrics, Mapping):
            raise ValueError("metrics must be a mapping")
        object.__setattr__(self, "strength", float(self.strength))


def _ap40_table(
    payload: Mapping[str, object],
) -> dict[tuple[str, str], float]:
    classes = payload.get("classes")
    if not isinstance(classes, Mapping):
        raise ValueError("metrics must contain a classes mapping")
    result: dict[tuple[str, str], float] = {}
    for class_name, difficulties in classes.items():
        if not isinstance(class_name, str) or not isinstance(
            difficulties,
            Mapping,
        ):
            raise ValueError("invalid class metrics")
        for difficulty, values in difficulties.items():
            if not isinstance(difficulty, str) or not isinstance(
                values,
                Mapping,
            ):
                raise ValueError("invalid difficulty metrics")
            ap40 = values.get("ap40")
            if (
                isinstance(ap40, bool)
                or not isinstance(ap40, (int, float))
                or not math.isfinite(float(ap40))
            ):
                raise ValueError(
                    f"invalid ap40 for {class_name}/{difficulty}"
                )
            result[(class_name, difficulty)] = float(ap40)
    if not result:
        raise ValueError("metrics classes must not be empty")
    return result


def _trapezoid(
    strengths: Sequence[float],
    values: Sequence[float],
) -> float:
    return sum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for left_x, right_x, left_y, right_y in zip(
            strengths,
            strengths[1:],
            values,
            values[1:],
        )
    )


def summarize_robustness(
    conditions: Sequence[RobustnessCondition],
) -> dict[str, object]:
    """Aggregate seed-repeated AP40 degradation curves.

    ``normalized_auc`` is the area under the mean AP40 retention curve,
    normalized by both the measured strength interval and clean AP40. A value
    of 1 therefore means no performance loss across the intervention range.
    """

    if not conditions:
        raise ValueError("at least one robustness condition is required")
    grouped: dict[str, list[RobustnessCondition]] = defaultdict(list)
    for condition in conditions:
        if not isinstance(condition, RobustnessCondition):
            raise ValueError("conditions must contain RobustnessCondition")
        grouped[condition.kind].append(condition)

    curves: dict[str, object] = {}
    for kind in sorted(grouped):
        kind_conditions = grouped[kind]
        grids_by_seed: dict[int, set[float]] = defaultdict(set)
        metric_tables: dict[tuple[float, int], dict[tuple[str, str], float]] = {}
        for condition in kind_conditions:
            key = (condition.strength, condition.seed)
            if key in metric_tables:
                raise ValueError(
                    f"duplicate condition for {kind}, strength "
                    f"{condition.strength}, seed {condition.seed}"
                )
            grids_by_seed[condition.seed].add(condition.strength)
            metric_tables[key] = _ap40_table(condition.metrics)

        seed_grids = tuple(grids_by_seed.values())
        expected_grid = seed_grids[0]
        if any(grid != expected_grid for grid in seed_grids[1:]):
            raise ValueError(
                f"all {kind} seeds must use the same strength grid"
            )
        strengths = sorted(expected_grid)
        if not strengths or strengths[0] != 0.0:
            raise ValueError(f"{kind} curve must include strength 0")
        if len(strengths) < 2 or strengths[-1] <= strengths[0]:
            raise ValueError(
                f"{kind} curve requires at least two distinct strengths"
            )

        reference_keys = set(next(iter(metric_tables.values())))
        if any(set(table) != reference_keys for table in metric_tables.values()):
            raise ValueError(
                f"all {kind} conditions must contain the same metrics"
            )
        seed_count = len(grids_by_seed)
        class_curves: dict[str, dict[str, object]] = {}
        for class_name, difficulty in sorted(reference_keys):
            mean_values = [
                sum(
                    metric_tables[(strength, seed)][
                        (class_name, difficulty)
                    ]
                    for seed in grids_by_seed
                )
                / seed_count
                for strength in strengths
            ]
            clean = mean_values[0]
            at_max = mean_values[-1]
            span = strengths[-1] - strengths[0]
            raw_auc = _trapezoid(strengths, mean_values) / span
            normalized_auc = raw_auc / clean if clean > 0.0 else None
            retention = at_max / clean if clean > 0.0 else None
            class_curves.setdefault(class_name, {})[difficulty] = {
                "strengths": strengths,
                "mean_ap40": mean_values,
                "seed_count": seed_count,
                "clean_ap40": clean,
                "ap40_at_max_strength": at_max,
                "absolute_drop": clean - at_max,
                "relative_retention": retention,
                "mean_auc_ap40": raw_auc,
                "normalized_auc": normalized_auc,
            }
        curves[kind] = class_curves

    return {
        "schema_version": 1,
        "metric": "KITTI_2D_AP40",
        "curves": curves,
    }
