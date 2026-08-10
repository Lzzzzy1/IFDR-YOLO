"""Fail-closed evidence gates for registered factor-repair conditions.

The module deliberately owns the statistical constants used for factor
selection.  Callers can provide endpoint evidence and, when available,
per-image endpoint values (or a ``recompute_endpoints`` callback).  F0 and a
candidate are then evaluated on the same image-cluster bootstrap draws.  The
absolute gate consumes only registered primary-node evidence; per-node and
per-seed confidence intervals remain diagnostic data and never select a
repair.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Callable

import numpy as np

from ifdr_yolo.eval.natural_factor_audit import partial_spearman, spearman


PRIMARY_ENDPOINTS = (
    "sampling_residual_spearman",
    "visibility_residual_spearman",
    "sampling_specificity_gap",
    "visibility_specificity_gap",
)
PRIMARY_NODE_IDS = (17, 20, 23, 26)
DIAGNOSTIC_NODE_IDS = (11, 14)

FACTOR_GATE_BOOTSTRAP_REPLICATES = 10_000
FACTOR_GATE_BOOTSTRAP_SEED = 20260805
FACTOR_GATE_BOOTSTRAP_PERCENTILES = (0.025, 0.975)

# Short aliases are useful to consumers that import the registered node
# identities from the evaluation module rather than the YAML parser.
PRIMARY_NODES = PRIMARY_NODE_IDS
DIAGNOSTIC_NODES = DIAGNOSTIC_NODE_IDS

_SELECTION_TIE_TOLERANCE = 1e-12
_REVERSE_RHO_THRESHOLD = 0.1
_SEVERITY_THRESHOLD = 0.8
_MISSING = object()


def _jsonable(value: object) -> object:
    """Convert nested evidence to deterministic JSON-compatible values."""

    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=repr)
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        # JSON's representation is stable for ordinary finite float values;
        # explicitly retain non-finite markers for a deterministic failure
        # digest instead of allowing platform-specific reprs.
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


class _FrozenEndpointRow(tuple):
    """Tuple representation that also supports ``row[endpoint]`` access."""

    def __new__(cls, values: Iterable[tuple[str, float]]) -> "_FrozenEndpointRow":
        return super().__new__(cls, tuple(values))

    def __getitem__(self, key: object) -> object:
        if isinstance(key, str):
            for name, value in self:
                if name == key:
                    return value
            raise KeyError(key)
        return super().__getitem__(key)

    def keys(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self)

    def items(self) -> tuple[tuple[str, float], ...]:
        return tuple(self)


class _FrozenEndpointTable(tuple):
    """Stable sorted tuple table with read-only mapping-style conveniences."""

    def __new__(
        cls,
        values: Iterable[tuple[str, _FrozenEndpointRow]],
    ) -> "_FrozenEndpointTable":
        return super().__new__(cls, tuple(values))

    def __getitem__(self, key: object) -> object:
        if isinstance(key, str):
            for name, value in self:
                if name == key:
                    return value
            raise KeyError(key)
        return super().__getitem__(key)

    def keys(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self)

    def items(self) -> tuple[tuple[str, _FrozenEndpointRow], ...]:
        return tuple(self)


def _freeze_endpoint_table(value: object) -> _FrozenEndpointTable:
    if isinstance(value, Mapping):
        raw_items = list(value.items())
    elif isinstance(value, (tuple, list)):
        raw_items = list(value)
    else:
        raise ValueError("endpoint_table must be a mapping or sorted tuple table")

    rows: list[tuple[str, _FrozenEndpointRow]] = []
    for item in raw_items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("endpoint_table entries must be condition/value pairs")
        condition, raw_endpoints = item
        if not isinstance(condition, str):
            raise ValueError("endpoint_table conditions must be strings")
        if not isinstance(raw_endpoints, Mapping):
            if isinstance(raw_endpoints, (tuple, list)):
                raw_endpoints = dict(raw_endpoints)
            else:
                raise ValueError("endpoint_table values must be endpoint mappings")
        endpoint_items: list[tuple[str, float]] = []
        for name, raw_value in raw_endpoints.items():
            if not isinstance(name, str):
                raise ValueError("endpoint names must be strings")
            if isinstance(raw_value, Mapping):
                raw_value = _extract_numeric(raw_value, ("value", "rho", "estimate", "point"))
            try:
                endpoint_items.append((name, float(raw_value)))
            except (TypeError, ValueError) as error:
                raise ValueError("endpoint values must be numeric") from error
        rows.append((condition, _FrozenEndpointRow(sorted(endpoint_items))))
    return _FrozenEndpointTable(sorted(rows, key=lambda item: item[0]))


def _endpoint_table_to_mapping(value: object) -> dict[str, dict[str, float]]:
    table = _freeze_endpoint_table(value)
    return {
        condition: {name: float(endpoint) for name, endpoint in endpoints}
        for condition, endpoints in table
    }


def _extract_numeric(value: object, keys: Sequence[str]) -> float | None:
    if isinstance(value, Mapping):
        for key in keys:
            if key in value:
                return _extract_numeric(value[key], keys)
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    return float(value)


@dataclass(frozen=True)
class FactorRepairGateDecision:
    """Immutable absolute factor-gate result."""

    passed: bool
    stage: str
    primary_nodes: tuple[int, ...]
    diagnostic_nodes: tuple[int, ...]
    checks: Mapping[str, bool]
    failures: tuple[str, ...]
    evidence_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "stage", str(self.stage))
        object.__setattr__(
            self,
            "primary_nodes",
            tuple(int(node) for node in self.primary_nodes),
        )
        object.__setattr__(
            self,
            "diagnostic_nodes",
            tuple(int(node) for node in self.diagnostic_nodes),
        )
        object.__setattr__(
            self,
            "checks",
            MappingProxyType(
                {str(name): bool(value) for name, value in self.checks.items()}
            ),
        )
        object.__setattr__(self, "failures", tuple(str(item) for item in self.failures))
        object.__setattr__(self, "evidence_sha256", str(self.evidence_sha256))

    @property
    def gate_passed(self) -> bool:
        return self.passed

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "stage": self.stage,
            "primary_nodes": list(self.primary_nodes),
            "diagnostic_nodes": list(self.diagnostic_nodes),
            "checks": dict(self.checks),
            "failures": list(self.failures),
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True)
class FactorRepairEvidence:
    """Convenience immutable evidence record accepted by the selector.

    Existing producers may pass a mapping or an arbitrary object with the
    same attributes instead; this record simply documents the consumer
    contract and supplies a safe default representation for local callers.
    """

    condition: str
    image_ids_hash: str
    image_ids: tuple[str, ...]
    endpoints: Mapping[str, float]
    evidence_sha256: str
    absolute_gate_passed: bool = True
    complete: bool = True
    endpoint_samples: Mapping[str, Sequence[float]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.condition, str) or not self.condition:
            raise ValueError("evidence condition must be a non-empty string")
        if not isinstance(self.image_ids_hash, str) or not self.image_ids_hash:
            raise ValueError("image_ids_hash must be a non-empty string")
        image_ids = tuple(sorted(str(image_id) for image_id in self.image_ids))
        if not image_ids:
            raise ValueError("evidence image_ids must not be empty")
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("evidence image_ids must be unique")
        object.__setattr__(self, "image_ids", image_ids)
        object.__setattr__(self, "endpoints", _freeze(self.endpoints))
        object.__setattr__(self, "absolute_gate_passed", bool(self.absolute_gate_passed))
        object.__setattr__(self, "complete", bool(self.complete))
        if self.endpoint_samples is not None:
            object.__setattr__(self, "endpoint_samples", _freeze(self.endpoint_samples))


@dataclass(frozen=True)
class FactorRepairSelectionDecision:
    """Immutable F0-relative selection decision."""

    reference_condition: str
    selected_condition: str
    delta_s_point: float
    delta_s_ci95: tuple[float, float]
    endpoint_table: object
    reference_evidence_sha256: str
    selected_evidence_sha256: str
    decision_sha256: str

    def __post_init__(self) -> None:
        if self.reference_condition != "F0":
            raise ValueError("selection reference condition must be F0")
        if self.selected_condition not in {"F1", "F2", "F3"}:
            raise ValueError("selection condition must be F1, F2, or F3")
        point = float(self.delta_s_point)
        ci = tuple(float(value) for value in self.delta_s_ci95)
        if len(ci) != 2 or not all(math.isfinite(value) for value in (point, *ci)):
            raise ValueError("selection delta and CI must be finite")
        if ci[0] > ci[1]:
            raise ValueError("selection CI lower bound must not exceed upper bound")
        object.__setattr__(self, "delta_s_point", point)
        object.__setattr__(self, "delta_s_ci95", (ci[0], ci[1]))
        object.__setattr__(self, "endpoint_table", _freeze_endpoint_table(self.endpoint_table))
        object.__setattr__(self, "reference_evidence_sha256", str(self.reference_evidence_sha256))
        object.__setattr__(self, "selected_evidence_sha256", str(self.selected_evidence_sha256))
        object.__setattr__(self, "decision_sha256", str(self.decision_sha256))

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_condition": self.reference_condition,
            "selected_condition": self.selected_condition,
            "delta_s_point": self.delta_s_point,
            "delta_s_ci95": list(self.delta_s_ci95),
            "endpoint_table": _endpoint_table_to_mapping(self.endpoint_table),
            "reference_evidence_sha256": self.reference_evidence_sha256,
            "selected_evidence_sha256": self.selected_evidence_sha256,
            "decision_sha256": self.decision_sha256,
        }

    def verify_digest(self) -> bool:
        expected = digest_selection_decision(
            self.reference_condition,
            self.selected_condition,
            self.delta_s_point,
            self.delta_s_ci95,
            self.endpoint_table,
            self.reference_evidence_sha256,
            self.selected_evidence_sha256,
        )
        return expected == self.decision_sha256


@dataclass(frozen=True)
class PairedDelta:
    point: float
    ci95: tuple[float, float]
    candidate_endpoints: Mapping[str, float]
    candidate_evidence_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", float(self.point))
        ci = tuple(float(value) for value in self.ci95)
        if len(ci) != 2:
            raise ValueError("paired CI must have two values")
        object.__setattr__(self, "ci95", (ci[0], ci[1]))
        object.__setattr__(self, "candidate_endpoints", _freeze(self.candidate_endpoints))
        object.__setattr__(self, "candidate_evidence_sha256", str(self.candidate_evidence_sha256))


def _read(value: object, key: str, default: object = _MISSING) -> object:
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
    else:
        try:
            return getattr(value, key)
        except AttributeError:
            pass
    return default


def _first(value: object, keys: Sequence[str], default: object = _MISSING) -> object:
    for key in keys:
        found = _read(value, key, _MISSING)
        if found is not _MISSING:
            return found
    return default


def _normalise_endpoint_mapping(value: object) -> dict[str, object]:
    raw = _read(value, "endpoints", _MISSING)
    if raw is _MISSING or raw is None:
        if isinstance(value, Mapping) and any(name in value for name in PRIMARY_ENDPOINTS):
            raw = value
        else:
            raw = {}
    if not isinstance(raw, Mapping):
        return {}
    return {
        name: raw[name]
        for name in PRIMARY_ENDPOINTS
        if name in raw
    }


def _normalise_gate_row(value: object, index: int) -> dict[str, object]:
    endpoints = _normalise_endpoint_mapping(value)
    endpoint_values: dict[str, float | None] = {}
    for name, raw in endpoints.items():
        endpoint_values[name] = _extract_numeric(raw, ("value", "rho", "estimate", "point"))
    seed = _first(value, ("seed", "seed_id", "development_seed"), None)
    node = _first(value, ("node_id", "node", "reliability_node"), None)

    def numeric(keys: Sequence[str]) -> float | None:
        found = _first(value, keys, _MISSING)
        if found is _MISSING:
            for section in ("intervention", "specificity", "metrics", "checks"):
                nested = _read(value, section, _MISSING)
                if nested is not _MISSING:
                    found = _first(nested, keys, _MISSING)
                    if found is not _MISSING:
                        break
        return _extract_numeric(found, ("value", "rho", "estimate", "point", "gap"))

    direction = numeric(("direction", "rho", "residual_rho", "composite_score"))
    if direction is None and all(endpoint_values.get(name) is not None for name in PRIMARY_ENDPOINTS):
        direction = sum(float(endpoint_values[name]) for name in PRIMARY_ENDPOINTS) / 4.0
    severity = numeric(("severity_ordering", "ordered_pair_rate", "ordering_rate", "severity_rate"))
    target_response = numeric(("target_response", "paired_target_response", "target_mean_response"))
    background_response = numeric(("background_response", "paired_background_response", "background_mean_response"))
    gap = numeric(("background_gap", "specificity_gap", "paired_specificity_gap"))
    if gap is None and target_response is not None and background_response is not None:
        gap = target_response - background_response
    residual_rho = numeric(("residual_rho", "residual_spearman", "rho"))
    if residual_rho is None:
        # Diagnostic reverse-association evidence is commonly emitted as the
        # first residual endpoint rather than a separate ``rho`` field.
        residual_rho = endpoint_values.get("sampling_residual_spearman")
        if residual_rho is None:
            residual_rho = endpoint_values.get("visibility_residual_spearman")
    malformed = _first(value, ("malformed", "malformed_count", "malformed_pairs"), 0)
    malformed_count = _extract_numeric(malformed, ("value", "count"))
    return {
        "index": index,
        "seed": int(seed) if isinstance(seed, Integral) and not isinstance(seed, (bool, np.bool_)) else None,
        "node": int(node) if isinstance(node, Integral) and not isinstance(node, (bool, np.bool_)) else None,
        "endpoints": endpoint_values,
        "direction": direction,
        "residual_rho": residual_rho,
        "severity": severity,
        "target_response": target_response,
        "background_response": background_response,
        "gap": gap,
        "malformed": malformed_count,
    }


def _row_payload(row: Mapping[str, object]) -> dict[str, object]:
    return {
        key: row[key]
        for key in (
            "seed", "node", "endpoints", "direction", "residual_rho",
            "severity", "target_response", "background_response", "gap", "malformed",
        )
    }


def _unique_failures(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def _ci_bounds(value: object) -> tuple[float, float] | None:
    """Read a registered confidence interval without inventing one."""

    if isinstance(value, Mapping):
        nested = _first(value, ("ci95", "confidence_interval", "interval"), _MISSING)
        if nested is not _MISSING and nested is not value:
            return _ci_bounds(nested)
        lower = _first(value, ("ci_lower", "lower", "lower_bound"), _MISSING)
        upper = _first(value, ("ci_upper", "upper", "upper_bound"), _MISSING)
        if lower is _MISSING or upper is _MISSING:
            return None
        lower_number = _extract_numeric(lower, ("value", "estimate", "point"))
        upper_number = _extract_numeric(upper, ("value", "estimate", "point"))
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        lower_number = _extract_numeric(value[0], ("value", "estimate", "point"))
        upper_number = _extract_numeric(value[1], ("value", "estimate", "point"))
    else:
        return None
    if lower_number is None or upper_number is None:
        return None
    return float(lower_number), float(upper_number)


def _stat_point(value: object) -> float | None:
    return _extract_numeric(value, ("rho", "estimate", "value", "point"))


def _stat_success(value: object) -> bool:
    if not isinstance(value, Mapping) and not is_dataclass(value):
        return False
    status = _read(value, "status", _MISSING)
    if status is _MISSING or str(status) != "ok":
        return False
    success = _read(value, "success", _MISSING)
    if success is not _MISSING and bool(success) is not True:
        return False
    return True


def _normalise_audit_root(value: object) -> Mapping[str, object] | None:
    """Return the factor-result mapping emitted by ``natural_factor_audit``."""

    if value is _MISSING or value is None:
        return None
    factor_results = _read(value, "factor_results", _MISSING)
    if factor_results is not _MISSING:
        value = factor_results
    elif isinstance(value, Mapping):
        nested = _first(value, ("factors", "audit", "natural_factor_audit"), _MISSING)
        if nested is not _MISSING:
            return _normalise_audit_root(nested)
    if not isinstance(value, Mapping):
        return None
    return value


def _audit_factor_section(root: Mapping[str, object], factor: str) -> Mapping[str, object] | None:
    value = root.get(factor, _MISSING)
    if value is _MISSING or not isinstance(value, Mapping):
        return None
    return value


def _audit_alignment(section: Mapping[str, object]) -> Mapping[str, object]:
    value = _first(section, ("alignment", "natural_factor_alignment"), _MISSING)
    if isinstance(value, Mapping):
        return value
    return section


def _audit_intervention(section: Mapping[str, object]) -> Mapping[str, object] | None:
    value = _first(section, ("intervention", "intervention_statistics"), _MISSING)
    return value if isinstance(value, Mapping) else None


def _validate_pooled_stat(
    section: Mapping[str, object],
    name: str,
    failures: list[str],
) -> bool:
    stat = section.get(name, _MISSING)
    if stat is _MISSING or not _stat_success(stat):
        failures.append(f"{name}_missing_or_invalid")
        return False
    point = _stat_point(stat)
    if point is None or not math.isfinite(point):
        failures.append(f"{name}_nonfinite")
        return False
    if not -1.0 <= point <= 1.0:
        failures.append(f"{name}_out_of_bounds")
        return False
    interval = _ci_bounds(section.get(f"{name}_ci", _MISSING))
    if interval is None:
        # Natural-factor-audit names these fields ``pooled_*_ci``; accept a
        # direct ``ci95`` nested on the point statistic only as a producer
        # convenience, while still requiring an explicit interval.
        interval = _ci_bounds(stat)
    if interval is None:
        failures.append(f"{name}_ci_missing")
        return False
    lower, upper = interval
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower > upper
        or lower < -1.0
        or upper > 1.0
    ):
        failures.append(f"{name}_ci_nonfinite_or_invalid")
        return False
    if point <= 0.0 or lower <= 0.0:
        failures.append(f"{name}_ci_crosses_zero")
        return False
    return True


def _evaluate_audit_evidence(
    audit: object,
    *,
    expected_seeds: Sequence[int],
) -> tuple[dict[str, bool], tuple[str, ...]]:
    """Validate pooled natural-audit statistics without re-aggregating rows."""

    root = _normalise_audit_root(audit)
    failures: list[str] = []
    checks: dict[str, bool] = {}
    if root is None:
        return {"pooled_primary_ci": False, "intervention_statistics": False}, (
            "missing_pooled_audit_evidence",
        )

    pooled_ok = True
    intervention_ok = True
    for factor in ("sampling", "visibility"):
        section = _audit_factor_section(root, factor)
        if section is None:
            pooled_ok = False
            intervention_ok = False
            failures.append(f"{factor}_audit_missing")
            continue
        alignment = _audit_alignment(section)
        raw_ok = _validate_pooled_stat(alignment, "pooled_raw", failures)
        residual_ok = _validate_pooled_stat(alignment, "pooled_residual", failures)
        pooled_ok = pooled_ok and raw_ok and residual_ok

        intervention = _audit_intervention(section)
        if intervention is None:
            intervention_ok = False
            failures.append(f"{factor}_intervention_statistics_missing")
            continue
        status = intervention.get("status", _MISSING)
        malformed = _extract_numeric(intervention.get("malformed", _MISSING), ("count", "value"))
        ordered = _extract_numeric(intervention.get("ordered_pair_rate", _MISSING), ("value", "rate"))
        target = _extract_numeric(intervention.get("target_mean_response", _MISSING), ("value", "estimate"))
        background = _extract_numeric(intervention.get("background_mean_response", _MISSING), ("value", "estimate"))
        paired = _extract_numeric(intervention.get("paired_mean", _MISSING), ("value", "estimate"))
        factor_intervention_ok = True
        if status != "ok":
            failures.append(f"{factor}_intervention_status_not_ok")
            factor_intervention_ok = False
        if malformed is None or not math.isfinite(malformed) or malformed != 0.0:
            failures.append(f"{factor}_malformed_intervention_pairs")
            factor_intervention_ok = False
        if ordered is None or not math.isfinite(ordered) or ordered < _SEVERITY_THRESHOLD:
            failures.append(f"{factor}_intervention_order_below_threshold")
            factor_intervention_ok = False
        if (
            target is None
            or background is None
            or not math.isfinite(target)
            or not math.isfinite(background)
            or target <= background
        ):
            failures.append(f"{factor}_target_response_not_stronger_than_background")
            factor_intervention_ok = False
        if paired is None or not math.isfinite(paired) or paired <= 0.0:
            failures.append(f"{factor}_paired_response_not_positive")
            factor_intervention_ok = False
        explicit_gap = _first(
            intervention,
            ("specificity_gap", "background_gap", "paired_specificity_gap"),
            _MISSING,
        )
        if explicit_gap is not _MISSING:
            gap = _extract_numeric(explicit_gap, ("value", "estimate", "point", "gap"))
            if (
                gap is None
                or target is None
                or background is None
                or not math.isfinite(gap)
                or abs(gap - (target - background)) > 1e-12
            ):
                failures.append(f"{factor}_specificity_gap_conflict")
                factor_intervention_ok = False
        eligible = intervention.get("eligible_by_seed_node", _MISSING)
        if not isinstance(eligible, Mapping):
            failures.append(f"{factor}_missing_intervention_seed_node")
            factor_intervention_ok = False
        else:
            for seed in expected_seeds:
                for node in (*PRIMARY_NODE_IDS, *DIAGNOSTIC_NODE_IDS):
                    count = _extract_numeric(eligible.get(f"{seed}:{node}", _MISSING), ("count", "value"))
                    if count is None or not math.isfinite(count) or count < 1.0:
                        failures.append(f"{factor}_missing_intervention_seed_{seed}_node_{node}")
                        factor_intervention_ok = False
        intervention_ok = intervention_ok and factor_intervention_ok

        seed_node = alignment.get("seed_node", _MISSING)
        if not isinstance(seed_node, Mapping):
            failures.append(f"{factor}_seed_node_statistics_missing")
            pooled_ok = False
            continue
        for seed in expected_seeds:
            for node in DIAGNOSTIC_NODE_IDS:
                node_result = seed_node.get(f"{seed}:{node}", _MISSING)
                if not isinstance(node_result, Mapping):
                    failures.append(f"{factor}_diagnostic_statistics_missing")
                    pooled_ok = False
                    continue
                residual = node_result.get("residual", node_result)
                point = _stat_point(residual)
                if point is None or not math.isfinite(point) or not -1.0 <= point <= 1.0:
                    failures.append(f"{factor}_diagnostic_point_missing_or_invalid")
                    pooled_ok = False
                    continue
                if point <= -_REVERSE_RHO_THRESHOLD:
                    failures.append("diagnostic_reverse_association")
                    pooled_ok = False

    checks["pooled_primary_ci"] = pooled_ok
    checks["intervention_statistics"] = intervention_ok
    checks["severity_ordering"] = intervention_ok
    checks["paired_target_response"] = intervention_ok
    checks["background_specificity_gap"] = intervention_ok
    checks["diagnostic_reverse_absence"] = "diagnostic_reverse_association" not in failures
    return checks, _unique_failures(failures)


def evaluate_factor_repair_gate(
    rows: Iterable[object] | Mapping[str, object],
    *,
    stage: str,
) -> FactorRepairGateDecision:
    """Evaluate registered primary/diagnostic evidence with fail-closed rules."""

    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a non-empty string")
    audit = _MISSING
    if isinstance(rows, Mapping):
        audit = _first(rows, ("audit", "natural_factor_audit", "factor_results", "factors"), _MISSING)
        if "rows" in rows:
            rows = rows["rows"]  # type: ignore[index]
        elif audit is not _MISSING:
            rows = ()
        else:
            rows = (rows,)
    else:
        audit = _read(rows, "audit", _MISSING)
    try:
        raw_rows = tuple(rows)
    except TypeError as error:
        raise ValueError("factor gate rows must be iterable") from error
    if stage == "development":
        expected = (17,)
    elif stage == "formal":
        expected = (17, 29, 41)
    else:
        raise ValueError("stage must be development or formal")
    audit_checks, audit_failures = _evaluate_audit_evidence(
        audit,
        expected_seeds=expected,
    )
    normalised = tuple(_normalise_gate_row(row, index) for index, row in enumerate(raw_rows))
    failures: list[str] = list(audit_failures)
    required_pairs = tuple((seed, node) for seed in expected for node in (*PRIMARY_NODE_IDS, *DIAGNOSTIC_NODE_IDS))
    by_pair: dict[tuple[int, int], dict[str, object]] = {}
    duplicate = False
    endpoint_ok = True
    malformed_zero = True
    for row in normalised:
        seed = row["seed"]
        node = row["node"]
        if seed is None or node is None:
            failures.append("missing_seed_or_node")
        else:
            key = (int(seed), int(node))
            if key in by_pair:
                duplicate = True
            by_pair[key] = row
        endpoint_values = row["endpoints"]
        assert isinstance(endpoint_values, Mapping)
        if set(endpoint_values) != set(PRIMARY_ENDPOINTS):
            endpoint_ok = False
        for name in PRIMARY_ENDPOINTS:
            number = endpoint_values.get(name)
            if number is None:
                endpoint_ok = False
                continue
            if not math.isfinite(float(number)):
                endpoint_ok = False
            elif not -1.0 <= float(number) <= 1.0:
                endpoint_ok = False
        malformed = row["malformed"]
        if malformed is None or not math.isfinite(float(malformed)) or float(malformed) > 0.0:
            malformed_zero = False
    missing_pairs = [pair for pair in required_pairs if pair not in by_pair]
    unexpected_pairs = sorted(set(by_pair) - set(required_pairs))
    if missing_pairs:
        failures.append("missing_seed_or_node")
    if unexpected_pairs:
        failures.append("unexpected_seed_or_node")
    if duplicate:
        failures.append("duplicate_seed_node")
    if not endpoint_ok:
        failures.append("missing_or_invalid_endpoint")
    if not malformed_zero:
        failures.append("malformed_intervention_rows")

    primary_rows = [
        by_pair[(seed, node)]
        for seed, node in required_pairs
        if node in PRIMARY_NODE_IDS and (seed, node) in by_pair
    ]
    positive_directions = sum(
        1
        for row in primary_rows
        if row["direction"] is not None and math.isfinite(float(row["direction"])) and float(row["direction"]) > 0.0
    )
    if stage == "development":
        primary_direction_pass = positive_directions >= 3 and len(primary_rows) >= 4
        if not primary_direction_pass:
            failures.append("seed17_primary_directions_below_3_of_4")
    else:
        primary_direction_pass = positive_directions >= 10 and len(primary_rows) >= 12
        if not primary_direction_pass:
            failures.append("formal_primary_directions_below_10_of_12")

    if audit is _MISSING:
        severity_ordering_pass = False
        target_pass = False
        gap_pass = False
        diagnostic_reverse = False
        failures.append("missing_pooled_audit_evidence")
    else:
        severity_ordering_pass = bool(audit_checks.get("severity_ordering", False))
        target_pass = bool(audit_checks.get("paired_target_response", False))
        gap_pass = bool(audit_checks.get("background_specificity_gap", False))
        diagnostic_reverse = not bool(audit_checks.get("diagnostic_reverse_absence", False))
    if not severity_ordering_pass and "severity_ordering_below_0.8" not in failures:
        failures.append("severity_ordering_below_0.8")
    if not target_pass and "paired_target_response_not_positive" not in failures:
        failures.append("paired_target_response_not_positive")
    if not gap_pass and "background_specificity_gap_not_positive" not in failures:
        failures.append("background_specificity_gap_not_positive")
    if diagnostic_reverse and "diagnostic_reverse_association" not in failures:
        failures.append("diagnostic_reverse_association")

    checks = {
        "complete_seed_node_matrix": not missing_pairs and not unexpected_pairs and not duplicate,
        "endpoints_finite_bounded": endpoint_ok,
        "malformed_zero": malformed_zero,
        "primary_direction": primary_direction_pass,
        "pooled_primary_ci": bool(audit_checks.get("pooled_primary_ci", False)),
        "intervention_statistics": bool(audit_checks.get("intervention_statistics", False)),
        "severity_ordering": severity_ordering_pass,
        "paired_target_response": target_pass,
        "background_specificity_gap": gap_pass,
        "diagnostic_reverse_absence": not diagnostic_reverse,
    }
    evidence_payload = {
        "stage": stage,
        "expected_seeds": expected,
        "primary_nodes": PRIMARY_NODE_IDS,
        "diagnostic_nodes": DIAGNOSTIC_NODE_IDS,
        "rows": tuple(_row_payload(row) for row in normalised),
        "audit": _jsonable(audit) if audit is not _MISSING else None,
        "checks": checks,
        "failures": _unique_failures(failures),
    }
    return FactorRepairGateDecision(
        passed=not failures,
        stage=stage,
        primary_nodes=PRIMARY_NODE_IDS,
        diagnostic_nodes=DIAGNOSTIC_NODE_IDS,
        checks=checks,
        failures=_unique_failures(failures),
        evidence_sha256=_canonical_digest(evidence_payload),
    )


def _coerce_evidence(value: object) -> FactorRepairEvidence:
    if isinstance(value, FactorRepairEvidence):
        return value
    condition = _read(value, "condition", _MISSING)
    image_ids_hash = _read(value, "image_ids_hash", _MISSING)
    image_ids = _read(value, "image_ids", _MISSING)
    endpoints = _read(value, "endpoints", _MISSING)
    if condition is _MISSING or image_ids_hash is _MISSING or image_ids is _MISSING or endpoints is _MISSING:
        raise ValueError("factor evidence must include condition, image IDs, and endpoints")
    evidence_hash = _read(value, "evidence_sha256", _MISSING)
    if evidence_hash is _MISSING:
        evidence_hash = _canonical_digest({"condition": condition, "image_ids": tuple(image_ids), "endpoints": endpoints})
    absolute = _first(value, ("absolute_gate_passed", "gate_passed", "passed"), True)
    complete = _read(value, "complete", True)
    samples = _first(value, ("endpoint_samples", "per_image_endpoints", "image_endpoint_table", "image_endpoints"), None)
    return FactorRepairEvidence(
        condition=str(condition),
        image_ids_hash=str(image_ids_hash),
        image_ids=tuple(str(item) for item in image_ids),
        endpoints=endpoints,
        evidence_sha256=str(evidence_hash),
        absolute_gate_passed=bool(absolute),
        complete=bool(complete),
        endpoint_samples=samples,
    )


def _validated_endpoints(evidence: FactorRepairEvidence) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in PRIMARY_ENDPOINTS:
        if name not in evidence.endpoints:
            raise ValueError("factor endpoint evidence is incomplete")
        raw = evidence.endpoints[name]
        number = _extract_numeric(raw, ("value", "rho", "estimate", "point"))
        if number is None or not math.isfinite(number) or not -1.0 <= number <= 1.0:
            raise ValueError("factor endpoint must be finite and bounded")
        result[name] = float(number)
    return result


def composite_mechanism_score(evidence: Mapping[str, object] | FactorRepairEvidence) -> float:
    """Average the four registered, bounded primary endpoints."""

    source: object = evidence.endpoints if isinstance(evidence, FactorRepairEvidence) else evidence
    values: list[float] = []
    for name in PRIMARY_ENDPOINTS:
        if not isinstance(source, Mapping) or name not in source:
            raise ValueError("factor endpoint evidence is incomplete")
        number = _extract_numeric(source[name], ("value", "rho", "estimate", "point"))
        if number is None or not math.isfinite(number) or not -1.0 <= number <= 1.0:
            raise ValueError("factor endpoint must be finite and bounded")
        values.append(float(number))
    return sum(values) / 4.0


def paired_resample_indices(
    *,
    stage: str,
    image_ids_hash: str,
    image_count: int,
    replicate_index: int,
) -> tuple[int, ...]:
    """Return deterministic image-cluster indices for the registered key.

    The candidate condition is intentionally absent from the key.  Therefore
    F0 and every candidate consume exactly the same cluster draw for a given
    stage, image manifest, and replicate index.
    """

    if not isinstance(stage, str) or not stage:
        raise ValueError("stage must be a non-empty string")
    if not isinstance(image_ids_hash, str) or not image_ids_hash:
        raise ValueError("image_ids_hash must be a non-empty string")
    if not isinstance(image_count, Integral) or isinstance(image_count, (bool, np.bool_)) or int(image_count) < 0:
        raise ValueError("image_count must be a non-negative integer")
    if not isinstance(replicate_index, Integral) or isinstance(replicate_index, (bool, np.bool_)) or int(replicate_index) < 0:
        raise ValueError("replicate_index must be a non-negative integer")
    if int(image_count) == 0:
        return ()
    key = (FACTOR_GATE_BOOTSTRAP_SEED, stage, image_ids_hash, int(replicate_index))
    digest = hashlib.sha256(
        json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, int(image_count), size=int(image_count), dtype=np.int64)
    return tuple(int(index) for index in indices)


def _sample_endpoint_values(
    evidence: FactorRepairEvidence,
    indices: Sequence[int],
) -> dict[str, float] | None:
    samples = evidence.endpoint_samples
    if samples is None:
        return None
    if not isinstance(samples, Mapping):
        return None
    image_count = len(evidence.image_ids)
    output: dict[str, float] = {}
    for name in PRIMARY_ENDPOINTS:
        if name not in samples:
            return None
        raw_values = samples[name]
        try:
            values = np.asarray(tuple(float(value) for value in raw_values), dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if len(values) != image_count or not np.isfinite(values).all():
            return None
        selected = values[np.asarray(tuple(indices), dtype=np.intp)]
        output[name] = float(np.mean(selected))
    return output


def recompute_endpoints(
    evidence: FactorRepairEvidence | object,
    indices: Sequence[int],
) -> Mapping[str, float]:
    """Recompute all primary endpoints for one paired image bootstrap draw."""

    coerced = _coerce_evidence(evidence)
    callback = _read(evidence, "recompute_endpoints", _MISSING)
    if callback is not _MISSING and callable(callback):
        values = callback(tuple(indices))
        if not isinstance(values, Mapping):
            raise ValueError("recompute_endpoints must return an endpoint mapping")
        # Validate through the same endpoint checker used for point estimates.
        return _validated_endpoints(
            FactorRepairEvidence(
                condition=coerced.condition,
                image_ids_hash=coerced.image_ids_hash,
                image_ids=coerced.image_ids,
                endpoints=values,
                evidence_sha256=coerced.evidence_sha256,
            )
        )
    sampled = _sample_endpoint_values(coerced, indices)
    if sampled is not None:
        return _validated_endpoints(
            FactorRepairEvidence(
                condition=coerced.condition,
                image_ids_hash=coerced.image_ids_hash,
                image_ids=coerced.image_ids,
                endpoints=sampled,
                evidence_sha256=coerced.evidence_sha256,
            )
        )
    return _validated_endpoints(coerced)


def _validate_paired_evidence(candidate: object, f0: object) -> tuple[FactorRepairEvidence, FactorRepairEvidence]:
    candidate_evidence = _coerce_evidence(candidate)
    f0_evidence = _coerce_evidence(f0)
    if candidate_evidence.image_ids_hash != f0_evidence.image_ids_hash:
        raise ValueError("candidate/F0 evidence image IDs mismatch")
    if tuple(candidate_evidence.image_ids) != tuple(f0_evidence.image_ids):
        raise ValueError("candidate/F0 evidence image IDs mismatch")
    if not candidate_evidence.complete:
        raise ValueError("incomplete candidate evidence")
    if not f0_evidence.complete:
        raise ValueError("incomplete F0 evidence")
    return candidate_evidence, f0_evidence


def _paired_image_cluster_replicate_from_evidence(
    candidate: FactorRepairEvidence,
    f0: FactorRepairEvidence,
    replicate_index: int,
    *,
    indices: Sequence[int] | None = None,
    reference_draw: Mapping[str, float] | None = None,
    candidate_source: object | None = None,
    f0_source: object | None = None,
) -> float:
    if indices is None:
        indices = paired_resample_indices(
            stage="development",
            image_ids_hash=f0.image_ids_hash,
            image_count=len(f0.image_ids),
            replicate_index=replicate_index,
        )
    else:
        indices = tuple(indices)
        if len(indices) != len(f0.image_ids):
            raise ValueError("image-cluster draw length does not match evidence image count")
    # Keep the original producer objects when they expose a raw-row callback.
    # ``_coerce_evidence`` intentionally returns the small immutable point
    # record for arbitrary producers; using that record here would silently
    # discard its ``recompute_endpoints`` method and bootstrap the point value
    # instead of the registered image-cluster statistic.
    candidate_draw = recompute_endpoints(
        candidate if candidate_source is None else candidate_source,
        indices,
    )
    reference_values = (
        reference_draw
        if reference_draw is not None
        else recompute_endpoints(
            f0 if f0_source is None else f0_source,
            indices,
        )
    )
    delta = composite_mechanism_score(candidate_draw) - composite_mechanism_score(reference_values)
    if not math.isfinite(delta):
        raise ValueError("paired bootstrap produced a non-finite delta")
    return float(delta)


def paired_image_cluster_replicate(
    candidate: object,
    f0: object,
    replicate_index: int,
    *,
    indices: Sequence[int] | None = None,
    reference_draw: Mapping[str, float] | None = None,
) -> float:
    """Compute one deterministic candidate-minus-F0 paired replicate.

    ``reference_draw`` is an optional already-computed F0 endpoint mapping.
    Supplying it lets callers evaluate F1/F2/F3 while computing the shared F0
    draw exactly once per replicate; omitting it preserves the legacy pure
    two-evidence behavior.
    """

    if not isinstance(replicate_index, Integral) or isinstance(replicate_index, (bool, np.bool_)):
        raise ValueError("replicate_index must be a non-negative integer")
    replicate_index = int(replicate_index)
    if replicate_index < 0:
        raise ValueError("replicate_index must be a non-negative integer")
    candidate_evidence, f0_evidence = _validate_paired_evidence(candidate, f0)
    return _paired_image_cluster_replicate_from_evidence(
        candidate_evidence,
        f0_evidence,
        replicate_index,
        indices=indices,
        reference_draw=reference_draw,
        candidate_source=candidate,
        f0_source=f0,
    )


def paired_image_cluster_delta(candidate: object, f0: object) -> PairedDelta:
    """Compute candidate-minus-F0 composite delta on paired image clusters."""

    candidate_evidence, f0_evidence = _validate_paired_evidence(candidate, f0)
    candidate_point_endpoints = _validated_endpoints(candidate_evidence)
    f0_point_endpoints = _validated_endpoints(f0_evidence)
    point = composite_mechanism_score(candidate_point_endpoints) - composite_mechanism_score(f0_point_endpoints)
    image_count = len(f0_evidence.image_ids)
    replicates: list[float] = []
    for replicate_index in range(FACTOR_GATE_BOOTSTRAP_REPLICATES):
        indices = paired_resample_indices(
            stage="development",
            image_ids_hash=f0_evidence.image_ids_hash,
            image_count=image_count,
            replicate_index=replicate_index,
        )
        replicates.append(
            _paired_image_cluster_replicate_from_evidence(
                candidate_evidence,
                f0_evidence,
                replicate_index,
                indices=indices,
                candidate_source=candidate,
                f0_source=f0,
            )
        )
    ci = tuple(
        float(value)
        for value in np.quantile(
            np.asarray(replicates, dtype=np.float64),
            FACTOR_GATE_BOOTSTRAP_PERCENTILES,
            method="linear",
        )
    )
    return PairedDelta(
        point=point,
        ci95=(ci[0], ci[1]),
        candidate_endpoints=candidate_point_endpoints,
        candidate_evidence_sha256=candidate_evidence.evidence_sha256,
    )


def digest_selection_decision(
    reference_condition: str,
    selected_condition: str,
    delta_s_point: float,
    delta_s_ci95: Sequence[float],
    endpoint_table: object,
    reference_evidence_sha256: str,
    selected_evidence_sha256: str,
) -> str:
    """Return the canonical SHA-256 digest consumed by decision verifiers."""

    table = _freeze_endpoint_table(endpoint_table)
    payload = {
        "reference_condition": str(reference_condition),
        "selected_condition": str(selected_condition),
        "delta_s_point": float(delta_s_point),
        "delta_s_ci95": tuple(float(value) for value in delta_s_ci95),
        "endpoint_table": table,
        "reference_evidence_sha256": str(reference_evidence_sha256),
        "selected_evidence_sha256": str(selected_evidence_sha256),
    }
    return _canonical_digest(payload)


def _select_from_paired_deltas(
    reference: FactorRepairEvidence,
    candidates: Sequence[object],
    paired_deltas: Mapping[str, object],
) -> FactorRepairSelectionDecision | None:
    """Apply the registered eligibility and tie-break rule to precomputed deltas."""

    if reference.condition != "F0":
        raise ValueError("reference evidence condition must be F0")
    if not reference.complete:
        raise ValueError("incomplete F0 evidence")
    reference_endpoints = _validated_endpoints(reference)
    # Validate the composite once before iterating candidates so malformed F0
    # evidence cannot be hidden by an empty candidate list.
    composite_mechanism_score(reference_endpoints)
    eligible: list[tuple[float, float, str, PairedDelta, FactorRepairEvidence]] = []
    for candidate_raw in candidates:
        candidate = _coerce_evidence(candidate_raw)
        if candidate.condition not in {"F1", "F2", "F3"}:
            raise ValueError("selection candidates must be F1, F2, or F3")
        if (
            tuple(candidate.image_ids) != tuple(reference.image_ids)
            or candidate.image_ids_hash != reference.image_ids_hash
        ):
            raise ValueError("candidate/F0 evidence image IDs mismatch")
        if not candidate.complete or not candidate.absolute_gate_passed:
            continue
        paired_raw = paired_deltas.get(candidate.condition)
        if paired_raw is None:
            raise ValueError(f"precomputed paired delta is missing: {candidate.condition}")
        candidate_hash = getattr(paired_raw, "candidate_evidence_sha256", None)
        if candidate_hash is not None and str(candidate_hash) != candidate.evidence_sha256:
            raise ValueError(f"precomputed paired delta evidence hash mismatch: {candidate.condition}")
        if not hasattr(paired_raw, "ci95") or not hasattr(paired_raw, "point"):
            raise ValueError(f"precomputed paired delta is malformed: {candidate.condition}")
        paired = paired_raw  # type: ignore[assignment]
        lower, upper = tuple(float(value) for value in paired.ci95)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError("paired DeltaS CI must be finite and ordered")
        point = float(paired.point)
        if not math.isfinite(point):
            raise ValueError("paired DeltaS point must be finite")
        if lower > 0.0:
            eligible.append((float(lower), point, candidate.condition, paired, candidate))
    if not eligible:
        return None
    best_lower = max(item[0] for item in eligible)
    lower_tied = [item for item in eligible if abs(item[0] - best_lower) <= _SELECTION_TIE_TOLERANCE]
    best_point = max(item[1] for item in lower_tied)
    point_tied = [item for item in lower_tied if abs(item[1] - best_point) <= _SELECTION_TIE_TOLERANCE]
    _, point, condition, paired, candidate = min(point_tied, key=lambda item: item[2])
    endpoint_table = {
        "F0": reference_endpoints,
        condition: dict(paired.candidate_endpoints),
    }
    digest = digest_selection_decision(
        "F0",
        condition,
        point,
        paired.ci95,
        endpoint_table,
        reference.evidence_sha256,
        candidate.evidence_sha256,
    )
    return FactorRepairSelectionDecision(
        reference_condition="F0",
        selected_condition=condition,
        delta_s_point=point,
        delta_s_ci95=paired.ci95,
        endpoint_table=endpoint_table,
        reference_evidence_sha256=reference.evidence_sha256,
        selected_evidence_sha256=candidate.evidence_sha256,
        decision_sha256=digest,
    )


def select_repair_against_f0(
    f0: object,
    candidates: Iterable[object],
    *,
    paired_deltas: Mapping[str, object] | None = None,
) -> FactorRepairSelectionDecision | None:
    """Select at most one eligible F1/F2/F3 candidate against complete F0.

    ``paired_deltas`` is an optional execution seam for checkpointed callers;
    when omitted, this preserves the historical in-function bootstrap.
    """

    candidate_values = tuple(candidates)
    reference = _coerce_evidence(f0)
    if paired_deltas is None:
        computed: dict[str, object] = {}
        for candidate_raw in candidate_values:
            candidate = _coerce_evidence(candidate_raw)
            if candidate.condition not in {"F1", "F2", "F3"}:
                raise ValueError("selection candidates must be F1, F2, or F3")
            if not candidate.complete or not candidate.absolute_gate_passed:
                continue
            computed[candidate.condition] = paired_image_cluster_delta(candidate_raw, f0)
        paired_deltas = computed
    return _select_from_paired_deltas(reference, candidate_values, paired_deltas)


def select_repair_against_f0_precomputed(
    f0: object,
    candidates: Iterable[object],
    paired_deltas: Mapping[str, object],
) -> FactorRepairSelectionDecision | None:
    """Explicit name for selection from already checkpointed candidate deltas."""

    return select_repair_against_f0(f0, candidates, paired_deltas=paired_deltas)


def require_factor_guided_advancement(*, pre: object, post: object) -> None:
    """Require both pre- and post-adaptation absolute gates to pass."""

    if isinstance(pre, str) or not hasattr(pre, "passed") or not hasattr(post, "passed"):
        raise ValueError("selection decision required")
    if not bool(getattr(pre, "passed")):
        raise ValueError("pre-adaptation factor gate failed")
    if not bool(getattr(post, "passed")):
        raise ValueError("post-adaptation factor gate failed")


__all__ = [
    "PRIMARY_ENDPOINTS",
    "PRIMARY_NODE_IDS",
    "DIAGNOSTIC_NODE_IDS",
    "PRIMARY_NODES",
    "DIAGNOSTIC_NODES",
    "FACTOR_GATE_BOOTSTRAP_REPLICATES",
    "FACTOR_GATE_BOOTSTRAP_SEED",
    "FACTOR_GATE_BOOTSTRAP_PERCENTILES",
    "FactorRepairGateDecision",
    "FactorRepairEvidence",
    "FactorRepairSelectionDecision",
    "PairedDelta",
    "evaluate_factor_repair_gate",
    "composite_mechanism_score",
    "paired_resample_indices",
    "recompute_endpoints",
    "paired_image_cluster_replicate",
    "paired_image_cluster_delta",
    "digest_selection_decision",
    "select_repair_against_f0",
    "select_repair_against_f0_precomputed",
    "require_factor_guided_advancement",
    "spearman",
    "partial_spearman",
]
