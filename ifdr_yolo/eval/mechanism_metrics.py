from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from ifdr_yolo.models.gated_fusion import ReliabilityContext


FACTOR_INDEX = {"sampling": 0, "visibility": 1}
VALID_ROLES = {"object", "background"}


def _finite_float(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be finite")
    return float(value)


@dataclass(frozen=True)
class MechanismResponse:
    kind: str
    role: str
    strength: float
    node: int
    target_response: float
    expected_response: float
    target_mae: float
    leakage: float
    selectivity: float | None
    routing_shift: float
    gate_strength: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in FACTOR_INDEX:
            raise ValueError("kind must be sampling or visibility")
        if self.role not in VALID_ROLES:
            raise ValueError("role must be object or background")
        strength = _finite_float(self.strength, "strength")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be within [0, 1]")
        if isinstance(self.node, bool) or not isinstance(self.node, int):
            raise ValueError("node must be an integer")
        for field in (
            "target_response",
            "expected_response",
            "target_mae",
            "leakage",
            "routing_shift",
            "gate_strength",
        ):
            value = _finite_float(getattr(self, field), field)
            if field in {
                "target_mae",
                "leakage",
                "routing_shift",
                "gate_strength",
            } and value < 0.0:
                raise ValueError(f"{field} must be non-negative")
            object.__setattr__(self, field, value)
        if self.selectivity is not None:
            selectivity = _finite_float(self.selectivity, "selectivity")
            if not 0.0 <= selectivity <= 1.0:
                raise ValueError("selectivity must be within [0, 1]")
            object.__setattr__(self, "selectivity", selectivity)
        object.__setattr__(self, "strength", strength)

    @property
    def effective_routing_shift(self) -> float:
        return self.routing_shift * self.gate_strength


def _validated_targets(
    delta_target: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        not isinstance(delta_target, torch.Tensor)
        or not isinstance(weight, torch.Tensor)
        or delta_target.shape != weight.shape
        or delta_target.ndim != 4
        or delta_target.shape[1] != 2
        or not delta_target.is_floating_point()
        or not weight.is_floating_point()
    ):
        raise ValueError("delta_target and weight must be matching B2HW tensors")
    if (
        not torch.isfinite(delta_target).all()
        or not torch.isfinite(weight).all()
        or torch.any(weight < 0.0)
    ):
        raise ValueError("targets must be finite with non-negative weights")
    return delta_target.detach(), weight.detach()


def _validated_context_pair(
    intervention: object,
    clean: object,
    *,
    batch_size: int,
    node: int,
) -> tuple[ReliabilityContext, ReliabilityContext]:
    if not isinstance(intervention, ReliabilityContext) or not isinstance(
        clean,
        ReliabilityContext,
    ):
        raise ValueError(f"node {node} must contain reliability contexts")
    tensors = (
        intervention.factors,
        clean.factors,
        intervention.branch_weights,
        clean.branch_weights,
    )
    if any(
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.ndim != 4
        or value.shape[0] != batch_size
        or value.shape[1] != 2
        for value in tensors
    ):
        raise ValueError(f"node {node} contexts must contain B2HW tensors")
    if any(value.shape != tensors[0].shape for value in tensors[1:]):
        raise ValueError(f"node {node} context shapes must match")
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError(f"node {node} contexts must be finite")
    return intervention, clean


def _weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> float:
    denominator = weight.sum()
    if denominator <= 0.0:
        raise ValueError("mechanism response requires positive target support")
    return float((values * weight).sum() / denominator)


def measure_paired_mechanism_response(
    intervention_contexts: Mapping[int, ReliabilityContext],
    clean_contexts: Mapping[int, ReliabilityContext],
    delta_target: torch.Tensor,
    weight: torch.Tensor,
    *,
    kind: str,
    role: str,
    strength: float,
) -> tuple[MechanismResponse, ...]:
    if not isinstance(intervention_contexts, Mapping) or not isinstance(
        clean_contexts,
        Mapping,
    ):
        raise ValueError("contexts must be mappings")
    if not intervention_contexts or set(intervention_contexts) != set(
        clean_contexts
    ):
        raise ValueError("intervention and clean contexts must use the same nodes")
    if kind not in FACTOR_INDEX:
        raise ValueError("kind must be sampling or visibility")
    if role not in VALID_ROLES:
        raise ValueError("role must be object or background")
    strength = _finite_float(strength, "strength")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be within [0, 1]")
    delta_target, weight = _validated_targets(delta_target, weight)
    factor_index = FACTOR_INDEX[kind]
    other_index = 1 - factor_index
    responses: list[MechanismResponse] = []
    for node in sorted(intervention_contexts):
        intervention, clean = _validated_context_pair(
            intervention_contexts[node],
            clean_contexts[node],
            batch_size=delta_target.shape[0],
            node=node,
        )
        size = intervention.factors.shape[-2:]
        scaled_target = F.interpolate(delta_target, size=size, mode="area")
        scaled_weight = F.interpolate(weight, size=size, mode="area")
        predicted_delta = intervention.factors - clean.factors
        target_weight = scaled_weight[:, factor_index]
        other_weight = scaled_weight[:, other_index]
        target_response = _weighted_mean(
            predicted_delta[:, factor_index],
            target_weight,
        )
        expected_response = _weighted_mean(
            scaled_target[:, factor_index],
            target_weight,
        )
        target_mae = _weighted_mean(
            (
                predicted_delta[:, factor_index]
                - scaled_target[:, factor_index]
            ).abs(),
            target_weight,
        )
        leakage = _weighted_mean(
            predicted_delta[:, other_index].abs(),
            other_weight,
        )
        support = scaled_weight.amax(dim=1)
        routing_shift = _weighted_mean(
            (
                intervention.branch_weights - clean.branch_weights
            ).abs().mean(dim=1),
            support,
        )
        selectivity_denominator = abs(target_response) + leakage
        selectivity = (
            abs(target_response) / selectivity_denominator
            if selectivity_denominator > 0.0
            else None
        )
        responses.append(
            MechanismResponse(
                kind=kind,
                role=role,
                strength=strength,
                node=node,
                target_response=target_response,
                expected_response=expected_response,
                target_mae=target_mae,
                leakage=leakage,
                selectivity=selectivity,
                routing_shift=routing_shift,
                gate_strength=abs(intervention.gate_strength),
            )
        )
    return tuple(responses)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index, _ in indexed[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        return None
    first_mean = _mean(first)
    second_mean = _mean(second)
    first_delta = [value - first_mean for value in first]
    second_delta = [value - second_mean for value in second]
    denominator = math.sqrt(
        sum(value * value for value in first_delta)
        * sum(value * value for value in second_delta)
    )
    if denominator == 0.0:
        return None
    result = sum(
        left * right for left, right in zip(first_delta, second_delta)
    ) / denominator
    return min(1.0, max(-1.0, result))


def _curve(records: Sequence[MechanismResponse]) -> dict[str, object]:
    grouped: defaultdict[float, list[MechanismResponse]] = defaultdict(list)
    for record in records:
        grouped[record.strength].append(record)
    strengths = sorted(grouped)

    def level_mean(field: str) -> list[float | None]:
        result: list[float | None] = []
        for strength in strengths:
            values = [getattr(record, field) for record in grouped[strength]]
            defined = [float(value) for value in values if value is not None]
            result.append(_mean(defined) if defined else None)
        return result

    target_response = [float(value) for value in level_mean("target_response")]
    expected_response = [
        float(value) for value in level_mean("expected_response")
    ]
    target_mae = [float(value) for value in level_mean("target_mae")]
    leakage = [float(value) for value in level_mean("leakage")]
    selectivity = level_mean("selectivity")
    routing_shift = [float(value) for value in level_mean("routing_shift")]
    effective_routing_shift = [
        _mean(
            [
                record.effective_routing_shift
                for record in grouped[strength]
            ]
        )
        for strength in strengths
    ]
    directional = [
        record.target_response * record.expected_response > 0.0
        for record in records
        if record.expected_response != 0.0
    ]
    response_span = expected_response[-1] - expected_response[0]
    response_gain = (
        (target_response[-1] - target_response[0]) / response_span
        if response_span != 0.0
        else None
    )
    return {
        "strengths": strengths,
        "samples_per_strength": [len(grouped[value]) for value in strengths],
        "target_response": target_response,
        "expected_response": expected_response,
        "target_mae": target_mae,
        "leakage": leakage,
        "selectivity": selectivity,
        "routing_shift": routing_shift,
        "effective_routing_shift": effective_routing_shift,
        "spearman": _pearson(
            _average_ranks(strengths),
            _average_ranks(target_response),
        ),
        "monotonic_violations": sum(
            right < left
            for left, right in zip(target_response, target_response[1:])
        ),
        "direction_agreement": (
            sum(directional) / len(directional) if directional else None
        ),
        "response_gain": response_gain,
    }


def summarize_mechanism_responses(
    responses: Sequence[MechanismResponse],
) -> dict[str, object]:
    if not responses or any(
        not isinstance(response, MechanismResponse) for response in responses
    ):
        raise ValueError("responses must contain MechanismResponse values")
    grouped: defaultdict[tuple[str, str], list[MechanismResponse]] = (
        defaultdict(list)
    )
    for response in responses:
        grouped[(response.kind, response.role)].append(response)
    conditions: dict[str, dict[str, object]] = {}
    for (kind, role), records in sorted(grouped.items()):
        by_node: defaultdict[int, list[MechanismResponse]] = defaultdict(list)
        for record in records:
            by_node[record.node].append(record)
        conditions.setdefault(kind, {})[role] = {
            "aggregate": _curve(records),
            "nodes": {
                str(node): _curve(node_records)
                for node, node_records in sorted(by_node.items())
            },
        }
    return {
        "schema_version": 1,
        "conditions": conditions,
    }
