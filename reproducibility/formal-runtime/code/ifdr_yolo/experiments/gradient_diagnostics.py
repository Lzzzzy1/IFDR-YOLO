from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
import math

import torch


def _validated_parameters(
    parameters: Sequence[torch.nn.Parameter],
) -> tuple[torch.nn.Parameter, ...]:
    result = tuple(parameters)
    if not result:
        raise ValueError("gradient diagnostics require parameters")
    if any(
        not isinstance(parameter, torch.nn.Parameter)
        or not parameter.requires_grad
        for parameter in result
    ):
        raise ValueError(
            "gradient diagnostic parameters must require gradients"
        )
    if len({id(parameter) for parameter in result}) != len(result):
        raise ValueError("gradient diagnostic parameters must be unique")
    return result


def _validated_gradient_inputs(
    tensors: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    result = tuple(tensors)
    if not result:
        raise ValueError("gradient diagnostics require tensors")
    if any(
        not isinstance(tensor, torch.Tensor)
        or not tensor.requires_grad
        or not tensor.is_floating_point()
        for tensor in result
    ):
        raise ValueError(
            "gradient diagnostic tensors must be differentiable floating tensors"
        )
    if len({id(tensor) for tensor in result}) != len(result):
        raise ValueError("gradient diagnostic tensors must be unique")
    return result


def _validated_losses(
    losses: Mapping[str, torch.Tensor],
) -> None:
    if not isinstance(losses, Mapping) or len(losses) < 2:
        raise ValueError("gradient diagnostics require at least two losses")
    if any(
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(loss, torch.Tensor)
        or not loss.requires_grad
        for name, loss in losses.items()
    ):
        raise ValueError(
            "gradient diagnostic losses must be named differentiable tensors"
        )
    for name, loss in losses.items():
        if not torch.isfinite(loss.detach()).all():
            raise FloatingPointError(
                f"non-finite diagnostic loss {name}"
            )


def _gradient_conflict_snapshot(
    losses: Mapping[str, torch.Tensor],
    tensors: Sequence[torch.Tensor],
) -> dict[str, object]:
    _validated_losses(losses)
    diagnostic_tensors = _validated_gradient_inputs(tensors)
    vectors: dict[str, torch.Tensor] = {}
    norms: dict[str, float] = {}
    for name, loss in losses.items():
        gradients = torch.autograd.grad(
            loss.sum(),
            diagnostic_tensors,
            retain_graph=True,
            allow_unused=True,
        )
        vector = torch.cat(
            tuple(
                torch.zeros_like(tensor).reshape(-1)
                if gradient is None
                else gradient.reshape(-1)
                for tensor, gradient in zip(diagnostic_tensors, gradients)
            )
        ).float()
        if not torch.isfinite(vector).all():
            raise FloatingPointError(
                f"non-finite gradient in diagnostic loss {name}"
            )
        vectors[name] = vector
        norms[name] = float(torch.linalg.vector_norm(vector))

    pairs: dict[str, dict[str, object]] = {}
    for first, second in combinations(sorted(vectors), 2):
        denominator = norms[first] * norms[second]
        cosine = None
        if denominator != 0.0:
            raw_cosine = float(
                torch.dot(vectors[first], vectors[second]) / denominator
            )
            cosine = min(1.0, max(-1.0, raw_cosine))
        if cosine is not None and not math.isfinite(cosine):
            raise FloatingPointError(
                f"non-finite gradient cosine for {first} and {second}"
            )
        pairs[f"{first}::{second}"] = {
            "cosine": cosine,
            "conflict": cosine is not None and cosine < 0.0,
        }
    return {
        "schema_version": 1,
        "gradient_norms": norms,
        "pairs": pairs,
    }


def gradient_conflict_snapshot(
    losses: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
) -> dict[str, object]:
    return _gradient_conflict_snapshot(
        losses,
        _validated_parameters(parameters),
    )


def node_gradient_conflict_snapshot(
    losses: Mapping[str, torch.Tensor],
    node_tensors: Mapping[int, Sequence[torch.Tensor] | torch.Tensor],
    *,
    required_nodes: Sequence[int] = (17, 20, 23, 26),
    required_components: Sequence[str] = (
        "detection_base",
        "dcli_conditioning",
        "dcli_calibration",
        "factor_supervision",
    ),
) -> dict[str, object]:
    """Record component gradients independently for each final pyramid node."""

    try:
        expected_nodes = tuple(required_nodes)
    except TypeError as error:
        raise ValueError(
            "required diagnostic nodes must be unique integers"
        ) from error
    if (
        not expected_nodes
        or len(set(expected_nodes)) != len(expected_nodes)
        or any(
            isinstance(node, bool) or not isinstance(node, int)
            for node in expected_nodes
        )
    ):
        raise ValueError("required diagnostic nodes must be unique integers")
    if (
        not isinstance(node_tensors, Mapping)
        or set(node_tensors) != set(expected_nodes)
    ):
        raise ValueError(
            "node diagnostics require exactly the registered nodes"
        )
    _validated_losses(losses)
    try:
        expected_components = tuple(required_components)
    except TypeError as error:
        raise ValueError(
            "node diagnostics are missing required loss components"
        ) from error
    if (
        not expected_components
        or len(set(expected_components)) != len(expected_components)
        or set(expected_components) - set(losses)
    ):
        raise ValueError("node diagnostics are missing required loss components")
    node_snapshots: dict[int, dict[str, object]] = {}
    for node in expected_nodes:
        tensors = node_tensors[node]
        if isinstance(tensors, torch.Tensor):
            tensors = (tensors,)
        if not isinstance(tensors, Sequence) or isinstance(tensors, (str, bytes)):
            raise ValueError(
                f"node {node} diagnostics require tensor sequence"
            )
        validated = _validated_gradient_inputs(tensors)
        if any(
            not torch.isfinite(tensor.detach()).all()
            for tensor in validated
        ):
            raise FloatingPointError(
                f"non-finite diagnostic tensor at node {node}"
            )
        node_snapshots[node] = _gradient_conflict_snapshot(losses, validated)
    return {"schema_version": 1, "nodes": node_snapshots}


def grouped_gradient_conflict_snapshot(
    losses: Mapping[str, torch.Tensor],
    parameter_groups: Mapping[str, Sequence[torch.nn.Parameter]],
) -> dict[str, object]:
    if not isinstance(parameter_groups, Mapping) or not parameter_groups:
        raise ValueError("gradient diagnostics require parameter groups")
    if any(
        not isinstance(name, str) or not name.strip()
        for name in parameter_groups
    ):
        raise ValueError("gradient diagnostic group names must be non-empty")
    groups = {
        name: gradient_conflict_snapshot(losses, parameters)
        for name, parameters in parameter_groups.items()
    }
    return {
        "schema_version": 2,
        "parameter_groups": groups,
    }

class GradientConflictAccumulator:
    def __init__(self) -> None:
        self._observations: defaultdict[str, int] = defaultdict(int)
        self._defined: defaultdict[str, int] = defaultdict(int)
        self._conflicts: defaultdict[str, int] = defaultdict(int)
        self._negative_sum: defaultdict[str, float] = defaultdict(float)

    def update(self, snapshot: Mapping[str, object]) -> None:
        pairs = snapshot.get("pairs")
        if not isinstance(pairs, Mapping):
            raise ValueError("gradient snapshot must contain pairs")
        for name, payload in pairs.items():
            if not isinstance(name, str) or not isinstance(payload, Mapping):
                raise ValueError("invalid gradient pair payload")
            cosine = payload.get("cosine")
            conflict = payload.get("conflict")
            if cosine is not None and (
                isinstance(cosine, bool)
                or not isinstance(cosine, (int, float))
                or not math.isfinite(float(cosine))
            ):
                raise ValueError("gradient cosine must be finite or null")
            if not isinstance(conflict, bool):
                raise ValueError("gradient conflict flag must be boolean")
            if conflict != (cosine is not None and float(cosine) < 0.0):
                raise ValueError("gradient conflict flag contradicts cosine")
            self._observations[name] += 1
            if cosine is not None:
                self._defined[name] += 1
            if conflict:
                self._conflicts[name] += 1
                self._negative_sum[name] += float(cosine)

    def summary(self) -> dict[str, object]:
        pairs: dict[str, object] = {}
        for name in sorted(self._observations):
            observations = self._observations[name]
            conflicts = self._conflicts[name]
            pairs[name] = {
                "observations": observations,
                "defined_cosines": self._defined[name],
                "conflict_frequency": conflicts / observations,
                "mean_negative_cosine": (
                    self._negative_sum[name] / conflicts
                    if conflicts
                    else None
                ),
            }
        return {"schema_version": 1, "pairs": pairs}


class ScheduledGradientDiagnostics:
    def __init__(self, *, interval: int = 0) -> None:
        if (
            isinstance(interval, bool)
            or not isinstance(interval, int)
            or interval < 0
        ):
            raise ValueError(
                "gradient diagnostic interval must be a non-negative integer"
            )
        self.interval = interval
        self._step = 0
        self._records: list[dict[str, object]] = []

    def observe(
        self,
        losses: Mapping[str, torch.Tensor],
        parameters: Sequence[torch.nn.Parameter],
    ) -> dict[str, object] | None:
        self._step += 1
        if self.interval == 0 or self._step % self.interval:
            return None
        record = gradient_conflict_snapshot(losses, parameters)
        record["step"] = self._step
        self._records.append(record)
        return record

    def observe_groups(
        self,
        losses: Mapping[str, torch.Tensor],
        parameter_groups: Mapping[
            str,
            Sequence[torch.nn.Parameter],
        ],
        *,
        node_losses: Mapping[str, torch.Tensor] | None = None,
        node_tensors: Mapping[int, Sequence[torch.Tensor] | torch.Tensor]
        | None = None,
    ) -> dict[str, object] | None:
        if (node_losses is None) != (node_tensors is None):
            raise ValueError(
                "node diagnostic losses and tensors must be provided together"
            )
        self._step += 1
        if self.interval == 0 or self._step % self.interval:
            return None
        grouped_losses = losses
        if (
            isinstance(losses, Mapping)
            and len(losses) < 2
            and node_losses is not None
        ):
            grouped_losses = node_losses
        record = grouped_gradient_conflict_snapshot(
            grouped_losses,
            parameter_groups,
        )
        if node_losses is not None and node_tensors is not None:
            record["node_diagnostics"] = node_gradient_conflict_snapshot(
                node_losses,
                node_tensors,
            )
        record["step"] = self._step
        self._records.append(record)
        return record

    def drain(self) -> tuple[dict[str, object], ...]:
        records = tuple(self._records)
        self._records.clear()
        return records
