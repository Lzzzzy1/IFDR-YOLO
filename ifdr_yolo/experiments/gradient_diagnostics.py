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


def gradient_conflict_snapshot(
    losses: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
) -> dict[str, object]:
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
    diagnostic_parameters = _validated_parameters(parameters)
    vectors: dict[str, torch.Tensor] = {}
    norms: dict[str, float] = {}
    for name, loss in losses.items():
        gradients = torch.autograd.grad(
            loss.sum(),
            diagnostic_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        vector = torch.cat(
            tuple(
                torch.zeros_like(parameter).reshape(-1)
                if gradient is None
                else gradient.reshape(-1)
                for parameter, gradient in zip(
                    diagnostic_parameters,
                    gradients,
                )
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
    ) -> dict[str, object] | None:
        self._step += 1
        if self.interval == 0 or self._step % self.interval:
            return None
        record = grouped_gradient_conflict_snapshot(
            losses,
            parameter_groups,
        )
        record["step"] = self._step
        self._records.append(record)
        return record

    def drain(self) -> tuple[dict[str, object], ...]:
        records = tuple(self._records)
        self._records.clear()
        return records
