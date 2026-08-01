from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class ReliabilityContext:
    factors: torch.Tensor
    branch_weights: torch.Tensor
    gate_strength: float


def _positive_channels(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


class ResidualSemanticMapAdapter(nn.Module):
    """Zero-residual task adapter for protected BCHW semantics."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = _positive_channels(channels, "channels")
        hidden = max(4, channels // 4)
        self.channels = channels
        self.delta = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, semantics: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(semantics, torch.Tensor)
            or semantics.ndim != 4
            or semantics.shape[1] != self.channels
        ):
            raise ValueError(
                f"semantics must have shape [batch, {self.channels}, h, w]"
            )
        return semantics + self.delta(semantics)


class ResidualFactorAdapter(nn.Module):
    """Bounded zero-residual adapter for sampling/visibility factors."""

    def __init__(self, hidden_channels: int = 8) -> None:
        super().__init__()
        hidden = _positive_channels(hidden_channels, "hidden_channels")
        self.delta = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, factors: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(factors, torch.Tensor)
            or not factors.is_floating_point()
            or factors.ndim < 1
            or factors.shape[-1] != 2
        ):
            raise ValueError("factors must be floating point and end with 2")
        residual = 0.25 * self.delta(factors).tanh()
        return (factors + residual).clamp(0.0, 1.0)


class ReliabilityEstimator(nn.Module):
    """Cross-scale estimator that gives both factors one shared meaning."""

    def __init__(self, reliability_channels: int = 32) -> None:
        super().__init__()
        channels = _positive_channels(
            reliability_channels,
            "reliability_channels",
        )
        self.reliability_channels = channels
        combined = channels * 2
        self.shared_core = nn.Sequential(
            nn.Conv2d(
                combined,
                combined,
                3,
                padding=1,
                groups=combined,
                bias=False,
            ),
            nn.GroupNorm(1, combined),
            nn.SiLU(),
            nn.Conv2d(combined, channels, 1, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
        )
        self.factor_head = nn.Conv2d(channels, 2, 1)

    def forward(self, projected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reliability = self.shared_core(projected)
        return reliability, self.factor_head(reliability).sigmoid()


class ReliabilityGatedConcat(nn.Module):
    def __init__(
        self,
        *,
        input_channels: tuple[int, int],
        reliability_channels: int = 32,
        reliability_estimator: ReliabilityEstimator | None = None,
        semantic_protection: bool = False,
    ) -> None:
        super().__init__()
        if (
            not isinstance(input_channels, tuple)
            or len(input_channels) != 2
        ):
            raise ValueError("input_channels must contain exactly two values")
        self.input_channels = tuple(
            _positive_channels(value, f"input_channels[{index}]")
            for index, value in enumerate(input_channels)
        )
        channels = _positive_channels(
            reliability_channels,
            "reliability_channels",
        )
        self.reliability_channels = channels
        if reliability_estimator is None:
            reliability_estimator = ReliabilityEstimator(channels)
        if (
            not isinstance(reliability_estimator, ReliabilityEstimator)
            or reliability_estimator.reliability_channels != channels
        ):
            raise ValueError(
                "reliability_estimator must match reliability_channels"
            )
        self.reliability_estimator = reliability_estimator
        if not isinstance(semantic_protection, bool):
            raise ValueError("semantic_protection must be a boolean")
        self.semantic_protection = semantic_protection
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(input_channel, channels, 1, bias=False),
                nn.GroupNorm(1, channels),
                nn.SiLU(),
            )
            for input_channel in self.input_channels
        )
        self.router = nn.Conv2d(channels + 2, 2, 1)
        self.fusion_adapter = (
            ResidualSemanticMapAdapter(channels + 2)
            if semantic_protection
            else None
        )
        self.gate_logit = nn.Parameter(torch.tensor(0.0))
        self.register_buffer(
            "_schedule",
            torch.tensor(0.0),
            persistent=True,
        )
        self._context: ReliabilityContext | None = None

    @property
    def shared_core(self) -> nn.Module:
        return self.reliability_estimator.shared_core

    @property
    def factor_head(self) -> nn.Module:
        return self.reliability_estimator.factor_head

    def set_schedule(self, value: float) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError("schedule must be finite and within [0, 1]")
        self._schedule.fill_(float(value))

    def _validate_inputs(
        self,
        inputs: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError("fusion inputs must contain exactly two tensors")
        first, second = inputs
        if not isinstance(first, torch.Tensor) or not isinstance(
            second,
            torch.Tensor,
        ):
            raise ValueError("fusion inputs must be tensors")
        if first.ndim != 4 or second.ndim != 4:
            raise ValueError("fusion inputs must be BCHW tensors")
        if first.shape[0] != second.shape[0]:
            raise ValueError("fusion inputs must share batch size")
        if first.shape[2:] != second.shape[2:]:
            raise ValueError("fusion inputs must share spatial dimensions")
        for index, (tensor, expected) in enumerate(
            zip((first, second), self.input_channels)
        ):
            if tensor.shape[1] != expected:
                raise ValueError(
                    f"fusion input {index} channels must equal {expected}"
                )
        return first, second

    def forward(
        self,
        inputs: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        first, second = self._validate_inputs(inputs)
        projected = torch.cat(
            (
                self.projections[0](first),
                self.projections[1](second),
            ),
            dim=1,
        )
        reliability, factors = self.reliability_estimator(projected)
        semantics = torch.cat((reliability, factors), dim=1)
        if self.semantic_protection:
            assert self.fusion_adapter is not None
            semantics = self.fusion_adapter(semantics.detach())
        branch_weights = self.router(semantics).softmax(dim=1)
        gate_strength = self._schedule * self.gate_logit.sigmoid()
        first_scale = 1.0 + gate_strength * (
            branch_weights[:, 0:1] - 0.5
        )
        second_scale = 1.0 + gate_strength * (
            branch_weights[:, 1:2] - 0.5
        )
        output = torch.cat(
            (first * first_scale, second * second_scale),
            dim=1,
        )
        self._context = ReliabilityContext(
            factors=factors,
            branch_weights=branch_weights,
            gate_strength=float(gate_strength.detach().cpu()),
        )
        return output

    def consume_context(self) -> ReliabilityContext:
        context = self._context
        if context is None:
            raise RuntimeError("no reliability context is available")
        self._context = None
        return context
