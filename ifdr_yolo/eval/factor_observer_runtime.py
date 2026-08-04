"""Checkpoint loading and reliability-context pooling for factor observation.

This module deliberately stops at validated model/context inputs.  Inference,
interventions, and JSONL journaling remain owned by the observer runtime built
on top of these primitives.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ifdr_yolo.eval.factor_observer import (
    DEFAULT_REQUIRED_NODES,
    LetterboxGeometry,
    map_box_to_feature_roi,
)


def _sha256_hex(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a 64-character SHA-256 hex digest")
    if value != value.lower() or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be lowercase hexadecimal")
    return value


def _finite_unit(value: object, field: str) -> float:
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        if not value.is_floating_point() and value.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
            raise ValueError(f"{field} must be a finite number")
        value = float(value.detach().cpu())
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be finite and within [0, 1]")
    return result


@dataclass(frozen=True)
class LoadedIFDRCheckpoint:
    """A trusted IFDR module and the hash of its exact checkpoint bytes."""

    model: nn.Module
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, nn.Module):
            raise ValueError("model must be a torch.nn.Module")
        consume = getattr(self.model, "consume_reliability_context", None)
        if not callable(consume):
            raise ValueError("model must expose callable consume_reliability_context")
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _sha256_hex(self.checkpoint_sha256, "checkpoint_sha256"),
        )


def load_ifdr_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> LoadedIFDRCheckpoint:
    """Load an IFDR checkpoint, preferring its EMA module when present."""

    try:
        checkpoint_path = Path(path)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint path is invalid") from exc
    if not checkpoint_path.is_file():
        raise ValueError(f"checkpoint path does not exist: {checkpoint_path}")
    try:
        raw = checkpoint_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"unable to read checkpoint: {checkpoint_path}") from exc
    if not raw:
        raise ValueError("checkpoint is empty")
    checkpoint_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except Exception as exc:
        raise ValueError(f"unable to load checkpoint: {checkpoint_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must be a mapping containing ema or model")
    candidate = payload.get("ema")
    if candidate is None:
        candidate = payload.get("model")
    if candidate is None:
        raise ValueError("checkpoint must contain a non-null ema or model")
    if not isinstance(candidate, nn.Module):
        raise ValueError("checkpoint ema/model must be a torch.nn.Module")
    consume = getattr(candidate, "consume_reliability_context", None)
    if not callable(consume):
        raise ValueError("checkpoint model must expose callable consume_reliability_context")
    try:
        candidate = candidate.to(device)
        candidate.eval()
    except Exception as exc:
        raise ValueError(f"unable to prepare checkpoint model on device {device!r}") from exc
    return LoadedIFDRCheckpoint(
        model=candidate,
        checkpoint_sha256=checkpoint_sha256,
    )


@dataclass(frozen=True)
class PooledReliability:
    """One reliability node's mean factor and routing values over an ROI."""

    node: int
    roi_xyxy: tuple[int, int, int, int]
    feature_shape: tuple[int, int]
    sampling: float
    visibility: float
    branch_weights: tuple[float, float]
    gate_strength: float

    def __post_init__(self) -> None:
        if isinstance(self.node, bool) or not isinstance(self.node, int):
            raise ValueError("node must be an integer")
        if not isinstance(self.roi_xyxy, tuple) or len(self.roi_xyxy) != 4:
            raise ValueError("roi_xyxy must contain four integer coordinates")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in self.roi_xyxy):
            raise ValueError("roi_xyxy must contain four integer coordinates")
        x1, y1, x2, y2 = self.roi_xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("roi_xyxy must have positive area")
        if not isinstance(self.feature_shape, tuple) or len(self.feature_shape) != 2:
            raise ValueError("feature_shape must contain height and width")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.feature_shape):
            raise ValueError("feature_shape must contain positive integers")
        sampling = _finite_unit(self.sampling, "sampling")
        visibility = _finite_unit(self.visibility, "visibility")
        if not isinstance(self.branch_weights, tuple) or len(self.branch_weights) != 2:
            raise ValueError("branch_weights must contain two values")
        branches = tuple(_finite_unit(value, f"branch_weights[{index}]") for index, value in enumerate(self.branch_weights))
        if abs(sum(branches) - 1.0) > 1e-6:
            raise ValueError("branch_weights must sum to 1 within 1e-6")
        gate_strength = _finite_unit(self.gate_strength, "gate_strength")
        object.__setattr__(self, "sampling", sampling)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "branch_weights", branches)
        object.__setattr__(self, "gate_strength", gate_strength)

    @property
    def node_id(self) -> int:
        return self.node

    @property
    def roi(self) -> tuple[int, int, int, int]:
        return self.roi_xyxy

    @property
    def predicted_sampling(self) -> float:
        return self.sampling

    @property
    def predicted_visibility(self) -> float:
        return self.visibility


def _validate_context(
    context: object,
    *,
    node: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if isinstance(context, Mapping):
        factors = context.get("factors")
        branches = context.get("branch_weights")
        gate_strength = context.get("gate_strength")
    else:
        factors = getattr(context, "factors", None)
        branches = getattr(context, "branch_weights", None)
        gate_strength = getattr(context, "gate_strength", None)
    if (
        not isinstance(factors, torch.Tensor)
        or not isinstance(branches, torch.Tensor)
        or factors.ndim != 4
        or branches.ndim != 4
        or factors.shape[1] != 2
        or branches.shape[1] != 2
        or factors.shape != branches.shape
        or not factors.is_floating_point()
        or not branches.is_floating_point()
    ):
        raise ValueError(f"node {node} contexts must contain matching floating B2HW tensors")
    if not torch.isfinite(factors).all() or not torch.isfinite(branches).all():
        raise ValueError(f"node {node} contexts must be finite")
    if torch.any(factors < 0.0) or torch.any(factors > 1.0):
        raise ValueError(f"node {node} factors must be within [0, 1]")
    if torch.any(branches < 0.0) or torch.any(branches > 1.0):
        raise ValueError(f"node {node} branch_weights must be within [0, 1]")
    if not torch.allclose(
        branches.sum(dim=1),
        torch.ones_like(branches[:, 0]),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(f"node {node} branch_weights must sum to 1")
    gate = _finite_unit(gate_strength, f"node {node} gate_strength")
    return factors.detach(), branches.detach(), gate


def pool_reliability_contexts(
    contexts: Mapping[int, object],
    *,
    batch_index: int,
    bbox_xyxy: Sequence[float],
    geometry: LetterboxGeometry,
    required_nodes: Sequence[int] = DEFAULT_REQUIRED_NODES,
) -> tuple[PooledReliability, ...]:
    """Pool six-node reliability maps over one original-image ROI."""

    try:
        nodes = tuple(required_nodes)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"required_nodes must equal {DEFAULT_REQUIRED_NODES}") from exc
    if nodes != DEFAULT_REQUIRED_NODES:
        raise ValueError(f"required_nodes must equal {DEFAULT_REQUIRED_NODES}")
    if not isinstance(contexts, Mapping) or set(contexts) != set(nodes):
        raise ValueError("contexts must contain exactly the required nodes")
    if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 0:
        raise ValueError("batch_index must be a non-negative integer")
    pooled: list[PooledReliability] = []
    for node in nodes:
        factors, branches, gate = _validate_context(contexts[node], node=node)
        batch_size, _, feature_height, feature_width = factors.shape
        if batch_index >= batch_size:
            raise ValueError("batch_index is outside context batch dimension")
        roi = map_box_to_feature_roi(
            bbox_xyxy,
            geometry,
            (feature_height, feature_width),
        )
        x1, y1, x2, y2 = roi
        factor_mean = factors[batch_index, :, y1:y2, x1:x2].mean(dim=(1, 2))
        branch_mean = branches[batch_index, :, y1:y2, x1:x2].mean(dim=(1, 2))
        pooled.append(
            PooledReliability(
                node=node,
                roi_xyxy=roi,
                feature_shape=(feature_height, feature_width),
                sampling=float(factor_mean[0]),
                visibility=float(factor_mean[1]),
                branch_weights=(float(branch_mean[0]), float(branch_mean[1])),
                gate_strength=gate,
            )
        )
    return tuple(pooled)


__all__ = [
    "LoadedIFDRCheckpoint",
    "PooledReliability",
    "load_ifdr_checkpoint",
    "pool_reliability_contexts",
]
