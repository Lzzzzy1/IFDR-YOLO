"""Checkpoint loading and reliability-context pooling for factor observation.

This module deliberately stops at validated model/context inputs.  Inference,
interventions, and JSONL journaling remain owned by the observer runtime built
on top of these primitives.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from ifdr_yolo.data.interventions import (
    InterventionKind,
    InterventionRole,
    InterventionSpec,
    apply_intervention,
    factor_target_for_spec,
)
from ifdr_yolo.eval.factor_observer import (
    DEFAULT_REQUIRED_NODES,
    FactorObservationJournal,
    FactorObservationManifest,
    ImageObservationPlan,
    LetterboxGeometry,
    ObservationCondition,
    map_box_to_feature_roi,
    letterbox_image,
)
from ifdr_yolo.eval.natural_factor_audit import NaturalFactorObservation


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
            io.BytesIO(raw),
            map_location="cpu",
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
        candidate = candidate.float()
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
        or factors.device != branches.device
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


def _required_context_nodes(required_nodes: Sequence[int]) -> tuple[int, ...]:
    try:
        nodes = tuple(required_nodes)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"required_nodes must equal {DEFAULT_REQUIRED_NODES}") from exc
    if nodes != DEFAULT_REQUIRED_NODES:
        raise ValueError(f"required_nodes must equal {DEFAULT_REQUIRED_NODES}")
    return nodes


def _prepare_reliability_contexts(
    contexts: Mapping[int, object],
    *,
    required_nodes: Sequence[int] = DEFAULT_REQUIRED_NODES,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor, float]], int]:
    """Validate one model context batch once for reuse across multiple ROIs."""

    nodes = _required_context_nodes(required_nodes)
    if not isinstance(contexts, Mapping) or set(contexts) != set(nodes):
        raise ValueError("contexts must contain exactly the required nodes")
    prepared: dict[int, tuple[torch.Tensor, torch.Tensor, float]] = {}
    expected_batch_size: int | None = None
    expected_device: torch.device | None = None
    for node in nodes:
        factors, branches, gate = _validate_context(contexts[node], node=node)
        batch_size = factors.shape[0]
        if expected_batch_size is None:
            expected_batch_size = batch_size
            expected_device = factors.device
        elif batch_size != expected_batch_size:
            raise ValueError("contexts must share batch size across nodes")
        elif factors.device != expected_device:
            raise ValueError("contexts must share device across nodes")
        prepared[node] = (factors, branches, gate)
    assert expected_batch_size is not None
    return prepared, expected_batch_size


def _pool_prepared_reliability_contexts(
    prepared: Mapping[int, tuple[torch.Tensor, torch.Tensor, float]],
    batch_size: int,
    *,
    batch_index: int,
    bbox_xyxy: Sequence[float],
    geometry: LetterboxGeometry,
    required_nodes: Sequence[int] = DEFAULT_REQUIRED_NODES,
) -> tuple[PooledReliability, ...]:
    nodes = _required_context_nodes(required_nodes)
    if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 0:
        raise ValueError("batch_index must be a non-negative integer")
    if batch_index >= batch_size:
        raise ValueError("batch_index is outside context batch dimension")
    pooled: list[PooledReliability] = []
    for node in nodes:
        factors, branches, gate = prepared[node]
        _, _, feature_height, feature_width = factors.shape
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


def pool_reliability_contexts(
    contexts: Mapping[int, object],
    *,
    batch_index: int,
    bbox_xyxy: Sequence[float],
    geometry: LetterboxGeometry,
    required_nodes: Sequence[int] = DEFAULT_REQUIRED_NODES,
) -> tuple[PooledReliability, ...]:
    """Pool six-node reliability maps over one original-image ROI."""

    prepared, batch_size = _prepare_reliability_contexts(
        contexts,
        required_nodes=required_nodes,
    )
    return _pool_prepared_reliability_contexts(
        prepared,
        batch_size,
        batch_index=batch_index,
        bbox_xyxy=bbox_xyxy,
        geometry=geometry,
        required_nodes=required_nodes,
    )


def _model_device(model: nn.Module) -> torch.device:
    for parameter in model.parameters():
        return parameter.device
    for buffer in model.buffers():
        return buffer.device
    return torch.device("cpu")


def _read_png_once(path: str | Path) -> tuple[np.ndarray, int, int, str]:
    """Read and decode one PNG while retaining the bytes used for hashing."""

    image_path = Path(path)
    try:
        raw = image_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"unable to read image path: {image_path}") from exc
    if not raw:
        raise ValueError(f"image is empty: {image_path}")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image is not a readable PNG: {image_path}")
    return image, int(image.shape[1]), int(image.shape[0]), hashlib.sha256(raw).hexdigest()


def _normalized_bbox(
    bbox_xyxy: Sequence[float],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    return (x1 / width, y1 / height, x2 / width, y2 / height)


_OBSERVATION_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_sha256",
        "observation_id",
        "condition_id",
        "transform_id",
        "checkpoint_sha256",
        "source_sha256",
        "seed",
        "transform_seed",
        "node_id",
        "image_id",
        "object_id",
        "class_id",
        "class_name",
        "bbox_xyxy",
        "box_height",
        "natural_sampling",
        "natural_visibility",
        "region_xyxy",
        "region_role",
        "intervention_kind",
        "intervention_factor",
        "intervention_severity",
        "pair_id",
        "matched_background_bbox",
        "predicted_sampling",
        "predicted_visibility",
        "branch_weights",
        "gate_strength",
        "feature_roi_xyxy",
        "feature_shape",
        "input_shape",
    }
)


def _transform_seed_for_condition(condition: ObservationCondition) -> int | None:
    """Derive the fixed common-random-number seed for one intervention pair.

    The protocol is the first eight digest bytes, interpreted big-endian and
    masked to 63 bits, over ``b"ifdr-observer-transform-v1\\0" + pair_id``.
    Natural and clean source transforms deliberately have no random seed.
    """

    if condition.intervention_kind in {"natural", "clean"}:
        return None
    pair_id = condition.pair_id
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("intervention condition must have a pair_id")
    digest = hashlib.sha256(
        b"ifdr-observer-transform-v1\0" + pair_id.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _plan_letterbox_geometry(plan: ImageObservationPlan, input_size: int) -> LetterboxGeometry:
    scale = min(input_size / plan.width, input_size / plan.height)
    resized_width = max(1, int(round(plan.width * scale)))
    resized_height = max(1, int(round(plan.height * scale)))
    pad_width = input_size - resized_width
    pad_height = input_size - resized_height
    pad_left = pad_width // 2
    pad_top = pad_height // 2
    return LetterboxGeometry(
        original_width=plan.width,
        original_height=plan.height,
        input_size=input_size,
        scale=scale,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_height // 2,
        pad_right=pad_width - pad_left,
        pad_bottom=pad_height - pad_top,
    )


def _condition_observation_ids(
    plan: ImageObservationPlan,
    nodes: Sequence[int],
) -> dict[tuple[str, int], str]:
    expected: dict[tuple[str, int], str] = {}
    width = len(tuple(nodes))
    if len(plan.expected_observation_ids) != len(plan.conditions) * width:
        raise ValueError("plan expected observation IDs have an invalid count")
    for condition_index, condition in enumerate(plan.conditions):
        for node_index, node in enumerate(nodes):
            expected[(condition.condition_id, node)] = plan.expected_observation_ids[
                condition_index * width + node_index
            ]
    return expected


def _require_row_mapping(row: object) -> Mapping[str, object]:
    if not isinstance(row, Mapping) or any(not isinstance(key, str) for key in row):
        raise ValueError("observation row must be a JSON object with string keys")
    return row


def _row_number(value: object, field: str, *, unit: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (unit and not 0.0 <= result <= 1.0):
        if unit:
            raise ValueError(f"{field} must be finite and within [0, 1]")
        raise ValueError(f"{field} must be finite")
    return result


def _row_integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _row_box(value: object, field: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError(f"{field} must contain four coordinates")
    result = tuple(_row_number(item, f"{field}[{index}]") for index, item in enumerate(value))
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"{field} must have positive area")
    return result


def _row_equal(actual: object, expected: object, field: str) -> None:
    if actual != expected:
        raise ValueError(f"observation {field} does not match manifest")


def validate_observation_row(
    row: Mapping[str, object],
    *,
    manifest: FactorObservationManifest,
    plan: ImageObservationPlan,
    condition: ObservationCondition,
    node_id: int,
    observation_id: str,
    checkpoint_sha256: str,
    manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Validate one JSON-native row against its immutable plan identity."""

    if manifest_sha256 is None:
        manifest_sha256 = manifest.hash()
    value = _require_row_mapping(row)
    if set(value) != _OBSERVATION_ROW_FIELDS:
        raise ValueError("observation row fields do not match schema")
    if value.get("schema_version") != 1:
        raise ValueError("observation schema_version must be 1")
    _row_equal(value.get("manifest_sha256"), manifest_sha256, "manifest_sha256")
    _row_equal(value.get("observation_id"), observation_id, "observation_id")
    for field, expected in (
        ("condition_id", condition.condition_id),
        ("transform_id", condition.transform_id),
        ("checkpoint_sha256", checkpoint_sha256),
        ("source_sha256", condition.source_sha256),
        ("image_id", condition.image_id),
        ("seed", condition.seed),
        ("transform_seed", _transform_seed_for_condition(condition)),
        ("object_id", condition.object_id),
        ("class_id", condition.class_id),
        ("class_name", condition.class_name),
        ("region_role", condition.region_role),
        ("intervention_kind", condition.intervention_kind),
        ("intervention_factor", condition.intervention_factor),
        ("pair_id", condition.pair_id),
    ):
        _row_equal(value.get(field), expected, field)
    if _row_integer(value.get("node_id"), "node_id") != node_id:
        raise ValueError("observation node_id does not match manifest")
    bbox = _row_box(value.get("bbox_xyxy"), "bbox_xyxy")
    if bbox != condition.bbox_xyxy:
        raise ValueError("observation bbox does not match manifest")
    if not math.isclose(
        _row_number(value.get("box_height"), "box_height"),
        condition.box_height,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("observation box_height does not match manifest")
    for field, expected in (
        ("natural_sampling", condition.natural_sampling),
        ("natural_visibility", condition.natural_visibility),
        ("intervention_severity", condition.intervention_severity),
    ):
        if not math.isclose(
            _row_number(value.get(field), field),
            expected,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"observation {field} does not match manifest")
    matched = value.get("matched_background_bbox")
    expected_matched = condition.matched_background_bbox
    if expected_matched is None:
        if matched is not None:
            raise ValueError("observation matched background must be null")
    elif _row_box(matched, "matched_background_bbox") != expected_matched:
        raise ValueError("observation matched background does not match manifest")

    region = _row_box(value.get("region_xyxy"), "region_xyxy")
    expected_region = _normalized_bbox(
        condition.bbox_xyxy,
        width=plan.width,
        height=plan.height,
    )
    if any(not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9) for left, right in zip(region, expected_region)):
        raise ValueError("observation region does not match manifest")
    input_shape = value.get("input_shape")
    expected_input_shape = [3, manifest.input_size, manifest.input_size]
    if input_shape != expected_input_shape:
        raise ValueError("input_shape does not match manifest input_size")
    roi_values = _row_box(value.get("feature_roi_xyxy"), "feature_roi_xyxy")
    roi_int = tuple(_row_integer(int(item), "feature ROI coordinate") for item in roi_values)
    if any(float(item) != value for item, value in zip(roi_int, roi_values)):
        raise ValueError("feature ROI coordinates must be integers")
    shape = value.get("feature_shape")
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError("feature_shape must contain height and width")
    feature_shape = tuple(_row_integer(item, "feature_shape", minimum=1) for item in shape)
    if roi_int[2] <= roi_int[0] or roi_int[3] <= roi_int[1]:
        raise ValueError("feature ROI must have positive area")
    sampling = _row_number(value.get("predicted_sampling"), "predicted_sampling", unit=True)
    visibility = _row_number(value.get("predicted_visibility"), "predicted_visibility", unit=True)
    branch_value = value.get("branch_weights")
    if not isinstance(branch_value, (tuple, list)) or len(branch_value) != 2:
        raise ValueError("branch_weights must contain two values")
    branches = tuple(_row_number(item, f"branch_weights[{index}]", unit=True) for index, item in enumerate(branch_value))
    if abs(sum(branches) - 1.0) > 1e-6:
        raise ValueError("branch_weights must sum to 1")
    gate = _row_number(value.get("gate_strength"), "gate_strength", unit=True)
    geometry = _plan_letterbox_geometry(plan, manifest.input_size)
    expected_roi = map_box_to_feature_roi(condition.bbox_xyxy, geometry, feature_shape)
    if roi_int != expected_roi:
        raise ValueError("feature ROI does not match letterbox geometry")
    NaturalFactorObservation(
        seed=condition.seed,
        node_id=node_id,
        image_id=condition.image_id,
        object_id=condition.object_id,
        class_id=condition.class_id,
        class_name=condition.class_name,
        box_height=condition.box_height,
        region_role=condition.region_role,
        intervention_kind=condition.intervention_kind,
        intervention_severity=condition.intervention_severity,
        pair_id=condition.pair_id,
        natural_sampling=condition.natural_sampling,
        natural_visibility=condition.natural_visibility,
        predicted_sampling=sampling,
        predicted_visibility=visibility,
        branch_weights=branches,
        intervention_factor=condition.intervention_factor,
    )
    normalized = dict(value)
    normalized.update(
        {
            "feature_roi_xyxy": list(roi_int),
            "feature_shape": list(feature_shape),
            "predicted_sampling": sampling,
            "predicted_visibility": visibility,
            "branch_weights": list(branches),
            "gate_strength": gate,
        }
    )
    return normalized


def validate_observation_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    manifest: FactorObservationManifest,
    plan: ImageObservationPlan,
    checkpoint_sha256: str,
    manifest_sha256: str | None = None,
) -> tuple[dict[str, object], ...]:
    """Validate all rows for one image and require exact condition×node coverage."""

    if manifest_sha256 is None:
        manifest_sha256 = manifest.hash()
    materialized = tuple(rows)
    expected = _condition_observation_ids(plan, DEFAULT_REQUIRED_NODES)
    if len(materialized) != len(expected):
        raise ValueError("observation row count does not match manifest")
    condition_by_id = {condition.condition_id: condition for condition in plan.conditions}
    node_set = set(DEFAULT_REQUIRED_NODES)
    seen: set[str] = set()
    validated: list[dict[str, object]] = []
    for row in materialized:
        value = _require_row_mapping(row)
        condition_id = value.get("condition_id")
        condition = condition_by_id.get(condition_id) if isinstance(condition_id, str) else None
        node_id = value.get("node_id")
        if condition is None or isinstance(node_id, bool) or not isinstance(node_id, int) or node_id not in node_set:
            raise ValueError("observation condition-by-node does not match manifest")
        observation_id = expected[(condition_id, node_id)]
        if observation_id in seen:
            raise ValueError("duplicate observation_id in image rows")
        seen.add(observation_id)
        validated.append(
            validate_observation_row(
                value,
                manifest=manifest,
                plan=plan,
                condition=condition,
                node_id=node_id,
                observation_id=observation_id,
                checkpoint_sha256=checkpoint_sha256,
                manifest_sha256=manifest_sha256,
            )
        )
    if seen != set(expected.values()):
        raise ValueError("observation rows do not match manifest identities")
    return tuple(validated)


def _read_existing_rows(
    journal: FactorObservationJournal,
    manifest: FactorObservationManifest,
    checkpoint_sha256: str,
    manifest_sha256: str,
) -> None:
    plan_by_image = {plan.image_id: plan for plan in manifest.plans}
    completed = journal.completed_image_ids
    seen_images: set[str] = set()
    current_image_id: str | None = None
    current_rows: list[dict[str, object]] = []

    def validate_block(image_id: str | None, rows: list[dict[str, object]]) -> None:
        if image_id is None:
            return
        if image_id in seen_images:
            raise ValueError("observation rows for one image are not contiguous")
        plan = plan_by_image.get(image_id)
        if plan is None:
            raise ValueError("observation JSONL contains an unknown image")
        if image_id not in completed:
            raise ValueError("observation rows exist for an uncommitted image")
        if not rows:
            raise ValueError("completed image has no observation rows")
        validate_observation_rows(
            rows,
            manifest=manifest,
            plan=plan,
            checkpoint_sha256=checkpoint_sha256,
            manifest_sha256=manifest_sha256,
        )
        seen_images.add(image_id)

    try:
        handle = journal.output_path.open("rb")
    except OSError as exc:
        raise ValueError("unable to read observation JSONL") from exc
    with handle:
        for raw_line in handle:
            if not raw_line.endswith(b"\n"):
                raise ValueError("observation JSONL contains an unterminated line")
            try:
                row = json.loads(
                    raw_line.decode("utf-8"),
                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
                )
            except (UnicodeDecodeError, TypeError, ValueError) as exc:
                raise ValueError("observation JSONL contains malformed JSON") from exc
            value = _require_row_mapping(row)
            image_id = value.get("image_id")
            if not isinstance(image_id, str):
                raise ValueError("observation row image_id is invalid")
            if image_id != current_image_id:
                validate_block(current_image_id, current_rows)
                current_image_id = image_id
                current_rows = []
            current_rows.append(dict(value))
    validate_block(current_image_id, current_rows)
    missing = completed - seen_images
    if missing:
        raise ValueError("completed image has no observation rows")


def _intervention_spec(
    condition: ObservationCondition,
    *,
    width: int,
    height: int,
) -> InterventionSpec:
    role = (
        InterventionRole.OBJECT
        if condition.region_role == "target"
        else InterventionRole.BACKGROUND
    )
    kind = (
        InterventionKind.SAMPLING
        if condition.intervention_kind == "sampling"
        else InterventionKind.VISIBILITY
    )
    return InterventionSpec(
        image_id=condition.image_id,
        kind=kind,
        role=role,
        strength=condition.intervention_severity,
        seed=_transform_seed_for_condition(condition),
        object_id=condition.object_id if role is InterventionRole.OBJECT else None,
        region_xyxy=_normalized_bbox(
            condition.bbox_xyxy,
            width=width,
            height=height,
        ),
    )


def _observation_row(
    *,
    manifest: FactorObservationManifest,
    manifest_sha256: str,
    plan: ImageObservationPlan,
    condition: ObservationCondition,
    node: PooledReliability,
    observation_id: str,
    checkpoint_sha256: str,
) -> dict[str, object]:
    region = _normalized_bbox(condition.bbox_xyxy, width=plan.width, height=plan.height)
    matched = condition.matched_background_bbox
    row: dict[str, object] = {
        "schema_version": 1,
        "manifest_sha256": manifest_sha256,
        "observation_id": observation_id,
        "condition_id": condition.condition_id,
        "transform_id": condition.transform_id,
        "checkpoint_sha256": checkpoint_sha256,
        "source_sha256": condition.source_sha256,
        "seed": condition.seed,
        "transform_seed": _transform_seed_for_condition(condition),
        "node_id": node.node,
        "image_id": condition.image_id,
        "object_id": condition.object_id,
        "class_id": condition.class_id,
        "class_name": condition.class_name,
        "bbox_xyxy": list(condition.bbox_xyxy),
        "box_height": condition.box_height,
        "natural_sampling": condition.natural_sampling,
        "natural_visibility": condition.natural_visibility,
        "region_xyxy": list(region),
        "region_role": condition.region_role,
        "intervention_kind": condition.intervention_kind,
        "intervention_factor": condition.intervention_factor,
        "intervention_severity": condition.intervention_severity,
        "pair_id": condition.pair_id,
        "matched_background_bbox": None if matched is None else list(matched),
        "predicted_sampling": node.sampling,
        "predicted_visibility": node.visibility,
        "branch_weights": list(node.branch_weights),
        "gate_strength": node.gate_strength,
        "feature_roi_xyxy": list(node.roi_xyxy),
        "feature_shape": list(node.feature_shape),
        "input_shape": [3, manifest.input_size, manifest.input_size],
    }
    return row


def run_factor_observer(
    loaded: LoadedIFDRCheckpoint,
    manifest: FactorObservationManifest,
    journal: FactorObservationJournal,
    *,
    transform_batch_size: int = 8,
) -> dict[str, object]:
    """Execute deterministic transform-level IFDR observations exactly once."""

    if not isinstance(loaded, LoadedIFDRCheckpoint):
        raise ValueError("loaded must be a LoadedIFDRCheckpoint")
    if not isinstance(manifest, FactorObservationManifest):
        raise ValueError("manifest must be a FactorObservationManifest")
    if not isinstance(journal, FactorObservationJournal):
        raise ValueError("journal must be a FactorObservationJournal")
    manifest_sha256 = manifest.hash()
    if loaded.checkpoint_sha256 != manifest.checkpoint_sha256:
        raise ValueError("checkpoint hash does not match manifest")
    if journal.manifest != manifest:
        raise ValueError("journal manifest does not match manifest")
    if journal.manifest.checkpoint_sha256 != loaded.checkpoint_sha256:
        raise ValueError("journal checkpoint hash does not match checkpoint")
    if isinstance(transform_batch_size, bool) or type(transform_batch_size) is not int or transform_batch_size <= 0:
        raise ValueError("transform_batch_size must be a positive integer")

    _read_existing_rows(
        journal,
        manifest,
        loaded.checkpoint_sha256,
        manifest_sha256,
    )
    model = loaded.model
    device = _model_device(model)
    model.eval()
    nodes = manifest.required_nodes
    for plan in manifest.plans:
        image, width, height, source_sha256 = _read_png_once(plan.image_path)
        if width != plan.width or height != plan.height:
            raise ValueError(f"image dimensions do not match manifest for {plan.image_id}")
        if source_sha256 != plan.source_sha256:
            raise ValueError(f"image source hash does not match manifest for {plan.image_id}")
        if journal.is_completed(plan.image_id):
            continue
        grouped: dict[str, list[ObservationCondition]] = defaultdict(list)
        for condition in plan.conditions:
            grouped[condition.transform_id].append(condition)
        transform_ids = sorted(grouped)
        rows: list[dict[str, object]] = []
        expected_ids = _condition_observation_ids(plan, nodes)
        for start in range(0, len(transform_ids), transform_batch_size):
            chunk_ids = transform_ids[start : start + transform_batch_size]
            chunk: list[tuple[LetterboxGeometry, tuple[ObservationCondition, ...]]] = []
            tensors: list[torch.Tensor] = []
            for transform_id in chunk_ids:
                conditions = tuple(grouped[transform_id])
                if any(condition.image_id != plan.image_id or condition.source_sha256 != plan.source_sha256 for condition in conditions):
                    raise ValueError("transform group metadata does not match image plan")
                representative = conditions[0]
                transformed_image = image
                if representative.intervention_kind in {"sampling", "visibility"}:
                    spec = _intervention_spec(representative, width=width, height=height)
                    target = factor_target_for_spec(
                        spec,
                        natural_sampling=representative.natural_sampling,
                        natural_occlusion=representative.natural_visibility,
                    )
                    transformed_image = apply_intervention(image, spec, target).image
                elif representative.intervention_kind not in {"natural", "clean"}:
                    raise ValueError("unsupported condition intervention kind")
                tensor, geometry = letterbox_image(transformed_image, manifest.input_size)
                tensors.append(tensor)
                chunk.append((geometry, conditions))
            batch = torch.stack(tensors, dim=0).to(device=device, dtype=torch.float32)
            del tensors
            with torch.inference_mode():
                model(batch)
            contexts = model.consume_reliability_context()
            prepared_contexts, context_batch_size = _prepare_reliability_contexts(
                contexts,
                required_nodes=nodes,
            )
            for batch_index, (geometry, conditions) in enumerate(chunk):
                for condition in conditions:
                    pooled = _pool_prepared_reliability_contexts(
                        prepared_contexts,
                        context_batch_size,
                        batch_index=batch_index,
                        bbox_xyxy=condition.bbox_xyxy,
                        geometry=geometry,
                        required_nodes=nodes,
                    )
                    for node in pooled:
                        rows.append(
                            _observation_row(
                                manifest=manifest,
                                manifest_sha256=manifest_sha256,
                                plan=plan,
                                condition=condition,
                                node=node,
                                observation_id=expected_ids[(condition.condition_id, node.node)],
                                checkpoint_sha256=loaded.checkpoint_sha256,
                            )
                        )
            del contexts, prepared_contexts, context_batch_size, chunk
            del batch
        validated = validate_observation_rows(
            rows,
            manifest=manifest,
            plan=plan,
            checkpoint_sha256=loaded.checkpoint_sha256,
            manifest_sha256=manifest_sha256,
        )
        if len(validated) != len(plan.expected_observation_ids):
            raise ValueError("generated observation rows do not match manifest")
        journal.commit_image(plan.image_id, validated)
    return journal.finalize()


__all__ = [
    "LoadedIFDRCheckpoint",
    "PooledReliability",
    "load_ifdr_checkpoint",
    "pool_reliability_contexts",
    "run_factor_observer",
    "validate_observation_row",
    "validate_observation_rows",
]
