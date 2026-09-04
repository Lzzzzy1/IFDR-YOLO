"""Audit-only replay of the registered P2--P5 TaskAlignedAssigner.

The observer calls the original assigner first and returns its object untouched.
When enabled, it replays the public assigner helpers on detached clones and
records only diagnostic values.  The small runner at the bottom of this module
is deliberately synthetic: it exercises the journal/identity contract without
silently becoming a training entry point.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import torch

from ifdr_yolo.data.splits import sha256_file
from ifdr_yolo.experiments.config import load_baseline_config
from ifdr_yolo.experiments.p2_fit_reference import (
    validate_fit_development_split,
    validate_plain_p2_model,
    validate_primary_checkpoint,
)


LEVEL_NAMES = ("P2", "P3", "P4", "P5")
SUPPORTED_ULTRALYTICS_VERSION = "8.4.98"
# These are independent Moderate strata.  A GT may belong to more than one
# (for example, a small object can also be far); never collapse them into a
# mutually-exclusive ``stratum`` label.
MODERATE_STRATA = ("small_25_40", "large_gt_80", "far_gt_40m", "near_0_20m")
PRE_REGISTERED_CONTRASTS = (
    {
        "name": "small_25_40-vs-large_gt_80",
        "target": "small_25_40",
        "control": "large_gt_80",
        "estimand": "target_zero_p2_positive_rate - control_zero_p2_positive_rate",
        "target_worse_positive": True,
    },
    {
        "name": "far_gt_40m-vs-near_0_20m",
        "target": "far_gt_40m",
        "control": "near_0_20m",
        "estimand": "target_zero_p2_positive_rate - control_zero_p2_positive_rate",
        "target_worse_positive": True,
    },
)
IDENTITY_FIELDS = (
    "fit_ids_sha256",
    "development_ids_sha256",
    "checkpoint_sha256",
    "config_sha256",
    "code_sha256",
)


@dataclass(frozen=True)
class LevelSlice:
    """A half-open slice in the flattened anchor order."""

    name: str
    start: int
    stop: int
    height: int
    width: int

    @property
    def count(self) -> int:
        return self.stop - self.start


def _feature_hw(feature: Any) -> tuple[int, int]:
    shape = getattr(feature, "shape", feature)
    try:
        height, width = int(shape[-2]), int(shape[-1])
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError("feature level must expose H and W as its last two dimensions") from error
    if height <= 0 or width <= 0:
        raise ValueError("feature level dimensions must be positive")
    return height, width


def derive_level_slices(
    feature_maps: Sequence[Any], names: Sequence[str] = LEVEL_NAMES, *, strides: Sequence[int] | None = None
) -> tuple[LevelSlice, ...]:
    """Derive level boundaries from feature-map H*W, never from hard-coded counts."""

    if len(feature_maps) != len(names) or not feature_maps:
        raise ValueError("feature-map and level-name counts must match and be non-zero")
    if strides is not None:
        validate_level_strides(strides, expected_count=len(names))
    cursor = 0
    result: list[LevelSlice] = []
    seen: set[str] = set()
    for name, feature in zip(names, feature_maps):
        name = str(name)
        if not name or name in seen:
            raise ValueError("level names must be unique and non-empty")
        seen.add(name)
        height, width = _feature_hw(feature)
        count = height * width
        result.append(LevelSlice(name, cursor, cursor + count, height, width))
        cursor += count
    return tuple(result)


def validate_level_strides(
    strides: Sequence[Any], *, expected: Sequence[int] = (4, 8, 16, 32), expected_count: int | None = None
) -> tuple[int, ...]:
    """Bind audit level order to the plain-P2 Detect stride contract."""

    observed = tuple(int(value.item()) if isinstance(value, torch.Tensor) else int(value) for value in strides)
    wanted = tuple(int(value) for value in expected)
    if expected_count is not None and len(observed) != expected_count:
        raise ValueError("feature stride count does not match feature levels")
    if observed != wanted:
        raise ValueError(f"feature strides must be {wanted}, got {observed}")
    return observed


def _sha256_state(value: Any) -> str:
    """Hash model parameters/buffers without serializing optimizer/runtime state."""

    digest = hashlib.sha256()
    state = value.state_dict() if hasattr(value, "state_dict") else {}
    for name in sorted(state):
        digest.update(str(name).encode("utf-8"))
        item = state[name]
        if isinstance(item, torch.Tensor):
            digest.update(str(item.dtype).encode("ascii"))
            digest.update(str(tuple(item.shape)).encode("ascii"))
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(repr(item).encode("utf-8"))
    return digest.hexdigest()


def _rng_state() -> tuple[bytes, ...]:
    states = [torch.get_rng_state().cpu().numpy().tobytes()]
    if torch.cuda.is_available():
        states.extend(state.cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all())
    return tuple(states)


def _torch_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(_torch_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_torch_equal(left[key], right[key]) for key in left)
    return left == right


def _prediction_mapping(preds: Any) -> Mapping[str, Any]:
    if isinstance(preds, Mapping):
        return preds
    if isinstance(preds, tuple) and len(preds) > 1 and isinstance(preds[1], Mapping):
        return preds[1]
    raise ValueError("model output must expose a prediction mapping with feats")


def _detect_strides(model: Any) -> tuple[int, ...]:
    candidates: list[Any] = []
    modules = model.modules() if hasattr(model, "modules") else ()
    for module in modules:
        if module.__class__.__name__ == "Detect" and hasattr(module, "stride"):
            candidates = list(module.stride)
            break
    if not candidates and hasattr(model, "strides"):
        candidates = list(model.strides)
    if not candidates and hasattr(model, "stride"):
        candidates = list(model.stride)
    if not candidates:
        raise ValueError("could not find Detect feature strides")
    return validate_level_strides(candidates)


def _normalize_device(value: object) -> str:
    raw = str(value).strip()
    if not raw:
        return "cpu"
    if raw.isdigit():
        return f"cuda:{raw}"
    return raw


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _detached(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("assigner replay inputs must be torch tensors")
    return value.detach().clone()


def _extract_inputs(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> tuple[torch.Tensor, ...]:
    names = ("pd_scores", "pd_bboxes", "anc_points", "gt_labels", "gt_bboxes", "mask_gt")
    if args and kwargs:
        raise TypeError("assigner audit accepts either positional or keyword inputs, not both")
    if args:
        if len(args) != len(names):
            raise TypeError("assigner audit expects six assignment tensors")
        values = args
    else:
        try:
            values = tuple(kwargs[name] for name in names)
        except KeyError as error:
            raise TypeError(f"missing assignment input: {error.args[0]}") from error
    return tuple(_detached(value) for value in values)  # type: ignore[return-value]


def _ordinary_iou(gt_boxes: torch.Tensor, pd_boxes: torch.Tensor) -> torch.Tensor:
    """Compute ordinary IoU; this is intentionally distinct from TAL CIoU."""

    left_top = torch.maximum(gt_boxes[..., :2], pd_boxes[..., :2])
    right_bottom = torch.minimum(gt_boxes[..., 2:], pd_boxes[..., 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(-1)
    gt_area = (gt_boxes[..., 2:] - gt_boxes[..., :2]).clamp_min(0).prod(-1)
    pd_area = (pd_boxes[..., 2:] - pd_boxes[..., :2]).clamp_min(0).prod(-1)
    return intersection / (gt_area + pd_area - intersection).clamp_min(1e-12)


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask.bool()]
    return 0.0 if selected.numel() == 0 else float(selected.max().item())


def _metadata_value(values: Any, batch_index: int, gt_index: int, default: str) -> str:
    if values is None:
        return default
    if isinstance(values, Mapping):
        raw = values.get((batch_index, gt_index), values.get(str(batch_index), default))
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            return str(raw[gt_index]) if gt_index < len(raw) else default
        return str(raw)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if batch_index < len(values):
            raw = values[batch_index]
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                return str(raw[gt_index]) if gt_index < len(raw) else default
            return str(raw)
    return default


def _metadata_mapping(values: Any, batch_index: int, gt_index: int) -> Mapping[str, object]:
    if isinstance(values, Mapping):
        raw = values.get((batch_index, gt_index), values.get(str(batch_index)))
        if isinstance(raw, Mapping):
            return raw
        return {}
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and batch_index < len(values):
        raw = values[batch_index]
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and gt_index < len(raw):
            item = raw[gt_index]
            return item if isinstance(item, Mapping) else {}
    return {}


def replay_assignment(
    assigner: Any,
    inputs: Sequence[torch.Tensor],
    original_output: Sequence[torch.Tensor],
    level_slices: Sequence[LevelSlice],
    *,
    image_ids: Sequence[str] | None = None,
    gt_strata: Any = None,
    gt_metadata: Any = None,
    class_names: Mapping[int, str] | None = None,
) -> list[dict[str, object]]:
    """Replay public assigner helpers on clones and return per-GT/per-level rows."""

    if len(inputs) != 6 or len(original_output) < 5:
        raise ValueError("assignment replay requires six inputs and five assigner outputs")
    pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt = inputs
    batch_size, anchor_count = int(pd_scores.shape[0]), int(pd_scores.shape[1])
    gt_count = int(gt_bboxes.shape[1])
    if int(pd_bboxes.shape[0]) != batch_size or int(pd_bboxes.shape[1]) != anchor_count:
        raise ValueError("prediction tensor shapes disagree")
    if sum(item.count for item in level_slices) != anchor_count:
        raise ValueError("dynamic level slices do not cover flattened anchors")
    if tuple(int(item) for item in anc_points.shape[:1]) != (anchor_count,):
        raise ValueError("anchor point count does not match flattened predictions")
    original_fg = original_output[3]
    original_gt_idx = original_output[4]
    if original_fg.shape[:2] != (batch_size, anchor_count) or original_gt_idx.shape[:2] != (batch_size, anchor_count):
        raise ValueError("assigner output foreground/index shapes do not match inputs")
    if gt_count == 0:
        if bool(original_fg.bool().any().item()):
            raise ValueError("assigner returned foreground positives for an empty GT batch")
        return []

    # The original call has already established these fields on Ultralytics'
    # assigner.  Preserve and restore them around helper replay so diagnostics
    # cannot leak mutable state into subsequent batches.
    previous_state = {
        key: getattr(assigner, key)
        for key in ("bs", "n_max_boxes")
        if hasattr(assigner, key)
    }
    missing_state = {key for key in ("bs", "n_max_boxes") if key not in previous_state}
    try:
        assigner.bs = batch_size
        assigner.n_max_boxes = gt_count
        replay_inputs = tuple(_detached(value) for value in inputs)
        replay_scores, replay_boxes, replay_anchors, replay_labels, replay_gt, replay_mask = replay_inputs
        legal = assigner.select_candidates_in_gts(replay_anchors, replay_gt, replay_mask)
        before, align_metric, tal_ciou = assigner.get_pos_mask(
            replay_scores, replay_boxes, replay_labels, replay_gt, replay_anchors, replay_mask
        )
        replay_gt_idx, replay_fg, after = assigner.select_highest_overlaps(
            before.detach().clone(), tal_ciou, gt_count, align_metric
        )
    finally:
        for key, value in previous_state.items():
            setattr(assigner, key, value)
        for key in missing_state:
            if hasattr(assigner, key):
                delattr(assigner, key)

    if not torch.equal(replay_fg.bool(), original_fg.bool()) or not torch.equal(replay_gt_idx, original_gt_idx):
        raise ValueError("detached assignment replay differs from original output")

    rows: list[dict[str, object]] = []
    names = class_names or {}
    for batch_index in range(batch_size):
        image_id = image_ids[batch_index] if image_ids is not None and batch_index < len(image_ids) else str(batch_index)
        for gt_index in range(gt_count):
            if not bool(replay_mask[batch_index, gt_index].item()):
                continue
            class_id = int(replay_labels[batch_index, gt_index, 0].item())
            class_name = str(names.get(class_id, class_id))
            height = float((replay_gt[batch_index, gt_index, 3] - replay_gt[batch_index, gt_index, 1]).item())
            stratum = _metadata_value(gt_strata, batch_index, gt_index, "unknown")
            metadata = _metadata_mapping(gt_metadata, batch_index, gt_index)
            all_alignment = align_metric[batch_index, gt_index]
            for level in level_slices:
                view = slice(level.start, level.stop)
                level_legal = legal[batch_index, gt_index, view].bool()
                level_alignment = align_metric[batch_index, gt_index, view]
                best_value = _masked_max(level_alignment, level_legal)
                rank: int | None = None
                rank_low: int | None = None
                rank_high: int | None = None
                tie_count = 0
                if bool(level_legal.any().item()):
                    rank_low = int((all_alignment > best_value).sum().item()) + 1
                    tie_count = int((all_alignment == best_value).sum().item())
                    rank_high = rank_low + tie_count - 1
                    if best_value > 0.0:
                        rank = rank_low
                legal_boxes = replay_boxes[batch_index, view, :][level_legal]
                ordinary_level = (
                    _ordinary_iou(replay_gt[batch_index, gt_index].expand(legal_boxes.shape[0], -1), legal_boxes)
                    if legal_boxes.numel()
                    else torch.empty((0,), dtype=replay_boxes.dtype, device=replay_boxes.device)
                )
                rows.append(
                    {
                        "image_id": str(image_id),
                        "gt_index": gt_index,
                        "class_id": class_id,
                        "class_name": class_name,
                        "stratum": stratum,
                        "gt_height_px": metadata.get("height_px", height),
                        "gt_depth_m": metadata.get("depth_m"),
                        "moderate_valid": metadata.get("moderate_valid"),
                        "small_25_40": metadata.get("small_25_40"),
                        "large_gt_80": metadata.get("large_gt_80"),
                        "far_gt_40m": metadata.get("far_gt_40m"),
                        "near_0_20m": metadata.get("near_0_20m"),
                        "level": level.name,
                        "legal_anchor_count": int(level_legal.sum().item()),
                        "selected_before_collision": int(before[batch_index, gt_index, view].sum().item()),
                        "selected_after_collision": int(after[batch_index, gt_index, view].sum().item()),
                        "assigned_positive_count": int(
                            ((original_gt_idx[batch_index, view] == gt_index) & original_fg[batch_index, view].bool()).sum().item()
                        ),
                        "max_iou": 0.0 if ordinary_level.numel() == 0 else float(ordinary_level.max().item()),
                        "max_tal_ciou": _masked_max(tal_ciou[batch_index, gt_index, view], level_legal),
                        "max_task_alignment": best_value,
                        "best_alignment_rank": rank,
                        "rank_low": rank_low,
                        "rank_high": rank_high,
                        "tie_count": tie_count,
                    }
                )
    return rows


class AssignmentAuditObserver:
    """Forward-hook observer that never replaces the registered assigner."""

    def __init__(
        self,
        assigner: Any,
        *,
        enabled: bool = False,
        strict: bool = False,
        level_slices: Sequence[LevelSlice] | None = None,
        class_names: Mapping[int, str] | None = None,
    ) -> None:
        self.assigner = assigner
        self.enabled = bool(enabled)
        self.strict = bool(strict)
        self.level_slices = tuple(level_slices or ())
        self.class_names = dict(class_names or {})
        self.records: list[dict[str, object]] = []
        self._hook_handle: Any = None
        self._image_ids: Sequence[str] | None = None
        self._gt_strata: Any = None
        self._gt_metadata: Any = None
        self.last_output: Any = None

    def set_metadata(
        self,
        *,
        image_ids: Sequence[str] | None = None,
        gt_strata: Any = None,
        gt_metadata: Any = None,
    ) -> None:
        """Set metadata for the next direct assigner call without changing its signature."""

        self._image_ids = image_ids
        self._gt_strata = gt_strata
        self._gt_metadata = gt_metadata

    def attach(self) -> "AssignmentAuditObserver":
        if self._hook_handle is not None:
            return self
        if not hasattr(self.assigner, "register_forward_hook"):
            raise TypeError("assignment observer requires a torch module assigner")
        try:
            self._hook_handle = self.assigner.register_forward_hook(self._forward_hook_with_kwargs, with_kwargs=True)
        except TypeError:
            self._hook_handle = self.assigner.register_forward_hook(self._forward_hook)
        return self

    def detach(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def __enter__(self) -> "AssignmentAuditObserver":
        return self.attach()

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.detach()

    def _observe(self, args: tuple[Any, ...], kwargs: Mapping[str, Any], output: Any) -> None:
        self.last_output = output
        if not self.enabled:
            return
        if not self.level_slices:
            raise ValueError("enabled assignment audit requires dynamic level slices")
        inputs = _extract_inputs(args, kwargs)
        if self.strict:
            image_ids = self._image_ids
            if image_ids is None or isinstance(image_ids, (str, bytes)) or len(image_ids) != int(inputs[0].shape[0]):
                raise ValueError("strict assignment audit requires image_ids for every batch item")
            if any(not str(value).strip() for value in image_ids):
                raise ValueError("strict assignment audit requires non-empty image_ids")
            mask_gt = inputs[5].bool()
            labels = inputs[3]
            for batch_index in range(int(mask_gt.shape[0])):
                for gt_index in range(int(mask_gt.shape[1])):
                    if not bool(mask_gt[batch_index, gt_index].item()):
                        continue
                    if self._gt_strata is None or _metadata_value(
                        self._gt_strata, batch_index, gt_index, "__MISSING__"
                    ) == "__MISSING__":
                        raise ValueError("strict assignment audit requires GT strata metadata")
                    if self._gt_metadata is None:
                        raise ValueError("strict assignment audit requires complete GT metadata")
                    metadata = _metadata_mapping(self._gt_metadata, batch_index, gt_index)
                    required = {
                        "moderate_valid",
                        "height_px",
                        "depth_m",
                        "small_25_40",
                        "large_gt_80",
                        "far_gt_40m",
                        "near_0_20m",
                    }
                    if not required.issubset(metadata) or any(metadata.get(key) is None for key in required):
                        raise ValueError("strict assignment audit requires complete GT metadata")
                    class_id = int(labels[batch_index, gt_index, 0].item())
                    if class_id not in self.class_names:
                        raise ValueError("strict assignment audit requires class metadata")
        self.records.extend(
            replay_assignment(
                self.assigner,
                inputs,
                output,
                self.level_slices,
                image_ids=self._image_ids,
                gt_strata=self._gt_strata,
                gt_metadata=self._gt_metadata,
                class_names=self.class_names,
            )
        )

    def _forward_hook_with_kwargs(
        self, _module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any], output: Any
    ) -> None:
        self._observe(args, kwargs, output)
        return None

    def _forward_hook(self, _module: Any, args: tuple[Any, ...], output: Any) -> None:
        self._observe(args, {}, output)
        return None


def summarize_zero_p2_positive(records: Sequence[Mapping[str, object]]) -> dict[str, dict[str, dict[str, int | float]]]:
    """Summarize independent Moderate strata using legal P2 GT denominators."""

    # Deduplicate the four level-independent flags at each image/GT.  A GT is
    # intentionally counted in every flag it satisfies; no OR/AND composite
    # stratum is part of the estimand.
    groups: dict[tuple[str, str, str], dict[str, int]] = {}
    for record in records:
        if record.get("level") != "P2":
            continue
        class_name = str(record.get("class_name", "unknown"))
        gt_key = f"{record.get('image_id', '')}:{record.get('gt_index', '')}"
        legal = int(record.get("legal_anchor_count", 0))
        positive = int(record.get("assigned_positive_count", 0))
        for stratum in MODERATE_STRATA:
            if not bool(record.get(stratum, False)):
                continue
            key = (class_name, stratum, gt_key)
            item = groups.setdefault(key, {"legal_anchor_count": 0, "assigned_positive_count": 0})
            item["legal_anchor_count"] = max(item["legal_anchor_count"], legal)
            item["assigned_positive_count"] = max(item["assigned_positive_count"], positive)
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"denominator": 0, "zero_positive": 0})
    class_names = {"Car", "Pedestrian", "Cyclist"} | {key[0] for key in groups}
    for class_name in class_names:
        for stratum in MODERATE_STRATA:
            counts[(class_name, stratum)]
    for (class_name, stratum, _gt_key), item in groups.items():
        if item["legal_anchor_count"] <= 0:
            continue
        counts[(class_name, stratum)]["denominator"] += 1
        if item["assigned_positive_count"] == 0:
            counts[(class_name, stratum)]["zero_positive"] += 1
    result: dict[str, dict[str, dict[str, int | float]]] = defaultdict(dict)
    for (class_name, stratum), value in sorted(counts.items()):
        denominator = value["denominator"]
        result[class_name][stratum] = {
            **value,
            "rate": value["zero_positive"] / denominator if denominator else 0.0,
        }
    return dict(result)


def build_cyclist_image_cluster_frame(
    records: Sequence[Mapping[str, object]],
    selected_fit_ids: Sequence[str],
    *,
    levels: Sequence[str] = LEVEL_NAMES,
) -> list[dict[str, object]]:
    """Build a complete image-cluster frame for Cyclist zero-positive rates.

    Every selected image is retained, including images without an eligible
    Cyclist GT.  Numerators/denominators are counted once per image/GT/level;
    the four Moderate flags remain independent, so one GT can contribute to
    more than one stratum.
    """

    image_ids = tuple(str(value) for value in selected_fit_ids)
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("sampling frame image IDs must be unique")
    level_names = tuple(str(value) for value in levels)
    if not level_names or len(level_names) != len(set(level_names)):
        raise ValueError("sampling frame levels must be unique and non-empty")
    image_set = set(image_ids)
    frame: dict[str, dict[str, object]] = {}
    for image_id in image_ids:
        frame[image_id] = {
            "image_id": image_id,
            "strata": {
                stratum: {
                    level: {"numerator": 0, "denominator": 0}
                    for level in level_names
                }
                for stratum in MODERATE_STRATA
            },
        }
    observations: dict[tuple[str, int, str, str], tuple[int, int]] = {}
    for row in records:
        image_id = str(row.get("image_id", ""))
        if image_id not in image_set:
            raise ValueError(f"journal image ID is outside the sampling frame: {image_id}")
        level = str(row.get("level", ""))
        if level not in level_names:
            raise ValueError(f"journal level is outside the configured levels: {level}")
        if str(row.get("class_name", "")) != "Cyclist":
            continue
        if row.get("moderate_valid") is False:
            continue
        gt_index = int(row.get("gt_index", -1))
        legal = int(row.get("legal_anchor_count", 0))
        zero_positive = int(row.get("assigned_positive_count", 0)) == 0
        for stratum in MODERATE_STRATA:
            if not bool(row.get(stratum, False)):
                continue
            key = (image_id, gt_index, level, stratum)
            observed = (1 if legal > 0 else 0, 1 if legal > 0 and zero_positive else 0)
            previous = observations.get(key)
            if previous is not None and previous != observed:
                raise ValueError("duplicate journal GT/level rows disagree")
            observations[key] = observed
    for (image_id, _gt_index, level, stratum), (denominator, numerator) in observations.items():
        cell = frame[image_id]["strata"][stratum][level]  # type: ignore[index]
        cell["denominator"] += denominator
        cell["numerator"] += numerator
    return [frame[image_id] for image_id in image_ids]


def _validate_input_hash(name: str, value: object) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA256")
    return normalized


def bootstrap_cyclist_zero_p2_contrasts(
    frame: Sequence[Mapping[str, object]],
    *,
    reps: int = 10_000,
    seed: int = 20260812,
    levels: Sequence[str] = LEVEL_NAMES,
    full_dataset: bool = True,
    journal_sha256: str,
    identity_sha256: str,
    manifest_sha256: str,
    chunk_size: int = 64,
) -> dict[str, object]:
    """Compute the preregistered image-cluster contrasts and gate.

    Each bootstrap draw samples the complete image frame with replacement and
    sums numerators/denominators before taking a ratio.  Both contrasts and all
    levels reuse exactly the same sampled image indices in each chunk.
    """

    import numpy as np

    level_names = tuple(str(value) for value in levels)
    rows = list(frame)
    if not rows:
        raise ValueError("sampling frame must contain at least one image")
    if len({str(row.get("image_id", "")) for row in rows}) != len(rows):
        raise ValueError("sampling frame image IDs must be unique")
    reps = int(reps)
    if reps <= 0:
        raise ValueError("bootstrap replicate count must be positive")
    chunk_size = max(1, int(chunk_size))
    input_hashes = {
        "journal_sha256": _validate_input_hash("journal_sha256", journal_sha256),
        "identity_sha256": _validate_input_hash("identity_sha256", identity_sha256),
        "manifest_sha256": _validate_input_hash("manifest_sha256", manifest_sha256),
    }
    array = np.zeros((len(rows), len(MODERATE_STRATA), len(level_names), 2), dtype=np.int64)
    stratum_index = {name: index for index, name in enumerate(MODERATE_STRATA)}
    level_index = {name: index for index, name in enumerate(level_names)}
    for image_index, row in enumerate(rows):
        strata = row.get("strata")
        if not isinstance(strata, Mapping):
            raise ValueError("sampling frame row is missing strata")
        for stratum in MODERATE_STRATA:
            values = strata.get(stratum)
            if not isinstance(values, Mapping):
                raise ValueError(f"sampling frame row is missing {stratum}")
            for level in level_names:
                cell = values.get(level)
                if not isinstance(cell, Mapping):
                    raise ValueError(f"sampling frame row is missing {stratum}/{level}")
                denominator = int(cell.get("denominator", 0))
                numerator = int(cell.get("numerator", 0))
                if denominator < 0 or numerator < 0 or numerator > denominator:
                    raise ValueError("sampling frame numerator/denominator is invalid")
                array[image_index, stratum_index[stratum], level_index[level]] = (numerator, denominator)
    if not full_dataset:
        return {
            "schema_version": 1,
            "gate_state": "not_evaluated_smoke",
            "sampling_frame_image_count": len(rows),
            "bootstrap_replicates": reps,
            "bootstrap_seed": int(seed),
            "input_hashes": input_hashes,
        }
    contrast_specs = {str(item["name"]): item for item in PRE_REGISTERED_CONTRASTS}
    aggregate = array.sum(axis=0)
    for spec in PRE_REGISTERED_CONTRASTS:
        for level in level_names:
            target = aggregate[stratum_index[str(spec["target"])], level_index[level]]
            control = aggregate[stratum_index[str(spec["control"])], level_index[level]]
            if int(target[1]) <= 0 or int(control[1]) <= 0:
                raise ValueError("zero aggregate denominator")

    # Keep one effect vector per (level, contrast), while sharing draw indices
    # across all contrasts and levels within each chunk.
    effect_samples = {
        (str(spec["name"]), level): np.empty(reps, dtype=np.float64)
        for spec in PRE_REGISTERED_CONTRASTS
        for level in level_names
    }
    rng = np.random.default_rng(int(seed))
    cursor = 0
    while cursor < reps:
        count = min(chunk_size, reps - cursor)
        draw_indices = rng.integers(0, len(rows), size=(count, len(rows)), dtype=np.int64)
        drawn = array[draw_indices].sum(axis=1)
        for spec in PRE_REGISTERED_CONTRASTS:
            name = str(spec["name"])
            target_index = stratum_index[str(spec["target"])]
            control_index = stratum_index[str(spec["control"])]
            for level in level_names:
                position = level_index[level]
                target = drawn[:, target_index, position, :]
                control = drawn[:, control_index, position, :]
                if np.any(target[:, 1] <= 0) or np.any(control[:, 1] <= 0):
                    raise ValueError("zero bootstrap denominator")
                effects = target[:, 0] / target[:, 1] - control[:, 0] / control[:, 1]
                if not np.all(np.isfinite(effects)):
                    raise ValueError("non-finite bootstrap effect")
                effect_samples[(name, level)][cursor : cursor + count] = effects
        cursor += count

    def stats(spec: Mapping[str, object], level: str) -> dict[str, object]:
        name = str(spec["name"])
        position = level_index[level]
        target = aggregate[stratum_index[str(spec["target"])], position]
        control = aggregate[stratum_index[str(spec["control"])], position]
        observed = float(target[0] / target[1] - control[0] / control[1])
        samples = effect_samples[(name, level)]
        ci95 = np.percentile(samples, [2.5, 97.5]).astype(float).tolist()
        ci975 = np.percentile(samples, [1.25, 98.75]).astype(float).tolist()
        if not np.isfinite(observed) or not np.all(np.isfinite(ci95)) or not np.all(np.isfinite(ci975)):
            raise ValueError("non-finite bootstrap summary")
        gate_pass = bool(observed >= 0.10 and ci975[0] > 0.0)
        return {
            "target": str(spec["target"]),
            "control": str(spec["control"]),
            "observed": observed,
            "observed_percentage_points": observed * 100.0,
            "target_numerator": int(target[0]),
            "target_denominator": int(target[1]),
            "control_numerator": int(control[0]),
            "control_denominator": int(control[1]),
            "ci95": ci95,
            "ci97_5_bonferroni": ci975,
            "gate_pass": gate_pass,
        }

    levels_result: dict[str, dict[str, dict[str, object]]] = {
        level: {str(spec["name"]): stats(spec, level) for spec in PRE_REGISTERED_CONTRASTS}
        for level in level_names
    }
    p2_level = levels_result.get("P2")
    if p2_level is None:
        raise ValueError("cross-level bootstrap requires a P2 level")
    shared_all_levels = {
        name: bool(all(levels_result[level][name]["gate_pass"] for level in level_names))
        for name in contrast_specs
    }
    p2_gate = {name: bool(value["gate_pass"]) for name, value in p2_level.items()}
    assignment_go = any(p2_gate[name] and not shared_all_levels[name] for name in p2_gate)
    decision = "GO_A_ASSIGNMENT" if assignment_go else "NEXT_B_SCORE_AUDIT"
    return {
        "schema_version": 1,
        "gate_state": decision,
        "sampling_frame_image_count": len(rows),
        "bootstrap_replicates": reps,
        "bootstrap_seed": int(seed),
        "sampling_unit": "image_cluster",
        "ratio_of_sums": True,
        "same_sampled_image_indices_for_contrasts": True,
        "input_hashes": input_hashes,
        "contrasts": dict(p2_level),
        "cross_level_specificity": levels_result,
        "p2_gate": p2_gate,
        "shared_all_levels": shared_all_levels,
        "decision": decision,
    }


@dataclass(frozen=True)
class AuditIdentity:
    values: Mapping[str, str]

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "AuditIdentity":
        missing = [field for field in IDENTITY_FIELDS if not str(values.get(field, "")).strip()]
        if missing:
            raise ValueError(f"identity missing fields: {', '.join(missing)}")
        normalized = {str(key): str(value) for key, value in values.items()}
        invalid = [
            field
            for field in IDENTITY_FIELDS
            if len(normalized[field]) != 64 or any(char not in "0123456789abcdefABCDEF" for char in normalized[field])
        ]
        if invalid:
            raise ValueError(f"identity fields must be SHA256 values: {', '.join(invalid)}")
        return cls(normalized)

    def as_dict(self) -> dict[str, str]:
        return dict(sorted(self.values.items()))

    @property
    def sha256(self) -> str:
        return _payload_sha256(self.as_dict())


def validate_fit_development_ids(
    fit_ids: Sequence[str],
    development_ids: Sequence[str],
    *,
    expected_fit_count: int = 3341,
    expected_development_count: int = 371,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    fit = tuple(str(value).strip() for value in fit_ids if str(value).strip())
    development = tuple(str(value).strip() for value in development_ids if str(value).strip())
    if len(fit) != len(set(fit)) or len(development) != len(set(development)):
        raise ValueError("fit/development IDs must be unique")
    overlap = sorted(set(fit) & set(development))
    if overlap:
        raise ValueError(f"fit/development overlap: {overlap[:3]}")
    if len(fit) != expected_fit_count:
        raise ValueError(f"fit split count must be {expected_fit_count}")
    if len(development) != expected_development_count:
        raise ValueError(f"development split count must be {expected_development_count}")
    return fit, development


def _ids_sha256(ids: Sequence[str]) -> str:
    """Hash the canonical newline-delimited split representation."""

    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        try:
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _mirror_file(source: Path, mirror_root: Path) -> None:
    destination = mirror_root / source.name
    _atomic_write(destination, source.read_bytes())


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _journal_prefix(path: Path, completed: Sequence[str], identity_sha: str) -> bytes:
    """Return the checkpoint prefix and reject any altered completed record."""

    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    if len(lines) < len(completed):
        raise ValueError("audit journal is shorter than checkpoint")
    prefix = b"".join(lines[: len(completed)])
    observed: list[str] = []
    for index, line in enumerate(lines[: len(completed)]):
        if not line.endswith(b"\n"):
            raise ValueError("audit journal checkpoint prefix has an incomplete line")
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("audit journal checkpoint prefix is invalid") from error
        if record.get("identity_sha256") != identity_sha or record.get("image_id") != completed[index]:
            raise ValueError("audit journal checkpoint prefix mismatch")
        observed.append(str(record["image_id"]))
    if observed != list(completed):
        raise ValueError("audit journal checkpoint prefix mismatch")
    return prefix


def _prepare_recovery_artifacts(
    journal_path: Path,
    checkpoint_path: Path,
    mirror_root: Path,
    completed: Sequence[str],
    identity_sha: str,
) -> None:
    """Validate primary/mirror and trim a legal post-checkpoint tail."""

    mirror_journal = mirror_root / journal_path.name
    mirror_checkpoint = mirror_root / checkpoint_path.name
    if not journal_path.is_file() or not checkpoint_path.is_file() or not mirror_journal.is_file() or not mirror_checkpoint.is_file():
        raise ValueError("primary/mirror checkpoint and journal are required for resume")
    if checkpoint_path.read_bytes() != mirror_checkpoint.read_bytes():
        raise ValueError("primary/mirror checkpoint mismatch")
    primary_prefix = _journal_prefix(journal_path, completed, identity_sha)
    mirror_prefix = _journal_prefix(mirror_journal, completed, identity_sha)
    if primary_prefix != mirror_prefix:
        raise ValueError("primary/mirror journal prefix mismatch")
    if journal_path.read_bytes() != primary_prefix:
        _atomic_write(journal_path, primary_prefix)
    if mirror_journal.read_bytes() != primary_prefix:
        _atomic_write(mirror_journal, primary_prefix)


def _directory_sha256(path: Path) -> str:
    """Hash a file or a directory in canonical relative-path order."""

    path = Path(path).resolve()
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: item.relative_to(path).as_posix()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(sha256_file(child).encode("ascii"))
    return digest.hexdigest()


def _audit_code_sha256(repository_root: Path) -> str:
    """Hash every source file that can change this audit's semantics."""

    root = Path(repository_root).resolve()
    relative_files = (
        Path(__file__).relative_to(root),
        Path("scripts/run_p2_candidate_survival_audit.py"),
        Path("ifdr_yolo/experiments/config.py"),
        Path("ifdr_yolo/experiments/p2_fit_reference.py"),
        Path("ifdr_yolo/data/splits.py"),
        Path("ifdr_yolo/data/yolo_export.py"),
        Path("ifdr_yolo/data/kitti_types.py"),
        Path("ifdr_yolo/eval/kitti_ap40.py"),
        Path("ifdr_yolo/eval/prediction_io.py"),
    )
    digest = hashlib.sha256()
    for relative in relative_files:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _fit_image_manifest_sha256(resolved_data_path: Path, fit_ids: Sequence[str]) -> str:
    """Hash the content of every image addressed by the registered fit split."""

    try:
        import yaml
    except ImportError as error:  # pragma: no cover - formal runtime dependency
        raise RuntimeError("PyYAML is required to resolve the fit image manifest") from error
    payload = yaml.safe_load(Path(resolved_data_path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("resolved data YAML must be a mapping")
    root_value = payload.get("path", Path(resolved_data_path).parent)
    data_root = Path(str(root_value))
    if not data_root.is_absolute():
        data_root = Path(resolved_data_path).parent / data_root
    train_value = payload.get("train")
    if isinstance(train_value, Sequence) and not isinstance(train_value, (str, bytes)):
        train_value = train_value[0] if train_value else None
    if train_value is None:
        raise ValueError("resolved data YAML has no train image path")
    train_path = Path(str(train_value))
    if not train_path.is_absolute():
        train_path = data_root / train_path
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    files = tuple(sorted((item for item in train_path.rglob("*") if item.is_file()), key=lambda item: item.as_posix())) if train_path.is_dir() else (train_path,)
    by_id: dict[str, Path] = {}
    for item in files:
        stem = item.stem
        if stem in by_id and by_id[stem] != item:
            raise ValueError(f"fit image manifest contains duplicate image ID: {stem}")
        by_id[stem] = item
    missing = [str(image_id) for image_id in fit_ids if str(image_id) not in by_id]
    if missing:
        raise ValueError(f"fit image manifest is missing IDs: {missing[:3]}")
    manifest = [
        {"image_id": str(image_id), "sha256": sha256_file(by_id[str(image_id)])}
        for image_id in fit_ids
    ]
    return _payload_sha256(manifest)


def _upstream_source_hashes() -> tuple[str, dict[str, str]]:
    """Hash all installed Ultralytics Python sources and expose four critical files."""

    import importlib
    import inspect
    import ultralytics

    modules = {
        "tal.py": "ultralytics.utils.tal",
        "loss.py": "ultralytics.utils.loss",
        "dataset.py": "ultralytics.data.dataset",
        "augment.py": "ultralytics.data.augment",
    }
    critical_hashes: dict[str, str] = {}
    for name, module_name in modules.items():
        try:
            module = importlib.import_module(module_name)
            source_path = Path(inspect.getfile(module))
        except (ImportError, OSError, TypeError) as error:  # pragma: no cover - formal runtime dependency
            raise RuntimeError(f"cannot resolve installed Ultralytics source for {name}") from error
        if source_path.name != name or not source_path.is_file():
            raise ValueError(f"resolved upstream source does not match {name}: {source_path}")
        critical_hashes[name] = sha256_file(source_path)
    package_root = Path(inspect.getfile(ultralytics)).resolve().parent
    all_hashes: dict[str, str] = {}
    for source_path in sorted(package_root.rglob("*.py"), key=lambda item: item.relative_to(package_root).as_posix()):
        relative = source_path.relative_to(package_root).as_posix()
        all_hashes[relative] = sha256_file(source_path)
    source_hashes = {f"critical/{name}": value for name, value in sorted(critical_hashes.items())}
    source_hashes.update({f"all/{name}": value for name, value in sorted(all_hashes.items())})
    return _payload_sha256(source_hashes), source_hashes


class _UltralyticsMetadataDataset(torch.utils.data.Dataset):
    """Attach checked raw-label strata to the standard no-augmentation loader."""

    collate_fn = None

    def __init__(self, dataset: Any, expected_ids: Sequence[str], raw_label_dir: Path) -> None:
        self.dataset = dataset
        self.expected_ids = tuple(str(value) for value in expected_ids)
        self.raw_label_dir = Path(raw_label_dir)
        self.collate_fn = dataset.collate_fn
        files = tuple(str(value) for value in getattr(dataset, "im_files", ()))
        if len(files) != len(set(Path(value).stem for value in files)):
            raise ValueError("underlying loader image IDs are not unique")
        index_by_id = {Path(value).stem: index for index, value in enumerate(files)}
        missing = [value for value in self.expected_ids if value not in index_by_id]
        if missing:
            raise ValueError(f"underlying loader is missing fit IDs: {missing[:3]}")
        self._indices = tuple(index_by_id[value] for value in self.expected_ids)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        dataset_index = self._indices[index]
        sample = dict(self.dataset[dataset_index])
        image_id = Path(str(sample.get("im_file", ""))).stem
        if image_id not in self.expected_ids:
            raise ValueError(f"loader image ID outside fit selection: {image_id}")
        from ifdr_yolo.data.kitti_types import Difficulty, TRAIN_CLASS_TO_ID
        from ifdr_yolo.data.yolo_export import object_to_yolo
        from ifdr_yolo.eval.kitti_ap40 import is_valid_ground_truth
        from ifdr_yolo.eval.prediction_io import load_kitti_ground_truth

        ori_shape = sample.get("ori_shape")
        if not isinstance(ori_shape, (tuple, list)) or len(ori_shape) != 2:
            raise ValueError(f"missing ori_shape for {image_id}")
        height, width = int(ori_shape[0]), int(ori_shape[1])
        objects = load_kitti_ground_truth(self.raw_label_dir, (image_id,))[image_id]
        rows: list[tuple[int, float, float, float, float]] = []
        metadata: list[dict[str, object]] = []
        for obj in objects:
            row = object_to_yolo(obj, width, height)
            if row is None:
                continue
            rows.append(row.as_tuple())
            moderate = bool(is_valid_ground_truth(obj, obj.kind, Difficulty.MODERATE))
            height_px = float(obj.bbox.height)
            depth_m = float(obj.location_xyz[2])
            small_25_40 = moderate and (25.0 < height_px <= 40.0)
            large_gt_80 = moderate and height_px > 80.0
            far_gt_40m = moderate and depth_m > 40.0
            near_0_20m = moderate and (0.0 < depth_m <= 20.0)
            metadata.append(
                {
                    "class_name": obj.kind,
                    "class_id": int(TRAIN_CLASS_TO_ID[obj.kind]),
                    "moderate_valid": moderate,
                    "height_px": height_px,
                    "depth_m": depth_m,
                    "small_25_40": bool(small_25_40),
                    "large_gt_80": bool(large_gt_80),
                    "far_gt_40m": bool(far_gt_40m),
                    "near_0_20m": bool(near_0_20m),
                }
            )
        cls = sample.get("cls")
        if not isinstance(cls, torch.Tensor):
            raise ValueError(f"loader batch labels missing for {image_id}")
        label_entry = self.dataset.labels[dataset_index] if hasattr(self.dataset, "labels") else None
        source_cls = label_entry.get("cls") if isinstance(label_entry, Mapping) else cls
        source_bboxes = label_entry.get("bboxes") if isinstance(label_entry, Mapping) else None
        source_cls_tensor = torch.as_tensor(source_cls).reshape(-1)
        if len(rows) != int(cls.shape[0]) or len(rows) != int(source_cls_tensor.shape[0]):
            raise ValueError(f"raw/view GT count mismatch for {image_id}")
        for row_index, row in enumerate(rows):
            if int(cls[row_index].item()) != int(row[0]) or int(source_cls_tensor[row_index].item()) != int(row[0]):
                raise ValueError(f"raw/view GT mapping mismatch for {image_id} at row {row_index}")
            if source_bboxes is not None and not torch.allclose(
                torch.as_tensor(source_bboxes[row_index]).float(), torch.tensor(row[1:], dtype=torch.float32), atol=1e-5, rtol=0.0
            ):
                raise ValueError(f"raw/view bbox mapping mismatch for {image_id} at row {row_index}")
        sample["image_id"] = image_id
        sample["image_ids"] = image_id
        sample["gt_metadata"] = metadata
        sample["gt_strata"] = ["moderate" if bool(item["moderate_valid"]) else "non_moderate" for item in metadata]
        return sample


class _DefaultFitAuditRuntime:
    """Real Ultralytics runtime kept behind the injectable formal runner API."""

    def load_model(self, checkpoint: Path, device: str) -> Any:
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg

        handle = YOLO(str(checkpoint))
        model = handle.model
        if isinstance(getattr(model, "args", None), dict):
            model.args = get_cfg(overrides=model.args)
        model.to(device)
        model.eval()
        _detect_strides(model)
        return model

    def build_criterion(self, model: Any) -> Any:
        return model.init_criterion()

    def build_loader(
        self,
        *,
        config: Any,
        image_ids: Sequence[str],
        raw_label_dir: Path,
        batch_size: int,
        workers: int,
        resolved_data: Path,
        model: Any,
        **_unused: object,
    ) -> Any:
        from ultralytics.cfg import get_cfg
        from ultralytics.data import build_dataloader, build_yolo_dataset
        from ultralytics.data.utils import check_det_dataset

        data = check_det_dataset(str(resolved_data), autodownload=False, split="train")
        model_args = getattr(model, "args", {})
        cfg = get_cfg(overrides=model_args)
        cfg.mode = "val"
        cfg.imgsz = 640
        cfg.batch = int(batch_size)
        cfg.workers = int(workers)
        cfg.fraction = 1.0
        cfg.multi_scale = False
        cfg.rect = False
        cfg.cache = False
        cfg.augment = False
        cfg.single_cls = False
        cfg.classes = None
        dataset = build_yolo_dataset(
            cfg,
            data["train"],
            batch=int(batch_size),
            data=data,
            mode="val",
            rect=False,
            stride=32,
            fraction=1.0,
        )
        wrapped = _UltralyticsMetadataDataset(dataset, image_ids, raw_label_dir)
        return build_dataloader(wrapped, batch=int(batch_size), workers=int(workers), shuffle=False, drop_last=False)


def _runtime_loader(runtime: Any, **kwargs: object) -> Any:
    method = getattr(runtime, "build_loader")
    try:
        return method(**kwargs)
    except TypeError as error:
        if kwargs.keys() >= {"config", "image_ids", "raw_label_dir", "batch_size", "workers"}:
            return method(kwargs["image_ids"])
        raise error


def _write_formal_checkpoint(
    *,
    output: Path,
    mirror: Path,
    identity: Mapping[str, object],
    identity_sha: str,
    completed: Sequence[str],
    journal_path: Path,
    journal_offset: int,
    journal_prefix_sha256: str,
    elapsed_seconds: float,
    state: str,
) -> dict[str, object]:
    checkpoint = {
        "schema_version": 1,
        "state": state,
        "identity": dict(identity),
        "identity_sha256": identity_sha,
        "completed_image_ids": list(completed),
        "next_position": len(completed),
        "journal_offset": int(journal_offset),
        "journal_prefix_sha256": str(journal_prefix_sha256),
        "elapsed_seconds": float(elapsed_seconds),
        "output_paths": {"journal": journal_path.name, "checkpoint": "checkpoint.json"},
    }
    checkpoint_path = output / "checkpoint.json"
    mirror_journal = mirror / journal_path.name
    if journal_path.stat().st_size != int(journal_offset) or not mirror_journal.is_file() or mirror_journal.stat().st_size != int(journal_offset):
        raise ValueError("primary/mirror journal sizes differ before checkpoint commit")
    _atomic_write(checkpoint_path, (json.dumps(checkpoint, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    _mirror_file(checkpoint_path, mirror)
    return checkpoint


def _recover_formal_checkpoint(
    output: Path, mirror: Path, identity_sha: str, selected_ids: Sequence[str]
) -> tuple[list[str], float]:
    checkpoint_path = output / "checkpoint.json"
    journal_path = output / "assignment_audit.jsonl"
    mirror_checkpoint = mirror / checkpoint_path.name
    mirror_journal = mirror / journal_path.name
    if not checkpoint_path.is_file() or not journal_path.is_file() or not mirror_checkpoint.is_file() or not mirror_journal.is_file():
        raise ValueError("primary/mirror checkpoint and journal are required for resume")
    primary_checkpoint_bytes = checkpoint_path.read_bytes()
    mirror_checkpoint_bytes = mirror_checkpoint.read_bytes()
    primary_journal = journal_path.read_bytes()
    mirror_journal_bytes = mirror_journal.read_bytes()

    def validate_candidate(checkpoint: Mapping[str, object], journal: Path, raw: bytes) -> tuple[list[str], int, bytes]:
        if checkpoint.get("identity_sha256") != identity_sha:
            raise ValueError("audit identity mismatch on resume")
        completed = [str(value) for value in checkpoint.get("completed_image_ids", [])]
        if completed != list(selected_ids[: len(completed)]) or len(completed) != len(set(completed)):
            raise ValueError("checkpoint completed IDs are not a valid fit prefix")
        offset = int(checkpoint.get("journal_offset", -1))
        expected_prefix = str(checkpoint.get("journal_prefix_sha256", ""))
        if offset < 0 or offset > len(raw):
            raise ValueError("checkpoint journal offset is invalid")
        prefix = raw[:offset]
        if hashlib.sha256(prefix).hexdigest() != expected_prefix or _journal_prefix(journal, completed, identity_sha) != prefix:
            raise ValueError("journal checkpoint prefix mismatch")
        return completed, offset, prefix

    primary_checkpoint = _load_json(checkpoint_path)
    mirror_checkpoint_payload = _load_json(mirror_checkpoint)
    primary_completed, primary_offset, primary_prefix = validate_candidate(primary_checkpoint, journal_path, primary_journal)
    mirror_completed, mirror_offset, mirror_prefix = validate_candidate(mirror_checkpoint_payload, mirror_journal, mirror_journal_bytes)
    if primary_completed == mirror_completed:
        if (primary_offset, primary_prefix, primary_checkpoint.get("state"), primary_checkpoint.get("identity_sha256")) != (
            mirror_offset,
            mirror_prefix,
            mirror_checkpoint_payload.get("state"),
            mirror_checkpoint_payload.get("identity_sha256"),
        ):
            raise ValueError("primary/mirror checkpoint generation mismatch")
        selected_checkpoint_bytes = primary_checkpoint_bytes
        completed, prefix = primary_completed, primary_prefix
        selected_elapsed = float(primary_checkpoint.get("elapsed_seconds", 0.0))
    elif primary_completed == list(mirror_completed[: len(primary_completed)]):
        selected_checkpoint_bytes = primary_checkpoint_bytes
        completed, prefix = primary_completed, primary_prefix
        selected_elapsed = float(primary_checkpoint.get("elapsed_seconds", 0.0))
    elif mirror_completed == list(primary_completed[: len(mirror_completed)]):
        selected_checkpoint_bytes = mirror_checkpoint_bytes
        completed, prefix = mirror_completed, mirror_prefix
        selected_elapsed = float(mirror_checkpoint_payload.get("elapsed_seconds", 0.0))
    else:
        raise ValueError("primary/mirror checkpoint generations are not a common prefix")
    if not primary_journal.startswith(prefix) or not mirror_journal_bytes.startswith(prefix):
        raise ValueError("primary/mirror journals do not share the selected checkpoint prefix")
    _atomic_write(journal_path, prefix)
    _atomic_write(mirror_journal, prefix)
    _atomic_write(checkpoint_path, selected_checkpoint_bytes)
    _atomic_write(mirror_checkpoint, selected_checkpoint_bytes)
    # Keep the selected generation's cumulative elapsed value available to the
    # caller without changing the public completed-ID recovery contract.
    return completed, selected_elapsed


def _append_formal_record(journal_path: Path, mirror: Path, record: Mapping[str, object]) -> bytes:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    mirror.mkdir(parents=True, exist_ok=True)
    line = _canonical_bytes(record) + b"\n"
    for path in (journal_path, mirror / journal_path.name):
        with path.open("ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
    return line


def _summary_denominators(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for row in records:
        class_name = str(row.get("class_name", "unknown"))
        level = str(row.get("level", "unknown"))
        gt_key = str(row.get("image_id", "")) + ":" + str(row.get("gt_index", ""))
        for stratum in MODERATE_STRATA:
            if not bool(row.get(stratum, False)):
                continue
            key = (class_name, stratum, level, gt_key)
            item = groups.setdefault(
                key,
                {
                    "class_name": class_name,
                    "stratum": stratum,
                    "level": level,
                    "legal_anchor_count": 0,
                    "assigned_positive_count": 0,
                },
            )
            item["legal_anchor_count"] = max(int(item["legal_anchor_count"]), int(row.get("legal_anchor_count", 0)))
            item["assigned_positive_count"] = max(int(item["assigned_positive_count"]), int(row.get("assigned_positive_count", 0)))
    counts: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: {"gt_count": 0, "legal_denominator": 0, "zero_positive": 0})
    for item in groups.values():
        key = (str(item["class_name"]), str(item["stratum"]), str(item["level"]))
        counts[key]["gt_count"] += 1
        if int(item["legal_anchor_count"]) > 0:
            counts[key]["legal_denominator"] += 1
            if int(item["assigned_positive_count"]) == 0:
                counts[key]["zero_positive"] += 1
    observed_classes = {"Car", "Pedestrian", "Cyclist"} | {key[0] for key in groups}
    for class_name in observed_classes:
        for stratum in MODERATE_STRATA:
            for level in LEVEL_NAMES:
                counts[(class_name, stratum, level)]
    return [
        {"class_name": key[0], "stratum": key[1], "level": key[2], **value, "zero_positive_rate": value["zero_positive"] / value["legal_denominator"] if value["legal_denominator"] else 0.0}
        for key, value in sorted(counts.items())
    ]


def run_fit_assignment_audit(
    *,
    config_path: Path,
    resolved_data_path: Path,
    fit_ids_path: Path,
    development_ids_path: Path,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    raw_label_dir: Path,
    output_dir: Path,
    mirror_dir: Path,
    mode: str = "smoke",
    device: str = "cpu",
    batch: int = 1,
    workers: int = 0,
    resume: bool = False,
    runtime: Any | None = None,
    repository_root: Path | None = None,
    stop_after: int | None = None,
) -> dict[str, object]:
    """Run the fit-only candidate-survival audit against a real or injected runtime."""

    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    if int(batch) <= 0 or int(workers) < 0:
        raise ValueError("batch must be positive and workers non-negative")
    resolved_device = _normalize_device(device)
    root = Path(repository_root or Path(__file__).resolve().parents[2]).resolve()
    config_path = Path(config_path).resolve()
    resolved_data_path = Path(resolved_data_path).resolve()
    fit_ids_path = Path(fit_ids_path).resolve()
    development_ids_path = Path(development_ids_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    raw_label_dir = Path(raw_label_dir).resolve()
    output = Path(output_dir).resolve()
    mirror = Path(mirror_dir).resolve()
    if output == mirror or output.is_relative_to(mirror) or mirror.is_relative_to(output):
        raise ValueError("primary and mirror directories must be separate")
    if not resolved_data_path.is_file():
        raise FileNotFoundError(resolved_data_path)
    try:
        ultralytics_version = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError("Ultralytics version is unavailable; expected 8.4.98") from error
    if ultralytics_version != SUPPORTED_ULTRALYTICS_VERSION:
        raise ValueError(
            f"Ultralytics version must be {SUPPORTED_ULTRALYTICS_VERSION}, got {ultralytics_version}"
        )
    config = load_baseline_config(config_path, repository_root=root)
    validate_plain_p2_model(config)
    split = validate_fit_development_split(config, fit_ids_path, development_ids_path)
    checkpoint = validate_primary_checkpoint(checkpoint_path)
    fit_ids, development_ids = split.fit_ids, split.development_ids
    expected_checkpoint_sha256 = str(expected_checkpoint_sha256).strip().lower()
    if len(expected_checkpoint_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_checkpoint_sha256):
        raise ValueError("expected checkpoint SHA256 must be a 64-character hexadecimal digest")
    actual_checkpoint_sha256 = sha256_file(checkpoint)
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    selected_ids = fit_ids if mode == "full" else fit_ids[: min(2, len(fit_ids))]
    fit_image_manifest_sha256 = _fit_image_manifest_sha256(resolved_data_path, fit_ids)
    upstream_source_sha256, upstream_source_files = _upstream_source_hashes()
    identity: dict[str, object] = {
        "fit_ids_sha256": sha256_file(fit_ids_path),
        "development_ids_sha256": sha256_file(development_ids_path),
        "checkpoint_sha256": actual_checkpoint_sha256,
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "config_sha256": sha256_file(config_path),
        "code_sha256": _audit_code_sha256(root),
        "resolved_data_sha256": sha256_file(resolved_data_path),
        "raw_label_dir_sha256": _directory_sha256(raw_label_dir),
        "fit_image_manifest_sha256": fit_image_manifest_sha256,
        "upstream_source_sha256": upstream_source_sha256,
        "upstream_source_files": upstream_source_files,
        "ultralytics_version": ultralytics_version,
        "mode": mode,
        "processed_fit_count": len(selected_ids),
        "device": resolved_device,
        "batch": int(batch),
        "workers": int(workers),
        "imgsz": 640,
        "rect": False,
        "augment": False,
        "shuffle": False,
    }
    AuditIdentity.from_mapping(identity)
    identity_sha = _payload_sha256(identity)
    output.mkdir(parents=True, exist_ok=True)
    mirror.mkdir(parents=True, exist_ok=True)
    journal_path = output / "assignment_audit.jsonl"
    checkpoint_file = output / "checkpoint.json"
    completed: list[str] = []
    elapsed_seconds = 0.0
    run_started = time.monotonic()
    if checkpoint_file.exists():
        if not resume:
            raise ValueError("existing audit checkpoint requires resume=True")
        completed, elapsed_seconds = _recover_formal_checkpoint(output, mirror, identity_sha, selected_ids)
    elif resume:
        raise ValueError("resume requested without checkpoint")
    elif journal_path.exists() and journal_path.stat().st_size:
        raise ValueError("existing audit journal has no checkpoint")
    journal_offset = journal_path.stat().st_size if journal_path.exists() else 0
    journal_hasher = hashlib.sha256()
    if journal_offset:
        with journal_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                journal_hasher.update(block)
    runtime = runtime or _DefaultFitAuditRuntime()
    model = runtime.load_model(checkpoint, resolved_device)
    strides = _detect_strides(model)
    criterion = runtime.build_criterion(model)
    remaining_ids = tuple(value for value in selected_ids if value not in completed)
    loader = _runtime_loader(
        runtime,
        config=config,
        image_ids=remaining_ids,
        raw_label_dir=raw_label_dir,
        batch_size=int(batch),
        workers=int(workers),
        resolved_data=resolved_data_path,
        model=model,
        device=resolved_device,
    )
    assigner = getattr(criterion, "assigner", None)
    if assigner is None:
        raise ValueError("criterion does not expose its original assigner")
    observer: AssignmentAuditObserver | None = None
    all_records: list[dict[str, object]] = []
    actual_seen: list[str] = list(completed)
    first_batch = True
    model_before = _sha256_state(model)
    with torch.no_grad():
        for batch_data in loader:
            image_ids = tuple(str(value) for value in batch_data.get("image_ids", ()))
            if not image_ids or any(value not in selected_ids for value in image_ids) or len(set(image_ids)) != len(image_ids):
                raise ValueError("loader image_ids must be unique and inside the selected fit prefix")
            expected_batch = tuple(selected_ids[len(actual_seen) : len(actual_seen) + len(image_ids)])
            if image_ids != expected_batch:
                raise ValueError("loader image IDs do not match the expected fit prefix")
            actual_seen.extend(image_ids)
            if len(actual_seen) != len(set(actual_seen)):
                raise ValueError("loader yielded duplicate image IDs")
            strata = batch_data.get("gt_strata")
            class_names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
            torch_device = torch.device(resolved_device)
            for key in ("batch_idx", "cls", "bboxes"):
                value = batch_data.get(key)
                if isinstance(value, torch.Tensor):
                    batch_data[key] = value.to(torch_device)
            image_tensor = batch_data.get("img")
            if not isinstance(image_tensor, torch.Tensor):
                raise ValueError("loader image tensor is missing")
            batch_data["img"] = image_tensor.to(torch_device).float().div(255.0)
            preds = model(batch_data["img"])
            feature_maps = _prediction_mapping(preds).get("feats")
            if feature_maps is None:
                raise ValueError("model predictions do not expose feature levels")
            level_slices = derive_level_slices(feature_maps, strides=strides)
            if observer is None:
                observer = AssignmentAuditObserver(assigner, enabled=False, strict=True, level_slices=level_slices, class_names=class_names)
                observer.attach()
            observer.set_metadata(image_ids=image_ids, gt_strata=strata, gt_metadata=batch_data.get("gt_metadata"))
            if first_batch:
                state_before = _sha256_state(model)
                rng_before = _rng_state()
                first_batch_ok = False
                try:
                    observer.enabled = False
                    off = criterion(preds, batch_data)
                    off_assignments = observer.last_output
                    observer.enabled = True
                    observer.records.clear()
                    on = criterion(preds, batch_data)
                    on_assignments = observer.last_output
                    first_batch_ok = True
                finally:
                    if not first_batch_ok:
                        observer.detach()
                if not _torch_equal(off, on) or not _torch_equal(off_assignments, on_assignments):
                    raise ValueError("audit hook changed criterion loss or assignment outputs")
                if _rng_state() != rng_before or _sha256_state(model) != state_before:
                    raise ValueError("audit hook changed RNG or model state")
                first_batch = False
            else:
                observer.enabled = True
                criterion(preds, batch_data)
            batch_records = list(observer.records)
            observer.records.clear()
            records_by_image: dict[str, list[dict[str, object]]] = defaultdict(list)
            for row in batch_records:
                records_by_image[str(row["image_id"])].append(dict(row))
            batch_records: list[dict[str, object]] = []
            for image_id in image_ids:
                if image_id in completed:
                    raise ValueError("loader yielded an already committed image ID")
                batch_records.append(
                    {
                        "schema_version": 1,
                        "identity_sha256": identity_sha,
                        "image_id": image_id,
                        "rows": records_by_image.get(image_id, []),
                    }
                )
            for record in batch_records:
                line = _append_formal_record(journal_path, mirror, record)
                journal_hasher.update(line)
                journal_offset += len(line)
            completed.extend(image_ids)
            _write_formal_checkpoint(
                output=output,
                mirror=mirror,
                identity=identity,
                identity_sha=identity_sha,
                completed=completed,
                journal_path=journal_path,
                journal_offset=journal_offset,
                journal_prefix_sha256=journal_hasher.hexdigest(),
                elapsed_seconds=elapsed_seconds + (time.monotonic() - run_started),
                state="running",
            )
            if stop_after is not None and len(completed) >= int(stop_after) and len(completed) < len(selected_ids):
                raise RuntimeError("interrupted fit assignment audit")
    if observer is not None:
        observer.detach()
    if actual_seen != list(selected_ids):
        raise ValueError("loader did not provide exact selected fit coverage")
    if set(actual_seen) & set(development_ids):
        raise ValueError("actual loader read development IDs")
    if _sha256_state(model) != model_before:
        raise ValueError("formal audit changed model state")
    for line in journal_path.read_bytes().splitlines():
        payload = json.loads(line.decode("utf-8"))
        all_records.extend(payload.get("rows", []))
    denominators = _summary_denominators(all_records)
    journal_sha256 = _file_sha256(journal_path)
    manifest_binding_sha256 = _payload_sha256(
        {"identity_sha256": identity_sha, "journal_sha256": journal_sha256}
    )
    cyclist_frame = build_cyclist_image_cluster_frame(all_records, selected_ids)
    cyclist_bootstrap = bootstrap_cyclist_zero_p2_contrasts(
        cyclist_frame,
        full_dataset=(mode == "full" and len(fit_ids) == 3341 and len(selected_ids) == len(fit_ids)),
        journal_sha256=journal_sha256,
        identity_sha256=identity_sha,
        manifest_sha256=manifest_binding_sha256,
    )
    summary = {
        "schema_version": 1,
        "state": "complete",
        "identity": identity,
        "identity_sha256": identity_sha,
        "fit_count": len(fit_ids),
        "processed_fit_count": len(selected_ids),
        "development_count": len(development_ids),
        "intersection_count": 0,
        "actual_loader_fit_count": len(actual_seen),
        "denominators": denominators,
        "zero_p2_positive": summarize_zero_p2_positive(all_records),
        "strata": list(MODERATE_STRATA),
        "contrasts": [dict(item) for item in PRE_REGISTERED_CONTRASTS],
        "cyclist_p2_zero_positive_bootstrap": cyclist_bootstrap,
    }
    summary_path = output / "summary.json"
    summary_csv = output / "summary.csv"
    _atomic_write(summary_path, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=output, prefix=".summary.", suffix=".tmp") as stream:
        writer = csv.DictWriter(stream, fieldnames=["class_name", "stratum", "level", "gt_count", "legal_denominator", "zero_positive", "zero_positive_rate"])
        writer.writeheader()
        writer.writerows(denominators)
        stream.flush()
        os.fsync(stream.fileno())
        temporary_csv = Path(stream.name)
    os.replace(temporary_csv, summary_csv)
    _write_formal_checkpoint(
        output=output,
        mirror=mirror,
        identity=identity,
        identity_sha=identity_sha,
        completed=completed,
        journal_path=journal_path,
        journal_offset=journal_offset,
        journal_prefix_sha256=journal_hasher.hexdigest(),
        elapsed_seconds=elapsed_seconds + (time.monotonic() - run_started),
        state="complete",
    )
    manifest = {
        "schema_version": 1,
        "identity_sha256": identity_sha,
        "binding_sha256": manifest_binding_sha256,
        "bootstrap_input_hashes": cyclist_bootstrap["input_hashes"],
        "files": {name: _file_sha256(output / name) for name in ("assignment_audit.jsonl", "checkpoint.json", "summary.json", "summary.csv")},
    }
    manifest_path = output / "manifest.json"
    _atomic_write(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    for artifact in (journal_path, checkpoint_file, summary_path, summary_csv, manifest_path):
        _mirror_file(artifact, mirror)
    return summary


def run_synthetic_audit(
    *,
    fit_ids: Sequence[str],
    development_ids: Sequence[str],
    output_dir: Path,
    mirror_dir: Path,
    identity: AuditIdentity | Mapping[str, object],
    expected_fit_count: int = 3341,
    expected_development_count: int = 371,
    resume: bool = False,
    stop_after: int | None = None,
) -> dict[str, object]:
    """Run a deterministic synthetic journal; this never loads data or starts training."""

    fit, _ = validate_fit_development_ids(
        fit_ids,
        development_ids,
        expected_fit_count=expected_fit_count,
        expected_development_count=expected_development_count,
    )
    development = tuple(str(value).strip() for value in development_ids if str(value).strip())
    audit_identity = identity if isinstance(identity, AuditIdentity) else AuditIdentity.from_mapping(identity)
    if audit_identity.values.get("fit_ids_sha256") != _ids_sha256(fit):
        raise ValueError("fit IDs SHA256 does not match audit identity")
    if audit_identity.values.get("development_ids_sha256") != _ids_sha256(development):
        raise ValueError("development IDs SHA256 does not match audit identity")
    output = Path(output_dir).resolve()
    mirror = Path(mirror_dir).resolve()
    if output == mirror or output.is_relative_to(mirror) or mirror.is_relative_to(output):
        raise ValueError("primary and mirror directories must be separate")
    output.mkdir(parents=True, exist_ok=True)
    mirror.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.json"
    journal_path = output / "audit.jsonl"
    identity_sha = audit_identity.sha256
    completed: list[str] = []
    elapsed = 0.0
    if checkpoint_path.exists():
        if not resume:
            raise ValueError("existing audit checkpoint requires resume=True")
        checkpoint_bytes = checkpoint_path.read_bytes()
        if not (mirror / checkpoint_path.name).is_file():
            raise ValueError("mirror checkpoint is required for resume")
        checkpoint = _load_json(checkpoint_path)
        if checkpoint.get("identity_sha256") != identity_sha:
            raise ValueError("audit identity mismatch on resume")
        completed = [str(value) for value in checkpoint.get("completed_image_ids", [])]
        if completed != list(fit[: len(completed)]) or len(completed) != len(set(completed)):
            raise ValueError("checkpoint completed IDs are not a valid fit prefix")
        _prepare_recovery_artifacts(journal_path, checkpoint_path, mirror, completed, identity_sha)
        if checkpoint_path.read_bytes() != checkpoint_bytes:
            raise ValueError("checkpoint changed during recovery validation")
        elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
    elif resume:
        raise ValueError("resume requested without checkpoint")
    elif journal_path.exists() and journal_path.stat().st_size:
        raise ValueError("existing audit journal has no checkpoint")

    run_started = time.monotonic()
    with journal_path.open("a", encoding="utf-8") as journal:
        for position, image_id in enumerate(fit):
            if image_id in completed:
                continue
            record = {"schema_version": 1, "identity_sha256": identity_sha, "position": position, "image_id": image_id}
            journal.write(json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            completed.append(image_id)
            checkpoint = {
                "schema_version": 1,
                "state": "running",
                "identity": audit_identity.as_dict(),
                "identity_sha256": identity_sha,
                "completed_image_ids": completed,
                "next_position": position + 1,
                "elapsed_seconds": elapsed + (time.monotonic() - run_started),
                "output_paths": {"journal": str(journal_path), "checkpoint": str(checkpoint_path)},
            }
            _atomic_write(checkpoint_path, (json.dumps(checkpoint, indent=2, sort_keys=True) + "\n").encode("utf-8"))
            _mirror_file(journal_path, mirror)
            _mirror_file(checkpoint_path, mirror)
            if stop_after is not None and len(completed) >= stop_after and len(completed) < len(fit):
                raise RuntimeError("interrupted synthetic audit")
    summary = {
        "schema_version": 1,
        "state": "complete",
        "identity": audit_identity.as_dict(),
        "identity_sha256": identity_sha,
        "fit_count": len(fit),
        "development_count": expected_development_count,
        "intersection_count": 0,
        "completed_image_count": len(completed),
    }
    summary_path = output / "summary.json"
    summary_csv = output / "summary.csv"
    _atomic_write(summary_path, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    _atomic_write(summary_csv, b"state,fit_count,development_count,intersection_count\ncomplete,%d,%d,0\n" % (len(fit), expected_development_count))
    elapsed += time.monotonic() - run_started
    checkpoint = {
        **summary,
        "next_position": len(fit),
        "elapsed_seconds": elapsed,
        "output_paths": {"journal": str(journal_path), "checkpoint": str(checkpoint_path)},
    }
    _atomic_write(checkpoint_path, (json.dumps(checkpoint, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    manifest = {
        "schema_version": 1,
        "identity_sha256": identity_sha,
        "files": {name: _file_sha256(output / name) for name in ("audit.jsonl", "checkpoint.json", "summary.json", "summary.csv")},
    }
    manifest_path = output / "manifest.json"
    _atomic_write(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    for artifact in (journal_path, checkpoint_path, summary_path, summary_csv, manifest_path):
        _mirror_file(artifact, mirror)
    return summary


__all__ = [
    "AssignmentAuditObserver",
    "AuditIdentity",
    "build_cyclist_image_cluster_frame",
    "bootstrap_cyclist_zero_p2_contrasts",
    "IDENTITY_FIELDS",
    "LEVEL_NAMES",
    "MODERATE_STRATA",
    "PRE_REGISTERED_CONTRASTS",
    "LevelSlice",
    "derive_level_slices",
    "validate_level_strides",
    "replay_assignment",
    "run_fit_assignment_audit",
    "run_synthetic_audit",
    "summarize_zero_p2_positive",
    "validate_fit_development_ids",
    "validate_level_strides",
]
