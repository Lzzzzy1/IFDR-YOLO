"""Fit-only score/ranking and NMS survival audit.

The audit is deliberately observational: the registered Ultralytics NMS is
called on a detached tensor and its result is returned unchanged.  Stage
indices and suppression metadata are computed from the same decoded tensor;
they never alter predictions or labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import torch
import yaml

from ifdr_yolo.data.kitti_types import TRAIN_CLASS_TO_ID
from ifdr_yolo.eval.kitti_ap40 import is_valid_ground_truth
from ifdr_yolo.eval.prediction_io import load_kitti_ground_truth
from ifdr_yolo.experiments.config import load_baseline_config
from ifdr_yolo.experiments.p2_candidate_survival_audit import (
    _audit_code_sha256,
    _directory_sha256,
    _fit_image_manifest_sha256,
    _upstream_source_hashes,
    sha256_file,
)


LEVEL_NAMES = ("P2", "P3", "P4", "P5")
STAGES = ("raw", "conf", "max_nms", "nms", "max_det", "final")
SUPPORTED_ULTRALYTICS_VERSION = "8.4.98"
S_STAGE_SET = frozenset(("max_nms", "nms", "max_det", "final"))
N_STAGE_SET = frozenset(("nms", "max_det", "final"))
BENCHMARK32_RNG_SEED = 20260812
BENCHMARK32_SELECTION_RULE = "first_32_registered_ordered_fit_ids"
REGISTERED_FIT_COUNT = 3341
FULL_AUDIT_RNG_SEED = 20260812
FULL_AUDIT_SELECTION_RULE = "all_3341_registered_ordered_fit_ids"


class ScoreNMSInterrupted(RuntimeError):
    """Intentional stop after a durable prefix; never use StopIteration.

    Ultralytics' streaming predictor may catch ``StopIteration`` internally
    and continue loading batches.  A dedicated RuntimeError makes the
    stop-after boundary fail closed and lets the CLI/runner distinguish an
    intentional interruption from a completed audit.
    """


def _file_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _ordered_ids_sha(ids: Sequence[str]) -> str:
    return _file_digest(("\n".join(str(value) for value in ids) + "\n").encode("utf-8"))


def _select_fit_ids_for_mode(mode: str, fit_ids: Sequence[str]) -> tuple[str, ...]:
    """Select only the registered ordered fit-manifest prefix for each mode."""

    registered = tuple(str(value) for value in fit_ids)
    if mode == "smoke":
        return registered[:2]
    if mode == "benchmark32":
        if len(registered) < 32:
            raise ValueError("benchmark32 requires at least >= 32 registered ordered fit IDs")
        return registered[:32]
    if mode == "full":
        if len(registered) != REGISTERED_FIT_COUNT:
            raise ValueError("full requires exactly 3341 registered ordered fit IDs")
        return registered
    raise ValueError("mode must be smoke, benchmark32, or full")


def _benchmark32_identity_fields(selected_ids: Sequence[str]) -> dict[str, object]:
    selected = tuple(str(value) for value in selected_ids)
    if len(selected) != 32:
        raise ValueError("benchmark32 identity requires exactly 32 selected IDs")
    return {
        "selection_rule": BENCHMARK32_SELECTION_RULE,
        "selected_ids_count": len(selected),
        "selected_ids_ordered_sha256": _ordered_ids_sha(selected),
        "benchmark_rng_seed": BENCHMARK32_RNG_SEED,
    }


def _full_audit_identity_fields(selected_ids: Sequence[str]) -> dict[str, object]:
    selected = tuple(str(value) for value in selected_ids)
    if len(selected) != REGISTERED_FIT_COUNT:
        raise ValueError("full identity requires exactly 3341 selected IDs")
    return {
        "selection_rule": FULL_AUDIT_SELECTION_RULE,
        "selected_ids_count": len(selected),
        "selected_ids_ordered_sha256": _ordered_ids_sha(selected),
        "full_audit_rng_seed": FULL_AUDIT_RNG_SEED,
    }


def _validate_mode_stop_after(mode: str, stop_after: object, *, completed_count: int) -> int | None:
    if stop_after is None:
        return None
    terminal = 32 if mode == "benchmark32" else REGISTERED_FIT_COUNT if mode == "full" else None
    if terminal is None:
        raise ValueError("stop_after is supported only for benchmark32 or full")
    if isinstance(stop_after, bool) or not isinstance(stop_after, int) or not 1 <= stop_after < terminal:
        raise ValueError(f"{mode} stop_after must be an integer in 1..{terminal - 1}")
    if completed_count >= stop_after or completed_count >= terminal:
        raise ValueError(f"{mode} stop_after must target an uncompleted nonterminal prefix")
    return int(stop_after)


def _validate_benchmark32_stop_after(stop_after: object, *, completed_count: int) -> int | None:
    return _validate_mode_stop_after("benchmark32", stop_after, completed_count=completed_count)


def _source_code_identity() -> str:
    here = Path(__file__).resolve()
    cli = here.parents[2] / "scripts" / "run_p2_score_nms_survival_audit.py"
    payload = {"module": _file_digest(here.read_bytes()), "cli": _file_digest(cli.read_bytes())}
    return _canonical_sha(payload)


def _local_source_hashes(root: Path) -> dict[str, str]:
    """Bind every local source reused by the formal audit to its identity."""

    paths = (
        Path(__file__).resolve(),
        root / "scripts" / "run_p2_score_nms_survival_audit.py",
        root / "ifdr_yolo" / "experiments" / "p2_candidate_survival_audit.py",
        root / "ifdr_yolo" / "experiments" / "config.py",
        root / "ifdr_yolo" / "experiments" / "p2_fit_reference.py",
        root / "ifdr_yolo" / "eval" / "kitti_ap40.py",
        root / "ifdr_yolo" / "eval" / "prediction_io.py",
        root / "ifdr_yolo" / "data" / "kitti_types.py",
        root / "ifdr_yolo" / "data" / "splits.py",
    )
    result: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"audit source missing: {path}")
        result[path.relative_to(root).as_posix()] = _file_digest(path.read_bytes())
    return result


def _runtime_identity(device: str) -> dict[str, object]:
    """Record runtime facts that can alter floating-point inference."""

    payload: dict[str, object] = {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda or ""),
        "cudnn_version": int(torch.backends.cudnn.version() or 0),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_requested": str(device),
    }
    try:
        import importlib.metadata
        payload["torchvision_version"] = str(importlib.metadata.version("torchvision"))
    except Exception:
        payload["torchvision_version"] = "unavailable"
    if torch.cuda.is_available():
        try:
            selected = torch.device(str(device))
            index = int(selected.index if selected.index is not None else torch.cuda.current_device())
            payload.update({
                "cuda_device_index": index,
                "cuda_device_name": str(torch.cuda.get_device_name(index)),
                "cuda_device_capability": list(torch.cuda.get_device_capability(index)),
            })
        except Exception as error:  # pragma: no cover - device-specific
            payload["cuda_device_error"] = type(error).__name__
    return payload


def _torchvision_backend_identity() -> dict[str, object]:
    """Import torchvision explicitly so smoke/full use one NMS backend."""

    try:
        import importlib.metadata
        import torchvision  # noqa: F401
        version = str(importlib.metadata.version("torchvision"))
    except Exception as error:  # pragma: no cover - formal runtime
        raise RuntimeError("torchvision is required to fix the Ultralytics NMS backend") from error
    return {"torchvision_version": version, "backend": "torchvision"}


def _map_box_to_input(box: Sequence[float], *, orig_shape: Sequence[int], input_shape: Sequence[int]) -> tuple[float, float, float, float]:
    """Map original-image xyxy to the exact batch tensor letterbox coordinates."""

    oh, ow = float(orig_shape[0]), float(orig_shape[1])
    ih, iw = float(input_shape[0]), float(input_shape[1])
    ratio = min(iw / ow, ih / oh)
    # Ultralytics LetterBox uses rounded resized dimensions and the
    # ``round(... - 0.1)`` border convention.  Reusing its exact arithmetic
    # avoids an otherwise silent one-pixel GT ownership shift on odd padding.
    resized_w, resized_h = round(ow * ratio), round(oh * ratio)
    pad_x = round((iw - resized_w) / 2.0 - 0.1)
    pad_y = round((ih - resized_h) / 2.0 - 0.1)
    x1, y1, x2, y2 = (float(v) for v in box)
    return (x1 * ratio + pad_x, y1 * ratio + pad_y, x2 * ratio + pad_x, y2 * ratio + pad_y)


@dataclass(frozen=True)
class LevelSlice:
    name: str
    start: int
    stop: int

    @property
    def count(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class NMSReplay:
    output: list[torch.Tensor]
    kept_indices: list[torch.Tensor]
    stage_indices: list[dict[str, torch.Tensor]]
    suppression: list[list[dict[str, object]]]


def _feature_hw(feature: Any) -> tuple[int, int]:
    shape = getattr(feature, "shape", feature)
    if len(shape) < 2:
        raise ValueError("feature map must expose H and W")
    height, width = int(shape[-2]), int(shape[-1])
    if height <= 0 or width <= 0:
        raise ValueError("feature map dimensions must be positive")
    return height, width


def derive_level_slices(feature_maps: Sequence[Any], names: Sequence[str] = LEVEL_NAMES) -> tuple[LevelSlice, ...]:
    """Derive flattened P2--P5 boundaries from dynamic H*W sizes."""

    if len(feature_maps) != len(names) or not feature_maps:
        raise ValueError("feature-map and level-name counts must match and be non-zero")
    result: list[LevelSlice] = []
    cursor = 0
    seen: set[str] = set()
    for raw_name, feature in zip(names, feature_maps):
        name = str(raw_name)
        if not name or name in seen:
            raise ValueError("level names must be unique and non-empty")
        seen.add(name)
        height, width = _feature_hw(feature)
        count = height * width
        result.append(LevelSlice(name, cursor, cursor + count))
        cursor += count
    return tuple(result)


def _canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    result = boxes.clone()
    x, y, w, h = boxes.unbind(-1)
    result[..., 0] = x - w / 2
    result[..., 1] = y - h / 2
    result[..., 2] = x + w / 2
    result[..., 3] = y + h / 2
    return result


def _iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    left = torch.maximum(box[:2], boxes[:, :2])
    right = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (right - left).clamp_min(0).prod(-1)
    area_a = (box[2:] - box[:2]).clamp_min(0).prod()
    area_b = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(-1)
    return intersection / (area_a + area_b - intersection).clamp_min(1e-12)


def _iou_on_device(box: torch.Tensor, boxes: torch.Tensor, *, device: torch.device | str) -> torch.Tensor:
    """Compute ordinary IoU after one explicit device/dtype normalization.

    Formal decoded candidates define the compute device.  Ground-truth and
    provenance boxes may originate on CPU even when decoded boxes are CUDA;
    normalizing both operands once here prevents a device mismatch without
    changing the ordinary-IoU threshold or moving tensors back and forth in a
    per-candidate loop.
    """

    target = torch.device(device)
    normalized_box = box.to(device=target)
    normalized_boxes = boxes.to(device=target, dtype=normalized_box.dtype)
    return _iou(normalized_box, normalized_boxes)


def _pairwise_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Vectorized ordinary IoU for ``[N,4]`` × ``[M,4]`` boxes."""

    if boxes_a.ndim != 2 or boxes_b.ndim != 2 or boxes_a.shape[-1] != 4 or boxes_b.shape[-1] != 4:
        raise ValueError("pairwise IoU expects [N,4] and [M,4] tensors")
    left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = (right - left).clamp_min(0).prod(-1)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp_min(0).prod(-1)[:, None]
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp_min(0).prod(-1)[None, :]
    return intersection / (area_a + area_b - intersection).clamp_min(1e-12)


def _indices_for_prediction(
    image_prediction: torch.Tensor,
    *,
    conf: float,
    max_nms: int,
    classes: Sequence[int] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw/conf/max_nms candidate indices, boxes and best class scores."""

    if image_prediction.ndim != 2 or image_prediction.shape[1] < 5:
        raise ValueError("one-image prediction must be [N, 4 + classes]")
    count = image_prediction.shape[0]
    raw = torch.arange(count, device=image_prediction.device, dtype=torch.long)
    nc = image_prediction.shape[1] - 4
    scores = image_prediction[:, 4 : 4 + nc]
    candidate = scores.amax(1) > conf
    conf_indices = raw[candidate]
    selected = image_prediction[candidate]
    if selected.numel() == 0:
        empty = raw[:0]
        return raw, empty, empty, image_prediction[:0, :4], image_prediction[:0, 4:]
    best_score, best_class = selected[:, 4:].max(1)
    keep = best_score > conf
    if classes is not None:
        keep &= torch.isin(best_class, torch.as_tensor(tuple(classes), device=best_class.device))
    conf_indices = conf_indices[keep]
    selected = selected[keep]
    best_score = best_score[keep]
    best_class = best_class[keep]
    if selected.shape[0] > max_nms:
        order = best_score.argsort(descending=True)[:max_nms]
        max_indices = conf_indices[order]
        selected = selected[order]
    else:
        max_indices = conf_indices
    return raw, conf_indices, max_indices, _xywh2xyxy(selected[:, :4]), selected[:, 4:]


def replay_nms_with_stages(
    prediction: torch.Tensor,
    *,
    level_slices: Sequence[LevelSlice | tuple[str, int, int]],
    conf: float = 0.001,
    iou: float = 0.7,
    max_nms: int = 30000,
    max_det: int = 300,
    classes: Sequence[int] | None = None,
    agnostic: bool = False,
    trace_suppression: bool = False,
) -> NMSReplay:
    """Replay official NMS and expose raw/conf/max-NMS/NMS/max-det/final sets.

    ``prediction`` is never modified.  Final ``output`` and ``kept_indices``
    come from the installed Ultralytics implementation, making byte/tensor
    equality checks against the normal predictor possible.
    """

    if prediction.ndim != 3 or prediction.shape[1] < 5:
        raise ValueError("prediction must be [B, 4 + classes, N]")
    if not (0 <= conf <= 1 and 0 <= iou <= 1 and max_nms > 0 and max_det > 0):
        raise ValueError("invalid NMS thresholds or limits")
    slices = tuple(item if isinstance(item, LevelSlice) else LevelSlice(*item) for item in level_slices)
    if not slices or slices[0].start != 0 or slices[-1].stop != prediction.shape[-1]:
        raise ValueError("level slices must cover flattened predictions")
    if any(left.stop != right.start for left, right in zip(slices, slices[1:])):
        raise ValueError("level slices must be contiguous")

    try:
        from ultralytics.utils import nms
    except ModuleNotFoundError:  # focused tests can run without the formal runtime
        nms = None

    def fallback_nms(value: torch.Tensor, limit: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        outputs: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        for image in value:
            rows = image.transpose(0, 1).clone()
            boxes = _xywh2xyxy(rows[:, :4])
            scores, labels = rows[:, 4:].max(1)
            keep = scores > conf
            if classes is not None:
                keep &= torch.isin(labels, torch.as_tensor(tuple(classes), device=labels.device))
            original = torch.arange(rows.shape[0], device=rows.device)[keep]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if boxes.shape[0] > max_nms:
                order = scores.argsort(descending=True)[:max_nms]
                boxes, scores, labels, original = boxes[order], scores[order], labels[order], original[order]
            order = scores.argsort(descending=True)
            picks: list[int] = []
            while order.numel():
                first = int(order[0].item())
                picks.append(first)
                if order.numel() == 1:
                    break
                overlap = _iou(boxes[first], boxes[order[1:]])
                if agnostic:
                    keep_order = overlap <= iou
                else:
                    keep_order = (overlap <= iou) | (labels[order[1:]] != labels[first])
                order = order[1:][keep_order]
            picks = picks[:limit]
            picked = torch.as_tensor(picks, dtype=torch.long, device=rows.device)
            outputs.append(torch.cat((boxes[picked], scores[picked, None], labels[picked, None].float()), dim=1) if picks else rows.new_zeros((0, 6)))
            indices.append(original[picked] if picks else original[:0])
        return outputs, indices

    # The upstream function mutates its input while converting xywh to xyxy.
    detached = prediction.detach().clone()
    if nms is None:
        official_pre, official_pre_keep = fallback_nms(detached.clone(), max_nms)
        official_final = [item[:max_det] for item in official_pre]
        official_keep = [item[:max_det] for item in official_pre_keep]
    else:
        official_pre, official_pre_keep = nms.non_max_suppression(
            detached.clone(), conf, iou, classes=classes, agnostic=agnostic, max_det=max_nms, max_nms=max_nms, return_idxs=True
        )
        official_final = [item[:max_det] for item in official_pre]
        official_keep = [item[:max_det] for item in official_pre_keep]
    outputs: list[torch.Tensor] = []
    final_indices: list[torch.Tensor] = []
    stage_rows: list[dict[str, torch.Tensor]] = []
    suppression_rows: list[list[dict[str, object]]] = []
    level_by_index = {index: level.name for level in slices for index in range(level.start, level.stop)}
    for image_index in range(prediction.shape[0]):
        image = prediction[image_index].transpose(0, 1)
        raw, conf_indices, max_indices, max_boxes, max_scores = _indices_for_prediction(
            image, conf=conf, max_nms=max_nms, classes=classes
        )
        pre_indices = official_pre_keep[image_index].detach().clone().long()
        final = official_keep[image_index].detach().clone().long()
        # The upstream index tensors are 1-D, but normalise empty outputs too.
        pre_set = set(int(v) for v in pre_indices.tolist())
        stage = {
            "raw": raw,
            "conf": conf_indices,
            "max_nms": max_indices,
            "nms": pre_indices,
            "max_det": final,
            "final": final,
        }
        stage_rows.append(stage)
        outputs.append(official_final[image_index])
        final_indices.append(final)

        classes_by_index: dict[int, int] = {}
        scores_by_index: dict[int, float] = {}
        boxes_by_index: dict[int, torch.Tensor] = {}
        selected_indices = max_indices.tolist()
        selected_rows = image[max_indices] if selected_indices else image[:0]
        if selected_rows.numel():
            selected_boxes = _xywh2xyxy(selected_rows[:, :4])
            selected_scores, selected_classes = selected_rows[:, 4:].max(1)
            for idx, cls, score, box in zip(selected_indices, selected_classes.tolist(), selected_scores.tolist(), selected_boxes):
                classes_by_index[int(idx)] = int(cls)
                scores_by_index[int(idx)] = float(score)
                boxes_by_index[int(idx)] = box
        image_suppression: list[dict[str, object]] = []
        for candidate_index in selected_indices if trace_suppression else ():
            suppressed = candidate_index not in pre_set
            detail = {"candidate_index": int(candidate_index), "suppressed": bool(suppressed)}
            if suppressed and not agnostic:
                cls = classes_by_index.get(int(candidate_index))
                box = boxes_by_index.get(int(candidate_index))
                if cls is not None and box is not None:
                    for suppressor in pre_indices.tolist():
                        if classes_by_index.get(int(suppressor)) != cls:
                            continue
                        suppressor_box = boxes_by_index.get(int(suppressor))
                        if suppressor_box is None:
                            continue
                        overlap = float(_iou(suppressor_box, box.unsqueeze(0))[0].item())
                        if overlap > iou:
                            detail.update(
                                {
                                    "suppressor_index": int(suppressor),
                                    "suppressor_level": level_by_index.get(int(suppressor), "unknown"),
                                    "candidate_level": level_by_index.get(int(candidate_index), "unknown"),
                                    "pair_iou": overlap,
                                    "iou": overlap,
                                    "suppressor_class": cls,
                                    "candidate_score": scores_by_index.get(int(candidate_index), 0.0),
                                    "suppressor_score": scores_by_index.get(int(suppressor), 0.0),
                                }
                            )
                            break
            image_suppression.append(detail)
        suppression_rows.append(image_suppression)
    return NMSReplay(outputs, final_indices, stage_rows, suppression_rows)


def _candidate_stage(candidate: Mapping[str, object], *, conf: float, final_indices: set[int]) -> str:
    explicit = str(candidate.get("stage", "")).strip()
    score = float(candidate.get("gt_score", candidate.get("score", max(candidate.get("class_scores", (0.0,)), default=0.0))))
    index = int(candidate.get("index", -1))
    if index in final_indices:
        return "final"
    if explicit in STAGES and explicit != "raw":
        return explicit
    return "raw" if score <= conf else "max_nms"


def score_nms_survival_row(
    *,
    gt_box: Sequence[float],
    gt_class: int,
    candidates: Sequence[Mapping[str, object]],
    conf: float = 0.001,
    final_indices: set[int] | Sequence[int] = (),
    useful_iou: float = 0.5,
) -> dict[str, object]:
    """Summarise one Moderate GT without merging P2 and coarse levels."""

    final_set = set(int(item) for item in final_indices)
    gt = torch.as_tensor(tuple(float(v) for v in gt_box), dtype=torch.float32)
    enriched: list[dict[str, object]] = []
    for candidate in candidates:
        box = torch.as_tensor(tuple(float(v) for v in candidate["box"]), dtype=torch.float32)
        overlap = float(_iou(gt, box.unsqueeze(0))[0].item())
        scores = tuple(float(v) for v in candidate.get("class_scores", ()))
        gt_score = scores[int(gt_class)] if int(gt_class) < len(scores) else 0.0
        best_class = int(max(range(len(scores)), key=lambda i: scores[i])) if scores else -1
        item = dict(candidate)
        item.update({"iou": overlap, "gt_score": gt_score, "best_class": best_class, "best_score": max(scores, default=0.0)})
        enriched.append(item)
    p2 = [item for item in enriched if str(item.get("level")) == "P2" and float(item["iou"]) > useful_iou]
    coarse = [item for item in enriched if str(item.get("level")) in {"P3", "P4", "P5"} and float(item["iou"]) > useful_iou]
    conf_p2 = [item for item in p2 if float(item["gt_score"]) > conf]
    conf_coarse = [item for item in coarse if float(item["gt_score"]) > conf]
    best_p2 = max(p2, key=lambda item: float(item["gt_score"]), default=None)
    best_coarse = max(coarse, key=lambda item: float(item["gt_score"]), default=None)
    p2_suppressed = [item for item in p2 if int(item.get("index", -1)) not in final_set and item.get("suppressor_index") is not None]
    route = "none"
    if p2 and not conf_p2 and conf_coarse:
        route = "B"
    elif p2_suppressed:
        for item in p2_suppressed:
            if float(item["iou"]) > float(item.get("suppressor_iou", 0.0)):
                route = "C"
                break
    return {
        "gt_class": int(gt_class),
        "raw_useful_p2": bool(p2),
        "raw_useful_coarse": bool(coarse),
        "conf_useful_p2": bool(conf_p2),
        "conf_useful_coarse": bool(conf_coarse),
        "p2_best_class": None if best_p2 is None else int(best_p2["best_class"]),
        "coarse_best_class": None if best_coarse is None else int(best_coarse["best_class"]),
        "p2_stage": "none" if best_p2 is None else _candidate_stage(best_p2, conf=conf, final_indices=final_set),
        "coarse_stage": "none" if best_coarse is None else _candidate_stage(best_coarse, conf=conf, final_indices=final_set),
        "route": route,
        "p2_candidate_count": len(p2),
        "coarse_candidate_count": len(coarse),
    }


def assign_raw_candidates_to_gt(
    candidates: Sequence[Mapping[str, object]],
    gt_boxes: Sequence[Sequence[float]],
    gt_classes: Sequence[int],
    moderate_mask: Sequence[bool] | None = None,
    *,
    useful_iou: float = 0.5,
    evaluation_class: int | None = None,
) -> list[dict[str, object]]:
    """Permanently assign decoded candidates to the best same-class Moderate GT.

    Assignment is geometry-only and performed once at raw stage.  Wrong
    argmax candidates remain owned by the target GT so semantic score loss is
    not silently removed from the estimand.  Ties resolve to the lower GT
    index.
    """

    if len(gt_boxes) != len(gt_classes):
        raise ValueError("GT boxes/classes must have equal lengths")
    moderate = tuple(True for _ in gt_boxes) if moderate_mask is None else tuple(bool(v) for v in moderate_mask)
    if len(moderate) != len(gt_boxes):
        raise ValueError("moderate mask length mismatch")
    gt_tensors = [torch.as_tensor(tuple(float(v) for v in box), dtype=torch.float32) for box in gt_boxes]
    output: list[dict[str, object]] = []
    for candidate in candidates:
        box = torch.as_tensor(tuple(float(v) for v in candidate["box"]), dtype=torch.float32)
        best: tuple[float, int] | None = None
        for gt_index, (gt_box, gt_class, is_moderate) in enumerate(zip(gt_tensors, gt_classes, moderate)):
            if not is_moderate or (evaluation_class is not None and int(gt_class) != int(evaluation_class)):
                continue
            overlap = float(_iou(gt_box, box.unsqueeze(0))[0].item())
            if overlap <= useful_iou:
                continue
            if best is None or overlap > best[0] or (overlap == best[0] and gt_index < best[1]):
                best = (overlap, gt_index)
        row = dict(candidate)
        row["owner_gt_index"] = None if best is None else int(best[1])
        row["owner_iou"] = 0.0 if best is None else float(best[0])
        output.append(row)
    return output


def _candidate_gt_score(candidate: Mapping[str, object], gt_class: int) -> float:
    if "gt_score" in candidate:
        return float(candidate["gt_score"])
    scores = tuple(float(v) for v in candidate.get("class_scores", ()))
    return scores[gt_class] if 0 <= gt_class < len(scores) else 0.0


def _candidate_best_class(candidate: Mapping[str, object]) -> int:
    if "best_class" in candidate:
        return int(candidate["best_class"])
    scores = tuple(float(v) for v in candidate.get("class_scores", ()))
    return int(max(range(len(scores)), key=lambda index: scores[index])) if scores else -1


def _candidate_reaches(candidate: Mapping[str, object], stage_set: frozenset[str], *, conf: float, gt_class: int) -> bool:
    if _candidate_best_class(candidate) != int(gt_class) or not (_candidate_gt_score(candidate, gt_class) > conf):
        return False
    if "stage" in candidate:
        return str(candidate["stage"]) in stage_set
    key = "nms" if stage_set is N_STAGE_SET else "max_nms"
    if key in candidate:
        return bool(candidate[key])
    return False


def _group_only_nms_survives(
    index: int,
    level: str,
    p2_only: Sequence[int] | set[int],
    coarse_only: Sequence[int] | set[int],
) -> bool:
    """Return whether a candidate survives NMS in its level-only replay."""

    candidate_index = int(index)
    level_name = str(level)
    if level_name == "P2":
        return candidate_index in p2_only
    if level_name in {"P3", "P4", "P5"}:
        return candidate_index in coarse_only
    return False


def _attach_suppressor_box(item: Mapping[str, object], boxes_orig: torch.Tensor) -> dict[str, object]:
    """Persist the direct suppressor's original-image xyxy coordinates."""

    result = dict(item)
    suppressor_index = result.get("suppressor_index")
    if suppressor_index is None:
        return result
    index = int(suppressor_index)
    if boxes_orig.ndim != 2 or boxes_orig.shape[1] != 4 or index < 0 or index >= boxes_orig.shape[0]:
        raise ValueError("suppressor index is outside original-box tensor")
    result["suppressor_box"] = tuple(float(value) for value in boxes_orig[index].detach().cpu().tolist())
    return result


def _failure_reason(candidates: Sequence[Mapping[str, object]], *, conf: float, gt_class: int) -> str:
    if not candidates:
        return "no_raw_useful"
    correct = [item for item in candidates if _candidate_best_class(item) == int(gt_class)]
    if not correct:
        return "wrong_argmax"
    scored = [item for item in correct if _candidate_gt_score(item, gt_class) > conf]
    if not scored:
        return "conf_leq_threshold"
    return "global_max_nms_rank_gt_30000"


def _target_population(record: Mapping[str, object]) -> bool:
    if str(record.get("class_name", record.get("class", ""))) not in {"Cyclist", "2"} and int(record.get("class_id", -1)) != 2:
        return False
    if not bool(record.get("moderate_valid", record.get("moderate", False))):
        return False
    return bool(record.get("small_25_40", False) or record.get("far_gt_40m", False))


def _row_candidates(record: Mapping[str, object], side: str) -> list[Mapping[str, object]]:
    direct = record.get(f"{side}_candidates")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        return [item for item in direct if isinstance(item, Mapping)]
    return []


def _report_candidate_iou(candidate: Mapping[str, object], *, formal: bool) -> float:
    """Read report-time raw ownership IoU without hiding malformed formal rows."""

    if "iou" not in candidate and "owner_iou" not in candidate:
        if formal:
            raise ValueError("formal report candidate IoU is missing")
        return 0.0
    try:
        value = float(candidate["iou"] if "iou" in candidate else candidate["owner_iou"])
    except (TypeError, ValueError) as error:
        if formal:
            raise ValueError("formal report candidate IoU is invalid") from error
        return 0.0
    if not math.isfinite(value):
        if formal:
            raise ValueError("formal report candidate IoU must be finite")
        return 0.0
    return value


def _eligible_score_row(record: Mapping[str, object], *, conf: float) -> dict[str, object] | None:
    gt_class = int(record.get("class_id", 2))
    p2 = _row_candidates(record, "p2")
    coarse = _row_candidates(record, "coarse")
    if not p2 or not coarse:
        return None
    p2_s = [item for item in p2 if float(item.get("iou", item.get("owner_iou", 0.0))) > 0.5]
    coarse_s = [item for item in coarse if float(item.get("iou", item.get("owner_iou", 0.0))) > 0.5]
    if not p2_s or not coarse_s:
        return None
    p2_fail = not any(_candidate_reaches(item, S_STAGE_SET, conf=conf, gt_class=gt_class) for item in p2_s)
    coarse_fail = not any(_candidate_reaches(item, S_STAGE_SET, conf=conf, gt_class=gt_class) for item in coarse_s)
    p2_enters_nms = any(_candidate_reaches(item, S_STAGE_SET, conf=conf, gt_class=gt_class) for item in p2_s)
    coarse_enters_nms = any(_candidate_reaches(item, S_STAGE_SET, conf=conf, gt_class=gt_class) for item in coarse_s)
    p2_harm = bool(record.get("p2_only_keeps_useful", record.get("p2_only_kept", False)) and not record.get("full_nms_any_useful", record.get("full_nms_keeps_any_useful", False)) and record.get("p2_direct_suppressed_by_coarse", False))
    coarse_harm = bool(record.get("coarse_only_keeps_useful", record.get("coarse_only_kept", False)) and not record.get("full_nms_any_useful", record.get("full_nms_keeps_any_useful", False)) and record.get("coarse_direct_suppressed_by_p2", False))
    return {
        "image_id": str(record.get("image_id", "")),
        "p2_failure": p2_fail,
        "coarse_failure": coarse_fail,
        "p2_reason": _failure_reason(p2_s, conf=conf, gt_class=gt_class) if p2_fail else "none",
        "coarse_reason": _failure_reason(coarse_s, conf=conf, gt_class=gt_class) if coarse_fail else "none",
        "p2_best_gt_score": max((_candidate_gt_score(item, gt_class) for item in p2_s), default=0.0),
        "coarse_best_gt_score": max((_candidate_gt_score(item, gt_class) for item in coarse_s), default=0.0),
        "score_margin": max((_candidate_gt_score(item, gt_class) for item in p2_s), default=0.0) - max((_candidate_gt_score(item, gt_class) for item in coarse_s), default=0.0),
        "p2_strict_ranks": [int(item.get("strict_rank", -1)) for item in p2_s],
        "coarse_strict_ranks": [int(item.get("strict_rank", -1)) for item in coarse_s],
        "p2_tie": any(int(item.get("tie_group_size", 1)) > 1 for item in p2_s),
        "coarse_tie": any(int(item.get("tie_group_size", 1)) > 1 for item in coarse_s),
        "p2_enters_nms": p2_enters_nms,
        "coarse_enters_nms": coarse_enters_nms,
        "p2_nms_useful": any(_candidate_reaches(item, N_STAGE_SET, conf=conf, gt_class=gt_class) for item in p2_s),
        "coarse_nms_useful": any(_candidate_reaches(item, N_STAGE_SET, conf=conf, gt_class=gt_class) for item in coarse_s),
        "p2_harm": p2_harm,
        "coarse_harm": coarse_harm,
    }


def _candidate_identity_payload(candidate: Mapping[str, object], *, default_level: str, evaluation_class: int) -> dict[str, object]:
    """Return the immutable fields used to deduplicate a candidate identity.

    A decoded candidate may occur in more than one GT row after the raw
    geometry assignment.  Ownership metadata is intentionally excluded: it
    is row-local, while the class/level/flat-index candidate identity and its
    decoded/scoring fields must be identical in every occurrence.
    """

    level = str(candidate.get("level", default_level))
    index = int(candidate.get("index", -1))
    if index < 0:
        raise ValueError("greedy candidate index must be non-negative")
    payload: dict[str, object] = {
        "index": index,
        "level": level,
        "box": tuple(float(value) for value in candidate.get("box", ())),
        "input_box": tuple(float(value) for value in candidate.get("input_box", ())),
        "class_scores": tuple(float(value) for value in candidate.get("class_scores", ())),
        "best_class": int(candidate.get("best_class", _candidate_best_class(candidate))),
        "score": float(candidate.get("score", candidate.get("gt_score", 0.0))),
        "gt_score": float(candidate.get("gt_score", _candidate_gt_score(candidate, int(evaluation_class)))),
        "stage": str(candidate.get("stage", "raw")),
        "strict_rank": int(candidate.get("strict_rank", -1)),
        "tie_group_size": int(candidate.get("tie_group_size", 1)),
        "group_only_nms_survives": bool(candidate.get("group_only_nms_survives", False)),
        "suppressor_index": candidate.get("suppressor_index"),
        "suppressor_level": str(candidate.get("suppressor_level", "")),
        "suppressor_class": candidate.get("suppressor_class"),
        "suppressor_box": tuple(float(value) for value in candidate["suppressor_box"]) if isinstance(candidate.get("suppressor_box"), Sequence) and not isinstance(candidate.get("suppressor_box"), (str, bytes)) else None,
        "pair_iou": float(candidate.get("pair_iou", 0.0)),
    }
    return payload


def _dedupe_class_candidates(
    gt_rows: Sequence[Mapping[str, object]],
    *,
    side: str,
) -> dict[int, list[dict[str, object]]]:
    """Deduplicate candidates by ``(evaluation_class, level, flat_index)``.

    The same candidate may be repeated in rows for several GTs.  Repetition
    is harmless only when all immutable candidate fields agree; a mismatch is
    a fail-closed identity error rather than silently choosing one occurrence.
    """

    side_name = str(side)
    default_level = "P2" if side_name in {"p2", "P2", "p2_candidates"} else "P3"
    by_key: dict[tuple[int, str, int], dict[str, object]] = {}
    for row in gt_rows:
        if not isinstance(row, Mapping):
            raise ValueError("greedy GT rows must be mappings")
        if not bool(row.get("moderate_valid", row.get("moderate", False))):
            continue
        gt_class = int(row.get("class_id", -1))
        direct = row.get(side_name if side_name.endswith("_candidates") else f"{side_name}_candidates", ())
        if not isinstance(direct, Sequence) or isinstance(direct, (str, bytes)):
            continue
        for raw in direct:
            if not isinstance(raw, Mapping):
                raise ValueError("greedy candidates must be mappings")
            payload = _candidate_identity_payload(raw, default_level=default_level, evaluation_class=gt_class)
            level = str(payload["level"])
            index = int(payload["index"])
            key = (gt_class, level, index)
            prior = by_key.get(key)
            if prior is not None and _canonical_sha(prior) != _canonical_sha(payload):
                raise ValueError(f"inconsistent duplicate greedy candidate identity: {key}")
            by_key.setdefault(key, payload)
    output: dict[int, list[dict[str, object]]] = {}
    for (gt_class, _level, _index), candidate in by_key.items():
        output.setdefault(gt_class, []).append(dict(candidate))
    for values in output.values():
        values.sort(key=lambda item: (int(item["index"]), str(item["level"])))
    return output


def _candidate_stage_allowed(candidate: Mapping[str, object], stage: object) -> bool:
    if stage is None:
        return True
    if isinstance(stage, str):
        allowed = {stage}
    elif isinstance(stage, Sequence) and not isinstance(stage, (str, bytes)):
        allowed = {str(value) for value in stage}
    else:
        allowed = {str(stage)}
    return str(candidate.get("stage", "raw")) in allowed


def _greedy_assign_candidates(
    gt_rows: Sequence[Mapping[str, object]],
    *,
    side: str,
    stage: object = None,
    useful_iou: float = 0.5,
) -> dict[str, object]:
    """Assign score-ranked candidates to same-class Moderate GTs once.

    Score ties are not broken by flat index.  A tie is ambiguous only when
    tied candidates have overlapping currently-unmatched GT option sets;
    disjoint ties are serialized by the stable identity key.
    """

    valid_rows = [
        row for row in gt_rows
        if isinstance(row, Mapping) and bool(row.get("moderate_valid", row.get("moderate", False)))
        and "gt_box" in row and "gt_index" in row
    ]
    deduped = _dedupe_class_candidates(valid_rows, side=side)
    gt_by_class: dict[int, list[tuple[int, torch.Tensor]]] = {}
    for row in valid_rows:
        cls = int(row.get("class_id", -1))
        gt_by_class.setdefault(cls, []).append((int(row["gt_index"]), torch.as_tensor(tuple(float(value) for value in row["gt_box"]), dtype=torch.float32)))
    for values in gt_by_class.values():
        values.sort(key=lambda item: item[0])
    matched: set[int] = set()
    matches: list[dict[str, object]] = []
    ambiguous_groups: list[dict[str, object]] = []
    for gt_class, candidates in sorted(deduped.items()):
        gt_values = gt_by_class.get(int(gt_class), [])
        if not gt_values:
            continue
        filtered = [item for item in candidates if _candidate_stage_allowed(item, stage)]
        scored = [(float(item.get("gt_score", 0.0)), item) for item in filtered if math.isfinite(float(item.get("gt_score", 0.0)))]
        scored.sort(key=lambda pair: -pair[0])
        position = 0
        while position < len(scored):
            end = position + 1
            score = scored[position][0]
            while end < len(scored) and scored[end][0] == score:
                end += 1
            tie_candidates = [item for _value, item in scored[position:end]]
            option_sets: list[set[int]] = []
            for item in tie_candidates:
                box = torch.as_tensor(tuple(float(value) for value in item["box"]), dtype=torch.float32)
                options = {
                    gt_index for gt_index, gt_box in gt_values
                    if gt_index not in matched and float(_iou(gt_box, box.unsqueeze(0))[0].item()) > float(useful_iou)
                }
                option_sets.append(options)
            overlap = any(left.intersection(right) for left_index, left in enumerate(option_sets) for right in option_sets[left_index + 1:])
            if overlap:
                ambiguous_groups.append({"class_id": int(gt_class), "score": score, "candidate_indices": [int(item["index"]) for item in tie_candidates], "option_sets": [sorted(values) for values in option_sets]})
                return {"state": "TIE_AMBIGUOUS", "covered": sorted(matched), "matches": matches, "ambiguous_groups": ambiguous_groups}
            # Disjoint ties are deterministic without pretending that one
            # candidate outranks another: canonical identity only serializes.
            tie_candidates.sort(key=lambda item: (str(item.get("level", "")), int(item["index"])))
            for item in tie_candidates:
                box = torch.as_tensor(tuple(float(value) for value in item["box"]), dtype=torch.float32)
                options: list[tuple[float, int]] = []
                for gt_index, gt_box in gt_values:
                    if gt_index in matched:
                        continue
                    overlap_value = float(_iou(gt_box, box.unsqueeze(0))[0].item())
                    if overlap_value > float(useful_iou):
                        options.append((overlap_value, gt_index))
                if options:
                    overlap_value, gt_index = max(options, key=lambda value: (value[0], -value[1]))
                    matched.add(gt_index)
                    matches.append({"candidate_index": int(item["index"]), "level": str(item.get("level", "")), "gt_index": int(gt_index), "iou": overlap_value, "score": float(item.get("gt_score", 0.0)), "class_id": int(gt_class), "candidate": dict(item)})
            position = end
    return {"state": "PASS", "covered": sorted(matched), "matches": matches, "ambiguous_groups": ambiguous_groups}


def _greedy_assignment_reaches(
    assignment: Mapping[str, object],
    *,
    gt_index: int,
    stage: frozenset[str] | set[str] | Sequence[str],
    gt_class: int,
    conf: float,
) -> bool:
    """Evaluate stage reachability on the already-fixed raw match identity."""

    for match in assignment.get("matches", ()):
        if not isinstance(match, Mapping) or int(match.get("gt_index", -1)) != int(gt_index):
            continue
        candidate = match.get("candidate")
        return isinstance(candidate, Mapping) and _candidate_reaches(candidate, frozenset(str(value) for value in stage), conf=conf, gt_class=gt_class)
    return False


def standard_greedy_one_to_one_coverage(
    gt_rows: Sequence[Mapping[str, object]],
    *,
    side: str,
    useful_iou: float = 0.5,
) -> set[int]:
    """KITTI-style score-greedy one-to-one GT coverage sensitivity.

    Candidate ownership is deliberately *not* used here.  All candidate rows
    from the requested level group compete globally by descending GT-class
    score, then flat index, and each Moderate GT can be covered once.  The
    function is a sensitivity check only; it never replaces the frozen
    class-indexed ownership estimand.
    """

    result = _greedy_assign_candidates(gt_rows, side=side, useful_iou=useful_iou)
    return set(int(value) for value in result.get("covered", ()))


def _greedy_match_candidate(
    gt_rows: Sequence[Mapping[str, object]],
    *,
    side: str,
    match: Mapping[str, object],
) -> Mapping[str, object] | None:
    target_index = int(match.get("candidate_index", -1))
    target_level = str(match.get("level", ""))
    target_class = int(match.get("class_id", -1))
    for row in gt_rows:
        if int(row.get("class_id", -1)) != target_class:
            continue
        for item in _row_candidates(row, side):
            if int(item.get("index", -1)) == target_index and str(item.get("level", "")) == target_level:
                return item
    return None


def _greedy_full_useful_gt_indices(
    gt_rows: Sequence[Mapping[str, object]],
    *,
    p2_assignment: Mapping[str, object],
    coarse_assignment: Mapping[str, object],
) -> set[int]:
    """Coverage from fixed raw matches that actually survive full NMS."""

    covered: set[int] = set()
    for assignment in (p2_assignment, coarse_assignment):
        for match in assignment.get("matches", ()):
            if not isinstance(match, Mapping):
                continue
            candidate = match.get("candidate")
            if isinstance(candidate, Mapping) and _candidate_reaches(candidate, N_STAGE_SET, conf=0.001, gt_class=int(match.get("class_id", -1))):
                covered.add(int(match.get("gt_index", -1)))
    return covered


def _greedy_n_harm_gt_indices(
    gt_rows: Sequence[Mapping[str, object]],
    *,
    side: str,
    assignment: Mapping[str, object],
    full_useful_gt_indices: set[int] | None = None,
) -> set[int]:
    """Return only direct opposite-level, same-class greedy NMS harms.

    ``full_nms_any_useful`` is required to be explicitly false, so a harmless
    swap to another useful detection cannot be mislabeled as harm.  The
    suppressor IoU is recomputed from its persisted original-image box and
    the greedy-matched GT box rather than trusting a row-local scalar.
    """

    opposite = {"P2"} if side in {"coarse", "coarse_candidates"} else {"P3", "P4", "P5"}
    rows_by_gt = {int(row.get("gt_index", -1)): row for row in gt_rows if isinstance(row, Mapping)}
    harmed: set[int] = set()
    for match in assignment.get("matches", ()):
        if not isinstance(match, Mapping):
            continue
        gt_index = int(match.get("gt_index", -1))
        row = rows_by_gt.get(gt_index)
        if row is None:
            continue
        if full_useful_gt_indices is not None and gt_index in full_useful_gt_indices:
            continue
        candidate = _greedy_match_candidate(gt_rows, side=side, match=match)
        if candidate is None or not bool(candidate.get("group_only_nms_survives", False)):
            continue
        suppressor_index = candidate.get("suppressor_index")
        if not bool(candidate.get("suppressed", False)) or not isinstance(suppressor_index, int) or int(suppressor_index) < 0:
            continue
        gt_class = int(row.get("class_id", -1))
        candidate_class = int(candidate.get("best_class", _candidate_best_class(candidate)))
        suppressor_class = candidate.get("suppressor_class")
        if candidate_class != gt_class or suppressor_class is None or int(suppressor_class) != gt_class:
            continue
        suppressor_level = str(candidate.get("suppressor_level", ""))
        if suppressor_level not in opposite:
            continue
        pair_iou = float(candidate.get("pair_iou", 0.0))
        if not math.isfinite(pair_iou) or pair_iou <= 0.7:
            continue
        suppressor_box = candidate.get("suppressor_box")
        gt_box = row.get("gt_box")
        if not isinstance(suppressor_box, Sequence) or isinstance(suppressor_box, (str, bytes)) or not isinstance(gt_box, Sequence) or isinstance(gt_box, (str, bytes)):
            continue
        recomputed = float(_iou(torch.as_tensor(tuple(float(value) for value in gt_box), dtype=torch.float32), torch.as_tensor(tuple(float(value) for value in suppressor_box), dtype=torch.float32).unsqueeze(0))[0].item())
        if not math.isfinite(recomputed):
            continue
        # If the suppressor itself is a useful same-GT match, the full run
        # would not be a loss; the explicit full flag above normally excludes
        # this, while this check guards synthetic/legacy rows.
        if recomputed > 0.5:
            continue
        harmed.add(gt_index)
    return harmed


def _greedy_sensitivity(rows_by_image: Mapping[str, Sequence[Mapping[str, object]]]) -> dict[str, object]:
    """Compute the score-greedy sensitivity counterpart of primary S/N.

    This function is deliberately conservative: it uses the same per-GT
    raw-useful and stage-entry predicates as the primary estimator, but runs a
    one-to-one assignment on each level group.  Any contested exact-score tie
    is a veto rather than an arbitrary flat-index decision.  Sparse fixtures
    therefore return ``NOT_ESTIMABLE`` and cannot promote a primary result.
    """

    frames: list[dict[str, object]] = []
    tie_images: list[str] = []
    for image_id, raw_rows in rows_by_image.items():
        # Primary sensitivity is the Cyclist estimand.  Keep every Moderate
        # Cyclist in the assignment pool (target and non-target) so matching
        # competition is faithful, but keep Pedestrian rows in a separate
        # future secondary rather than letting their ties veto Cyclist.
        gt_rows = [row for row in raw_rows if isinstance(row, Mapping) and bool(row.get("moderate_valid", row.get("moderate", False))) and int(row.get("class_id", -1)) == 2]
        # One raw assignment per level group.  All later S/N flags follow this
        # candidate identity; no stage-specific re-matching is permitted.
        p2_raw = _greedy_assign_candidates(gt_rows, side="p2", stage=None)
        coarse_raw = _greedy_assign_candidates(gt_rows, side="coarse", stage=None)
        if p2_raw.get("state") == "TIE_AMBIGUOUS" or coarse_raw.get("state") == "TIE_AMBIGUOUS":
            tie_images.append(str(image_id))
            frames.append({"image_id": str(image_id), "s_delta": 0, "s_denominator": 0, "s_discordant": 0, "n_delta": 0, "n_denominator": 0, "n_discordant": 0})
            continue
        p2_matches = {int(item.get("gt_index", -1)): item for item in p2_raw.get("matches", ()) if isinstance(item, Mapping)}
        coarse_matches = {int(item.get("gt_index", -1)): item for item in coarse_raw.get("matches", ()) if isinstance(item, Mapping)}
        full_useful_indices = _greedy_full_useful_gt_indices(gt_rows, p2_assignment=p2_raw, coarse_assignment=coarse_raw)
        p2_harm_indices = _greedy_n_harm_gt_indices(gt_rows, side="p2", assignment=p2_raw, full_useful_gt_indices=full_useful_indices)
        coarse_harm_indices = _greedy_n_harm_gt_indices(gt_rows, side="coarse", assignment=coarse_raw, full_useful_gt_indices=full_useful_indices)
        s_p2_failure = s_coarse_failure = s_discordant = s_denominator = 0
        n_p2_loss = n_coarse_loss = n_discordant = n_denominator = 0
        for row in gt_rows:
            if not bool(row.get("moderate_valid", row.get("moderate", False))):
                continue
            if not _target_population(row):
                continue
            gt_index = int(row.get("gt_index", -1))
            # S eligibility is the intersection of both raw greedy maps, not
            # merely the presence of row-local owner candidates.
            if gt_index not in p2_matches or gt_index not in coarse_matches:
                continue
            # S eligibility is identical to primary: both groups have raw
            # useful candidates.  The raw assignment has already fixed the
            # one candidate identity used for both stage comparisons.
            s_denominator += 1
            gt_class = int(row.get("class_id", 2))
            p2_failure = not _greedy_assignment_reaches(p2_raw, gt_index=gt_index, stage=S_STAGE_SET, gt_class=gt_class, conf=0.001)
            coarse_failure = not _greedy_assignment_reaches(coarse_raw, gt_index=gt_index, stage=S_STAGE_SET, gt_class=gt_class, conf=0.001)
            s_p2_failure += int(p2_failure)
            s_coarse_failure += int(coarse_failure)
            s_discordant += int(p2_failure != coarse_failure)
            p2_enters = _greedy_assignment_reaches(p2_raw, gt_index=gt_index, stage=S_STAGE_SET, gt_class=gt_class, conf=0.001)
            coarse_enters = _greedy_assignment_reaches(coarse_raw, gt_index=gt_index, stage=S_STAGE_SET, gt_class=gt_class, conf=0.001)
            # N denominator follows primary S_pre: both groups enter NMS.
            if p2_enters and coarse_enters:
                n_denominator += 1
                p2_loss = gt_index in p2_harm_indices
                coarse_loss = gt_index in coarse_harm_indices
                n_p2_loss += int(p2_loss)
                n_coarse_loss += int(coarse_loss)
                n_discordant += int(p2_loss != coarse_loss)
        frames.append({"image_id": str(image_id), "s_delta": s_p2_failure - s_coarse_failure, "s_p2_failure": s_p2_failure, "s_coarse_failure": s_coarse_failure, "s_denominator": s_denominator, "s_discordant": s_discordant, "n_delta": n_p2_loss - n_coarse_loss, "n_p2_loss": n_p2_loss, "n_coarse_loss": n_coarse_loss, "n_denominator": n_denominator, "n_discordant": n_discordant})

    def estimate(numerator: str, denominator: str, discordant: str) -> dict[str, object]:
        eligible = sum(int(frame.get(denominator, 0)) for frame in frames)
        images = sum(int(frame.get(denominator, 0)) > 0 for frame in frames)
        discord = sum(int(frame.get(discordant, 0)) for frame in frames)
        result: dict[str, object] = {"eligible_gt": eligible, "eligible_images": images, "discordant": discord}
        if eligible < 30 or images < 20 or discord < 10 or tie_images:
            result.update({"state": "TIE_AMBIGUOUS" if tie_images else "NOT_ESTIMABLE", "reason": "minimum denominator/image/discordance gate" if not tie_images else "contested exact score tie", "observed": None, "ci95": None, "ci97_5_bonferroni": None})
            return result
        try:
            boot = _cluster_bootstrap(frames, reps=10000, seed=20260812, numerator=numerator, denominator=denominator)
        except ValueError as error:
            result.update({"state": "NOT_ESTIMABLE", "reason": str(error)})
            return result
        result.update(boot)
        result["state"] = "estimable"
        result["passes"] = bool(float(boot["observed"]) >= 0.10 and float(boot["ci97_5_bonferroni"][0]) > 0.0)
        return result

    sensitivity_s = estimate("s_delta", "s_denominator", "s_discordant")
    sensitivity_n = estimate("n_delta", "n_denominator", "n_discordant")
    state = "PASS" if sensitivity_s.get("state") == "estimable" and sensitivity_n.get("state") == "estimable" else ("TIE_AMBIGUOUS" if tie_images else "NOT_ESTIMABLE")
    return {
        "state": state,
        "S": sensitivity_s,
        "N": sensitivity_n,
        "frames": frames,
        "direction_reversal": None,
        "route_authorized": False,
        "veto": state != "PASS",
        "must_pass_before_go": True,
        "tie_images": tie_images,
        "reps": 10000,
        "seed": 20260812,
    }


def _greedy_veto_for_decision(primary: Mapping[str, object], greedy: Mapping[str, object]) -> dict[str, object]:
    """Apply greedy sensitivity as a veto-only check on the primary route.

    The primary five-way decision is immutable.  A greedy result can reject a
    primary GO route when its corresponding endpoint is non-positive,
    non-estimable, tied, or reverses direction; it can never turn NO_GO or an
    ambiguous primary result into a GO.
    """

    decision = str(primary.get("decision", ""))
    target = "S" if decision == "GO_B_SCORE_OWNERSHIP" else "N" if decision == "GO_C_NMS_OWNERSHIP" else None
    direction_reversal: dict[str, bool] = {}
    for endpoint in ("S", "N"):
        primary_item = primary.get(endpoint)
        greedy_item = greedy.get(endpoint)
        p_value = primary_item.get("observed") if isinstance(primary_item, Mapping) else None
        g_value = greedy_item.get("observed") if isinstance(greedy_item, Mapping) else None
        try:
            p_float, g_float = float(p_value), float(g_value)
            direction_reversal[endpoint] = bool(math.isfinite(p_float) and math.isfinite(g_float) and p_float * g_float < 0.0)
        except (TypeError, ValueError):
            direction_reversal[endpoint] = False
    if target is None:
        return {"primary_decision": decision, "direction_reversal": direction_reversal, "greedy_veto": False, "route_authorized": False, "target_endpoint": None}
    endpoint_item = greedy.get(target)
    endpoint_state = endpoint_item.get("state") if isinstance(endpoint_item, Mapping) else None
    endpoint_value = endpoint_item.get("observed") if isinstance(endpoint_item, Mapping) else None
    try:
        valid_positive = bool(endpoint_state == "estimable" and math.isfinite(float(endpoint_value)) and float(endpoint_value) > 0.0)
    except (TypeError, ValueError):
        valid_positive = False
    # Only the endpoint corresponding to the primary route is authoritative;
    # an off-target S/N endpoint may be sparse without vetoing this route.
    greedy_veto = bool(not valid_positive or direction_reversal[target])
    return {"primary_decision": decision, "direction_reversal": direction_reversal, "greedy_veto": greedy_veto, "route_authorized": not greedy_veto, "target_endpoint": target}


def _primary_target_rows(records: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for record in records:
        nested = record.get("gt_rows") if isinstance(record.get("gt_rows"), Sequence) and not isinstance(record.get("gt_rows"), (str, bytes)) else (record,)
        for row in nested:
            if isinstance(row, Mapping) and _target_population(row):
                rows.append(row)
    return rows


def _denominator_ledger_and_first_loss(
    records: Sequence[Mapping[str, object]],
    fit_image_ids: Sequence[str],
    *,
    conf: float,
) -> dict[str, object]:
    """Build mutually exclusive primary target denominators and loss counts."""

    rows = _primary_target_rows(records)
    by_image: dict[str, list[Mapping[str, object]]] = {str(image_id): [] for image_id in fit_image_ids}
    for row in rows:
        by_image.setdefault(str(row.get("image_id", "")), []).append(row)
    ledger: dict[str, object] = {
        "target_gt": len(rows),
        "target_images": len({str(row.get("image_id", "")) for row in rows}),
        "raw_p2_useful_gt": 0,
        "raw_p2_useful_images": 0,
        "raw_coarse_useful_gt": 0,
        "raw_coarse_useful_images": 0,
        "raw_both_gt": 0,
        "raw_both_images": 0,
        "p2_enters_nms_input_gt": 0,
        "p2_enters_nms_input_images": 0,
        "coarse_enters_nms_input_gt": 0,
        "coarse_enters_nms_input_images": 0,
        "both_s_pre_gt": 0,
        "both_s_pre_images": 0,
        "n_eligible_gt": 0,
        "n_eligible_images": 0,
        "exclusions": {
            "no_raw_both": {"gt": 0, "images": 0},
            "p2_missing_coarse_present": {"gt": 0, "images": 0},
            "coarse_missing_p2_present": {"gt": 0, "images": 0},
            "both_raw_neither_enters": {"gt": 0, "images": 0},
            "both_raw_p2_not_enter": {"gt": 0, "images": 0},
            "both_raw_coarse_not_enter": {"gt": 0, "images": 0},
        },
    }
    stage_image_sets: dict[str, set[str]] = {key: set() for key in (
        "raw_p2_useful", "raw_coarse_useful", "raw_both", "p2_enters_nms_input",
        "coarse_enters_nms_input", "both_s_pre", "n_eligible",
    )}
    exclusion_image_sets: dict[str, set[str]] = {key: set() for key in ledger["exclusions"]}
    first_loss = {
        "p2": {"wrong_argmax": 0, "conf_leq_threshold": 0, "global_max_nms_rank_gt_30000": 0},
        "coarse": {"wrong_argmax": 0, "conf_leq_threshold": 0, "global_max_nms_rank_gt_30000": 0},
    }
    tie_descriptive = {"p2": {"gt": 0, "images": 0}, "coarse": {"gt": 0, "images": 0}}
    tie_image_sets = {"p2": set(), "coarse": set()}
    def bump_stage(key: str) -> None:
        ledger[f"{key}_gt"] = int(ledger.get(f"{key}_gt", 0)) + 1
        stage_image_sets[key].add(str(current_image))
    def bump_exclusion(key: str) -> None:
        bucket = ledger["exclusions"][key]
        bucket["gt"] += 1
        exclusion_image_sets[key].add(str(current_image))
    for row in rows:
        current_image = str(row.get("image_id", ""))
        gt_class = int(row.get("class_id", 2))
        sides: dict[str, list[Mapping[str, object]]] = {}
        for side in ("p2", "coarse"):
            candidates = _row_candidates(row, side)
            useful = [item for item in candidates if float(item.get("owner_iou", item.get("iou", 0.0))) > 0.5]
            sides[side] = useful
            raw_key = "raw_p2_useful" if side == "p2" else "raw_coarse_useful"
            if useful:
                bump_stage(raw_key)
                enters = any(_candidate_reaches(item, S_STAGE_SET, conf=conf, gt_class=gt_class) for item in useful)
                enter_key = "p2_enters_nms_input" if side == "p2" else "coarse_enters_nms_input"
                if enters:
                    bump_stage(enter_key)
                if not enters:
                    reason = _failure_reason(useful, conf=conf, gt_class=gt_class)
                    if reason != "no_raw_useful":
                        first_loss[side][reason] = int(first_loss[side].get(reason, 0)) + 1
                if any(int(item.get("tie_group_size", 1)) > 1 for item in useful):
                    tie_descriptive[side]["gt"] += 1
                    tie_image_sets[side].add(current_image)
            # no_raw_useful intentionally remains ledger-only.
        p2_has, coarse_has = bool(sides["p2"]), bool(sides["coarse"])
        p2_enters = any(_candidate_reaches(item, S_STAGE_SET, conf=conf, gt_class=gt_class) for item in sides["p2"])
        coarse_enters = any(_candidate_reaches(item, S_STAGE_SET, conf=conf, gt_class=gt_class) for item in sides["coarse"])
        if p2_has and coarse_has:
            bump_stage("raw_both")
            if p2_enters and coarse_enters:
                bump_stage("both_s_pre")
                bump_stage("n_eligible")
                continue
            if not p2_enters and not coarse_enters:
                exclusion = "both_raw_neither_enters"
            elif not p2_enters:
                exclusion = "both_raw_p2_not_enter"
            else:
                exclusion = "both_raw_coarse_not_enter"
            bump_exclusion(exclusion)
        elif not p2_has and not coarse_has:
            bump_exclusion("no_raw_both")
        elif not p2_has:
            bump_exclusion("p2_missing_coarse_present")
        else:
            bump_exclusion("coarse_missing_p2_present")
    for key, image_set in stage_image_sets.items():
        ledger[f"{key}_images"] = len(image_set)
    for key, image_set in exclusion_image_sets.items():
        ledger["exclusions"][key]["images"] = len(image_set)
    for side, image_set in tie_image_sets.items():
        tie_descriptive[side]["images"] = len(image_set)
    return {"ledger": ledger, "first_loss": first_loss, "tie_descriptive": tie_descriptive}


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(value) for value in ordered):
        raise ValueError("bootstrap values must be finite and non-empty")
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _continuous_summary_stats(values: Sequence[float]) -> dict[str, object]:
    numbers = tuple(float(value) for value in values)
    if not numbers or any(not math.isfinite(value) for value in numbers):
        raise ValueError("continuous summary values must be finite and non-empty")
    return {
        "n": len(numbers),
        "mean": sum(numbers) / len(numbers),
        "median": _percentile(numbers, 0.5),
        "q25": _percentile(numbers, 0.25),
        "q75": _percentile(numbers, 0.75),
    }


def _continuous_target_rows(records: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for record in records:
        nested = record.get("gt_rows")
        source = nested if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)) else (record,)
        for row in source:
            if isinstance(row, Mapping) and _target_population(row):
                rows.append(row)
    return rows


def _continuous_selected_candidate(
    row: Mapping[str, object],
    *,
    side: str,
) -> Mapping[str, object] | None:
    """Select one raw useful candidate by evaluation-class score without tie breaking."""

    candidates = _row_candidates(row, side)
    if not candidates:
        return None
    formal = "class_id" in row and "gt_index" in row
    gt_class = int(row.get("class_id", 2))
    useful: list[tuple[float, Mapping[str, object]]] = []
    for candidate in candidates:
        overlap = _report_candidate_iou(candidate, formal=formal)
        if overlap <= 0.5:
            continue
        has_scores = "class_scores" in candidate
        if formal and not has_scores:
            raise ValueError("continuous formal candidate class_scores are missing")
        if not (formal and has_scores):
            # Focused pre-formal fixtures have neither the evaluation-class
            # score vector nor rank provenance, so this local addendum is not
            # estimable rather than retrofitting those records.
            return None
        raw_scores = candidate["class_scores"]
        if not isinstance(raw_scores, Sequence) or isinstance(raw_scores, (str, bytes)) or gt_class < 0 or gt_class >= len(raw_scores):
            raise ValueError("continuous formal candidate class_scores are invalid")
        try:
            scores = tuple(float(value) for value in raw_scores)
            score = scores[gt_class]
            raw_rank = float(candidate["strict_rank"])
            raw_tie = float(candidate["tie_group_size"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("continuous formal candidate rank/tie metadata is invalid") from error
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in scores):
            raise ValueError("continuous formal class scores must be finite and in [0, 1]")
        if not math.isfinite(raw_rank) or not math.isfinite(raw_tie) or not raw_rank.is_integer() or not raw_tie.is_integer():
            raise ValueError("continuous formal candidate rank/tie metadata is invalid")
        rank, tie = int(raw_rank), int(raw_tie)
        if rank < 1 or tie < 1:
            raise ValueError("continuous formal candidate rank/tie metadata must be positive")
        useful.append((score, candidate))
    if not useful:
        return None
    best_score = max(score for score, _ in useful)
    best = [candidate for score, candidate in useful if score == best_score]
    metadata = {(int(candidate["strict_rank"]), int(candidate["tie_group_size"])) for candidate in best}
    if len(metadata) != 1:
        raise ValueError("continuous exact GT-score tie metadata disagrees")
    return best[0]


def _continuous_support(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Report paired raw-score/rank descriptions without affecting decisions."""

    population = {
        "definition": "primary S-eligible Moderate Cyclist small_25_40 OR far_gt_40m; raw useful permanent-owner candidates on P2 and P3-P5",
        "eligible_gt": 0,
    }
    base: dict[str, object] = {
        "population": population,
        "decision_role": "descriptive_only",
        "state": "NOT_ESTIMABLE",
        "reason": "zero primary S-eligible rows with formal continuous provenance",
    }
    p2_scores: list[float] = []
    coarse_scores: list[float] = []
    p2_ranks: list[float] = []
    coarse_ranks: list[float] = []
    p2_ties: list[float] = []
    coarse_ties: list[float] = []
    for row in _continuous_target_rows(records):
        p2 = _continuous_selected_candidate(row, side="p2")
        coarse = _continuous_selected_candidate(row, side="coarse")
        if p2 is None or coarse is None:
            continue
        gt_class = int(row["class_id"])
        p2_scores.append(float(p2["class_scores"][gt_class]))
        coarse_scores.append(float(coarse["class_scores"][gt_class]))
        p2_ranks.append(float(int(p2["strict_rank"])))
        coarse_ranks.append(float(int(coarse["strict_rank"])))
        p2_ties.append(float(int(p2["tie_group_size"])))
        coarse_ties.append(float(int(coarse["tie_group_size"])))
    population["eligible_gt"] = len(p2_scores)
    if not p2_scores:
        return base
    epsilon = float(torch.finfo(torch.float32).eps)
    p2_clamped = [min(1.0 - epsilon, max(epsilon, score)) for score in p2_scores]
    coarse_clamped = [min(1.0 - epsilon, max(epsilon, score)) for score in coarse_scores]
    p2_clamp_count = sum(int(original != clamped) for original, clamped in zip(p2_scores, p2_clamped))
    coarse_clamp_count = sum(int(original != clamped) for original, clamped in zip(coarse_scores, coarse_clamped))
    logits = [math.log(score / (1.0 - score)) for score in p2_clamped]
    coarse_logits = [math.log(score / (1.0 - score)) for score in coarse_clamped]
    score_margins = [p2 - coarse for p2, coarse in zip(p2_scores, coarse_scores)]
    logit_margins = [p2 - coarse for p2, coarse in zip(logits, coarse_logits)]
    base.update({
        "state": "estimable",
        "reason": None,
        "p2_best_gt_class_score": _continuous_summary_stats(p2_scores),
        "combined_coarse_best_gt_class_score": _continuous_summary_stats(coarse_scores),
        "p2_strict_rank": _continuous_summary_stats(p2_ranks),
        "combined_coarse_strict_rank": _continuous_summary_stats(coarse_ranks),
        "p2_tie_group_size": _continuous_summary_stats(p2_ties),
        "combined_coarse_tie_group_size": _continuous_summary_stats(coarse_ties),
        "p2_minus_coarse_score_margin": _continuous_summary_stats(score_margins),
        "p2_minus_coarse_logit_margin": _continuous_summary_stats(logit_margins),
        "logit_epsilon": epsilon,
        "clamp_counts": {"p2": p2_clamp_count, "combined_coarse": coarse_clamp_count, "total": p2_clamp_count + coarse_clamp_count},
    })
    return base


def _cluster_bootstrap(
    frames: Sequence[Mapping[str, object]],
    *,
    reps: int,
    seed: int,
    numerator: str,
    denominator: str,
) -> dict[str, object]:
    if not frames:
        raise ValueError("zero aggregate denominator")
    if sum(int(frame.get(denominator, 0)) for frame in frames) <= 0:
        raise ValueError("zero aggregate denominator")
    observed = sum(float(frame.get(numerator, 0)) for frame in frames) / sum(int(frame.get(denominator, 0)) for frame in frames)
    generator = torch.Generator().manual_seed(int(seed))
    values: list[float] = []
    for _ in range(int(reps)):
        indices = torch.randint(0, len(frames), (len(frames),), generator=generator).tolist()
        den = sum(int(frames[index].get(denominator, 0)) for index in indices)
        if den <= 0:
            raise ValueError("zero bootstrap denominator")
        values.append(sum(float(frames[index].get(numerator, 0)) for index in indices) / den)
    return {"observed": observed, "ci95": [_percentile(values, 0.025), _percentile(values, 0.975)], "ci97_5_bonferroni": [_percentile(values, 0.0125), _percentile(values, 0.9875)], "bootstrap_replicates": int(reps), "bootstrap_seed": int(seed), "draws": values}


_DESCRIPTIVE_STRATA: tuple[tuple[str, str], ...] = (
    ("small_25_40", "25 < height_px <= 40"),
    ("large_gt_80", "height_px > 80"),
    ("near_0_20m", "0 < depth_m <= 20"),
    ("far_gt_40m", "depth_m > 40"),
)


def _descriptive_strata(
    records: Sequence[Mapping[str, object]],
    fit_image_ids: Sequence[str],
) -> dict[str, dict[str, dict[str, object]]]:
    """Describe independent Moderate Pedestrian/Cyclist geometry strata.

    These are descriptive slices only.  Their bounds are evaluated from the
    recorded GT geometry rather than the primary target booleans, and each
    slice keeps its own GT and unique-image denominator.  A GT may contribute
    to more than one slice (for example, small and far); no complement or
    ``else`` branch is used.
    """

    fit_set = {str(value) for value in fit_image_ids}
    image_sets: dict[str, dict[str, set[str]]] = {
        class_name: {stratum: set() for stratum, _ in _DESCRIPTIVE_STRATA}
        for class_name in ("Pedestrian", "Cyclist")
    }
    counts: dict[str, dict[str, int]] = {
        class_name: {stratum: 0 for stratum, _ in _DESCRIPTIVE_STRATA}
        for class_name in ("Pedestrian", "Cyclist")
    }
    class_names = {1: "Pedestrian", 2: "Cyclist"}
    for record in records:
        parent_image_id = str(record.get("image_id", ""))
        nested = record.get("gt_rows")
        formal_nested = isinstance(nested, Sequence) and not isinstance(nested, (str, bytes))
        rows = nested if formal_nested else (record,)
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            image_id = str(row.get("image_id", parent_image_id))
            if image_id not in fit_set or not bool(row.get("moderate_valid", row.get("moderate", False))):
                continue
            try:
                class_id = int(row.get("class_id", -1))
            except (TypeError, ValueError):
                continue
            class_name = class_names.get(class_id)
            if class_name is None:
                continue
            formal_row = formal_nested or ("gt_index" in row and "class_id" in row)
            try:
                height = float(row["height_px"])
                depth = float(row["depth_m"])
            except (KeyError, TypeError, ValueError) as error:
                if formal_row:
                    raise ValueError("descriptive strata geometry is missing or invalid") from error
                continue
            if not (math.isfinite(height) and math.isfinite(depth)):
                if formal_row:
                    raise ValueError("descriptive strata geometry is missing or invalid")
                continue
            memberships = {
                "small_25_40": 25.0 < height <= 40.0,
                "large_gt_80": height > 80.0,
                "near_0_20m": 0.0 < depth <= 20.0,
                "far_gt_40m": depth > 40.0,
            }
            for stratum, _definition in _DESCRIPTIVE_STRATA:
                if memberships[stratum]:
                    counts[class_name][stratum] += 1
                    if image_id:
                        image_sets[class_name][stratum].add(image_id)
    output: dict[str, dict[str, dict[str, object]]] = {}
    for class_name in ("Pedestrian", "Cyclist"):
        output[class_name] = {}
        for stratum, definition in _DESCRIPTIVE_STRATA:
            gt_count = counts[class_name][stratum]
            image_count = len(image_sets[class_name][stratum])
            item: dict[str, object] = {
                "class": class_name,
                "stratum": stratum,
                "definition": definition,
                "role": "secondary_descriptive",
                "mutually_exclusive": False,
                "gt": gt_count,
                "eligible_gt": gt_count,
                "unique_images": image_count,
                "images": image_count,
            }
            if gt_count == 0:
                item.update({"state": "NOT_ESTIMABLE", "reason": "zero slice denominator"})
            else:
                item["state"] = "estimable"
            output[class_name][stratum] = item
    return output


_NEGATIVE_CONTROL_CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("small_25_40_minus_large_gt_80", "small_25_40", "large_gt_80"),
    ("far_gt_40m_minus_near_0_20m", "far_gt_40m", "near_0_20m"),
)


def _negative_control_memberships(row: Mapping[str, object]) -> dict[str, bool] | None:
    """Evaluate the registered geometry slices without using legacy flags."""

    try:
        height = float(row["height_px"])
        depth = float(row["depth_m"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(height) and math.isfinite(depth)):
        raise ValueError("descriptive strata geometry is missing or invalid")
    return {
        "small_25_40": 25.0 < height <= 40.0,
        "large_gt_80": height > 80.0,
        "near_0_20m": 0.0 < depth <= 20.0,
        "far_gt_40m": depth > 40.0,
    }


def _negative_control_bootstrap(
    frames: Sequence[Mapping[str, object]],
    *,
    target_numerator: str,
    target_denominator: str,
    control_numerator: str,
    control_denominator: str,
    reps: int,
    seed: int,
) -> dict[str, object]:
    """Cluster bootstrap a target-minus-control ratio-of-sums contrast."""

    generator = torch.Generator().manual_seed(int(seed))
    draws: list[float] = []
    for _ in range(int(reps)):
        indices = torch.randint(0, len(frames), (len(frames),), generator=generator).tolist()
        target_den = sum(int(frames[index][target_denominator]) for index in indices)
        control_den = sum(int(frames[index][control_denominator]) for index in indices)
        if target_den <= 0 or control_den <= 0:
            raise ValueError("zero bootstrap contrast denominator")
        target_num = sum(int(frames[index][target_numerator]) for index in indices)
        control_num = sum(int(frames[index][control_numerator]) for index in indices)
        draws.append(target_num / target_den - control_num / control_den)
    return {
        "ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
        "bootstrap_replicates": int(reps),
        "bootstrap_seed": int(seed),
    }


def _negative_score_controls(
    records: Sequence[Mapping[str, object]],
    fit_image_ids: Sequence[str],
    *,
    reps: int,
    seed: int,
    conf: float,
) -> dict[str, object]:
    """Report non-decisional score-failure geometry contrasts by side/class."""

    fit_ids = tuple(str(value) for value in fit_image_ids)
    frames: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for class_name in ("Pedestrian", "Cyclist"):
        for side in ("P2", "combined_coarse"):
            frames[class_name, side] = {
                image_id: {"image_id": image_id, **{f"{stratum}_{field}": 0 for stratum, _ in _DESCRIPTIVE_STRATA for field in ("num", "den")}}
                for image_id in fit_ids
            }

    class_names = {1: "Pedestrian", 2: "Cyclist"}
    for record in records:
        parent_image_id = str(record.get("image_id", ""))
        nested = record.get("gt_rows")
        formal_nested = isinstance(nested, Sequence) and not isinstance(nested, (str, bytes))
        rows = nested if formal_nested else (record,)
        for row in rows:
            if not isinstance(row, Mapping) or not bool(row.get("moderate_valid", row.get("moderate", False))):
                continue
            image_id = str(row.get("image_id", parent_image_id))
            if image_id not in fit_ids:
                continue
            try:
                class_name = class_names.get(int(row.get("class_id", -1)))
            except (TypeError, ValueError):
                class_name = None
            if class_name is None:
                continue
            memberships = _negative_control_memberships(row)
            if memberships is None:
                if formal_nested or ("gt_index" in row and "class_id" in row):
                    raise ValueError("descriptive strata geometry is missing or invalid")
                continue
            gt_class = int(row["class_id"])
            for output_side, candidate_side in (("P2", "p2"), ("combined_coarse", "coarse")):
                formal = formal_nested or ("gt_index" in row and "class_id" in row)
                useful = [candidate for candidate in _row_candidates(row, candidate_side) if _report_candidate_iou(candidate, formal=formal) > 0.5]
                if not useful:
                    continue
                failure = not any(_candidate_reaches(candidate, S_STAGE_SET, conf=conf, gt_class=gt_class) for candidate in useful)
                frame = frames[class_name, output_side][image_id]
                for stratum, _definition in _DESCRIPTIVE_STRATA:
                    if memberships[stratum]:
                        frame[f"{stratum}_den"] = int(frame[f"{stratum}_den"]) + 1
                        frame[f"{stratum}_num"] = int(frame[f"{stratum}_num"]) + int(failure)

    output: dict[str, object] = {"decision_role": "auxiliary_only"}
    for class_name in ("Pedestrian", "Cyclist"):
        output[class_name] = {}
        for side in ("P2", "combined_coarse"):
            side_frames = list(frames[class_name, side].values())
            contrasts: dict[str, object] = {}
            for name, target_stratum, control_stratum in _NEGATIVE_CONTROL_CONTRASTS:
                target_num = sum(int(frame[f"{target_stratum}_num"]) for frame in side_frames)
                target_den = sum(int(frame[f"{target_stratum}_den"]) for frame in side_frames)
                control_num = sum(int(frame[f"{control_stratum}_num"]) for frame in side_frames)
                control_den = sum(int(frame[f"{control_stratum}_den"]) for frame in side_frames)
                target = {"stratum": target_stratum, "num": target_num, "den": target_den, "rate": None if target_den == 0 else target_num / target_den, "eligible_unique_images": sum(int(frame[f"{target_stratum}_den"]) > 0 for frame in side_frames)}
                control = {"stratum": control_stratum, "num": control_num, "den": control_den, "rate": None if control_den == 0 else control_num / control_den, "eligible_unique_images": sum(int(frame[f"{control_stratum}_den"]) > 0 for frame in side_frames)}
                result: dict[str, object] = {
                    "decision_role": "auxiliary_only",
                    "target": target,
                    "control": control,
                    "observed_rate_difference": None if target_den == 0 or control_den == 0 else float(target["rate"]) - float(control["rate"]),
                    "ci95": None,
                    "bootstrap_replicates": int(reps),
                    "bootstrap_seed": int(seed),
                    "all_fit_image_ids": list(fit_ids),
                }
                if target_den == 0 or control_den == 0:
                    result.update({"state": "NOT_ESTIMABLE", "reason": "zero original contrast denominator"})
                else:
                    try:
                        result.update(_negative_control_bootstrap(
                            side_frames,
                            target_numerator=f"{target_stratum}_num",
                            target_denominator=f"{target_stratum}_den",
                            control_numerator=f"{control_stratum}_num",
                            control_denominator=f"{control_stratum}_den",
                            reps=reps,
                            seed=seed,
                        ))
                    except ValueError as error:
                        result.update({"state": "NOT_ESTIMABLE", "reason": str(error)})
                    else:
                        result["state"] = "estimable"
                contrasts[name] = result
            output[class_name][side] = contrasts
    return output


def summarize_score_nms_estimands(
    records: Sequence[Mapping[str, object]],
    fit_image_ids: Sequence[str],
    *,
    reps: int = 10000,
    seed: int = 20260812,
    conf: float = 0.001,
) -> dict[str, object]:
    """Compute frozen paired-A S/N estimands and fail closed on sparse strata."""

    fit_ids = tuple(str(value) for value in fit_image_ids)
    if not fit_ids or len(fit_ids) != len(set(fit_ids)):
        raise ValueError("fit image IDs must be non-empty and unique")
    fit_set = set(fit_ids)
    seen_gt: set[tuple[str, int, int]] = set()
    for record in records:
        image_id = str(record.get("image_id", ""))
        if image_id and image_id not in fit_set:
            raise ValueError(f"score/NMS row references non-fit image: {image_id}")
        nested = record.get("gt_rows")
        candidates = nested if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)) else (record,)
        for row in candidates:
            if not isinstance(row, Mapping):
                raise ValueError("score/NMS GT rows must be mappings")
            row_image = str(row.get("image_id", image_id))
            if row_image not in fit_set:
                raise ValueError(f"score/NMS GT row references non-fit image: {row_image}")
            if "class_id" not in row or "gt_index" not in row:
                # Legacy focused fixtures are one GT per image.  Formal rows
                # always carry explicit class/index and therefore cannot hit
                # this default silently.
                if nested is not None:
                    raise ValueError("formal score/NMS GT row lacks class_id or gt_index")
                continue
            key = (row_image, int(row["class_id"]), int(row["gt_index"]))
            if key in seen_gt:
                raise ValueError(f"duplicate score/NMS GT row: {key}")
            seen_gt.add(key)

    rows = [row for row in (_eligible_score_row(record, conf=conf) for record in records if _target_population(record)) if row is not None]
    by_image: dict[str, list[dict[str, object]]] = {str(image_id): [] for image_id in fit_ids}
    for row in rows:
        by_image.setdefault(str(row["image_id"]), []).append(row)
    frames: list[dict[str, object]] = []
    for image_id in fit_ids:
        image_rows = by_image.get(str(image_id), [])
        s_delta = sum(int(bool(row["p2_failure"]) - bool(row["coarse_failure"])) for row in image_rows)
        s_discord = sum(int(bool(row["p2_failure"]) != bool(row["coarse_failure"])) for row in image_rows)
        # Eligibility is both groups entering the NMS input (after conf and
        # max_nms), not both surviving NMS; otherwise harmful suppression is
        # structurally excluded.
        n_rows = [row for row in image_rows if row["p2_enters_nms"] and row["coarse_enters_nms"]]
        p2_loss = sum(int(row["p2_harm"]) for row in n_rows)
        coarse_loss = sum(int(row["coarse_harm"]) for row in n_rows)
        n_discord = sum(int(bool(row["p2_harm"]) != bool(row["coarse_harm"])) for row in n_rows)
        frames.append({"image_id": str(image_id), "s_delta": s_delta, "s_p2_failure": sum(int(row["p2_failure"]) for row in image_rows), "s_coarse_failure": sum(int(row["coarse_failure"]) for row in image_rows), "s_denominator": len(image_rows), "s_discordant": s_discord, "n_delta": p2_loss - coarse_loss, "n_p2_loss": p2_loss, "n_coarse_loss": coarse_loss, "n_denominator": len(n_rows), "n_discordant": n_discord})

    def finish(name: str, numerator: str, denominator: str, discordant: str) -> dict[str, object]:
        denominator_value = sum(int(frame[denominator]) for frame in frames)
        images_value = sum(int(frame[denominator]) > 0 for frame in frames)
        discordant_value = sum(int(frame[discordant]) for frame in frames)
        base = {"estimand": name, "eligible_gt": denominator_value, "eligible_images": images_value, "discordant": discordant_value}
        if denominator_value < 30 or images_value < 20 or discordant_value < 10:
            base.update({"state": "NOT_ESTIMABLE", "reason": "minimum denominator/image/discordance gate"})
            return base
        try:
            result = _cluster_bootstrap(frames, reps=reps, seed=seed, numerator=numerator, denominator=denominator)
        except ValueError as error:
            base.update({"state": "NOT_ESTIMABLE", "reason": str(error)})
            return base
        base.update(result)
        base["state"] = "estimable"
        base["passes"] = bool(result["observed"] >= 0.10 and result["ci97_5_bonferroni"][0] > 0.0)
        return base

    score = finish("S", "s_delta", "s_denominator", "s_discordant")
    # N is a paired direction difference computed per bootstrap draw.
    n_base = finish("N_p2_minus_coarse", "n_delta", "n_denominator", "n_discordant")
    score_pass, n_pass = score.get("passes") is True, n_base.get("passes") is True
    if score.get("state") != "estimable" or n_base.get("state") != "estimable":
        decision = "NO_GO_INSUFFICIENT_EVIDENCE"
    elif score_pass and n_pass:
        decision = "NO_GO_AMBIGUOUS"
    elif score_pass:
        decision = "GO_B_SCORE_OWNERSHIP"
    elif n_pass:
        decision = "GO_C_NMS_OWNERSHIP"
    else:
        decision = "NO_GO_BC"
    descriptive_strata = _descriptive_strata(records, fit_ids)
    negative_controls = _negative_score_controls(records, fit_ids, reps=reps, seed=seed, conf=conf)
    raw_rows_by_image: dict[str, list[Mapping[str, object]]] = {image_id: [] for image_id in fit_ids}
    for record in records:
        image_id = str(record.get("image_id", ""))
        nested = record.get("gt_rows") if isinstance(record.get("gt_rows"), Sequence) and not isinstance(record.get("gt_rows"), (str, bytes)) else (record,)
        for row in nested:
            if isinstance(row, Mapping) and image_id in raw_rows_by_image:
                raw_rows_by_image[image_id].append(row)
    sensitivity = _greedy_sensitivity(raw_rows_by_image)
    sensitivity["must_pass_before_go"] = True
    primary = {"decision": decision, "S": score, "N": n_base}
    veto = _greedy_veto_for_decision(primary, sensitivity)
    sensitivity.update(veto)
    ledger_result = _denominator_ledger_and_first_loss(records, fit_ids, conf=conf)
    continuous_support = _continuous_support(records)
    return {"schema_version": 1, "target_population": "Moderate Cyclist small_25_40 OR far_gt_40m (deduplicated)", "useful_iou": ">0.5", "conf": conf, "bootstrap": {"reps": reps, "seed": seed, "all_fit_image_ids": list(fit_ids), "same_sampled_indices_for_S_and_N": True}, "frames": frames, "S": score, "N": n_base, "descriptive_strata": descriptive_strata, "negative_controls": negative_controls, "continuous_support": continuous_support, "greedy_one_to_one_sensitivity": sensitivity, "direction_reversal": veto["direction_reversal"], "greedy_veto": veto["greedy_veto"], "route_authorized": veto["route_authorized"], "decision": decision, "denominator_ledger": ledger_result["ledger"], "first_loss": ledger_result["first_loss"], "tie_descriptive": ledger_result["tie_descriptive"]}


def _seal_benchmark32_summary(summary: Mapping[str, object]) -> dict[str, object]:
    """Keep benchmark diagnostics while preventing a 32-image route decision."""

    sealed = dict(summary)
    computed_decision = sealed.get("decision")
    if not isinstance(computed_decision, str):
        raise ValueError("benchmark32 computed decision is invalid")
    sealed.update({
        "state": "benchmark32_nonformal",
        "evaluation_role": "benchmark32_nonformal",
        "benchmark_computed_decision": computed_decision,
        "decision": "BENCHMARK32_NOT_FOR_ROUTE_DECISION",
        "route_authorized": False,
        "benchmark_veto": True,
    })
    return sealed


LONG_COLUMNS = (
    "family", "population", "class", "stratum", "side", "contrast",
    "statistic", "value", "state", "numerator", "denominator",
    "eligible_images", "discordant", "ci_level", "ci_low", "ci_high",
    "reps", "seed", "decision_role",
)


def summary_long_rows(summary: Mapping[str, object]) -> list[dict[str, object]]:
    """Flatten only registered aggregate report families into fixed long rows.

    Per-image frames, bootstrap draw arrays, and ID lists intentionally remain
    JSON evidence.  Every registered scalar aggregate must either appear in a
    long row or make publication fail closed.
    """

    if not isinstance(summary, Mapping):
        raise ValueError("summary must be a mapping")
    if summary.get("state") == "smoke_not_evaluated":
        allowed = {
            "state", "processed_fit_count", "default_vs_audit_labels",
            "default_vs_audit_results", "non_interference",
            "default_reference_labels", "default_reference_boxes",
        }
        unexpected = set(summary) - allowed
        if unexpected:
            raise ValueError(f"unmapped registered aggregate statistic: {sorted(unexpected)[0]}")
        processed = summary.get("processed_fit_count")
        if isinstance(processed, bool) or not isinstance(processed, (int, float)) or not math.isfinite(float(processed)) or int(processed) != processed or int(processed) < 0:
            raise ValueError("nonfinite numeric output")
        def validate_gate(value: object) -> None:
            if isinstance(value, Mapping):
                if any(not isinstance(key, str) for key in value):
                    raise ValueError("unmapped registered aggregate statistic: invalid smoke gate mapping")
                for nested in value.values():
                    validate_gate(nested)
                return
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for nested in value:
                    validate_gate(nested)
                return
            if value is None or isinstance(value, (str, bool)):
                return
            if isinstance(value, (int, float)):
                if not math.isfinite(float(value)):
                    raise ValueError("nonfinite numeric output")
                return
            raise ValueError("unmapped registered aggregate statistic: invalid smoke gate value")

        for key in (
            "default_vs_audit_labels", "default_vs_audit_results", "non_interference",
            "default_reference_labels", "default_reference_boxes",
        ):
            if key in summary:
                gate = summary[key]
                allowed_gate_keys = {"state", "reason", "image_count", "result_count", "bitwise_boxes_data", "paired_rng_final_equal", "paired_model_post_equal", "mismatches", "reference"}
                if not isinstance(gate, Mapping) or set(gate) - allowed_gate_keys or gate.get("state") != "PASS":
                    raise ValueError(f"unmapped registered aggregate statistic: {key}")
                if "mismatches" in gate and (not isinstance(gate["mismatches"], Sequence) or isinstance(gate["mismatches"], (str, bytes))):
                    raise ValueError(f"unmapped registered aggregate statistic: {key}.mismatches")
                validate_gate(gate)
        return [dict(zip(LONG_COLUMNS, ("decision", "", "", "", "", "", "smoke_processed_fit_count", processed, "smoke_not_evaluated", "", "", "", "", "", "", "", "", "", "smoke_only")))]
    rows: list[dict[str, object]] = []
    consumed: set[tuple[str, ...]] = set()

    def row(
        family: str,
        statistic: str,
        value: object = None,
        *,
        path: tuple[str, ...],
        population: str = "",
        class_name: str = "",
        stratum: str = "",
        side: str = "",
        contrast: str = "",
        state: object = "",
        numerator: object = "",
        denominator: object = "",
        eligible_images: object = "",
        discordant: object = "",
        ci_level: object = "",
        ci_low: object = "",
        ci_high: object = "",
        reps: object = "",
        seed: object = "",
        decision_role: object = "",
    ) -> None:
        for number in (value, numerator, denominator, eligible_images, discordant, ci_low, ci_high, reps, seed):
            if isinstance(number, (int, float)) and not isinstance(number, bool) and not math.isfinite(float(number)):
                raise ValueError("nonfinite numeric output")
        rows.append(dict(zip(LONG_COLUMNS, (family, population, class_name, stratum, side, contrast, statistic, value, state, numerator, denominator, eligible_images, discordant, ci_level, ci_low, ci_high, reps, seed, decision_role))))
        consumed.add(path)

    def endpoint(family: str, side: str, item: object, path: tuple[str, ...], role: object = "") -> None:
        if not isinstance(item, Mapping):
            raise ValueError(f"unmapped registered aggregate statistic: {'.'.join(path)}")
        allowed = {"estimand", "eligible_gt", "eligible_images", "discordant", "state", "reason", "observed", "ci95", "ci97_5_bonferroni", "bootstrap_replicates", "bootstrap_seed", "draws", "passes"}
        unknown = set(item) - allowed
        if unknown:
            raise ValueError(f"unmapped registered aggregate statistic: {'.'.join(path + (sorted(unknown)[0],))}")
        state, denominator = item.get("state", ""), item.get("eligible_gt", "")
        common = dict(state=state, denominator=denominator, eligible_images=item.get("eligible_images", ""), discordant=item.get("discordant", ""), reps=item.get("bootstrap_replicates", ""), seed=item.get("bootstrap_seed", ""), decision_role=role)
        if "observed" in item:
            ci95 = item.get("ci95")
            ci975 = item.get("ci97_5_bonferroni")
            if ci95 is not None and (not isinstance(ci95, Sequence) or isinstance(ci95, (str, bytes)) or len(ci95) != 2):
                raise ValueError("unmapped registered aggregate statistic: invalid ci95")
            if ci975 is not None and (not isinstance(ci975, Sequence) or isinstance(ci975, (str, bytes)) or len(ci975) != 2):
                raise ValueError("unmapped registered aggregate statistic: invalid ci97_5_bonferroni")
            row(family, "observed", item["observed"], path=path + ("observed",), side=side, ci_level="95%", ci_low="" if ci95 is None else ci95[0], ci_high="" if ci95 is None else ci95[1], **common)
            if ci95 is not None:
                consumed.update({path + ("ci95", "0"), path + ("ci95", "1")})
            if ci975 is not None:
                row(family, "ci97_5_bonferroni_low", ci975[0], path=path + ("ci97_5_bonferroni", "0"), side=side, ci_level="97.5% Bonferroni", ci_low=ci975[0], ci_high=ci975[1], **common)
                consumed.add(path + ("ci97_5_bonferroni", "1"))
        for key in ("eligible_gt", "eligible_images", "discordant", "bootstrap_replicates", "bootstrap_seed", "passes"):
            if key in item:
                row(family, key, item[key], path=path + (key,), side=side, **common)
        consumed.update(path + (key,) for key in ("estimand", "state", "reason") if key in item)
        draws = item.get("draws")
        if draws is not None:
            if not isinstance(draws, Sequence) or isinstance(draws, (str, bytes)):
                raise ValueError("unmapped registered aggregate statistic: invalid draws")
            for draw in draws:
                if isinstance(draw, bool) or not isinstance(draw, (int, float)):
                    raise ValueError("unmapped registered aggregate statistic: invalid draws")
                if not math.isfinite(float(draw)):
                    raise ValueError("nonfinite numeric output")
            consumed.update(path + ("draws", str(index)) for index in range(len(draws)))

    endpoint("primary", "S", summary.get("S"), ("S",), "primary")
    endpoint("primary", "N", summary.get("N"), ("N",), "primary")
    greedy = summary.get("greedy_one_to_one_sensitivity")
    if not isinstance(greedy, Mapping):
        raise ValueError("unmapped registered aggregate statistic: greedy_one_to_one_sensitivity")
    endpoint("greedy", "S", greedy.get("S"), ("greedy_one_to_one_sensitivity", "S"), "veto_only")
    endpoint("greedy", "N", greedy.get("N"), ("greedy_one_to_one_sensitivity", "N"), "veto_only")
    for key in ("state", "reps", "seed", "must_pass_before_go", "veto", "route_authorized", "greedy_veto"):
        if key in greedy:
            if key == "route_authorized" and summary.get("evaluation_role") == "benchmark32_nonformal":
                consumed.add(("greedy_one_to_one_sensitivity", key))
            else:
                row("greedy", key, greedy[key], path=("greedy_one_to_one_sensitivity", key), state=greedy.get("state", ""), decision_role="veto_only")
    consumed.update(("greedy_one_to_one_sensitivity", key) for key in ("frames", "tie_images", "primary_decision", "target_endpoint") if key in greedy)
    if isinstance(greedy.get("direction_reversal"), Mapping):
        consumed.update(("greedy_one_to_one_sensitivity", "direction_reversal", str(key)) for key in greedy["direction_reversal"])

    ledger = summary.get("denominator_ledger")
    if not isinstance(ledger, Mapping):
        raise ValueError("unmapped registered aggregate statistic: denominator_ledger")
    ledger_keys = {
        "target_gt", "target_images", "raw_p2_useful_gt", "raw_p2_useful_images",
        "raw_coarse_useful_gt", "raw_coarse_useful_images", "raw_both_gt",
        "raw_both_images", "p2_enters_nms_input_gt", "p2_enters_nms_input_images",
        "coarse_enters_nms_input_gt", "coarse_enters_nms_input_images",
        "both_s_pre_gt", "both_s_pre_images", "n_eligible_gt", "n_eligible_images",
        "exclusions",
    }
    unexpected_ledger = set(ledger) - ledger_keys
    if unexpected_ledger:
        raise ValueError(f"unmapped registered aggregate statistic: denominator_ledger.{sorted(unexpected_ledger)[0]}")
    for key, value in ledger.items():
        if key == "exclusions":
            if not isinstance(value, Mapping):
                raise ValueError("unmapped registered aggregate statistic: denominator_ledger.exclusions")
            for exclusion, counts in value.items():
                if not isinstance(counts, Mapping):
                    raise ValueError(f"unmapped registered aggregate statistic: denominator_ledger.exclusions.{exclusion}")
                if set(counts) != {"gt", "images"}:
                    raise ValueError(f"unmapped registered aggregate statistic: denominator_ledger.exclusions.{exclusion}")
                for count_key, count_value in counts.items():
                    row("ledger", f"exclusions.{exclusion}.{count_key}", count_value, path=("denominator_ledger", "exclusions", str(exclusion), str(count_key)))
        else:
            row("ledger", str(key), value, path=("denominator_ledger", str(key)))

    for family, key in (("first_loss", "first_loss"), ("tie_descriptive", "tie_descriptive")):
        item = summary.get(key)
        if not isinstance(item, Mapping):
            raise ValueError(f"unmapped registered aggregate statistic: {key}")
        for side, values in item.items():
            if not isinstance(values, Mapping):
                raise ValueError(f"unmapped registered aggregate statistic: {key}.{side}")
            for statistic, value in values.items():
                row(family, str(statistic), value, path=(key, str(side), str(statistic)), side=str(side), decision_role="descriptive_only" if family == "tie_descriptive" else "")

    strata = summary.get("descriptive_strata")
    if not isinstance(strata, Mapping):
        raise ValueError("unmapped registered aggregate statistic: descriptive_strata")
    for class_name, strata_by_name in strata.items():
        if not isinstance(strata_by_name, Mapping):
            raise ValueError(f"unmapped registered aggregate statistic: descriptive_strata.{class_name}")
        for stratum, item in strata_by_name.items():
            if not isinstance(item, Mapping):
                raise ValueError(f"unmapped registered aggregate statistic: descriptive_strata.{class_name}.{stratum}")
            for statistic, value in item.items():
                if statistic in {"class", "stratum", "definition", "role", "mutually_exclusive", "state", "reason"}:
                    consumed.add(("descriptive_strata", str(class_name), str(stratum), str(statistic)))
                else:
                    row("descriptive_strata", str(statistic), value, path=("descriptive_strata", str(class_name), str(stratum), str(statistic)), class_name=str(class_name), stratum=str(stratum), state=item.get("state", ""), decision_role=item.get("role", "descriptive_only"))

    controls = summary.get("negative_controls")
    if not isinstance(controls, Mapping):
        raise ValueError("unmapped registered aggregate statistic: negative_controls")
    consumed.add(("negative_controls", "decision_role"))
    for class_name, sides in controls.items():
        if class_name == "decision_role":
            continue
        if not isinstance(sides, Mapping):
            raise ValueError(f"unmapped registered aggregate statistic: negative_controls.{class_name}")
        for side, contrasts in sides.items():
            if not isinstance(contrasts, Mapping):
                raise ValueError(f"unmapped registered aggregate statistic: negative_controls.{class_name}.{side}")
            for contrast, item in contrasts.items():
                if not isinstance(item, Mapping):
                    raise ValueError(f"unmapped registered aggregate statistic: negative_controls.{class_name}.{side}.{contrast}")
                state, role = item.get("state", ""), item.get("decision_role", controls.get("decision_role", ""))
                for group in ("target", "control"):
                    values = item.get(group)
                    if not isinstance(values, Mapping):
                        raise ValueError(f"unmapped registered aggregate statistic: negative_controls.{class_name}.{side}.{contrast}.{group}")
                    for statistic, value in values.items():
                        if statistic == "stratum":
                            consumed.add(("negative_controls", str(class_name), str(side), str(contrast), group, statistic))
                        else:
                            row("negative_controls", f"{group}_{statistic}", value, path=("negative_controls", str(class_name), str(side), str(contrast), group, statistic), class_name=str(class_name), stratum=str(values.get("stratum", "")), side=str(side), contrast=str(contrast), state=state, numerator=values.get("num", ""), denominator=values.get("den", ""), eligible_images=values.get("eligible_unique_images", ""), decision_role=role)
                for statistic in ("observed_rate_difference", "bootstrap_replicates", "bootstrap_seed"):
                    if statistic in item:
                        row("negative_controls", statistic, item[statistic], path=("negative_controls", str(class_name), str(side), str(contrast), statistic), class_name=str(class_name), side=str(side), contrast=str(contrast), state=state, ci_level="95%", ci_low="" if item.get("ci95") is None else item["ci95"][0], ci_high="" if item.get("ci95") is None else item["ci95"][1], reps=item.get("bootstrap_replicates", ""), seed=item.get("bootstrap_seed", ""), decision_role=role)
                consumed.update(("negative_controls", str(class_name), str(side), str(contrast), key) for key in ("state", "reason", "decision_role", "all_fit_image_ids") if key in item)
                if item.get("ci95") is not None:
                    consumed.update({
                        ("negative_controls", str(class_name), str(side), str(contrast), "ci95", "0"),
                        ("negative_controls", str(class_name), str(side), str(contrast), "ci95", "1"),
                    })

    support = summary.get("continuous_support")
    if not isinstance(support, Mapping):
        raise ValueError("unmapped registered aggregate statistic: continuous_support")
    support_state, support_role = support.get("state", ""), support.get("decision_role", "descriptive_only")
    for name, value in support.items():
        if name in {"state", "reason", "decision_role"}:
            consumed.add(("continuous_support", name))
        elif name == "population":
            if not isinstance(value, Mapping):
                raise ValueError("unmapped registered aggregate statistic: continuous_support.population")
            for statistic, number in value.items():
                if statistic == "definition":
                    consumed.add(("continuous_support", "population", statistic))
                else:
                    row("continuous_support", f"population_{statistic}", number, path=("continuous_support", "population", str(statistic)), state=support_state, decision_role=support_role)
        elif isinstance(value, Mapping):
            for statistic, number in value.items():
                row("continuous_support", f"{name}_{statistic}", number, path=("continuous_support", str(name), str(statistic)), state=support_state, decision_role=support_role)
        else:
            row("continuous_support", name, value, path=("continuous_support", str(name)), state=support_state, decision_role=support_role)

    benchmark_role = summary.get("evaluation_role")
    if benchmark_role not in (None, "benchmark32_nonformal"):
        raise ValueError("unmapped registered aggregate statistic: evaluation_role")
    decision_role = "benchmark32_nonformal" if benchmark_role == "benchmark32_nonformal" else "primary"
    decision_values: list[tuple[str, object]] = [
        ("primary_decision", summary.get("decision")),
        ("direction_reversal_S", summary.get("direction_reversal", {}).get("S") if isinstance(summary.get("direction_reversal"), Mapping) else None),
        ("direction_reversal_N", summary.get("direction_reversal", {}).get("N") if isinstance(summary.get("direction_reversal"), Mapping) else None),
        ("greedy_veto", summary.get("greedy_veto")),
        ("route_authorized", summary.get("route_authorized")),
    ]
    if benchmark_role == "benchmark32_nonformal":
        if summary.get("state") != "benchmark32_nonformal" or summary.get("decision") != "BENCHMARK32_NOT_FOR_ROUTE_DECISION" or summary.get("route_authorized") is not False or summary.get("benchmark_veto") is not True:
            raise ValueError("benchmark32 nonformal decision seal is invalid")
        decision_values.extend((("benchmark_computed_decision", summary.get("benchmark_computed_decision")), ("benchmark_veto", summary.get("benchmark_veto"))))
    for statistic, value in decision_values:
        if statistic == "primary_decision":
            row("decision", statistic, "", path=("decision",), state=value, decision_role=decision_role)
        elif statistic == "benchmark_computed_decision":
            row("decision", statistic, "", path=(statistic,), state=value, decision_role=decision_role)
        else:
            source = "direction_reversal" if statistic.startswith("direction_reversal_") else statistic
            leaf = statistic.rsplit("_", 1)[-1] if source == "direction_reversal" else source
            row("decision", statistic, value, path=(source, leaf) if source == "direction_reversal" else (source,), decision_role=decision_role)

    def numeric_paths(value: object, path: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
        if isinstance(value, Mapping):
            result: set[tuple[str, ...]] = set()
            for key, item in value.items():
                result.update(numeric_paths(item, path + (str(key),)))
            return result
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return set().union(*(numeric_paths(item, path + (str(index),)) for index, item in enumerate(value))) if value else set()
        return {path} if isinstance(value, (int, float, bool)) else set()

    registered = ("S", "N", "greedy_one_to_one_sensitivity", "denominator_ledger", "first_loss", "tie_descriptive", "descriptive_strata", "negative_controls", "continuous_support", "decision", "direction_reversal", "greedy_veto", "route_authorized", "state", "evaluation_role", "benchmark_computed_decision", "benchmark_veto")
    ignored_prefixes = (("S", "draws"), ("N", "draws"), ("greedy_one_to_one_sensitivity", "frames"), ("negative_controls",))
    numeric = set().union(*(numeric_paths(summary[name], (name,)) for name in registered if name in summary))
    for path in sorted(numeric):
        if path in consumed or path[:2] in ignored_prefixes or path[:1] == ("negative_controls",) and path[-1].isdigit() and "all_fit_image_ids" in path:
            continue
        raise ValueError(f"unmapped registered aggregate statistic: {'.'.join(path)}")
    seen: set[tuple[object, ...]] = set()
    for item in rows:
        key = tuple(item[column] for column in ("family", "population", "class", "stratum", "side", "contrast", "statistic"))
        if key in seen:
            raise ValueError("duplicate long CSV row key")
        seen.add(key)
    return sorted(rows, key=lambda item: tuple(str(item[column]) for column in LONG_COLUMNS[:7]))


def _summary_long_csv(summary: Mapping[str, object]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=LONG_COLUMNS, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(summary_long_rows(summary))
    return stream.getvalue().encode("utf-8")


def _stage_for_index(index: int, stage_indices: Mapping[str, torch.Tensor], *, conf_score: float, conf: float) -> str:
    def as_set(value: object) -> set[int]:
        if isinstance(value, set):
            return value
        if isinstance(value, torch.Tensor):
            return {int(item) for item in value.tolist()}
        return set()

    if index in as_set(stage_indices.get("final", torch.empty(0))):
        return "final"
    if index in as_set(stage_indices.get("nms", torch.empty(0))):
        return "nms"
    if index in as_set(stage_indices.get("max_nms", torch.empty(0))):
        return "max_nms"
    if conf_score > conf and index in as_set(stage_indices.get("conf", torch.empty(0))):
        return "conf"
    return "raw"


def _nms_subset_indices(
    decoded: torch.Tensor,
    indices: Sequence[int],
    *,
    conf: float,
    iou: float,
    max_nms: int,
    max_det: int,
) -> set[int]:
    """Use the registered NMS on a level-masked clone and preserve flat IDs."""

    if not indices:
        return set()
    masked = decoded.detach().clone()
    selected_tensor = torch.as_tensor(tuple(int(value) for value in indices), dtype=torch.long, device=decoded.device)
    keep = torch.zeros(int(decoded.shape[-1]), dtype=torch.bool, device=decoded.device)
    keep[selected_tensor] = True
    masked[:, 4:] = masked[:, 4:] * keep.view(1, 1, -1)
    replay = replay_nms_with_stages(masked, level_slices=(("all", 0, decoded.shape[-1]),), conf=conf, iou=iou, max_nms=max_nms, max_det=max_det, trace_suppression=False)
    return set(int(value) for value in replay.stage_indices[0]["nms"].tolist())


def _targeted_suppression_detail(
    *,
    candidate_index: int,
    decoded: torch.Tensor,
    class_scores: torch.Tensor,
    boxes_input: torch.Tensor,
    pre_indices: Sequence[int],
    level_by_index: Mapping[int, str],
    iou: float,
) -> dict[str, object]:
    """Find the first same-class pre-NMS suppressor for one useful candidate.

    Only GT-owned useful candidates call this function.  It therefore avoids
    the former 30k×30k Python scan while retaining exact same-class and
    pair-IoU semantics for the N estimand.
    """

    candidate_index = int(candidate_index)
    candidate_class = int(class_scores[candidate_index].argmax().item())
    candidate_score = float(class_scores[candidate_index, candidate_class].item())
    pre = torch.as_tensor(tuple(int(value) for value in pre_indices), dtype=torch.long, device=decoded.device)
    if pre.numel() == 0:
        return {"candidate_index": candidate_index, "suppressed": True, "candidate_score": candidate_score}
    same = class_scores[pre].argmax(dim=1) == candidate_class
    same_indices = pre[same]
    if same_indices.numel() == 0:
        return {"candidate_index": candidate_index, "suppressed": True, "candidate_score": candidate_score}
    overlaps = _iou(boxes_input[candidate_index], boxes_input[same_indices])
    # NMS keeps the earlier official survivor; a later/equal-scored row is
    # not a valid direct suppressor.  The ordered pre-NMS list is authoritative
    # and tie order comes from upstream return_idxs.
    valid = torch.where((overlaps > float(iou)) & (same_indices != candidate_index))[0]
    if valid.numel() == 0:
        return {"candidate_index": candidate_index, "suppressed": True, "candidate_score": candidate_score}
    suppressor_index = int(same_indices[int(valid[0].item())].item())
    overlap = float(overlaps[int(valid[0].item())].item())
    suppressor_class = int(class_scores[suppressor_index].argmax().item())
    return {
        "candidate_index": candidate_index,
        "suppressed": True,
        "candidate_level": str(level_by_index.get(candidate_index, "unknown")),
        "candidate_score": candidate_score,
        "suppressor_index": suppressor_index,
        "suppressor_level": str(level_by_index.get(suppressor_index, "unknown")),
        "suppressor_score": float(class_scores[suppressor_index, suppressor_class].item()),
        "pair_iou": overlap,
        "iou": overlap,
        "suppressor_class": suppressor_class,
    }


def _gt_rows_for_image(
    *,
    image_id: str,
    decoded: torch.Tensor,
    orig_shape: Sequence[int],
    input_shape: Sequence[int],
    level_slices: Sequence[LevelSlice],
    stage_indices: Mapping[str, torch.Tensor],
    suppression: Sequence[Mapping[str, object]],
    raw_label_dir: Path,
    result: Any,
    conf: float,
    iou: float,
    max_nms: int,
    max_det: int,
) -> dict[str, object]:
    from ifdr_yolo.data.kitti_types import Difficulty

    objects = load_kitti_ground_truth(raw_label_dir, (image_id,))[image_id]
    valid_objects = [obj for obj in objects if obj.kind in {"Pedestrian", "Cyclist"} and is_valid_ground_truth(obj, obj.kind, Difficulty.MODERATE)]
    if not valid_objects:
        return {"image_id": image_id, "class_name": "", "class_id": -1, "moderate_valid": False, "small_25_40": False, "far_gt_40m": False, "p2_candidates": [], "coarse_candidates": []}
    # The primary summarizer consumes one row per GT.  A multi-GT image is
    # represented as a list under ``gt_rows`` while preserving image cluster.
    boxes_input = _xywh2xyxy(decoded[0, :4].transpose(0, 1).detach())
    boxes_orig = boxes_input.clone()
    from ultralytics.utils import ops

    boxes_orig = ops.scale_boxes(
        tuple(int(v) for v in input_shape),
        boxes_orig,
        tuple(int(v) for v in orig_shape),
    )
    class_scores = decoded[0, 4:].transpose(0, 1).detach()
    best_scores_global, best_classes_global = class_scores.max(dim=1)
    order_global = torch.argsort(best_scores_global, descending=True, stable=True)
    ordered_scores = best_scores_global[order_global]
    starts = torch.ones((ordered_scores.shape[0],), dtype=torch.bool, device=ordered_scores.device)
    if starts.numel() > 1:
        starts[1:] = ordered_scores[1:] != ordered_scores[:-1]
    group_ids = torch.cumsum(starts.to(torch.long), dim=0) - 1
    first_positions = torch.where(starts)[0]
    global_ranks_sorted = first_positions[group_ids] + 1
    group_counts = torch.bincount(group_ids)
    global_ties_sorted = group_counts[group_ids]
    global_ranks = torch.zeros_like(global_ranks_sorted)
    global_ties = torch.zeros_like(global_ties_sorted)
    global_ranks[order_global] = global_ranks_sorted
    global_ties[order_global] = global_ties_sorted
    stage_sets = {name: set(int(value) for value in values.tolist()) for name, values in stage_indices.items()}
    suppression_map = {int(item.get("candidate_index", -1)): dict(item) for item in suppression}
    level_by_index = {index: level.name for level in level_slices for index in range(level.start, level.stop)}
    gt_rows: list[dict[str, object]] = []
    all_gt_boxes = [obj.bbox.as_xyxy() for obj in valid_objects]
    all_gt_classes = [int(TRAIN_CLASS_TO_ID[obj.kind]) for obj in valid_objects]
    # Ownership is computed once per explicit evaluation class from a single
    # N×G ordinary-IoU matrix.  The previous implementation rebuilt Python
    # rows and scanned all 34k candidates for every GT, which made the formal
    # 3,341-image run infeasible.
    gt_orig_tensor = torch.as_tensor(all_gt_boxes, dtype=boxes_orig.dtype, device=boxes_orig.device)
    gt_input_tensor = torch.as_tensor(
        [_map_box_to_input(box, orig_shape=orig_shape, input_shape=input_shape) for box in all_gt_boxes],
        dtype=boxes_input.dtype,
        device=boxes_input.device,
    )
    gt_validation_input_tensor = torch.as_tensor(
        [_map_box_to_input(box, orig_shape=orig_shape, input_shape=input_shape) for box in all_gt_boxes],
        dtype=torch.float64,
        device=boxes_input.device,
    )
    gt_validation_orig_tensor = torch.as_tensor(all_gt_boxes, dtype=torch.float64, device=boxes_orig.device)
    try:
        roundtrip = ops.scale_boxes(tuple(int(v) for v in input_shape), gt_validation_input_tensor.clone(), tuple(int(v) for v in orig_shape))
    except Exception as error:
        raise ValueError("cannot round-trip GT boxes through official scale_boxes") from error
    clipped_orig = gt_validation_orig_tensor.clone()
    clipped_orig[:, [0, 2]] = clipped_orig[:, [0, 2]].clamp(0, int(orig_shape[1]))
    clipped_orig[:, [1, 3]] = clipped_orig[:, [1, 3]].clamp(0, int(orig_shape[0]))
    if not torch.allclose(roundtrip, clipped_orig, atol=1e-12, rtol=0.0):
        raise ValueError("GT LetterBox/scale_boxes round-trip mismatch")
    iou_matrix = _pairwise_iou(boxes_input, gt_input_tensor)
    ownership_by_class: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for evaluation_class in sorted(set(all_gt_classes)):
        class_indices = torch.as_tensor([idx for idx, value in enumerate(all_gt_classes) if value == evaluation_class], dtype=torch.long, device=iou_matrix.device)
        class_iou = iou_matrix[:, class_indices]
        class_best_iou, class_best_local = class_iou.max(dim=1)
        ownership_by_class[evaluation_class] = (class_best_iou, class_indices[class_best_local])
    global_max = set(int(value) for value in stage_indices.get("max_nms", torch.empty(0)).tolist())
    p2_indices = [index for level in level_slices if level.name == "P2" for index in range(level.start, level.stop) if index in global_max]
    coarse_indices = [index for level in level_slices if level.name in {"P3", "P4", "P5"} for index in range(level.start, level.stop) if index in global_max]
    p2_only = _nms_subset_indices(decoded, p2_indices, conf=conf, iou=iou, max_nms=max_nms, max_det=max_det)
    coarse_only = _nms_subset_indices(decoded, coarse_indices, conf=conf, iou=iou, max_nms=max_nms, max_det=max_det)
    full_pre_ordered = tuple(int(value) for value in stage_indices.get("nms", torch.empty(0)).tolist())
    full_pre = set(full_pre_ordered)
    for gt_index, obj in enumerate(valid_objects):
        gt_class = all_gt_classes[gt_index]
        best_iou, owners = ownership_by_class[gt_class]
        owned_indices = torch.where((owners == gt_index) & (best_iou > 0.5))[0].tolist()
        rows: list[dict[str, object]] = []
        for index in owned_indices:
            index = int(index)
            scores = tuple(float(v) for v in class_scores[index].tolist())
            best_class = int(max(range(len(scores)), key=lambda value: scores[value])) if scores else -1
            level_name = level_by_index.get(index, "unknown")
            row = {"index": index, "level": level_name, "box": tuple(float(v) for v in boxes_orig[index].tolist()), "input_box": tuple(float(v) for v in boxes_input[index].tolist()), "class_scores": scores, "best_class": best_class, "score": max(scores, default=0.0), "stage": _stage_for_index(index, stage_sets, conf_score=max(scores, default=0.0), conf=conf), "owner_gt_index": gt_index, "owner_iou": float(best_iou[index].item()), "strict_rank": int(global_ranks[index].item()), "tie_group_size": int(global_ties[index].item()), "group_only_nms_survives": _group_only_nms_survives(index, level_name, p2_only, coarse_only)}
            row.update(suppression_map.get(index, {}))
            if index not in full_pre and index in global_max and "suppressor_index" not in row:
                row.update(_targeted_suppression_detail(candidate_index=index, decoded=decoded, class_scores=class_scores, boxes_input=boxes_input, pre_indices=full_pre_ordered, level_by_index=level_by_index, iou=iou))
                if "suppressor_index" not in row:
                    raise ValueError(f"missing same-class suppressor provenance for dropped useful candidate {index}")
            if row.get("suppressor_index") is not None:
                row["suppressor_level"] = level_by_index.get(int(row["suppressor_index"]), "unknown")
            rows.append(row)
        p2_rows = [item for item in rows if item["level"] == "P2"]
        coarse_rows = [item for item in rows if item["level"] in {"P3", "P4", "P5"}]
        gt_box = obj.bbox.as_xyxy()
        # Candidate decoded/original boxes are the formal compute device; GT
        # labels are commonly loaded on CPU.  Keep the GT tensor there only
        # until this one explicit boundary, then compute IoU on boxes_orig's
        # device and persist a CPU scalar.
        gt_tensor = gt_orig_tensor[gt_index]

        def enrich(items: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
            enriched: list[dict[str, object]] = []
            for original in items:
                item = _attach_suppressor_box(original, boxes_orig)
                item["candidate_iou_to_gt"] = float(item.get("owner_iou", 0.0))
                suppressor_index = item.get("suppressor_index")
                if suppressor_index is not None:
                    suppressor_box = boxes_orig[int(suppressor_index)]
                    if suppressor_box.device != gt_tensor.device or suppressor_box.dtype != gt_tensor.dtype:
                        raise ValueError("suppressor/GT boxes must share the decoded compute device and dtype")
                    item["suppressor_iou_to_gt"] = float(_iou(gt_tensor, suppressor_box.unsqueeze(0))[0].item())
                enriched.append(item)
            return enriched

        p2_useful = enrich([item for item in p2_rows if item["owner_iou"] > 0.5])
        coarse_useful = enrich([item for item in coarse_rows if item["owner_iou"] > 0.5])
        def best_score(items: Sequence[Mapping[str, object]]) -> float:
            return max((float(item.get("class_scores", (0.0,) * 3)[gt_class]) if gt_class < len(item.get("class_scores", ())) else 0.0 for item in items), default=0.0)
        p2_best_score, coarse_best_score = best_score(p2_useful), best_score(coarse_useful)
        p2_final_useful = any(int(item["index"]) in full_pre and item["best_class"] == gt_class and item["score"] > conf for item in p2_useful)
        coarse_final_useful = any(int(item["index"]) in full_pre and item["best_class"] == gt_class and item["score"] > conf for item in coarse_useful)
        p2_only_useful = any(int(item["index"]) in p2_only and item["best_class"] == gt_class and item["score"] > conf for item in p2_useful)
        coarse_only_useful = any(int(item["index"]) in coarse_only and item["best_class"] == gt_class and item["score"] > conf for item in coarse_useful)
        p2_harm = any(
            int(item["index"]) in p2_only
            and int(item["index"]) not in full_pre
            and not (p2_final_useful or coarse_final_useful)
            and str(item.get("suppressor_level", "")) in {"P3", "P4", "P5"}
            and item["best_class"] == gt_class
            and item["score"] > conf
            for item in p2_useful
        )
        coarse_harm = any(
            int(item["index"]) in coarse_only
            and int(item["index"]) not in full_pre
            and not (p2_final_useful or coarse_final_useful)
            and str(item.get("suppressor_level", "")) == "P2"
            and item["best_class"] == gt_class
            and item["score"] > conf
            for item in coarse_useful
        )
        harm_events = []
        for item in (*p2_useful, *coarse_useful):
            if item.get("suppressor_index") is not None and item.get("suppressed"):
                harm_events.append({
                    "candidate_index": int(item["index"]),
                    "candidate_level": str(item.get("candidate_level", item.get("level", "unknown"))),
                    "candidate_score": float(item.get("candidate_score", item.get("score", 0.0))),
                    "candidate_iou_to_gt": float(item.get("candidate_iou_to_gt", 0.0)),
                    "suppressor_index": int(item["suppressor_index"]),
                    "suppressor_level": str(item.get("suppressor_level", "unknown")),
                    "suppressor_score": float(item.get("suppressor_score", 0.0)),
                    "suppressor_iou_to_gt": float(item.get("suppressor_iou_to_gt", 0.0)),
                    "suppressor_box": item.get("suppressor_box"),
                    "pair_iou": float(item.get("pair_iou", 0.0)),
                    "same_class": True,
                })
        gt_rows.append({
            "image_id": image_id,
            "gt_index": gt_index,
            "class_name": obj.kind,
            "class_id": gt_class,
            "gt_box": tuple(float(value) for value in gt_box),
            "moderate_valid": True,
            "height_px": float(obj.bbox.height),
            "depth_m": float(obj.location_xyz[2]),
            "small_25_40": 25.0 < float(obj.bbox.height) <= 40.0,
            "far_gt_40m": float(obj.location_xyz[2]) > 40.0,
            "p2_candidates": p2_useful,
            "coarse_candidates": coarse_useful,
            "p2_only_keeps_useful": p2_only_useful,
            "coarse_only_keeps_useful": coarse_only_useful,
            "full_nms_any_useful": p2_final_useful or coarse_final_useful,
            "p2_direct_suppressed_by_coarse": p2_harm,
            "coarse_direct_suppressed_by_p2": coarse_harm,
            "nms_provenance": harm_events,
            "p2_best_gt_score": p2_best_score,
            "coarse_best_gt_score": coarse_best_score,
            "p2_minus_coarse_score_margin": p2_best_score - coarse_best_score,
            "p2_strict_ranks": [int(item.get("strict_rank", -1)) for item in p2_useful],
            "coarse_strict_ranks": [int(item.get("strict_rank", -1)) for item in coarse_useful],
            "p2_tie_group_sizes": [int(item.get("tie_group_size", 1)) for item in p2_useful],
            "coarse_tie_group_sizes": [int(item.get("tie_group_size", 1)) for item in coarse_useful],
        })
    return {"image_id": image_id, "gt_rows": gt_rows}


class _ScoreNMSRunState:
    def __init__(self, *, output: Path, mirror: Path, identity: Mapping[str, object], identity_sha: str, selected_ids: Sequence[str], completed: Sequence[str], stop_after: int | None, rng_snapshot: Mapping[str, object] | None = None, generation_zero_pending: bool = False) -> None:
        self.output, self.mirror = Path(output), Path(mirror)
        self.identity, self.identity_sha = dict(identity), str(identity_sha)
        self.selected_ids = tuple(str(value) for value in selected_ids)
        self.completed = list(str(value) for value in completed)
        self.stop_after = None if stop_after is None else int(stop_after)
        self.rng_snapshot = _clone_rng_snapshot(rng_snapshot) if rng_snapshot is not None else None
        self._checkpoint_completed_count = len(self.completed)
        self._generation_zero_pending = bool(generation_zero_pending)
        self.journal = self.output / "score_nms_audit.jsonl"
        self.checkpoint = self.output / "checkpoint.json"
        self.mirror_journal = self.mirror / self.journal.name
        self.mirror_checkpoint = self.mirror / self.checkpoint.name
        self.output.mkdir(parents=True, exist_ok=True)
        self.mirror.mkdir(parents=True, exist_ok=True)

    def commit(self, record: Mapping[str, object], label_path: Path) -> None:
        image_id = str(record.get("image_id", ""))
        expected = self.selected_ids[len(self.completed)] if len(self.completed) < len(self.selected_ids) else None
        if image_id != expected:
            raise ValueError(f"image commit order mismatch: expected {expected}, got {image_id}")
        label_bytes = label_path.read_bytes() if label_path.is_file() else b""
        payload = dict(record)
        try:
            label_ref = label_path.relative_to(self.output).as_posix()
        except ValueError as error:
            raise ValueError("label path must be within audit output") from error
        payload.update({"schema_version": 1, "identity_sha256": self.identity_sha, "image_id": image_id, "label_path": label_ref, "label_missing_as_empty": not label_path.is_file(), "label_sha256": _file_digest(label_bytes), "label_size": len(label_bytes)})
        mirror_label = self.mirror / label_ref
        if label_path.is_file():
            mirror_label.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(mirror_label, label_bytes)
            if mirror_label.read_bytes() != label_bytes:
                raise ValueError(f"label mirror mismatch: {label_ref}")
        elif mirror_label.exists():
            mirror_label.unlink()
        _append_line(self.journal, self.mirror_journal, payload)
        self.completed.append(image_id)

    def set_rng_snapshot(self, snapshot: Mapping[str, object]) -> None:
        """Set the paired RNG state used at the next predictor boundary."""

        self.rng_snapshot = _clone_rng_snapshot(snapshot)

    def initialize_generation_zero(self, snapshot: Mapping[str, object]) -> None:
        """Durably create the empty prefix after predictor warmup.

        A crash after the paired default reference but before the first audit
        callback leaves no checkpoint.  This callback-boundary generation is
        reconstructed only from a valid identity-bound reference.  Identical
        uncommitted journal/label tails are discarded; asymmetric or
        tampered artifacts fail closed.
        """

        if self.completed or self._checkpoint_completed_count:
            raise ValueError("generation-zero initialization requires an empty prefix")
        journal_bytes = [path.read_bytes() if path.is_file() else b"" for path in (self.journal, self.mirror_journal)]
        if journal_bytes[0] != journal_bytes[1]:
            raise ValueError("generation-zero journal mirrors differ")
        label_sets: list[dict[str, bytes]] = []
        for base in (self.output, self.mirror):
            root = base / "predictions" / "labels"
            label_sets.append({path.relative_to(base).as_posix(): path.read_bytes() for path in root.glob("*.txt")} if root.is_dir() else {})
        if label_sets[0] != label_sets[1]:
            raise ValueError("generation-zero label mirrors differ")
        _atomic_write(self.journal, b"")
        _atomic_write(self.mirror_journal, b"")
        for base in (self.output, self.mirror):
            root = base / "predictions" / "labels"
            if root.is_dir():
                for path in root.glob("*.txt"):
                    path.unlink()
        snapshot = _clone_rng_snapshot(snapshot)
        self.rng_snapshot = snapshot
        payload = {
            "schema_version": 1,
            "state": "running",
            "identity": self.identity,
            "identity_sha256": self.identity_sha,
            "completed_image_ids": [],
            "next_position": 0,
            "journal_offset": 0,
            "journal_prefix_sha256": _file_digest(b""),
            "rng_completed_count": 0,
            "rng_snapshot": _rng_snapshot_payload(snapshot),
            "rng_digest": _rng_state_digest_from_snapshot(snapshot),
        }
        _atomic_write(self.checkpoint, (json.dumps(payload, sort_keys=True) + "\n").encode())
        _atomic_write(self.mirror_checkpoint, self.checkpoint.read_bytes())
        self._checkpoint_completed_count = 0
        self._generation_zero_pending = False

    def checkpoint_after_batch(self, snapshot: Mapping[str, object]) -> None:
        """Durably checkpoint one committed batch after all callbacks ran.

        ``write_results`` has already persisted the official label and journal
        record, but this method is intentionally called by the final
        ``on_predict_batch_end`` callback.  Thus an interrupted run can only
        advertise an image once the post-callback CPU/CUDA RNG state is bound
        to the same committed prefix.
        """

        if len(self.completed) != self._checkpoint_completed_count + 1:
            raise ValueError("score/NMS callback must commit exactly one batch")
        self.set_rng_snapshot(snapshot)
        checkpoint = {
            "schema_version": 1,
            "state": "running",
            "identity": self.identity,
            "identity_sha256": self.identity_sha,
            "completed_image_ids": self.completed,
            "next_position": len(self.completed),
            "journal_offset": self.journal.stat().st_size,
            "journal_prefix_sha256": _file_digest(self.journal.read_bytes()),
            "rng_completed_count": len(self.completed),
            "rng_snapshot": _rng_snapshot_payload(self.rng_snapshot),
            "rng_digest": _rng_state_digest_from_snapshot(self.rng_snapshot),
        }
        _atomic_write(self.checkpoint, (json.dumps(checkpoint, sort_keys=True) + "\n").encode())
        _atomic_write(self.mirror_checkpoint, self.checkpoint.read_bytes())
        self._checkpoint_completed_count = len(self.completed)
        if self.stop_after is not None and len(self.completed) >= self.stop_after and len(self.completed) < len(self.selected_ids):
            raise ScoreNMSInterrupted("interrupted score/NMS audit")

    def complete(self) -> None:
        if len(self.completed) != self._checkpoint_completed_count:
            raise ValueError("score/NMS completion requires callback checkpoint for every committed image")
        if self.rng_snapshot is None:
            raise ValueError("score/NMS completion requires a durable RNG snapshot")
        base = {"schema_version": 1, "identity": self.identity, "identity_sha256": self.identity_sha, "completed_image_ids": self.completed, "next_position": len(self.completed), "journal_offset": self.journal.stat().st_size, "journal_prefix_sha256": _file_digest(self.journal.read_bytes())}
        base.update({"rng_completed_count": len(self.completed), "rng_snapshot": _rng_snapshot_payload(self.rng_snapshot), "rng_digest": _rng_state_digest_from_snapshot(self.rng_snapshot)})
        # A publishing checkpoint is deliberately not a completion marker.
        # This makes a crash during label/summary/manifest publication fail
        # closed instead of looking like a finished scientific run.
        publishing = {**base, "state": "publishing", "publication_state": "publishing"}
        _atomic_write(self.checkpoint, (json.dumps(publishing, sort_keys=True) + "\n").encode())
        _atomic_write(self.mirror_checkpoint, self.checkpoint.read_bytes())
        labels: list[dict[str, object]] = []
        for label_path in sorted(self.output.rglob("labels/*.txt")):
            relative = label_path.relative_to(self.output).as_posix()
            mirror_path = self.mirror / relative
            mirror_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(mirror_path, label_path.read_bytes())
            raw = label_path.read_bytes()
            mirror_raw = mirror_path.read_bytes()
            if raw != mirror_raw:
                raise ValueError(f"label mirror mismatch: {relative}")
            labels.append({"path": relative, "size": len(raw), "sha256": _file_digest(raw)})
        files = {"score_nms_audit.jsonl": _file_digest(self.journal.read_bytes()), "checkpoint.json": _file_digest(self.checkpoint.read_bytes())}
        # Publication order is summary/CSV -> labels -> manifest.  Keeping
        # their hashes in the final manifest makes a partial result fail
        # closed instead of looking complete after a crash.
        for name in ("summary.json", "summary.csv", "default_reference.json"):
            path = self.output / name
            if path.is_file():
                mirror_path = self.mirror / name
                _atomic_write(mirror_path, path.read_bytes())
                if mirror_path.read_bytes() != path.read_bytes():
                    raise ValueError(f"summary mirror mismatch: {name}")
                files[name] = _file_digest(path.read_bytes())
        manifest = {"schema_version": 1, "publication_state": "publishing", "identity_sha256": self.identity_sha, "labels": labels, "files": files}
        _atomic_write(self.output / "manifest.json", (json.dumps(manifest, sort_keys=True) + "\n").encode())
        _atomic_write(self.mirror / "manifest.json", (self.output / "manifest.json").read_bytes())
        if (self.mirror / "manifest.json").read_bytes() != (self.output / "manifest.json").read_bytes():
            raise ValueError("manifest mirror mismatch during publishing")
        # The final completion marker is written only after all artifacts and
        # both manifests are durable.  The manifest is then rebound to the
        # final checkpoint hash, avoiding self-reference.
        complete_checkpoint = {**base, "state": "complete", "publication_state": "complete"}
        _atomic_write(self.checkpoint, (json.dumps(complete_checkpoint, sort_keys=True) + "\n").encode())
        _atomic_write(self.mirror_checkpoint, self.checkpoint.read_bytes())
        if self.mirror_checkpoint.read_bytes() != self.checkpoint.read_bytes():
            raise ValueError("checkpoint mirror mismatch at completion")
        files["checkpoint.json"] = _file_digest(self.checkpoint.read_bytes())
        final_manifest = {"schema_version": 1, "publication_state": "complete", "identity_sha256": self.identity_sha, "labels": labels, "files": files}
        _atomic_write(self.output / "manifest.json", (json.dumps(final_manifest, sort_keys=True) + "\n").encode())
        _atomic_write(self.mirror / "manifest.json", (self.output / "manifest.json").read_bytes())
        _validate_final_publication(self.output, self.mirror)


def build_score_nms_predictor_class(state: _ScoreNMSRunState, *, raw_label_dir: Path, conf: float = 0.001, iou: float = 0.7, max_nms: int = 30000, max_det: int = 300):
    """Build a DetectionPredictor subclass without modifying installed Ultralytics."""

    try:
        from ultralytics.models.yolo.detect.predict import DetectionPredictor
    except ImportError as error:  # pragma: no cover - formal runtime only
        raise RuntimeError("Ultralytics 8.4.98 is required for score/NMS predictor") from error

    class ScoreNMSPredictor(DetectionPredictor):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self._score_nms_pending: list[dict[str, object]] = []
            self._score_nms_rng_restored = False
            # Register after Ultralytics' integration callbacks so restoration
            # occurs after warmup/setup and checkpointing observes every prior
            # batch-end callback.  The protocol is batch=1, enforced below.
            self.add_callback("on_predict_start", self._score_nms_restore_rng)
            self.add_callback("on_predict_batch_end", self._score_nms_checkpoint_rng)

        def _score_nms_restore_rng(self, _predictor: object) -> None:
            if self._score_nms_rng_restored:
                return
            if state.rng_snapshot is None:
                raise RuntimeError("score/NMS predictor has no persisted RNG boundary")
            _restore_rng_state(state.rng_snapshot)
            if state._generation_zero_pending:
                state.initialize_generation_zero(state.rng_snapshot)
            self._score_nms_rng_restored = True

        def _score_nms_checkpoint_rng(self, _predictor: object) -> None:
            pending = len(state.completed) - state._checkpoint_completed_count
            if pending == 0:
                return
            if pending != 1:
                raise ValueError("score/NMS callback requires batch=1")
            state.checkpoint_after_batch(_capture_rng_state())

        def postprocess(self, preds: Any, img: torch.Tensor, orig_imgs: Any, **kwargs: object) -> list[Any]:
            decoded = preds[0] if isinstance(preds, (tuple, list)) else preds
            decoded_clone = decoded.detach().clone()
            mapping = preds[1] if isinstance(preds, (tuple, list)) and len(preds) > 1 and isinstance(preds[1], Mapping) else {}
            feats = mapping.get("feats") if isinstance(mapping, Mapping) else None
            result = super().postprocess(preds, img, orig_imgs, **kwargs)
            if not isinstance(orig_imgs, list):
                import numpy as np
                orig_imgs = [value for value in np.asarray(orig_imgs)]
            if feats is None:
                raise ValueError("prediction mapping must expose Detect feature levels")
            slices = derive_level_slices(feats)
            replay = replay_nms_with_stages(decoded_clone, level_slices=slices, conf=conf, iou=iou, max_nms=max_nms, max_det=max_det)
            from ultralytics.utils import ops
            for batch_index, item in enumerate(result):
                expected = replay.output[batch_index].detach().clone()
                if expected.numel():
                    expected[:, :4] = ops.scale_boxes(img.shape[2:], expected[:, :4], orig_imgs[batch_index].shape)
                observed = item.boxes.data.detach().cpu()
                if not torch.equal(expected.cpu(), observed):
                    raise ValueError("audit predictor changed official prediction tensor")
                image_id = Path(str(self.batch[0][batch_index])).stem
                pending = _gt_rows_for_image(image_id=image_id, decoded=decoded_clone[batch_index : batch_index + 1], orig_shape=orig_imgs[batch_index].shape[:2], input_shape=img.shape[2:], level_slices=slices, stage_indices=replay.stage_indices[batch_index], suppression=replay.suppression[batch_index], raw_label_dir=Path(raw_label_dir), result=item, conf=conf, iou=iou, max_nms=max_nms, max_det=max_det)
                pending["image_id"] = image_id
                pending["result_boxes"] = _result_boxes_payload(item)
                self._score_nms_pending.append(pending)
            return result

        def write_results(self, i: int, p: Path, im: torch.Tensor, s: list[str]) -> str:
            text = super().write_results(i, p, im, s)
            if i >= len(self._score_nms_pending):
                raise ValueError("missing pending score/NMS record")
            record = self._score_nms_pending.pop(0)
            # ``txt_path`` is assigned by BasePredictor immediately before
            # this method; commit only after the official label write.
            state.commit(record, Path(f"{self.txt_path}.txt"))
            return text

    return ScoreNMSPredictor


def build_rng_boundary_predictor_class(snapshot: Mapping[str, object]):
    """Build a standard-output predictor with a post-warmup RNG boundary.

    This is used only for the paired default reference.  It leaves
    ``DetectionPredictor`` postprocessing and label writing untouched while
    making the control and audit runs share the same callback-boundary RNG
    contract.
    """

    try:
        from ultralytics.models.yolo.detect.predict import DetectionPredictor
    except ImportError as error:  # pragma: no cover - formal runtime only
        raise RuntimeError("Ultralytics 8.4.98 is required for score/NMS predictor") from error
    persisted = _clone_rng_snapshot(snapshot)

    class RngBoundaryPredictor(DetectionPredictor):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self._rng_boundary_restored = False
            self.add_callback("on_predict_start", self._restore_boundary_rng)

        def _restore_boundary_rng(self, _predictor: object) -> None:
            if not self._rng_boundary_restored:
                _restore_rng_state(persisted)
                self._rng_boundary_restored = True

    return RngBoundaryPredictor


def _resolve_image_path(raw_images: Path, image_id: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = Path(raw_images) / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"raw image not found for {image_id} under {raw_images}")


def compare_saved_label_files(default_root: Path, audit_root: Path, image_ids: Sequence[str]) -> dict[str, object]:
    """Compare default and audit label bytes with missing-as-empty semantics."""

    def read(root: Path, image_id: str) -> bytes:
        path = Path(root) / "predictions" / "labels" / f"{image_id}.txt"
        return path.read_bytes() if path.is_file() else b""

    mismatches = [str(image_id) for image_id in image_ids if read(default_root, str(image_id)) != read(audit_root, str(image_id))]
    return {"state": "PASS" if not mismatches else "FAIL", "image_count": len(tuple(image_ids)), "mismatches": mismatches, "missing_as_empty": True}


def _validate_raw_label_view(raw_label_dir: Path, fit_ids: Sequence[str]) -> dict[str, object]:
    """Fail closed unless every fit image has one canonical KITTI label file."""

    root = Path(raw_label_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"raw-label directory missing: {root}")
    files = [path for path in root.rglob("*.txt") if path.is_file()]
    by_id: dict[str, Path] = {}
    for path in files:
        if path.stem in by_id and by_id[path.stem] != path:
            raise ValueError(f"raw-label view contains duplicate image ID: {path.stem}")
        by_id[path.stem] = path
    missing = [str(image_id) for image_id in fit_ids if str(image_id) not in by_id]
    if missing:
        raise ValueError(f"raw-label view is missing fit IDs: {missing[:5]}")
    # Return only registered fit references; extra labels (e.g. development)
    # are not read and do not alter the fit-only estimand.
    return {
        "root": str(root),
        "fit_count": len(tuple(fit_ids)),
        "fit_ids": [str(image_id) for image_id in fit_ids],
        "fit_label_sha256": {str(image_id): sha256_file(by_id[str(image_id)]) for image_id in fit_ids},
    }


def _rng_state_digest() -> dict[str, object]:
    state = {"cpu": _file_digest(torch.get_rng_state().cpu().numpy().tobytes())}
    if torch.cuda.is_available():
        state["cuda"] = [_file_digest(item.cpu().numpy().tobytes()) for item in torch.cuda.get_rng_state_all()]
    else:
        state["cuda"] = []
    return state


def _capture_rng_state() -> dict[str, object]:
    """Capture clonable CPU/CUDA RNG tensors for a controlled paired run."""

    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [item.clone() for item in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def _initialize_benchmark32_rng_boundary() -> dict[str, object]:
    """Create the fixed fresh-run RNG boundary after model construction."""

    torch.manual_seed(BENCHMARK32_RNG_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(BENCHMARK32_RNG_SEED)
    return _capture_rng_state()


def _initialize_full_audit_rng_boundary() -> dict[str, object]:
    """Create the fixed fresh full-run RNG boundary after model construction."""

    torch.manual_seed(FULL_AUDIT_RNG_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FULL_AUDIT_RNG_SEED)
    return _capture_rng_state()


def _clone_rng_snapshot(snapshot: Mapping[str, object]) -> dict[str, object]:
    """Clone a validated CPU/all-CUDA RNG snapshot for durable boundaries."""

    cpu = snapshot.get("cpu")
    cuda = snapshot.get("cuda", [])
    if not isinstance(cpu, torch.Tensor) or not isinstance(cuda, Sequence) or not all(isinstance(item, torch.Tensor) for item in cuda):
        raise ValueError("invalid RNG snapshot")
    return {"cpu": cpu.detach().clone(), "cuda": [item.detach().clone() for item in cuda]}


def _restore_rng_state(snapshot: Mapping[str, object]) -> None:
    cpu = snapshot.get("cpu")
    cuda = snapshot.get("cuda", [])
    if not isinstance(cpu, torch.Tensor) or not isinstance(cuda, Sequence):
        raise ValueError("invalid RNG snapshot")
    torch.set_rng_state(cpu.clone())
    if torch.cuda.is_available():
        if not all(isinstance(item, torch.Tensor) for item in cuda):
            raise ValueError("invalid CUDA RNG snapshot")
        torch.cuda.set_rng_state_all([item.clone() for item in cuda])


def _construct_resume_model(model_factory: Any, checkpoint: object, persisted_rng: Mapping[str, object]) -> Any:
    """Construct a resume model; restoration happens at ``on_predict_start``.

    Model loading, predictor setup, and warmup may consume ambient RNG.  The
    persisted post-prefix snapshot must therefore be restored only by the
    predictor's final ``on_predict_start`` callback, after those operations
    complete and immediately before the first remaining batch.  The snapshot
    argument is retained as an explicit contract check for callers/tests.
    """

    if not isinstance(persisted_rng, Mapping):
        raise ValueError("persisted RNG snapshot is required before model construction")
    return model_factory(str(checkpoint))


def _rng_state_digest_from_snapshot(snapshot: Mapping[str, object]) -> dict[str, object]:
    cpu = snapshot.get("cpu")
    cuda = snapshot.get("cuda", [])
    if not isinstance(cpu, torch.Tensor) or not isinstance(cuda, Sequence):
        raise ValueError("invalid RNG snapshot")
    return {
        "cpu": _file_digest(cpu.detach().cpu().numpy().tobytes()),
        "cuda": [_file_digest(item.detach().cpu().numpy().tobytes()) for item in cuda if isinstance(item, torch.Tensor)],
    }


def _rng_snapshot_payload(snapshot: Mapping[str, object]) -> dict[str, object]:
    cpu = snapshot.get("cpu")
    cuda = snapshot.get("cuda", [])
    if not isinstance(cpu, torch.Tensor) or not isinstance(cuda, Sequence):
        raise ValueError("invalid RNG snapshot")
    return {
        "cpu_hex": cpu.detach().cpu().numpy().tobytes().hex(),
        "cuda_hex": [item.detach().cpu().numpy().tobytes().hex() for item in cuda if isinstance(item, torch.Tensor)],
    }


def _rng_snapshot_from_payload(payload: Mapping[str, object]) -> dict[str, object]:
    cpu_hex = payload.get("cpu_hex")
    cuda_hex = payload.get("cuda_hex", [])
    if not isinstance(cpu_hex, str) or not isinstance(cuda_hex, Sequence):
        raise ValueError("invalid persisted RNG snapshot")
    return {
        "cpu": torch.frombuffer(bytearray.fromhex(cpu_hex), dtype=torch.uint8).clone(),
        "cuda": [torch.frombuffer(bytearray.fromhex(value), dtype=torch.uint8).clone() for value in cuda_hex if isinstance(value, str)],
    }


def _validated_checkpoint_rng(payload: Mapping[str, object], *, expected_count: int) -> dict[str, object]:
    """Decode and validate the RNG boundary bound to one journal prefix."""

    try:
        count = int(payload.get("rng_completed_count", -1))
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint RNG completed count is invalid") from error
    if count != int(expected_count):
        raise ValueError("checkpoint RNG completed count does not match journal prefix")
    snapshot_payload = payload.get("rng_snapshot")
    if not isinstance(snapshot_payload, Mapping):
        raise ValueError("checkpoint RNG snapshot is missing")
    snapshot = _rng_snapshot_from_payload(snapshot_payload)
    expected_digest = payload.get("rng_digest")
    if not isinstance(expected_digest, Mapping) or dict(expected_digest) != _rng_state_digest_from_snapshot(snapshot):
        raise ValueError("checkpoint RNG digest mismatch")
    return snapshot


def _result_boxes_payload(result: Any) -> dict[str, object]:
    boxes = getattr(result, "boxes", None)
    tensor = boxes.data.detach().cpu().contiguous() if boxes is not None and getattr(boxes, "data", None) is not None else torch.empty((0, 6), dtype=torch.float32)
    raw = tensor.numpy().tobytes()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "values": tensor.tolist(), "sha256": _file_digest(raw)}


def _label_payload(root: Path, image_id: str) -> dict[str, object]:
    path = Path(root) / "predictions" / "labels" / f"{image_id}.txt"
    if not path.is_file():
        return {"missing": True, "size": 0, "sha256": _file_digest(b"")}
    raw = path.read_bytes()
    return {"missing": False, "size": len(raw), "sha256": _file_digest(raw)}


def _write_default_reference(output: Path, mirror: Path, reference: Mapping[str, object]) -> None:
    raw = (json.dumps(dict(reference), sort_keys=True, separators=(",", ":")) + "\n").encode()
    _atomic_write(Path(output) / "default_reference.json", raw)
    _atomic_write(Path(mirror) / "default_reference.json", raw)
    if (Path(mirror) / "default_reference.json").read_bytes() != raw:
        raise ValueError("default reference mirror mismatch")


def _load_default_reference(output: Path, mirror: Path, identity_sha: str, selected_ids: Sequence[str]) -> dict[str, object]:
    primary, secondary = Path(output) / "default_reference.json", Path(mirror) / "default_reference.json"
    if not primary.is_file() or not secondary.is_file() or primary.read_bytes() != secondary.read_bytes():
        raise ValueError("default reference is missing or mirror differs")
    reference = json.loads(primary.read_text(encoding="utf-8"))
    if reference.get("identity_sha256") != identity_sha:
        raise ValueError("default reference identity mismatch")
    if [str(item) for item in reference.get("selected_ids", [])] != [str(item) for item in selected_ids]:
        raise ValueError("default reference selected IDs mismatch")
    return reference


def _load_persisted_smoke_summary(output: Path, mirror: Path, identity_sha: str) -> dict[str, object]:
    """Load smoke evidence from a publishing/complete resume generation.

    A fully processed publication may be interrupted after its checkpoint is
    complete but before the lagging mirror manifest is updated.  The summary
    is accepted only when both manifests bind the same bytes and all three
    non-interference/reference gates were already PASS; otherwise resume must
    fail closed instead of replacing evidence with ``None``.
    """

    output, mirror = Path(output), Path(mirror)
    manifest_paths = (output / "manifest.json", mirror / "manifest.json")
    if not all(path.is_file() for path in manifest_paths):
        raise ValueError("persisted smoke summary is missing its manifests")
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]
    for manifest in manifests:
        if manifest.get("identity_sha256") != identity_sha:
            raise ValueError("persisted smoke summary identity mismatch")
        if manifest.get("publication_state") not in {"publishing", "complete"}:
            raise ValueError("persisted smoke summary publication state is invalid")
        files = manifest.get("files")
        if not isinstance(files, Mapping) or "summary.json" not in files:
            raise ValueError("persisted smoke summary is not manifest-bound")
    primary_summary, mirror_summary = output / "summary.json", mirror / "summary.json"
    if not primary_summary.is_file() or not mirror_summary.is_file() or primary_summary.read_bytes() != mirror_summary.read_bytes():
        raise ValueError("persisted smoke summary mirror mismatch")
    summary_raw = primary_summary.read_bytes()
    for manifest in manifests:
        if manifest["files"].get("summary.json") != _file_digest(summary_raw):
            raise ValueError("persisted smoke summary hash mismatch")
    summary = json.loads(summary_raw.decode("utf-8"))
    if not isinstance(summary, Mapping):
        raise ValueError("persisted smoke summary is invalid")
    for key in ("non_interference", "default_vs_audit_labels", "default_vs_audit_results"):
        value = summary.get(key)
        if not isinstance(value, Mapping) or value.get("state") != "PASS":
            raise ValueError(f"persisted smoke summary {key} is not PASS")
    return dict(summary)


def _compare_reference_labels(reference: Mapping[str, object], audit_root: Path, image_ids: Sequence[str]) -> dict[str, object]:
    labels = reference.get("labels", {})
    mismatches: list[str] = []
    for image_id in image_ids:
        expected = labels.get(str(image_id)) if isinstance(labels, Mapping) else None
        if not isinstance(expected, Mapping) or dict(expected) != _label_payload(audit_root, str(image_id)):
            mismatches.append(str(image_id))
    return {"state": "PASS" if not mismatches else "FAIL", "image_count": len(tuple(image_ids)), "mismatches": mismatches, "reference": True}


def _compare_reference_boxes(reference: Mapping[str, object], records: Sequence[Mapping[str, object]], image_ids: Sequence[str]) -> dict[str, object]:
    expected = reference.get("boxes", {})
    actual = {str(record.get("image_id", "")): record.get("result_boxes") for record in records}
    mismatches = [str(image_id) for image_id in image_ids if not isinstance(expected, Mapping) or expected.get(str(image_id)) != actual.get(str(image_id))]
    return {"state": "PASS" if not mismatches else "FAIL", "image_count": len(tuple(image_ids)), "mismatches": mismatches, "reference": True}


def _model_runtime_facts(model: Any, *, device: str, backend: str) -> dict[str, object]:
    module = getattr(model, "model", None)
    if module is None:
        raise RuntimeError("loaded model does not expose runtime facts")
    parameters = [item for item in module.parameters() if hasattr(item, "device") and hasattr(item, "dtype")] if hasattr(module, "parameters") else []
    if not parameters:
        raise RuntimeError("runtime facts require at least one parameter device")
    parameter_devices = {str(torch.device(item.device)) for item in parameters}
    if len(parameter_devices) != 1:
        raise RuntimeError(f"runtime facts require all parameters on the same device, got {sorted(parameter_devices)}")
    fp32 = bool(parameters) and all(item.dtype == torch.float32 for item in parameters)
    fused_value = getattr(module, "is_fused", getattr(module, "fused", False))
    if callable(fused_value):
        try:
            fused_value = fused_value()
        except Exception:
            fused_value = False
    # Ultralytics DetectionModel has no reliable ``module.device`` property;
    # the parameter device is the runtime source of truth.
    actual_device = next(iter(parameter_devices))
    return {
        "backend": str(backend),
        "device": str(actual_device),
        "fp32": fp32,
        "eval": bool(not module.training),
        "fused": bool(fused_value),
    }


def _model_state_digest(model: Any) -> str:
    module = getattr(model, "model", None)
    state_dict = module.state_dict() if module is not None and hasattr(module, "state_dict") else None
    if not isinstance(state_dict, Mapping):
        raise RuntimeError("loaded model does not expose a state_dict for non-interference audit")
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items(), key=lambda item: str(item[0])):
        if not isinstance(tensor, torch.Tensor):
            continue
        digest.update(str(name).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _compare_controlled_run_gates(control: Mapping[str, object], audit: Mapping[str, object], *, expected_device: str | None = None) -> dict[str, object]:
    """Compare paired predictor runs after identical setup/warmup.

    Ultralytics may fuse layers and consume RNG during setup.  Comparing each
    run to its own pre-state therefore rejects valid executions; only the
    controlled control/audit pair is authoritative here.
    """

    for key in ("rng_initial", "rng_final", "model_post", "backend", "device", "fp32", "eval", "fused"):
        if control.get(key) != audit.get(key):
            return {"state": "FAIL", "reason": f"controlled_gate_mismatch:{key}", "key": key}
    if control.get("backend") != "torchvision" or not bool(control.get("fp32")) or not bool(control.get("eval")) or not bool(control.get("fused")):
        return {"state": "FAIL", "reason": "control_runtime_facts_not_frozen"}
    if expected_device is not None:
        requested = str(expected_device).lower()
        actual = str(control.get("device", "")).lower()
        if requested == "cpu" and actual != "cpu":
            return {"state": "FAIL", "reason": "resolved_device_mismatch"}
        if requested == "0" or requested.startswith("cuda"):
            if not actual.startswith("cuda"):
                return {"state": "FAIL", "reason": "resolved_device_mismatch"}
        if requested.startswith("cuda:") and actual != requested:
            return {"state": "FAIL", "reason": "resolved_device_mismatch"}
    return {"state": "PASS", "paired_rng_final_equal": True, "paired_model_post_equal": True}


def _publication_states() -> tuple[str, str, str]:
    """Return the only legal checkpoint publication progression."""

    return ("running", "publishing", "complete")


def _manifest_expected_paths(files: Mapping[str, object]) -> tuple[str, set[str]]:
    """Resolve the declared artifact set for either synthetic or formal output."""

    if not isinstance(files, Mapping):
        raise ValueError("publication manifest files are missing")
    journal_name = "score_nms_audit.jsonl" if "score_nms_audit.jsonl" in files else "audit.jsonl"
    expected = {journal_name, "checkpoint.json"}
    if journal_name == "score_nms_audit.jsonl":
        expected.update(("summary.json", "summary.csv"))
        # The smoke reference is optional for synthetic fixtures, but is
        # scientific identity for a formal smoke run when present.
        if "default_reference.json" in files:
            expected.add("default_reference.json")
    if set(str(name) for name in files) != expected:
        raise ValueError("publication manifest file set mismatch")
    return journal_name, expected


def _validate_publication_manifest_side(
    base: Path,
    manifest: Mapping[str, object],
    *,
    allow_publishing_checkpoint_mismatch: bool = False,
) -> dict[str, object]:
    """Validate one durable publication side before repairing its mirror.

    A publishing manifest may name the checkpoint written immediately before
    the final checkpoint generation.  That one hash is allowed to be stale
    only when the current checkpoint is itself a valid, same-identity complete
    checkpoint; every other declared artifact must still match byte-for-byte.
    """

    base = Path(base)
    state = str(manifest.get("publication_state", ""))
    if state not in {"publishing", "complete"}:
        raise ValueError("publication manifest state is invalid")
    identity_sha = str(manifest.get("identity_sha256", ""))
    if len(identity_sha) != 64 or any(char not in "0123456789abcdef" for char in identity_sha.lower()):
        raise ValueError("publication manifest identity is invalid")
    files = manifest.get("files")
    journal_name, expected_paths = _manifest_expected_paths(files if isinstance(files, Mapping) else {})
    for name in expected_paths:
        path = base / name
        if not path.is_file():
            raise ValueError(f"publication artifact is missing: {name}")
        declared = str(files.get(name, "")) if isinstance(files, Mapping) else ""
        if len(declared) != 64 or any(char not in "0123456789abcdef" for char in declared.lower()):
            raise ValueError(f"publication artifact hash is invalid: {name}")
        actual = _file_digest(path.read_bytes())
        if name != "checkpoint.json" or not (state == "publishing" and allow_publishing_checkpoint_mismatch):
            if declared != actual:
                raise ValueError(f"publication artifact hash mismatch: {name}")
    checkpoint_payload = json.loads((base / "checkpoint.json").read_text(encoding="utf-8"))
    if checkpoint_payload.get("identity_sha256") != identity_sha:
        raise ValueError("publication checkpoint identity mismatch")
    checkpoint_state = str(checkpoint_payload.get("state", ""))
    if state == "complete" and checkpoint_state != "complete":
        raise ValueError("complete publication has incomplete checkpoint")
    if state == "publishing" and checkpoint_state not in {"running", "publishing", "complete"}:
        raise ValueError("publishing publication has invalid checkpoint state")
    for line in (base / journal_name).read_text(encoding="utf-8").splitlines():
        if line and json.loads(line).get("identity_sha256") != identity_sha:
            raise ValueError("publication journal identity mismatch")
    labels = manifest.get("labels", [])
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        raise ValueError("publication labels are invalid")
    declared_labels = {str(item.get("path")): item for item in labels if isinstance(item, Mapping)}
    actual_labels = {
        path.relative_to(base).as_posix()
        for path in (base / "predictions" / "labels").glob("*.txt")
    }
    if actual_labels != set(declared_labels):
        raise ValueError("publication label set mismatch")
    for relative in sorted(actual_labels):
        raw = (base / relative).read_bytes()
        item = declared_labels[relative]
        if int(item.get("size", -1)) != len(raw) or item.get("sha256") != _file_digest(raw):
            raise ValueError(f"publication label hash mismatch: {relative}")
    return {"state": state, "identity_sha256": identity_sha, "journal_name": journal_name, "files": dict(files), "labels": list(labels)}


def _repair_mixed_publication(output: Path, mirror: Path) -> None:
    """Repair exactly one lagging final manifest after a mirror write crash."""

    output, mirror = Path(output), Path(mirror)
    primary_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    mirror_manifest = json.loads((mirror / "manifest.json").read_text(encoding="utf-8"))
    states = (str(primary_manifest.get("publication_state", "")), str(mirror_manifest.get("publication_state", "")))
    if set(states) != {"complete", "publishing"}:
        raise ValueError("mixed publication states are invalid")
    complete_base, complete_manifest = (output, primary_manifest) if states[0] == "complete" else (mirror, mirror_manifest)
    lagging_base, lagging_manifest = (mirror, mirror_manifest) if states[0] == "complete" else (output, primary_manifest)
    complete_info = _validate_publication_manifest_side(complete_base, complete_manifest)
    # The current checkpoints must already be the same complete generation;
    # only the final manifest write may be lagging.
    publishing_info = _validate_publication_manifest_side(
        lagging_base,
        lagging_manifest,
        allow_publishing_checkpoint_mismatch=True,
    )
    if complete_info["identity_sha256"] != publishing_info["identity_sha256"]:
        raise ValueError("mixed publication identity mismatch")
    if complete_info["journal_name"] != publishing_info["journal_name"]:
        raise ValueError("mixed publication journal mismatch")
    complete_files = complete_info["files"]
    publishing_files = publishing_info["files"]
    if set(complete_files) != set(publishing_files):
        raise ValueError("mixed publication artifact set mismatch")
    for name in set(complete_files) - {"checkpoint.json"}:
        if complete_files[name] != publishing_files[name] or (complete_base / name).read_bytes() != (lagging_base / name).read_bytes():
            raise ValueError(f"mixed publication artifact mismatch: {name}")
    if complete_info["labels"] != publishing_info["labels"]:
        raise ValueError("mixed publication label manifest mismatch")
    complete_checkpoint = complete_base / "checkpoint.json"
    lagging_checkpoint = lagging_base / "checkpoint.json"
    if complete_checkpoint.read_bytes() != lagging_checkpoint.read_bytes():
        raise ValueError("mixed publication checkpoint mismatch")
    complete_payload = json.loads(complete_checkpoint.read_text(encoding="utf-8"))
    if complete_payload.get("state") != "complete" or complete_payload.get("publication_state") != "complete":
        raise ValueError("mixed publication checkpoint is not complete")
    # Copy only the already validated complete marker; no inference or journal
    # mutation is needed for this crash window.
    _atomic_write(lagging_base / "manifest.json", (complete_base / "manifest.json").read_bytes())
    _validate_final_publication(output, mirror)


def _validate_final_publication(output: Path, mirror: Path) -> dict[str, object]:
    """Reject a partial or stale final publication before treating it complete."""

    output, mirror = Path(output), Path(mirror)
    manifest_path = output / "manifest.json"
    mirror_manifest_path = mirror / "manifest.json"
    checkpoint = output / "checkpoint.json"
    mirror_checkpoint = mirror / "checkpoint.json"
    if not all(path.is_file() for path in (manifest_path, mirror_manifest_path, checkpoint, mirror_checkpoint)):
        raise ValueError("final publication is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mirror_manifest = json.loads(mirror_manifest_path.read_text(encoding="utf-8"))
    if manifest != mirror_manifest or manifest.get("publication_state") != "complete":
        raise ValueError("final publication marker is incomplete or mirror differs")
    checkpoint_bytes = checkpoint.read_bytes()
    if checkpoint_bytes != mirror_checkpoint.read_bytes() or json.loads(checkpoint_bytes.decode("utf-8")).get("state") != "complete" or json.loads(checkpoint_bytes.decode("utf-8")).get("publication_state") != "complete":
        raise ValueError("final checkpoint is incomplete or mirror differs")
    checkpoint_payload = json.loads(checkpoint_bytes.decode("utf-8"))
    if checkpoint_payload.get("identity_sha256") != manifest.get("identity_sha256"):
        raise ValueError("final publication identity mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("final manifest files are missing")
    journal_name = "score_nms_audit.jsonl" if "score_nms_audit.jsonl" in files else "audit.jsonl"
    expected_paths = {journal_name, "checkpoint.json"}
    if journal_name == "score_nms_audit.jsonl":
        expected_paths.update(("summary.json", "summary.csv"))
        if (output / "default_reference.json").is_file() or (mirror / "default_reference.json").is_file():
            expected_paths.add("default_reference.json")
    if set(files) != expected_paths:
        raise ValueError("final manifest file set mismatch")
    for name in expected_paths:
        path = output / name
        mirror_path = mirror / name
        if not path.is_file() or not mirror_path.is_file() or path.read_bytes() != mirror_path.read_bytes() or files.get(name) != _file_digest(path.read_bytes()):
            raise ValueError(f"final publication hash mismatch: {name}")
    for line in (output / journal_name).read_text(encoding="utf-8").splitlines():
        if line:
            try:
                if json.loads(line).get("identity_sha256") != manifest.get("identity_sha256"):
                    raise ValueError("final journal identity mismatch")
            except json.JSONDecodeError as error:
                raise ValueError("final journal is not valid JSONL") from error
    labels = manifest.get("labels")
    if labels is None:
        labels = []
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        raise ValueError("final manifest labels are missing")
    expected_labels = {str(item.get("path")): item for item in labels if isinstance(item, Mapping)}
    primary_labels = {(path.relative_to(output).as_posix()) for path in (output / "predictions" / "labels").glob("*.txt")}
    mirror_labels = {(path.relative_to(mirror).as_posix()) for path in (mirror / "predictions" / "labels").glob("*.txt")}
    if primary_labels != mirror_labels:
        raise ValueError("final label set differs between primary and mirror")
    actual_labels: set[str] = set(primary_labels)
    for relative in sorted(actual_labels):
        path = output / relative
        mirror_path = mirror / relative
        raw = path.read_bytes()
        if mirror_path.read_bytes() != raw:
            raise ValueError(f"final label mirror mismatch: {relative}")
        item = expected_labels.get(relative)
        if item is None or int(item.get("size", -1)) != len(raw) or item.get("sha256") != _file_digest(raw):
            raise ValueError(f"final label hash mismatch: {relative}")
    if actual_labels != set(expected_labels):
        raise ValueError("final label set mismatch")
    return {"state": "PASS", "files": sorted(expected_paths), "labels": sorted(actual_labels)}


def _compare_result_boxes(default_results: Sequence[Any], audit_results: Sequence[Any]) -> dict[str, object]:
    if len(default_results) != len(audit_results):
        return {"state": "FAIL", "reason": "result_count_mismatch", "default_count": len(default_results), "audit_count": len(audit_results)}
    for index, (default, audit) in enumerate(zip(default_results, audit_results)):
        default_boxes = default.boxes.data.detach().cpu() if getattr(default, "boxes", None) is not None else torch.empty((0, 6))
        audit_boxes = audit.boxes.data.detach().cpu() if getattr(audit, "boxes", None) is not None else torch.empty((0, 6))
        if default_boxes.shape != audit_boxes.shape or not torch.equal(default_boxes, audit_boxes):
            return {"state": "FAIL", "reason": "results_boxes_data_mismatch", "batch_index": index}
    return {"state": "PASS", "result_count": len(default_results), "bitwise_boxes_data": True}


def _assert_output_fresh_or_resumable(output: Path, mirror: Path, *, resume: bool) -> None:
    """Reject stale artifacts before a new identity can append to them."""

    output = Path(output)
    mirror = Path(mirror)
    if output.exists() and not output.is_dir() or mirror.exists() and not mirror.is_dir():
        raise ValueError("score/NMS output and mirror must be directories")
    if resume:
        checkpoint_primary = (output / "checkpoint.json").is_file()
        checkpoint_mirror = (mirror / "checkpoint.json").is_file()
        if checkpoint_primary != checkpoint_mirror:
            raise ValueError("resume requires primary/mirror checkpoint")
        if not checkpoint_primary:
            # A smoke run may have durably written its identity-bound default
            # reference before the first audit callback.  The predictor will
            # recreate generation zero after warmup; any other checkpointless
            # output is not safely resumable.
            reference_primary = output / "default_reference.json"
            reference_mirror = mirror / "default_reference.json"
            if reference_primary.is_file() and reference_mirror.is_file() and reference_primary.read_bytes() == reference_mirror.read_bytes():
                return
            raise ValueError("resume requires primary/mirror checkpoint or paired default reference")
        manifests = [path for path in (output / "manifest.json", mirror / "manifest.json") if path.is_file()]
        if manifests:
            if len(manifests) != 2:
                raise ValueError("completed score/NMS output is immutable; publishing manifest mirror is incomplete")
            payloads = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
            states = {str(item.get("publication_state", "")) for item in payloads}
            if states == {"complete"}:
                raise ValueError("completed score/NMS output is immutable; choose a fresh identity")
            if states == {"complete", "publishing"}:
                _repair_mixed_publication(output, mirror)
                return
            if states != {"publishing"} or payloads[0] != payloads[1]:
                raise ValueError("publishing manifest mirror is invalid")
        return
    for root in (output, mirror):
        if root.exists():
            entries = [path.name for path in root.iterdir()]
            if entries:
                raise ValueError(f"score/NMS output is not fresh; stale artifacts={entries}")


def _resolved_train_image_dir(resolved_data_path: Path) -> Path:
    """Resolve the exact fit-view image directory from the resolved data YAML.

    The experiment config's ``raw_images`` points at the unfiltered source in
    some deployments.  The formal audit must instead read the ``train`` view
    bound by ``resolved_data.yaml`` so its image coverage is the registered
    fit-only view.  Relative YAML paths are resolved against the YAML parent.
    """

    path = Path(resolved_data_path).expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("train") is None:
        raise ValueError("resolved-data YAML must expose a train path")
    root_value = payload.get("path", path.parent)
    root = Path(str(root_value))
    if not root.is_absolute():
        root = path.parent / root
    train_value = payload["train"]
    if isinstance(train_value, Sequence) and not isinstance(train_value, (str, bytes)):
        train_value = train_value[0] if train_value else None
    if train_value is None:
        raise ValueError("resolved-data YAML train path is empty")
    train = Path(str(train_value))
    image_dir = train if train.is_absolute() else root / train
    image_dir = image_dir.resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"resolved-data train image directory missing: {image_dir}")
    return image_dir


def _recover_score_checkpoint(output: Path, mirror: Path, identity_sha: str, selected_ids: Sequence[str], *, expected_identity: Mapping[str, object]) -> list[str]:
    checkpoint = output / "checkpoint.json"
    journal = output / "score_nms_audit.jsonl"
    mirror_checkpoint, mirror_journal = mirror / checkpoint.name, mirror / journal.name
    if not all(path.is_file() for path in (checkpoint, journal, mirror_checkpoint, mirror_journal)):
        raise ValueError("primary/mirror score/NMS checkpoint and journal are required for resume")
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    mirror_payload = json.loads(mirror_checkpoint.read_text(encoding="utf-8"))
    if _canonical_sha(expected_identity) != str(identity_sha):
        raise ValueError("expected identity canonical SHA mismatch")
    expected_identity_dict = dict(expected_identity)
    for item in (payload, mirror_payload):
        if item.get("identity_sha256") != identity_sha:
            raise ValueError("score/NMS identity mismatch")
        embedded = item.get("identity")
        if not isinstance(embedded, Mapping) or dict(embedded) != expected_identity_dict or _canonical_sha(embedded) != str(identity_sha):
            raise ValueError("score/NMS embedded identity mismatch")
        ids = [str(value) for value in item.get("completed_image_ids", [])]
        if ids != list(selected_ids[: len(ids)]) or len(ids) != len(set(ids)):
            raise ValueError("score/NMS checkpoint completed IDs are not a valid prefix")
        _validated_checkpoint_rng(item, expected_count=len(ids))

    def read_side(base: Path, item: Mapping[str, object]) -> tuple[list[bytes], list[dict[str, object]], list[int]]:
        side_journal = base / "score_nms_audit.jsonl"
        raw = side_journal.read_bytes()
        offset = int(item.get("journal_offset", -1))
        if offset < 0 or offset > len(raw):
            raise ValueError("score/NMS journal offset is invalid")
        prefix = raw[:offset]
        expected = str(item.get("journal_prefix_sha256", ""))
        if not expected or expected != _file_digest(prefix):
            raise ValueError("score/NMS journal prefix hash mismatch")
        lines = prefix.splitlines(keepends=True)
        records: list[dict[str, object]] = []
        offsets = [0]
        for line in lines:
            record = json.loads(line.decode("utf-8"))
            if record.get("identity_sha256") != identity_sha:
                raise ValueError("score/NMS journal identity mismatch")
            records.append(record)
            offsets.append(offsets[-1] + len(line))
        declared = [str(value) for value in item.get("completed_image_ids", [])]
        if len(records) != len(declared) or [str(value.get("image_id", "")) for value in records] != declared:
            raise ValueError("score/NMS journal/checkpoint prefix mismatch")
        return lines, records, offsets

    primary_lines, primary_records, primary_offsets = read_side(output, payload)
    mirror_lines, mirror_records, mirror_offsets = read_side(mirror, mirror_payload)
    common = min(len(primary_records), len(mirror_records))
    # Find the oldest byte-identical, label-identical generation.  This makes
    # a primary-written/mirror-not-yet-written interruption recoverable.
    for index in range(common):
        if primary_lines[index] != mirror_lines[index]:
            common = index
            break
        for base, other, record in ((output, mirror, primary_records[index]), (mirror, output, mirror_records[index])):
            label_ref = str(record.get("label_path", ""))
            if not label_ref or Path(label_ref).is_absolute() or ".." in Path(label_ref).parts:
                raise ValueError("score/NMS label path is invalid")
            label = base / label_ref
            raw = label.read_bytes() if label.is_file() else b""
            if not label.is_file() and not bool(record.get("label_missing_as_empty", False)):
                raise ValueError("committed score/NMS label is missing")
            if _file_digest(raw) != record.get("label_sha256") or len(raw) != int(record.get("label_size", -1)):
                raise ValueError("score/NMS label hash mismatch")
            other_label = other / label_ref
            if label.is_file() != other_label.is_file():
                common = index
                break
            if label.is_file() and other_label.read_bytes() != raw:
                common = index
                break
        if common == index:
            break
    common_ids = list(selected_ids[:common])
    prefix_snapshots: list[dict[str, object]] = []
    for item in (payload, mirror_payload):
        if int(item.get("rng_completed_count", -1)) == common:
            prefix_snapshots.append(_validated_checkpoint_rng(item, expected_count=common))
    if not prefix_snapshots:
        raise ValueError("no checkpoint RNG snapshot matches common journal prefix")
    prefix_rng = prefix_snapshots[0]
    if any(_rng_state_digest_from_snapshot(item) != _rng_state_digest_from_snapshot(prefix_rng) for item in prefix_snapshots[1:]):
        raise ValueError("common-prefix checkpoint RNG snapshots differ")
    primary_prefix = b"".join(primary_lines[:common])
    mirror_prefix = b"".join(mirror_lines[:common])
    if primary_prefix != mirror_prefix:
        raise ValueError("score/NMS common journal prefix mismatch")
    for base, lines, offsets in ((output, primary_lines, primary_offsets), (mirror, mirror_lines, mirror_offsets)):
        _atomic_write(base / "score_nms_audit.jsonl", b"".join(lines[:common]))
        labels_root = base / "predictions" / "labels"
        committed_refs = {str(record.get("label_path", "")) for record in primary_records[:common]}
        if labels_root.is_dir():
            for label in labels_root.glob("*.txt"):
                if label.relative_to(base).as_posix() not in committed_refs:
                    label.unlink()
    # Recovery never emits a final marker.  Even a complete journal prefix
    # must pass the summary/label/manifest transaction again after a publish
    # interruption.
    recovered = {"schema_version": 1, "state": "running", "publication_state": "running", "identity": payload.get("identity", {}), "identity_sha256": identity_sha, "completed_image_ids": common_ids, "next_position": common, "journal_offset": len(primary_prefix), "journal_prefix_sha256": _file_digest(primary_prefix), "rng_completed_count": common, "rng_snapshot": _rng_snapshot_payload(prefix_rng), "rng_digest": _rng_state_digest_from_snapshot(prefix_rng)}
    _atomic_write(checkpoint, (json.dumps(recovered, sort_keys=True) + "\n").encode())
    _atomic_write(mirror_checkpoint, checkpoint.read_bytes())
    return common_ids


def run_fit_score_nms_audit(
    *,
    config_path: Path,
    resolved_data_path: Path,
    fit_ids_path: Path,
    development_ids_path: Path,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    raw_label_dir: Path,
    expected_raw_label_sha256: str,
    output_dir: Path,
    mirror_dir: Path,
    mode: str = "smoke",
    device: str = "cpu",
    resume: bool = False,
    stop_after: int | None = None,
) -> dict[str, object]:
    """Run the real clean-P2 fit-only score/NMS audit."""

    if mode not in {"smoke", "benchmark32", "full"}:
        raise ValueError("mode must be smoke, benchmark32, or full")
    try:
        import importlib.metadata
        version = importlib.metadata.version("ultralytics")
    except Exception as error:  # pragma: no cover
        raise ValueError("Ultralytics 8.4.98 is required") from error
    if version != SUPPORTED_ULTRALYTICS_VERSION:
        raise ValueError(f"Ultralytics version must be {SUPPORTED_ULTRALYTICS_VERSION}, got {version}")
    torchvision_identity = _torchvision_backend_identity()
    root = Path(__file__).resolve().parents[2]
    config = load_baseline_config(Path(config_path), repository_root=root)
    from ifdr_yolo.experiments.p2_fit_reference import validate_fit_development_split, validate_plain_p2_model, validate_primary_checkpoint
    validate_plain_p2_model(config)
    split = validate_fit_development_split(config, Path(fit_ids_path), Path(development_ids_path))
    checkpoint = validate_primary_checkpoint(Path(checkpoint_path))
    actual_checkpoint_sha = sha256_file(checkpoint)
    if actual_checkpoint_sha != str(expected_checkpoint_sha256).lower():
        raise ValueError("checkpoint SHA256 mismatch")
    fit_ids, development_ids = split.fit_ids, split.development_ids
    raw_label_identity = _validate_raw_label_view(Path(raw_label_dir), fit_ids)
    actual_raw_label_sha = _directory_sha256(Path(raw_label_dir))
    if actual_raw_label_sha != str(expected_raw_label_sha256).lower():
        raise ValueError("raw-label directory SHA256 mismatch")
    selected_ids = _select_fit_ids_for_mode(mode, fit_ids)
    train_image_dir = _resolved_train_image_dir(Path(resolved_data_path))
    image_paths = [path for path in train_image_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    id_paths: dict[str, Path] = {}
    for path in image_paths:
        if path.stem in id_paths and id_paths[path.stem] != path:
            raise ValueError(f"resolved-data train view contains duplicate image ID: {path.stem}")
        id_paths[path.stem] = path
    available_ids = set(id_paths)
    if available_ids != set(fit_ids):
        missing = sorted(set(fit_ids) - available_ids)[:5]
        unexpected = sorted(available_ids - set(fit_ids))[:5]
        raise ValueError(f"resolved-data train view does not exactly match fit IDs; missing={missing}, unexpected={unexpected}")
    upstream_sha, upstream_files = _upstream_source_hashes()
    identity = {
        "fit_ids_sha256": sha256_file(Path(fit_ids_path)),
        "fit_ids_ordered_sha256": _ordered_ids_sha(fit_ids),
        "fit_ids_count": len(fit_ids),
        "development_ids_sha256": sha256_file(Path(development_ids_path)),
        "development_ids_ordered_sha256": _ordered_ids_sha(development_ids),
        "development_ids_count": len(development_ids),
        "selected_ids_sha256": _ordered_ids_sha(selected_ids),
        "selected_ids_count": len(selected_ids),
        "mode": str(mode),
        "checkpoint_sha256": actual_checkpoint_sha,
        "config_sha256": sha256_file(Path(config_path)),
        "resolved_data_sha256": sha256_file(Path(resolved_data_path)),
        "resolved_train_image_dir": str(train_image_dir),
        "raw_label_dir_sha256": actual_raw_label_sha,
        "raw_label_view": raw_label_identity,
        "fit_image_manifest_sha256": _fit_image_manifest_sha256(Path(resolved_data_path), fit_ids),
        "code_sha256": _source_code_identity(),
        "local_source_files": _local_source_hashes(root),
        "upstream_source_sha256": upstream_sha,
        "upstream_source_files": upstream_files,
        "ultralytics_version": version,
        "runtime": {**_runtime_identity(device), **torchvision_identity},
        "protocol": {"batch": 1, "rect": True, "imgsz": 640, "augment": False, "conf": 0.001, "iou": 0.7, "max_nms": 30000, "max_det": 300, "fp32": True, "fit_count": len(fit_ids), "development_count": len(development_ids)},
    }
    if mode == "benchmark32":
        identity.update(_benchmark32_identity_fields(selected_ids))
    elif mode == "full":
        identity.update(_full_audit_identity_fields(selected_ids))
    identity_sha = _canonical_sha(identity)
    output, mirror = Path(output_dir).resolve(), Path(mirror_dir).resolve()
    if output == mirror or output.is_relative_to(mirror) or mirror.is_relative_to(output):
        raise ValueError("primary and mirror directories must be separate")
    _assert_output_fresh_or_resumable(output, mirror, resume=resume)
    output.mkdir(parents=True, exist_ok=True)
    mirror.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output / "checkpoint.json"
    if checkpoint_file.exists() and not resume:
        raise ValueError("existing score/NMS checkpoint requires resume=True")
    completed = _recover_score_checkpoint(output, mirror, identity_sha, selected_ids, expected_identity=identity) if checkpoint_file.exists() else []
    if mode in {"benchmark32", "full"}:
        stop_after = _validate_mode_stop_after(mode, stop_after, completed_count=len(completed))
    if resume and not checkpoint_file.exists() and mode != "smoke":
        raise ValueError("resume without checkpoint is supported only for smoke default-reference recovery")
    persisted_checkpoint_rng: dict[str, object] | None = None
    if checkpoint_file.exists():
        checkpoint_payload = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        persisted_checkpoint_rng = _validated_checkpoint_rng(checkpoint_payload, expected_count=len(completed))
    remaining = tuple(value for value in selected_ids if value not in completed)
    source_file = output / "source_remaining.txt"
    _atomic_write(source_file, ("\n".join(str(id_paths[value].resolve()) for value in remaining) + ("\n" if remaining else "")).encode())
    state = _ScoreNMSRunState(output=output, mirror=mirror, identity=identity, identity_sha=identity_sha, selected_ids=selected_ids, completed=completed, stop_after=stop_after, rng_snapshot=persisted_checkpoint_rng)
    default_root: Path | None = None
    default_results: list[Any] = []
    audit_results: list[Any] = []
    non_interference: dict[str, object] | None = None
    default_reference: dict[str, object] | None = None
    persisted_smoke_summary: dict[str, object] | None = None
    if resume and mode == "smoke":
        default_reference = _load_default_reference(output, mirror, identity_sha, selected_ids)
        snapshot_payload = default_reference.get("rng_initial_snapshot")
        if not isinstance(snapshot_payload, Mapping):
            raise ValueError("default reference RNG snapshot is missing")
        paired_rng_initial = _rng_snapshot_from_payload(snapshot_payload)
        if persisted_checkpoint_rng is None:
            state.set_rng_snapshot(paired_rng_initial)
            state._generation_zero_pending = True
        # If a crash happened after all image commits but before the mirror's
        # final-manifest write, no predictor will run on resume.  Preserve the
        # already verified smoke gates rather than replacing them with None.
        if not remaining:
            persisted_smoke_summary = _load_persisted_smoke_summary(output, mirror, identity_sha)
            non_interference = persisted_smoke_summary["non_interference"]
    if remaining:
        from ultralytics import YOLO
        if mode == "smoke" and not resume:
            default_root = output.parent / f"{output.name}.default"
            if default_root.exists():
                raise ValueError("default-vs-audit smoke directory already exists; use a fresh output identity")
            default_model = YOLO(str(checkpoint))
            audit_model = YOLO(str(checkpoint))
            paired_rng_initial = _capture_rng_state()
            default_predictor = build_rng_boundary_predictor_class(paired_rng_initial)
            default_stream = default_model.predict(source=str(source_file), predictor=default_predictor, project=str(default_root), name="predictions", exist_ok=True, batch=1, rect=True, imgsz=640, augment=False, conf=0.001, iou=0.7, max_det=300, save_txt=True, save_conf=True, device=device, stream=True, verbose=False)
            for item in default_stream:
                default_results.append(item)
            default_run = {
                "rng_initial": _rng_state_digest_from_snapshot(paired_rng_initial),
                "rng_final": _rng_state_digest(),
                "model_post": _model_state_digest(default_model),
                **_model_runtime_facts(default_model, device=device, backend=str(torchvision_identity["backend"])),
            }
            default_reference = {
                "schema_version": 1,
                "identity_sha256": identity_sha,
                "selected_ids": [str(value) for value in selected_ids],
                "boxes": {str(image_id): _result_boxes_payload(item) for image_id, item in zip(selected_ids, default_results)},
                "labels": {str(image_id): _label_payload(default_root, str(image_id)) for image_id in selected_ids},
                "control_run": default_run,
                "rng_initial_snapshot": _rng_snapshot_payload(paired_rng_initial),
            }
            _write_default_reference(output, mirror, default_reference)
            state.set_rng_snapshot(paired_rng_initial)
            state._generation_zero_pending = True
        elif mode == "benchmark32" and not resume:
            audit_model = YOLO(str(checkpoint))
            benchmark_rng_initial = _initialize_benchmark32_rng_boundary()
            state.set_rng_snapshot(benchmark_rng_initial)
            state._generation_zero_pending = True
        elif mode == "full" and not resume:
            audit_model = YOLO(str(checkpoint))
            full_rng_initial = _initialize_full_audit_rng_boundary()
            state.set_rng_snapshot(full_rng_initial)
            state._generation_zero_pending = True
        else:
            # Build/setup may consume ambient RNG.  The predictor's final
            # on_predict_start callback restores the persisted post-prefix
            # snapshot after warmup and immediately before the first batch.
            if persisted_checkpoint_rng is None:
                raise ValueError("score/NMS resume requires a durable RNG snapshot")
            audit_model = _construct_resume_model(YOLO, checkpoint, persisted_checkpoint_rng)
        predictor_class = build_score_nms_predictor_class(state, raw_label_dir=Path(raw_label_dir))
        model = audit_model
        stream = model.predict(source=str(source_file), predictor=predictor_class, project=str(output), name="predictions", exist_ok=True, batch=1, rect=True, imgsz=640, augment=False, conf=0.001, iou=0.7, max_det=300, save_txt=True, save_conf=True, device=device, stream=True, verbose=False)
        # ``stream=True`` keeps only one Results object in memory.  Do not
        # leave the generator lazy: every selected ID must reach the
        # postprocess/write_results commit boundary before returning.
        for item in stream:
            audit_results.append(item)
        if mode == "smoke":
            audit_run = {
                "rng_initial": _rng_state_digest_from_snapshot(paired_rng_initial),
                "rng_final": _rng_state_digest(),
                "model_post": _model_state_digest(model),
                **_model_runtime_facts(model, device=device, backend=str(torchvision_identity["backend"])),
            }
            control_run = default_run if not resume else default_reference.get("control_run") if isinstance(default_reference, Mapping) else None
            if not isinstance(control_run, Mapping):
                raise ValueError("default reference control runtime is missing")
            non_interference = _compare_controlled_run_gates(control_run, audit_run, expected_device=device)
            if non_interference["state"] != "PASS":
                raise RuntimeError("default/audit controlled runtime gate mismatch")
    if state.completed != list(selected_ids):
        raise ValueError("score/NMS loader did not provide exact selected fit coverage")
    records: list[dict[str, object]] = []
    journal = output / "score_nms_audit.jsonl"
    if journal.is_file():
        for line in journal.read_text(encoding="utf-8").splitlines():
            records.append(json.loads(line))
    flat_gt_rows = [gt for record in records for gt in record.get("gt_rows", [])]
    summary = summarize_score_nms_estimands(flat_gt_rows, selected_ids, reps=10000) if mode in {"benchmark32", "full"} else {"state": "smoke_not_evaluated", "processed_fit_count": len(selected_ids)}
    if mode == "benchmark32":
        summary = _seal_benchmark32_summary(summary)
    if mode == "smoke" and default_root is not None:
        summary["default_vs_audit_labels"] = compare_saved_label_files(default_root, output, selected_ids)
        if summary["default_vs_audit_labels"]["state"] != "PASS":
            raise ValueError("default-vs-audit label bytes differ")
        summary["default_vs_audit_results"] = _compare_result_boxes(default_results, audit_results)
        if summary["default_vs_audit_results"]["state"] != "PASS":
            raise ValueError("default-vs-audit Results.boxes.data differ")
        summary["non_interference"] = non_interference
    if mode == "smoke":
        if default_reference is None:
            raise ValueError("default reference is missing")
        summary["default_reference_labels"] = _compare_reference_labels(default_reference, output, selected_ids)
        summary["default_reference_boxes"] = _compare_reference_boxes(default_reference, records, selected_ids)
        if summary["default_reference_labels"]["state"] != "PASS" or summary["default_reference_boxes"]["state"] != "PASS":
            raise ValueError("default reference comparison failed")
        summary["default_vs_audit_labels"] = summary["default_reference_labels"]
        summary["default_vs_audit_results"] = summary["default_reference_boxes"]
        summary["non_interference"] = non_interference
        for key in ("default_vs_audit_labels", "default_vs_audit_results", "non_interference"):
            value = summary.get(key)
            if not isinstance(value, Mapping) or value.get("state") != "PASS":
                raise ValueError(f"smoke publication gate is not PASS: {key}")
    _atomic_write(output / "summary.json", (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
    _atomic_write(mirror / "summary.json", (output / "summary.json").read_bytes())
    _atomic_write(output / "summary.csv", _summary_long_csv(summary))
    _atomic_write(mirror / "summary.csv", (output / "summary.csv").read_bytes())
    state.complete()
    return {"state": "complete", "identity": identity, "identity_sha256": identity_sha, "fit_count": len(fit_ids), "processed_fit_count": len(selected_ids), "development_count": len(development_ids), "intersection_count": 0, "summary": summary, "output_dir": str(output), "mirror_dir": str(mirror)}


def bootstrap_cluster_contrast(
    frames: Sequence[Mapping[str, object]],
    *,
    numerator_key: str,
    denominator_key: str,
    reps: int = 10000,
    seed: int = 20260812,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Image-cluster bootstrap for a ratio-of-sums contrast."""

    if not frames or reps <= 0:
        raise ValueError("frames and reps must be positive")
    values = [(float(frame.get(numerator_key, 0.0)), float(frame.get(denominator_key, 0.0))) for frame in frames]
    if sum(item[1] for item in values) <= 0:
        raise ValueError("zero aggregate denominator")
    observed = sum(item[0] for item in values) / sum(item[1] for item in values)
    generator = torch.Generator().manual_seed(int(seed))
    draws: list[float] = []
    for _ in range(int(reps)):
        indices = torch.randint(0, len(values), (len(values),), generator=generator).tolist()
        denominator = sum(values[index][1] for index in indices)
        if denominator <= 0:
            raise ValueError("zero bootstrap denominator")
        draws.append(sum(values[index][0] for index in indices) / denominator)
    sorted_draws = sorted(draws)
    lo = sorted_draws[max(0, min(len(draws) - 1, math.floor((alpha / 2) * len(draws))))]
    hi = sorted_draws[max(0, min(len(draws) - 1, math.ceil((1 - alpha / 2) * len(draws)) - 1))]
    return {"observed": observed, "ci95": [lo, hi], "bootstrap_replicates": int(reps), "bootstrap_seed": int(seed), "estimand": f"sum({numerator_key})/sum({denominator_key})"}


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _append_line(primary: Path, mirror: Path, payload: Mapping[str, object]) -> bytes:
    line = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    primary.parent.mkdir(parents=True, exist_ok=True)
    with primary.open("ab") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
    mirror.parent.mkdir(parents=True, exist_ok=True)
    with mirror.open("ab") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
    return line


def run_synthetic_score_nms_audit(
    fit_ids: Sequence[str],
    output_dir: Path,
    mirror_dir: Path,
    identity: Mapping[str, object],
    *,
    resume: bool = False,
    stop_after: int | None = None,
) -> dict[str, object]:
    """Deterministic journal/resume harness used by focused tests and smoke."""

    selected = tuple(str(value) for value in fit_ids)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("fit IDs must be non-empty and unique")
    output, mirror = Path(output_dir), Path(mirror_dir)
    if output.resolve() == mirror.resolve() or output.resolve().is_relative_to(mirror.resolve()) or mirror.resolve().is_relative_to(output.resolve()):
        raise ValueError("primary and mirror directories must be separate")
    output.mkdir(parents=True, exist_ok=True)
    mirror.mkdir(parents=True, exist_ok=True)
    identity_sha = _canonical_sha(identity)
    journal, checkpoint = output / "audit.jsonl", output / "checkpoint.json"
    mirror_journal, mirror_checkpoint = mirror / journal.name, mirror / checkpoint.name
    completed: list[str] = []
    if checkpoint.exists():
        if not resume:
            raise ValueError("existing audit checkpoint requires resume=True")
        checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        if checkpoint_payload.get("identity_sha256") != identity_sha:
            raise ValueError("identity mismatch on resume")
        if not mirror_checkpoint.exists() or mirror_checkpoint.read_bytes() != checkpoint.read_bytes():
            raise ValueError("primary/mirror checkpoint mismatch")
        completed = [str(value) for value in checkpoint_payload.get("completed_image_ids", [])]
        if completed != list(selected[: len(completed)]):
            raise ValueError("checkpoint completed IDs are not a valid prefix")
        if not journal.exists() or not mirror_journal.exists():
            raise ValueError("checkpoint journal is missing")
        if journal.read_bytes() != mirror_journal.read_bytes():
            raise ValueError("primary/mirror journal mismatch")
        expected_lines = journal.read_bytes().splitlines()
        if len(expected_lines) != len(completed):
            raise ValueError("checkpoint journal prefix mismatch")
        for expected_id, raw_line in zip(completed, expected_lines):
            payload = json.loads(raw_line.decode("utf-8"))
            if payload.get("identity_sha256") != identity_sha or payload.get("image_id") != expected_id:
                raise ValueError("checkpoint journal prefix mismatch")
    elif resume:
        raise ValueError("resume requested without checkpoint")
    elif journal.exists() and journal.stat().st_size:
        raise ValueError("existing audit journal has no checkpoint")
    for image_id in selected[len(completed) :]:
        payload = {"schema_version": 1, "identity_sha256": identity_sha, "image_id": image_id, "rows": []}
        _append_line(journal, mirror_journal, payload)
        completed.append(image_id)
        checkpoint_payload = {"schema_version": 1, "state": "running", "identity": dict(identity), "identity_sha256": identity_sha, "completed_image_ids": completed, "journal_offset": journal.stat().st_size}
        _atomic_write(checkpoint, (json.dumps(checkpoint_payload, sort_keys=True) + "\n").encode())
        _atomic_write(mirror_checkpoint, checkpoint.read_bytes())
        if stop_after is not None and len(completed) >= int(stop_after) and len(completed) < len(selected):
            raise RuntimeError("interrupted score/NMS audit")
    checkpoint_payload["state"] = "publishing"
    checkpoint_payload["publication_state"] = "publishing"
    _atomic_write(checkpoint, (json.dumps(checkpoint_payload, sort_keys=True) + "\n").encode())
    _atomic_write(mirror_checkpoint, checkpoint.read_bytes())
    files = {"audit.jsonl": hashlib.sha256(journal.read_bytes()).hexdigest(), "checkpoint.json": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    manifest = {"schema_version": 1, "publication_state": "publishing", "identity_sha256": identity_sha, "files": files}
    _atomic_write(output / "manifest.json", (json.dumps(manifest, sort_keys=True) + "\n").encode())
    _atomic_write(mirror / "manifest.json", (output / "manifest.json").read_bytes())
    checkpoint_payload["state"] = "complete"
    checkpoint_payload["publication_state"] = "complete"
    _atomic_write(checkpoint, (json.dumps(checkpoint_payload, sort_keys=True) + "\n").encode())
    _atomic_write(mirror_checkpoint, checkpoint.read_bytes())
    files["checkpoint.json"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    final_manifest = {"schema_version": 1, "publication_state": "complete", "identity_sha256": identity_sha, "files": files}
    _atomic_write(output / "manifest.json", (json.dumps(final_manifest, sort_keys=True) + "\n").encode())
    _atomic_write(mirror / "manifest.json", (output / "manifest.json").read_bytes())
    _validate_final_publication(output, mirror)
    return {"state": "complete", "identity_sha256": identity_sha, "completed_image_ids": completed, "output_dir": str(output), "mirror_dir": str(mirror)}


def run_synthetic_benchmark32_audit(
    fit_ids: Sequence[str],
    output_dir: Path,
    mirror_dir: Path,
    identity: Mapping[str, object],
    *,
    resume: bool = False,
    stop_after: int | None = None,
) -> dict[str, object]:
    """Exercise the formal 32-image publication/recovery contract in tests."""

    selected = _select_fit_ids_for_mode("benchmark32", fit_ids)
    bound_identity = {**dict(identity), "mode": "benchmark32", **_benchmark32_identity_fields(selected)}
    identity_sha = _canonical_sha(bound_identity)
    output, mirror = Path(output_dir), Path(mirror_dir)
    if output.resolve() == mirror.resolve() or output.resolve().is_relative_to(mirror.resolve()) or mirror.resolve().is_relative_to(output.resolve()):
        raise ValueError("primary and mirror directories must be separate")
    output.mkdir(parents=True, exist_ok=True)
    mirror.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.json"
    if checkpoint.exists():
        if not resume:
            raise ValueError("existing benchmark32 checkpoint requires resume=True")
        completed = _recover_score_checkpoint(output, mirror, identity_sha, selected, expected_identity=bound_identity)
        checkpoint_rng = _validated_checkpoint_rng(json.loads(checkpoint.read_text(encoding="utf-8")), expected_count=len(completed))
        state = _ScoreNMSRunState(output=output, mirror=mirror, identity=bound_identity, identity_sha=identity_sha, selected_ids=selected, completed=completed, stop_after=_validate_benchmark32_stop_after(stop_after, completed_count=len(completed)), rng_snapshot=checkpoint_rng)
    else:
        if resume:
            raise ValueError("benchmark32 resume requested without checkpoint")
        initial_rng = _initialize_benchmark32_rng_boundary()
        state = _ScoreNMSRunState(output=output, mirror=mirror, identity=bound_identity, identity_sha=identity_sha, selected_ids=selected, completed=(), stop_after=_validate_benchmark32_stop_after(stop_after, completed_count=0), rng_snapshot=initial_rng, generation_zero_pending=True)
        state.initialize_generation_zero(initial_rng)
    for image_id in selected[len(state.completed) :]:
        label = output / "predictions" / "labels" / f"{image_id}.txt"
        _atomic_write(label, b"")
        state.commit({"image_id": image_id, "gt_rows": []}, label)
        torch.manual_seed(BENCHMARK32_RNG_SEED + len(state.completed))
        state.checkpoint_after_batch(_capture_rng_state())
    records = [json.loads(line) for line in (output / "score_nms_audit.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = _seal_benchmark32_summary(summarize_score_nms_estimands([gt for record in records for gt in record.get("gt_rows", [])], selected, reps=2))
    _atomic_write(output / "summary.json", (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
    _atomic_write(mirror / "summary.json", (output / "summary.json").read_bytes())
    _atomic_write(output / "summary.csv", _summary_long_csv(summary))
    _atomic_write(mirror / "summary.csv", (output / "summary.csv").read_bytes())
    state.complete()
    return {"state": "complete", "identity": bound_identity, "identity_sha256": identity_sha}


def run_synthetic_full_audit(
    fit_ids: Sequence[str],
    output_dir: Path,
    mirror_dir: Path,
    identity: Mapping[str, object],
    *,
    resume: bool = False,
    stop_after: int | None = None,
    synthetic_prefix_count: int | None = None,
) -> dict[str, object]:
    """Exercise the formal full-run recovery contract without model inference."""

    selected = _select_fit_ids_for_mode("full", fit_ids)
    if synthetic_prefix_count is None:
        synthetic_prefix_count = len(selected)
    if isinstance(synthetic_prefix_count, bool) or not isinstance(synthetic_prefix_count, int) or not 1 <= synthetic_prefix_count <= len(selected):
        raise ValueError("synthetic full prefix count must be in 1..3341")
    synthetic_ids = selected[:synthetic_prefix_count]
    bound_identity = {**dict(identity), "mode": "full", **_full_audit_identity_fields(selected)}
    identity_sha = _canonical_sha(bound_identity)
    output, mirror = Path(output_dir), Path(mirror_dir)
    if output.resolve() == mirror.resolve() or output.resolve().is_relative_to(mirror.resolve()) or mirror.resolve().is_relative_to(output.resolve()):
        raise ValueError("primary and mirror directories must be separate")
    output.mkdir(parents=True, exist_ok=True)
    mirror.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.json"
    if checkpoint.exists():
        if not resume:
            raise ValueError("existing full checkpoint requires resume=True")
        completed = _recover_score_checkpoint(output, mirror, identity_sha, synthetic_ids, expected_identity=bound_identity)
        checkpoint_rng = _validated_checkpoint_rng(json.loads(checkpoint.read_text(encoding="utf-8")), expected_count=len(completed))
        _restore_rng_state(checkpoint_rng)
        state = _ScoreNMSRunState(output=output, mirror=mirror, identity=bound_identity, identity_sha=identity_sha, selected_ids=synthetic_ids, completed=completed, stop_after=_validate_mode_stop_after("full", stop_after, completed_count=len(completed)), rng_snapshot=checkpoint_rng)
    else:
        if resume:
            raise ValueError("full resume requested without checkpoint")
        initial_rng = _initialize_full_audit_rng_boundary()
        state = _ScoreNMSRunState(output=output, mirror=mirror, identity=bound_identity, identity_sha=identity_sha, selected_ids=synthetic_ids, completed=(), stop_after=_validate_mode_stop_after("full", stop_after, completed_count=0), rng_snapshot=initial_rng, generation_zero_pending=True)
        state.initialize_generation_zero(initial_rng)
    for image_id in synthetic_ids[len(state.completed) :]:
        label = output / "predictions" / "labels" / f"{image_id}.txt"
        _atomic_write(label, b"")
        state.commit({"image_id": image_id, "gt_rows": []}, label)
        torch.rand(1)
        state.checkpoint_after_batch(_capture_rng_state())
    records = [json.loads(line) for line in (output / "score_nms_audit.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = summarize_score_nms_estimands([gt for record in records for gt in record.get("gt_rows", [])], synthetic_ids, reps=2)
    _atomic_write(output / "summary.json", (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
    _atomic_write(mirror / "summary.json", (output / "summary.json").read_bytes())
    _atomic_write(output / "summary.csv", _summary_long_csv(summary))
    _atomic_write(mirror / "summary.csv", (output / "summary.csv").read_bytes())
    state.complete()
    return {"state": "complete", "identity": bound_identity, "identity_sha256": identity_sha}


__all__ = [
    "LEVEL_NAMES",
    "STAGES",
    "LevelSlice",
    "NMSReplay",
    "derive_level_slices",
    "replay_nms_with_stages",
    "score_nms_survival_row",
    "bootstrap_cluster_contrast",
    "run_synthetic_score_nms_audit",
    "run_synthetic_benchmark32_audit",
    "run_synthetic_full_audit",
]
