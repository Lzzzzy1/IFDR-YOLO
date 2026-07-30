from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import colorstr

from ifdr_yolo.data.interventions.sampler import (
    DeterministicInterventionSampler,
    SamplingPolicy,
)
from ifdr_yolo.data.interventions.targets import factor_target_for_spec
from ifdr_yolo.data.interventions.transforms import apply_intervention


FACTOR_TARGET_KEY = "ifdr_factor_target"
FACTOR_WEIGHT_KEY = "ifdr_factor_weight"


def _epoch_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("epoch must be a non-negative integer")
    return value


class SharedEpoch:
    """Small process-shared epoch clock for persistent data workers."""

    def __init__(self, value: int = 0) -> None:
        self._value = multiprocessing.Value("q", _epoch_value(value))

    def get(self) -> int:
        with self._value.get_lock():
            return int(self._value.value)

    def set(self, value: int) -> None:
        value = _epoch_value(value)
        with self._value.get_lock():
            self._value.value = value


def _stable_index(size: int, *parts: object) -> int:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % size


def _normalized_xyxy(labels: dict[str, Any]) -> np.ndarray:
    image = labels["img"]
    height, width = image.shape[:2]
    instances = deepcopy(labels.get("instances"))
    if instances is None or len(instances) == 0:
        return np.empty((0, 4), dtype=np.float32)
    instances.convert_bbox(format="xyxy")
    boxes = instances.bboxes.astype(np.float32, copy=True)
    if not instances.normalized:
        boxes[:, [0, 2]] /= width
        boxes[:, [1, 3]] /= height
    return np.clip(boxes, 0.0, 1.0)


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    left = np.maximum(box[0], boxes[:, 0])
    top = np.maximum(box[1], boxes[:, 1])
    right = np.minimum(box[2], boxes[:, 2])
    bottom = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(right - left, 0.0) * np.maximum(
        bottom - top,
        0.0,
    )
    box_area = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
    boxes_area = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
        boxes[:, 3] - boxes[:, 1],
        0.0,
    )
    return intersection / np.maximum(box_area + boxes_area - intersection, 1e-9)


def _background_region(
    object_region: tuple[float, float, float, float],
    all_boxes: np.ndarray,
    *,
    selector: int,
) -> tuple[float, float, float, float]:
    width = max(object_region[2] - object_region[0], 0.05)
    height = max(object_region[3] - object_region[1], 0.05)
    width = min(width, 0.45)
    height = min(height, 0.45)
    candidates: list[tuple[float, float, float, float]] = []
    for center_y in np.linspace(height / 2, 1.0 - height / 2, 5):
        for center_x in np.linspace(width / 2, 1.0 - width / 2, 5):
            candidates.append(
                (
                    float(center_x - width / 2),
                    float(center_y - height / 2),
                    float(center_x + width / 2),
                    float(center_y + height / 2),
                )
            )
    overlaps = np.array(
        [
            float(_box_iou(np.asarray(candidate), all_boxes).max(initial=0.0))
            for candidate in candidates
        ]
    )
    minimum = overlaps.min()
    best = np.flatnonzero(np.isclose(overlaps, minimum, atol=1e-12))
    return candidates[int(best[selector % len(best)])]


def _sampling_proxy(box: tuple[float, float, float, float], height: int) -> float:
    box_height = max((box[3] - box[1]) * height, 0.0)
    return float(np.clip((64.0 - box_height) / 60.0, 0.0, 1.0))


class IFDRInterventionTransform:
    """Apply one reproducible object/background intervention to a YOLO sample."""

    def __init__(
        self,
        *,
        base_seed: int,
        epoch_state: SharedEpoch,
        enabled: bool,
        policy: SamplingPolicy | None = None,
    ) -> None:
        if not isinstance(epoch_state, SharedEpoch):
            raise ValueError("epoch_state must be SharedEpoch")
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        self.epoch_state = epoch_state
        self.enabled = enabled
        self.sampler = DeterministicInterventionSampler(
            base_seed=base_seed,
            policy=policy,
        )

    def _empty_maps(self, image: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = image.shape[:2]
        shape = (2, height, width)
        return (
            torch.zeros(shape, dtype=torch.float32),
            torch.zeros(shape, dtype=torch.float32),
        )

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        image = labels.get("img")
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise ValueError("IFDR transform requires a uint8 HWC image")
        if not self.enabled:
            target, weight = self._empty_maps(image)
            labels[FACTOR_TARGET_KEY] = target
            labels[FACTOR_WEIGHT_KEY] = weight
            labels["ifdr_spec"] = "disabled"
            return labels

        boxes = _normalized_xyxy(labels)
        if len(boxes) == 0:
            target, weight = self._empty_maps(image)
            labels[FACTOR_TARGET_KEY] = target
            labels[FACTOR_WEIGHT_KEY] = weight
            labels["ifdr_spec"] = "no_objects"
            return labels

        epoch = self.epoch_state.get()
        image_id = Path(str(labels.get("im_file", "unknown"))).stem
        object_id = _stable_index(
            len(boxes),
            "ifdr-object-v1",
            self.sampler.base_seed,
            image_id,
            epoch,
        )
        object_region = tuple(float(value) for value in boxes[object_id])
        selector = _stable_index(
            1 << 31,
            "ifdr-background-v1",
            self.sampler.base_seed,
            image_id,
            object_id,
            epoch,
        )
        background_region = _background_region(
            object_region,
            boxes,
            selector=selector,
        )
        object_spec, background_spec = self.sampler.sample_matched_pair(
            image_id=image_id,
            object_id=object_id,
            epoch=epoch,
            slot=0,
            object_region=object_region,
            background_region=background_region,
        )
        spec = object_spec if object_spec.seed % 2 == 0 else background_spec
        natural_sampling = (
            _sampling_proxy(object_region, image.shape[0])
            if spec is object_spec
            else 0.0
        )
        target = factor_target_for_spec(
            spec,
            natural_sampling=natural_sampling,
            natural_occlusion=0.0,
        )
        applied = apply_intervention(image, spec, target)
        labels["img"] = applied.image
        labels[FACTOR_TARGET_KEY] = torch.from_numpy(
            np.stack(
                (applied.sampling_target, applied.visibility_target),
                axis=0,
            )
        )
        labels[FACTOR_WEIGHT_KEY] = torch.from_numpy(
            np.stack(
                (applied.sampling_weight, applied.visibility_weight),
                axis=0,
            )
        )
        labels["ifdr_spec"] = json.dumps(
            spec.to_payload(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return labels


def collate_ifdr_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated = YOLODataset.collate_fn(batch)
    for key in (FACTOR_TARGET_KEY, FACTOR_WEIGHT_KEY):
        values = collated.get(key)
        if not isinstance(values, tuple):
            raise RuntimeError(f"{key} was not preserved by collation")
        collated[key] = torch.stack(values, dim=0)
    return collated


class IFDRYOLODataset(YOLODataset):
    """YOLO detection dataset with project-owned dense factor targets."""

    collate_fn = staticmethod(collate_ifdr_batch)

    def __init__(
        self,
        *args,
        intervention_seed: int,
        interventions_enabled: bool,
        intervention_policy: SamplingPolicy | None = None,
        **kwargs,
    ) -> None:
        self.epoch_state = SharedEpoch()
        self.intervention_transform = IFDRInterventionTransform(
            base_seed=intervention_seed,
            epoch_state=self.epoch_state,
            enabled=interventions_enabled,
            policy=intervention_policy,
        )
        super().__init__(*args, **kwargs)

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        transforms.insert(-1, self.intervention_transform)
        return transforms

    def set_epoch(self, epoch: int) -> None:
        self.epoch_state.set(epoch)


def build_ifdr_dataset(
    cfg: object,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    *,
    mode: str,
    rect: bool,
    stride: int,
    intervention_seed: int,
    interventions_enabled: bool,
) -> IFDRYOLODataset:
    """Build an IFDR dataset using the locked Ultralytics 8.4.98 contract."""

    if mode not in {"train", "val"}:
        raise ValueError("mode must be train or val")
    fraction = cfg.fraction if mode == "train" else 1.0
    return IFDRYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=fraction,
        intervention_seed=intervention_seed,
        interventions_enabled=interventions_enabled,
    )
