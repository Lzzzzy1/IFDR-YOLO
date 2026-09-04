from __future__ import annotations

from dataclasses import dataclass
import math

from ifdr_yolo.data.yolo_export import YoloRow


BDD100K_CLASS_TO_ID = {
    "car": 0,
    "pedestrian": 1,
    "rider": 2,
}
BDD100K_IGNORED_CATEGORIES = {
    "bicycle",
    "bus",
    "motorcycle",
    "traffic light",
    "traffic sign",
    "train",
    "truck",
}


@dataclass(frozen=True)
class BDD100KObject:
    category: str
    class_id: int
    xyxy: tuple[float, float, float, float]
    occluded: bool
    truncated: bool
    size_bin: str
    yolo_row: YoloRow


@dataclass(frozen=True)
class BDD100KFrame:
    name: str
    weather: str
    scene: str
    timeofday: str
    objects: tuple[BDD100KObject, ...]


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _coordinate(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be finite")
    return float(value)


def _size_bin(height: float) -> str:
    if height <= 40.0:
        return "small"
    if height <= 96.0:
        return "medium"
    return "large"


def parse_bdd100k_frame(
    frame: object,
    *,
    image_width: int,
    image_height: int,
) -> BDD100KFrame:
    if (
        isinstance(image_width, bool)
        or not isinstance(image_width, int)
        or image_width <= 0
        or isinstance(image_height, bool)
        or not isinstance(image_height, int)
        or image_height <= 0
    ):
        raise ValueError("image dimensions must be positive integers")
    if not isinstance(frame, dict):
        raise ValueError("BDD100K frame must be a mapping")
    attributes = frame.get("attributes")
    labels = frame.get("labels")
    if not isinstance(attributes, dict) or not isinstance(labels, list):
        raise ValueError("BDD100K frame requires attributes and labels")

    objects: list[BDD100KObject] = []
    for label in labels:
        if not isinstance(label, dict):
            raise ValueError("BDD100K label must be a mapping")
        category = _text(label.get("category"), "label.category")
        if category in BDD100K_IGNORED_CATEGORIES:
            continue
        if category not in BDD100K_CLASS_TO_ID:
            raise ValueError(f"unknown BDD100K category: {category}")
        box = label.get("box2d")
        object_attributes = label.get("attributes", {})
        if not isinstance(box, dict) or not isinstance(object_attributes, dict):
            raise ValueError("BDD100K detection requires box2d and attributes")
        raw = tuple(
            _coordinate(box.get(name), f"box2d.{name}")
            for name in ("x1", "y1", "x2", "y2")
        )
        x1 = min(max(raw[0], 0.0), float(image_width))
        y1 = min(max(raw[1], 0.0), float(image_height))
        x2 = min(max(raw[2], 0.0), float(image_width))
        y2 = min(max(raw[3], 0.0), float(image_height))
        width = x2 - x1
        height = y2 - y1
        if width <= 0.0 or height <= 0.0:
            raise ValueError("BDD100K box must have positive clipped area")
        occluded = object_attributes.get("occluded", False)
        truncated = object_attributes.get("truncated", False)
        if not isinstance(occluded, bool) or not isinstance(truncated, bool):
            raise ValueError("occluded and truncated must be booleans")
        class_id = BDD100K_CLASS_TO_ID[category]
        objects.append(
            BDD100KObject(
                category=category,
                class_id=class_id,
                xyxy=(x1, y1, x2, y2),
                occluded=occluded,
                truncated=truncated,
                size_bin=_size_bin(height),
                yolo_row=YoloRow(
                    class_id=class_id,
                    x_center=((x1 + x2) / 2.0) / image_width,
                    y_center=((y1 + y2) / 2.0) / image_height,
                    width=width / image_width,
                    height=height / image_height,
                ),
            )
        )

    return BDD100KFrame(
        name=_text(frame.get("name"), "frame.name"),
        weather=_text(attributes.get("weather"), "attributes.weather"),
        scene=_text(attributes.get("scene"), "attributes.scene"),
        timeofday=_text(
            attributes.get("timeofday"),
            "attributes.timeofday",
        ),
        objects=tuple(objects),
    )
