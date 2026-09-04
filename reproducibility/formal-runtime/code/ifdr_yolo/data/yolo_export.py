from __future__ import annotations

from dataclasses import dataclass

from ifdr_yolo.data.kitti_types import KittiObject, TRAIN_CLASS_TO_ID


@dataclass(frozen=True)
class YoloRow:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def as_tuple(self) -> tuple[int, float, float, float, float]:
        return (
            self.class_id,
            self.x_center,
            self.y_center,
            self.width,
            self.height,
        )

    def serialize(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.8f} {self.y_center:.8f} "
            f"{self.width:.8f} {self.height:.8f}"
        )


def object_to_yolo(
    obj: KittiObject,
    image_width: int,
    image_height: int,
) -> YoloRow | None:
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"image dimensions must be positive, got {image_width}x{image_height}"
        )
    class_id = TRAIN_CLASS_TO_ID.get(obj.kind)
    if class_id is None:
        return None

    x1 = min(max(obj.bbox.x1, 0.0), float(image_width))
    y1 = min(max(obj.bbox.y1, 0.0), float(image_height))
    x2 = min(max(obj.bbox.x2, 0.0), float(image_width))
    y2 = min(max(obj.bbox.y2, 0.0), float(image_height))
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0.0 or box_height <= 0.0:
        return None

    return YoloRow(
        class_id=class_id,
        x_center=((x1 + x2) / 2.0) / image_width,
        y_center=((y1 + y2) / 2.0) / image_height,
        width=box_width / image_width,
        height=box_height / image_height,
    )
