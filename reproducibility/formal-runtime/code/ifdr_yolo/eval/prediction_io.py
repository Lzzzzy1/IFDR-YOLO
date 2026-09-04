from __future__ import annotations

from pathlib import Path

from ifdr_yolo.data.kitti_parser import parse_kitti_file
from ifdr_yolo.data.kitti_types import (
    BoundingBox,
    Detection,
    KittiObject,
    TRAIN_CLASS_TO_ID,
)


CLASS_ID_TO_NAME = {
    class_id: class_name for class_name, class_id in TRAIN_CLASS_TO_ID.items()
}


def load_yolo_predictions(
    prediction_dir: Path,
    image_sizes: dict[str, tuple[int, int]],
) -> dict[str, tuple[Detection, ...]]:
    predictions: dict[str, tuple[Detection, ...]] = {}
    for image_id, (image_width, image_height) in image_sizes.items():
        if image_width <= 0 or image_height <= 0:
            raise ValueError(
                f"image dimensions must be positive for {image_id}: "
                f"{image_width}x{image_height}"
            )
        prediction_path = prediction_dir / f"{image_id}.txt"
        if not prediction_path.exists():
            predictions[image_id] = ()
            continue

        detections: list[Detection] = []
        for line_number, line in enumerate(
            prediction_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(
                    f"{prediction_path.name}:{line_number}: "
                    f"prediction must contain 6 fields, got {len(fields)}"
                )
            try:
                class_id = int(fields[0])
                x_center, y_center, width, height, score = (
                    float(value) for value in fields[1:]
                )
            except ValueError as error:
                raise ValueError(
                    f"{prediction_path.name}:{line_number}: invalid numeric field"
                ) from error
            if class_id not in CLASS_ID_TO_NAME:
                raise ValueError(
                    f"{prediction_path.name}:{line_number}: "
                    f"unknown class ID {class_id}"
                )

            pixel_width = width * image_width
            pixel_height = height * image_height
            pixel_x_center = x_center * image_width
            pixel_y_center = y_center * image_height
            x1 = min(
                max(pixel_x_center - pixel_width / 2.0, 0.0),
                float(image_width),
            )
            y1 = min(
                max(pixel_y_center - pixel_height / 2.0, 0.0),
                float(image_height),
            )
            x2 = min(
                max(pixel_x_center + pixel_width / 2.0, 0.0),
                float(image_width),
            )
            y2 = min(
                max(pixel_y_center + pixel_height / 2.0, 0.0),
                float(image_height),
            )
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    image_id=image_id,
                    kind=CLASS_ID_TO_NAME[class_id],
                    score=score,
                    bbox=BoundingBox(x1, y1, x2, y2),
                )
            )
        predictions[image_id] = tuple(detections)
    return predictions


def load_kitti_ground_truth(
    label_dir: Path,
    image_ids: tuple[str, ...],
) -> dict[str, tuple[KittiObject, ...]]:
    return {
        image_id: parse_kitti_file(label_dir / f"{image_id}.txt")
        for image_id in image_ids
    }
