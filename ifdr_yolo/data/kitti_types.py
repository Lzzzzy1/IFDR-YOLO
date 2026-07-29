from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


TRAIN_CLASS_TO_ID = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
}
EVAL_CLASSES = ("Car", "Pedestrian", "Cyclist")


class Difficulty(Enum):
    EASY = "easy"
    MODERATE = "moderate"
    HARD = "hard"


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in self.as_xyxy()):
            raise ValueError(f"bounding box coordinates must be finite: {self.as_xyxy()}")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"invalid bounding box: {self.as_xyxy()}")

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class KittiObject:
    kind: str
    truncated: float
    occluded: int
    alpha: float
    bbox: BoundingBox
    dimensions_hwl: tuple[float, float, float]
    location_xyz: tuple[float, float, float]
    rotation_y: float
    score: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.truncated):
            raise ValueError("truncated value must be finite")


@dataclass(frozen=True)
class Detection:
    image_id: str
    kind: str
    score: float
    bbox: BoundingBox
