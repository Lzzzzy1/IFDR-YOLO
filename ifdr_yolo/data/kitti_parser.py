from __future__ import annotations

from pathlib import Path

from ifdr_yolo.data.kitti_types import BoundingBox, KittiObject


def parse_kitti_line(line: str) -> KittiObject:
    fields = line.split()
    if len(fields) not in (15, 16):
        raise ValueError(
            f"KITTI label line must contain 15 or 16 fields, got {len(fields)}"
        )
    return KittiObject(
        kind=fields[0],
        truncated=float(fields[1]),
        occluded=int(fields[2]),
        alpha=float(fields[3]),
        bbox=BoundingBox(*(float(value) for value in fields[4:8])),
        dimensions_hwl=tuple(float(value) for value in fields[8:11]),
        location_xyz=tuple(float(value) for value in fields[11:14]),
        rotation_y=float(fields[14]),
        score=float(fields[15]) if len(fields) == 16 else None,
    )


def parse_kitti_file(path: Path) -> tuple[KittiObject, ...]:
    objects: list[KittiObject] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            objects.append(parse_kitti_line(line))
        except (ValueError, IndexError) as error:
            raise ValueError(f"{path.name}:{line_number}: {error}") from error
    return tuple(objects)
