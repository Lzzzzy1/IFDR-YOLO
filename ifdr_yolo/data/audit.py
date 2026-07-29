from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from PIL import Image

from ifdr_yolo.data.kitti_parser import parse_kitti_file
from ifdr_yolo.data.kitti_types import KittiObject, TRAIN_CLASS_TO_ID


@dataclass(frozen=True)
class FixedSizeAudit:
    assumed_width: int
    assumed_height: int
    image_count: int
    target_count: int
    affected_target_count: int
    mean_absolute_normalized_error: float
    max_absolute_normalized_error: float
    by_image_size: dict[str, dict[str, int | float]]


def _normalized_box(
    obj: KittiObject,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    return (
        ((obj.bbox.x1 + obj.bbox.x2) / 2.0) / width,
        ((obj.bbox.y1 + obj.bbox.y2) / 2.0) / height,
        obj.bbox.width / width,
        obj.bbox.height / height,
    )


def audit_fixed_size_assumption(
    *,
    image_dir: Path,
    label_dir: Path,
    image_ids: tuple[str, ...],
    assumed_width: int = 1242,
    assumed_height: int = 375,
) -> FixedSizeAudit:
    size_accumulators: dict[str, dict[str, int | float]] = {}
    target_count = 0
    affected_target_count = 0
    error_sum = 0.0
    max_error = 0.0

    for image_id in image_ids:
        with Image.open(image_dir / f"{image_id}.png") as image:
            width, height = image.size
        size_key = f"{width}x{height}"
        accumulator = size_accumulators.setdefault(
            size_key,
            {
                "images": 0,
                "targets": 0,
                "affected_targets": 0,
                "absolute_error_sum": 0.0,
                "coordinate_count": 0,
                "max_absolute_normalized_error": 0.0,
            },
        )
        accumulator["images"] = int(accumulator["images"]) + 1

        for obj in parse_kitti_file(label_dir / f"{image_id}.txt"):
            if obj.kind not in TRAIN_CLASS_TO_ID:
                continue
            actual = _normalized_box(obj, width, height)
            assumed = _normalized_box(obj, assumed_width, assumed_height)
            errors = tuple(
                abs(actual_value - assumed_value)
                for actual_value, assumed_value in zip(actual, assumed)
            )
            target_max = max(errors)
            target_count += 1
            accumulator["targets"] = int(accumulator["targets"]) + 1
            error_sum += sum(errors)
            accumulator["absolute_error_sum"] = (
                float(accumulator["absolute_error_sum"]) + sum(errors)
            )
            accumulator["coordinate_count"] = (
                int(accumulator["coordinate_count"]) + len(errors)
            )
            max_error = max(max_error, target_max)
            accumulator["max_absolute_normalized_error"] = max(
                float(accumulator["max_absolute_normalized_error"]),
                target_max,
            )
            if target_max > 1e-12:
                affected_target_count += 1
                accumulator["affected_targets"] = (
                    int(accumulator["affected_targets"]) + 1
                )

    by_image_size: dict[str, dict[str, int | float]] = {}
    for size_key, accumulator in sorted(size_accumulators.items()):
        coordinate_count = int(accumulator.pop("coordinate_count"))
        absolute_error_sum = float(accumulator.pop("absolute_error_sum"))
        by_image_size[size_key] = {
            **accumulator,
            "mean_absolute_normalized_error": (
                absolute_error_sum / coordinate_count
                if coordinate_count
                else 0.0
            ),
        }

    return FixedSizeAudit(
        assumed_width=assumed_width,
        assumed_height=assumed_height,
        image_count=len(image_ids),
        target_count=target_count,
        affected_target_count=affected_target_count,
        mean_absolute_normalized_error=(
            error_sum / (target_count * 4) if target_count else 0.0
        ),
        max_absolute_normalized_error=max_error,
        by_image_size=by_image_size,
    )


def write_audit_reports(
    audit: FixedSizeAudit,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(asdict(audit), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    rows = [
        "# KITTI Label Rebuild Audit",
        "",
        (
            f"Fixed {audit.assumed_width}x{audit.assumed_height} normalization "
            "was compared with each image's actual dimensions."
        ),
        "",
        f"- Images: {audit.image_count}",
        f"- Training targets: {audit.target_count}",
        f"- Affected targets: {audit.affected_target_count}",
        (
            "- Mean absolute normalized error: "
            f"{audit.mean_absolute_normalized_error:.10f}"
        ),
        (
            "- Maximum absolute normalized error: "
            f"{audit.max_absolute_normalized_error:.10f}"
        ),
        "",
        "| Image size | Images | Targets | Affected | Mean error | Max error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for size_key, values in audit.by_image_size.items():
        rows.append(
            f"| {size_key} | {values['images']} | {values['targets']} | "
            f"{values['affected_targets']} | "
            f"{float(values['mean_absolute_normalized_error']):.10f} | "
            f"{float(values['max_absolute_normalized_error']):.10f} |"
        )
    markdown_path.write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
