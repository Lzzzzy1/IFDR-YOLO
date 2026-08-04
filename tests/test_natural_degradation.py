import atexit
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest

from ifdr_yolo.data.natural_degradation import (
    NaturalDegradationRecord,
    compute_sampling_score,
    compute_visibility_score,
    load_natural_degradation_records,
)


_TEMPORARY_DIRECTORIES: list[tempfile.TemporaryDirectory[str]] = []


@atexit.register
def _cleanup_temporary_directories() -> None:
    while _TEMPORARY_DIRECTORIES:
        _TEMPORARY_DIRECTORIES.pop().cleanup()


def object_row(
    *,
    image_id: str = "000001",
    kind: str = "Car",
    bbox: dict[str, object] | None = None,
    truncated: object = 0.0,
    occluded: object = 0,
    location_xyz: object = (0.0, 0.0, 15.0),
    **extra: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "image_id": image_id,
        "kind": kind,
        "truncated": truncated,
        "occluded": occluded,
        "bbox": (
            bbox
            if bbox is not None
            else {"x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 64.0}
        ),
    }
    if location_xyz is not None:
        row["location_xyz"] = location_xyz
    row.update(extra)
    return row


def write_jsonl(lines: list[object]) -> Path:
    temporary_directory = tempfile.TemporaryDirectory()
    _TEMPORARY_DIRECTORIES.append(temporary_directory)
    path = Path(temporary_directory.name) / "objects.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            if isinstance(line, str):
                handle.write(line)
            else:
                handle.write(json.dumps(line))
            handle.write("\n")
    return path


class NaturalDegradationScoringTest(unittest.TestCase):
    def test_registered_score_endpoints(self) -> None:
        self.assertEqual(compute_sampling_score(64.0, 15.0), 0.0)
        self.assertEqual(compute_visibility_score(0, 0.0), 0.0)
        self.assertEqual(compute_sampling_score(4.0, 60.0), 1.0)
        self.assertEqual(compute_visibility_score(3, 1.0), 1.0)

    def test_sampling_is_clipped_and_monotonic_for_intermediate_values(self) -> None:
        self.assertAlmostEqual(compute_sampling_score(34.0, 37.5), 0.75)
        self.assertLess(
            compute_sampling_score(64.0, 15.0),
            compute_sampling_score(50.0, 25.0),
        )
        self.assertLess(
            compute_sampling_score(50.0, 25.0),
            compute_sampling_score(4.0, 60.0),
        )
        self.assertEqual(compute_sampling_score(1000.0, -10.0), 0.0)
        self.assertEqual(compute_sampling_score(-10.0, 1000.0), 1.0)

    def test_visibility_is_monotonic_for_intermediate_values(self) -> None:
        self.assertAlmostEqual(compute_visibility_score(0, 0.5), 0.5)
        self.assertAlmostEqual(compute_visibility_score(1, 0.0), 1.0 / 3.0)
        self.assertLess(
            compute_visibility_score(0, 0.0),
            compute_visibility_score(0, 0.5),
        )
        self.assertLess(
            compute_visibility_score(0, 0.5),
            compute_visibility_score(1, 0.5),
        )


class NaturalDegradationLoaderTest(unittest.TestCase):
    def test_loads_training_records_and_class_mapping(self) -> None:
        rows = [
            object_row(kind="Car"),
            object_row(kind="Pedestrian", image_id="000002"),
            object_row(kind="Cyclist", image_id="000003"),
        ]
        result = load_natural_degradation_records(write_jsonl(rows))

        self.assertEqual(result.skipped_non_training_count, 0)
        self.assertEqual(result.invalid_depth_count, 0)
        self.assertEqual(len(result.records), 3)
        self.assertEqual(
            [(record.class_name, record.class_id) for record in result.records],
            [("Car", 0), ("Pedestrian", 1), ("Cyclist", 2)],
        )
        self.assertIsInstance(result.records[0], NaturalDegradationRecord)
        self.assertIsInstance(result.records, tuple)

    def test_score_fields_use_height_depth_occlusion_and_truncation(self) -> None:
        rows = [
            object_row(
                bbox={"x1": 0.0, "y1": 0.0, "x2": 4.0, "y2": 64.0},
                location_xyz=(0.0, 0.0, 15.0),
            ),
            object_row(
                image_id="000002",
                bbox={"x1": 0.0, "y1": 0.0, "x2": 4.0, "y2": 4.0},
                location_xyz=(0.0, 0.0, 60.0),
            ),
            object_row(
                image_id="000003",
                occluded=3,
                truncated=1.0,
            ),
        ]
        result = load_natural_degradation_records(write_jsonl(rows))

        self.assertEqual(result.records[0].sampling_score, 0.0)
        self.assertEqual(result.records[0].visibility_score, 0.0)
        self.assertEqual(result.records[1].sampling_score, 1.0)
        self.assertEqual(result.records[2].visibility_score, 1.0)

    def test_missing_depth_is_explicit_and_never_interpolated(self) -> None:
        result = load_natural_degradation_records(
            write_jsonl(
                [
                    object_row(
                        location_xyz=None,
                        bbox={"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 34.0},
                    )
                ]
            )
        )

        record = result.records[0]
        self.assertIsNone(record.depth_m)
        self.assertFalse(record.depth_available)
        self.assertEqual(record.sampling_score, 0.5)
        self.assertEqual(result.invalid_depth_count, 0)

    def test_non_positive_training_depth_becomes_unavailable_and_is_counted(self) -> None:
        rows = [
            object_row(location_xyz=(0.0, 0.0, 0.0)),
            object_row(image_id="000002", location_xyz=(0.0, 0.0, -1.0)),
            object_row(image_id="000003", location_xyz=None),
        ]

        result = load_natural_degradation_records(write_jsonl(rows))

        self.assertEqual(result.invalid_depth_count, 2)
        self.assertEqual(len(result.records), 3)
        for record in result.records:
            if record.image_id == "000003":
                continue
            self.assertIsNone(record.depth_m)
            self.assertFalse(record.depth_available)
            self.assertEqual(record.sampling_score, 0.0)

    def test_non_training_rows_are_skipped_but_keep_per_image_positions(self) -> None:
        rows = [
            object_row(kind="Van"),
            object_row(kind="Car"),
            object_row(kind="Truck"),
            object_row(kind="Pedestrian"),
            object_row(image_id="000002", kind="Cyclist"),
        ]
        result = load_natural_degradation_records(write_jsonl(rows))

        self.assertEqual(result.skipped_non_training_count, 2)
        self.assertEqual(
            [(record.image_id, record.object_id) for record in result.records],
            [("000001", 1), ("000001", 3), ("000002", 0)],
        )

    def test_interleaved_images_keep_independent_row_positions(self) -> None:
        rows = [
            object_row(image_id="000001", kind="Car"),
            object_row(image_id="000002", kind="Car"),
            object_row(image_id="000001", kind="Van"),
            object_row(image_id="000002", kind="Truck"),
            object_row(image_id="000001", kind="Pedestrian"),
            object_row(image_id="000002", kind="Cyclist"),
        ]

        result = load_natural_degradation_records(write_jsonl(rows))

        self.assertEqual(
            [(record.image_id, record.object_id) for record in result.records],
            [("000001", 0), ("000002", 0), ("000001", 2), ("000002", 2)],
        )

    def test_skips_all_canonical_non_training_classes_including_dontcare_sentinel(
        self,
    ) -> None:
        rows = [
            object_row(kind="Van"),
            object_row(kind="Truck"),
            object_row(kind="Person_sitting"),
            object_row(kind="Tram"),
            object_row(kind="Misc"),
            object_row(
                kind="DontCare",
                truncated=-1.0,
                occluded=-1,
                location_xyz=(-1000.0, -1000.0, -1000.0),
            ),
        ]

        result = load_natural_degradation_records(write_jsonl(rows))

        self.assertEqual(result.records, ())
        self.assertEqual(result.skipped_non_training_count, 6)
        self.assertEqual(result.invalid_depth_count, 0)

    def test_rejects_explicit_implicit_object_id_collisions_with_line_context(self) -> None:
        cases = (
            [object_row(), object_row(object_id=0)],
            [object_row(object_id=1), object_row()],
        )
        for rows in cases:
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(ValueError, r"JSONL line 2"):
                    load_natural_degradation_records(write_jsonl(rows))

    def test_rejects_explicit_object_id_collisions_with_line_context(self) -> None:
        with self.assertRaisesRegex(ValueError, r"JSONL line 2"):
            load_natural_degradation_records(
                write_jsonl([object_row(object_id=3), object_row(object_id=3)])
            )

    def test_explicit_object_id_must_be_non_negative_integer(self) -> None:
        result = load_natural_degradation_records(
            write_jsonl([object_row(object_id=7)])
        )
        self.assertEqual(result.records[0].object_id, 7)
        for value in (-1, 1.0, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, r"JSONL line 1"):
                    load_natural_degradation_records(
                        write_jsonl([object_row(object_id=value)])
                    )

    def test_record_is_frozen(self) -> None:
        record = load_natural_degradation_records(
            write_jsonl([object_row()])
        ).records[0]
        with self.assertRaises(FrozenInstanceError):
            record.object_id = 2  # type: ignore[misc]

    def test_rejects_unknown_class_with_line_context(self) -> None:
        self.assert_rejected([object_row(kind="Alien")])

    def test_validates_non_training_rows_before_skipping(self) -> None:
        self.assert_rejected(
            [object_row(kind="Van", bbox={"x1": 0, "y1": 0, "x2": 0, "y2": 10})]
        )

    def test_rejects_empty_image_id_with_line_context(self) -> None:
        self.assert_rejected([object_row(image_id="")])

    def test_rejects_non_positive_or_non_finite_bbox(self) -> None:
        for bbox in (
            {"x1": 0, "y1": 0, "x2": 0, "y2": 10},
            {"x1": 0, "y1": 0, "x2": 10, "y2": -1},
            {"x1": float("nan"), "y1": 0, "x2": 10, "y2": 10},
            {"x1": float("inf"), "y1": 0, "x2": 10, "y2": 10},
        ):
            with self.subTest(bbox=bbox):
                self.assert_rejected([object_row(bbox=bbox)])

    def test_rejects_floating_point_bbox_width_overflow_with_line_context(self) -> None:
        self.assert_rejected(
            [
                object_row(
                    bbox={
                        "x1": -1.7e308,
                        "y1": -1.7e308,
                        "x2": 1.7e308,
                        "y2": 1.7e308,
                    }
                )
            ]
        )

    def test_rejects_invalid_depth_and_location(self) -> None:
        for location_xyz in ((0.0, 0.0, float("nan")), (0.0, 0.0, float("inf")), (0.0, 0.0)):
            with self.subTest(location_xyz=location_xyz):
                self.assert_rejected([object_row(location_xyz=location_xyz)])

    def test_rejects_invalid_occlusion_and_truncation(self) -> None:
        for kwargs in (
            {"occluded": True},
            {"occluded": 4},
            {"occluded": 1.0},
            {"truncated": True},
            {"truncated": -0.1},
            {"truncated": 1.1},
            {"truncated": float("nan")},
            {"truncated": float("inf")},
        ):
            with self.subTest(kwargs=kwargs):
                self.assert_rejected([object_row(**kwargs)])

    def test_rejects_boolean_numeric_fields(self) -> None:
        for row in (
            object_row(bbox={"x1": False, "y1": 0, "x2": 10, "y2": 10}),
            object_row(location_xyz=(0.0, 0.0, True)),
        ):
            with self.subTest(row=row):
                self.assert_rejected([row])

    def test_rejects_missing_required_fields(self) -> None:
        for field in ("image_id", "kind", "bbox", "truncated", "occluded"):
            with self.subTest(field=field):
                row = object_row()
                del row[field]
                self.assert_rejected([row])

    def test_rejects_malformed_json_with_line_context(self) -> None:
        path = write_jsonl([object_row()])
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"image_id":\n')
        with self.assertRaisesRegex(ValueError, r"JSONL line 2"):
            load_natural_degradation_records(path)

    def assert_rejected(self, rows: list[object]) -> None:
        with self.assertRaisesRegex(ValueError, r"JSONL line 1"):
            load_natural_degradation_records(write_jsonl(rows))


if __name__ == "__main__":
    unittest.main()
