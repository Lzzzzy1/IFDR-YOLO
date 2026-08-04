import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from ifdr_yolo.data.natural_degradation import NaturalDegradationRecord

_REAL_FSYNC = os.fsync

def os_fsync(fd: int) -> None:
    _REAL_FSYNC(fd)


class FactorObserverImportTest(unittest.TestCase):
    def test_canonical_api_imports(self) -> None:
        from ifdr_yolo.eval.factor_observer import (  # noqa: F401
            FactorObservationJournal,
            FactorObservationManifest,
            ImageObservationPlan,
            LetterboxGeometry,
            ObservationCondition,
            build_factor_observation_manifest,
            letterbox_image,
            map_box_to_feature_roi,
        )


def _record(image_id: str, object_id: int, bbox=(10.0, 10.0, 30.0, 30.0)) -> NaturalDegradationRecord:
    return NaturalDegradationRecord(
        image_id=image_id,
        object_id=object_id,
        class_id=0,
        class_name="Car",
        bbox_xyxy=bbox,
        box_height=bbox[3] - bbox[1],
        depth_m=20.0,
        depth_available=True,
        occlusion_level=0,
        truncation=0.0,
        sampling_score=0.1,
        visibility_score=0.0,
    )


class FactorObserverFoundationTest(unittest.TestCase):
    def _manifest(self, root: Path, *, nodes=(11,)):
        from ifdr_yolo.eval.factor_observer import build_factor_observation_manifest

        image_paths = {}
        for image_id, color in (("b", 31), ("a", 77)):
            path = root / f"{image_id}.png"
            image = np.full((80, 100, 3), color, dtype=np.uint8)
            encoded_ok, encoded = cv2.imencode(".png", image)
            self.assertTrue(encoded_ok)
            path.write_bytes(encoded.tobytes())
            image_paths[image_id] = path
        records = (_record("b", 2), _record("a", 1))
        checkpoint = "ab" * 32
        return build_factor_observation_manifest(
            records,
            image_paths,
            {("a", 1)},
            checkpoint,
            17,
            required_nodes=nodes,
            input_size=64,
        )

    def test_manifest_is_order_independent_and_registers_all_rows(self) -> None:
        from ifdr_yolo.eval.factor_observer import build_factor_observation_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._manifest(root)
            image_paths = {"a": root / "a.png", "b": root / "b.png"}
            second = build_factor_observation_manifest(
                (_record("a", 1), _record("b", 2)),
                image_paths,
                {("a", 1)},
                "ab" * 32,
                17,
                required_nodes=(11,),
                input_size=64,
            )
            self.assertEqual(first.hash(), second.hash())
            self.assertEqual(first.image_ids, ("a", "b"))
            self.assertEqual(len(first.plans[0].conditions), 21)
            self.assertEqual(first.expected_observation_count, 22)
            self.assertEqual(
                set(first.plans[0].expected_observation_ids),
                set(first.expected_observation_ids[:21]),
            )
            selected = [
                condition
                for condition in first.plans[0].conditions
                if condition.intervention_kind != "natural"
            ]
            self.assertTrue(selected)
            background = next(item.matched_background_bbox for item in selected)
            self.assertIsNotNone(background)
            self.assertEqual(background[2] - background[0], 20.0)
            self.assertEqual(background[3] - background[1], 20.0)
            all_boxes = [condition.bbox_xyxy for condition in first.plans[0].conditions if condition.intervention_kind == "natural"]
            self.assertLessEqual(
                max(
                    0.0
                    if box == background
                    else _iou(box, background)
                    for box in all_boxes
                ),
                0.05,
            )

    def test_manifest_rejects_identity_selection_and_background_failures(self) -> None:
        from ifdr_yolo.eval.factor_observer import build_factor_observation_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "a.png"
            encoded_ok, encoded = cv2.imencode(".png", np.zeros((20, 20, 3), dtype=np.uint8))
            self.assertTrue(encoded_ok)
            path.write_bytes(encoded.tobytes())
            kwargs = {
                "records": (_record("a", 1, (0.0, 0.0, 20.0, 20.0)),),
                "image_paths": {"a": path},
                "checkpoint_sha256": "ab" * 32,
                "seed": 17,
                "required_nodes": (11,),
                "input_size": 64,
            }
            with self.assertRaisesRegex(ValueError, "unknown"):
                build_factor_observation_manifest(selected_intervention_objects={("a", 9)}, **kwargs)
            with self.assertRaisesRegex(ValueError, "background"):
                build_factor_observation_manifest(selected_intervention_objects={("a", 1)}, **kwargs)
            duplicate = (_record("a", 1), _record("a", 1, (2.0, 2.0, 4.0, 4.0)))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                build_factor_observation_manifest(
                    duplicate,
                    kwargs["image_paths"],
                    set(),
                    kwargs["checkpoint_sha256"],
                    kwargs["seed"],
                    required_nodes=(11,),
                    input_size=64,
                )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                build_factor_observation_manifest(selected_intervention_objects=set(), checkpoint_sha256="bad", **{key: value for key, value in kwargs.items() if key != "checkpoint_sha256"})

    def test_letterbox_edges_and_multiple_feature_maps_are_clipped(self) -> None:
        from ifdr_yolo.eval.factor_observer import letterbox_image, map_box_to_feature_roi

        image = np.zeros((3, 7, 3), dtype=np.uint8)
        _, geometry = letterbox_image(image, 32)
        for feature_shape in ((1, 1), (3, 5), (17, 9)):
            roi = map_box_to_feature_roi((0.0, 0.0, 7.0, 3.0), geometry, feature_shape)
            self.assertEqual(roi[0], 0)
            self.assertEqual(roi[2], feature_shape[1])
            self.assertGreaterEqual(roi[3], roi[1] + 1)
            self.assertLessEqual(roi[3], feature_shape[0])
        self.assertEqual(map_box_to_feature_roi((6.999, 2.999, 7.0, 3.0), geometry, (1, 1)), (0, 0, 1, 1))

    def test_letterbox_and_roi_use_floor_ceil_and_rgb(self) -> None:
        from ifdr_yolo.eval.factor_observer import letterbox_image, map_box_to_feature_roi

        image = np.zeros((50, 100, 3), dtype=np.uint8)
        image[..., 0] = 10  # B
        image[..., 2] = 30  # R
        tensor, geometry = letterbox_image(image, 64)
        self.assertEqual(tuple(tensor.shape), (3, 64, 64))
        self.assertEqual(str(tensor.dtype), "torch.float32")
        self.assertAlmostEqual(float(tensor[0, 32, 32]), 30 / 255.0)
        self.assertEqual((geometry.pad_left, geometry.pad_top), (0, 16))
        self.assertEqual(map_box_to_feature_roi((0, 0, 100, 50), geometry, 8, 16), (0, 2, 16, 6))
        self.assertGreaterEqual(map_box_to_feature_roi((0, 0, 0.01, 0.01), geometry, 8, 16)[2], 1)

    def test_journal_crash_restart_exact_once_and_finalize(self) -> None:
        from ifdr_yolo.eval.factor_observer import FactorObservationJournal

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root, nodes=(11,))
            output = root / "rows.jsonl"
            progress = root / "progress.json"
            journal = FactorObservationJournal(manifest, output, progress)
            plan = manifest.plans[0]
            rows = [{"observation_id": item, "image_id": plan.image_id} for item in plan.expected_observation_ids]

            def crash(_phase: str) -> None:
                raise RuntimeError("power loss")

            with self.assertRaisesRegex(RuntimeError, "power loss"):
                journal.commit_image("a", rows, crash_hook=crash)
            self.assertTrue(json.loads(progress.read_text())["inflight"])
            restarted = FactorObservationJournal(manifest, output, progress)
            self.assertEqual(output.read_bytes(), b"")
            self.assertTrue(restarted.commit_image("a", rows))
            self.assertFalse(restarted.commit_image("a", list(reversed(rows))))
            other = manifest.plans[1]
            other_rows = [{"observation_id": item, "image_id": other.image_id} for item in other.expected_observation_ids]
            restarted.commit_image("b", other_rows)
            summary = restarted.finalize()
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(len(output.read_text().splitlines()), manifest.expected_observation_count)

            with output.open("ab") as handle:
                handle.write(b"{bad")
            with self.assertRaisesRegex(ValueError, "unterminated|malformed"):
                FactorObservationJournal(manifest, output, progress)

    def test_journal_rejects_hash_drift_identity_variants_and_missing_finalize(self) -> None:
        from ifdr_yolo.eval.factor_observer import FactorObservationJournal, build_factor_observation_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root, nodes=(11,))
            output, progress = root / "rows.jsonl", root / "progress.json"
            journal = FactorObservationJournal(manifest, output, progress)
            plan = manifest.plans[0]
            good = [{"observation_id": item, "image_id": plan.image_id} for item in plan.expected_observation_ids]
            with self.assertRaisesRegex(ValueError, "missing"):
                journal.commit_image("a", good[:-1])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                journal.commit_image("a", good[:-1] + [good[-2]])
            with self.assertRaisesRegex(ValueError, "extra|missing"):
                journal.commit_image("a", good + [{"observation_id": "ff" * 32, "image_id": "a"}])
            journal.commit_image("a", good)
            with self.assertRaisesRegex(ValueError, "conflicts"):
                journal.commit_image("a", [dict(row, value=1) for row in good])
            with self.assertRaisesRegex(ValueError, "missing"):
                journal.finalize()
            drift = build_factor_observation_manifest(
                (_record("a", 1), _record("b", 2)),
                {"a": root / "a.png", "b": root / "b.png"},
                {("a", 1)},
                "cd" * 32,
                17,
                required_nodes=(11,),
                input_size=64,
            )
            with self.assertRaisesRegex(ValueError, "hash"):
                FactorObservationJournal(drift, output, progress)

    def test_journal_uses_fsync_for_progress_and_rows(self) -> None:
        from unittest.mock import patch
        from ifdr_yolo.eval.factor_observer import FactorObservationJournal

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root, nodes=(11,))
            journal = FactorObservationJournal(manifest, root / "rows.jsonl", root / "progress.json")
            plan = manifest.plans[0]
            rows = [{"observation_id": item, "image_id": plan.image_id} for item in plan.expected_observation_ids]
            with patch("ifdr_yolo.eval.factor_observer.os.fsync", wraps=os_fsync) as fsync:
                journal.commit_image("a", rows)
            self.assertGreaterEqual(fsync.call_count, 2)


def _iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (area_left + area_right - intersection) if intersection else 0.0


if __name__ == "__main__":
    unittest.main()
