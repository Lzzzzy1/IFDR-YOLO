import hashlib
import json
import os
from dataclasses import replace
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
    @staticmethod
    def _condition_kwargs(**overrides):
        from ifdr_yolo.eval.factor_observer import _canonical_condition_id, _canonical_pair_id

        values = {
            "image_id": "a",
            "seed": 17,
            "object_id": 1,
            "class_id": 0,
            "class_name": "Car",
            "bbox_xyxy": (10.0, 10.0, 30.0, 30.0),
            "box_height": 20.0,
            "natural_sampling": 0.1,
            "natural_visibility": 0.2,
            "region_role": "target",
            "intervention_kind": "natural",
            "intervention_factor": None,
            "intervention_severity": 0.0,
            "pair_id": None,
            "condition_id": "cd" * 32,
            "transform_id": "ef" * 32,
            "source_sha256": "ab" * 32,
            "matched_background_bbox": None,
        }
        values.update(overrides)
        if values["intervention_kind"] != "natural" and values["pair_id"] is None:
            values["pair_id"] = _canonical_pair_id(
                image_id=values["image_id"],
                object_id=values["object_id"],
                factor=values["intervention_factor"],
                seed=values["seed"],
                source_sha256=values["source_sha256"],
            )
        if "condition_id" not in overrides:
            values["condition_id"] = _canonical_condition_id(
                image_id=values["image_id"],
                object_id=values["object_id"],
                class_id=values["class_id"],
                class_name=values["class_name"],
                bbox_xyxy=values["bbox_xyxy"],
                region_role=values["region_role"],
                intervention_kind=values["intervention_kind"],
                intervention_factor=values["intervention_factor"],
                intervention_severity=values["intervention_severity"],
                pair_id=values["pair_id"],
                source_sha256=values["source_sha256"],
                seed=values["seed"],
            )
        return values

    @staticmethod
    def _condition_id(condition, **overrides):
        from ifdr_yolo.eval.factor_observer import _canonical_condition_id

        values = {
            "image_id": condition.image_id,
            "object_id": condition.object_id,
            "class_id": condition.class_id,
            "class_name": condition.class_name,
            "bbox_xyxy": condition.bbox_xyxy,
            "region_role": condition.region_role,
            "intervention_kind": condition.intervention_kind,
            "intervention_factor": condition.intervention_factor,
            "intervention_severity": condition.intervention_severity,
            "pair_id": condition.pair_id,
            "source_sha256": condition.source_sha256,
            "seed": condition.seed,
        }
        values.update(overrides)
        return _canonical_condition_id(**values)

    @staticmethod
    def _pair_id(condition, **overrides):
        from ifdr_yolo.eval.factor_observer import _canonical_pair_id

        values = {
            "image_id": condition.image_id,
            "object_id": condition.object_id,
            "factor": condition.intervention_factor,
            "seed": condition.seed,
            "source_sha256": condition.source_sha256,
        }
        values.update(overrides)
        return _canonical_pair_id(**values)

    def test_direct_condition_schema_rejects_bad_ids_and_class_names(self) -> None:
        from ifdr_yolo.eval.factor_observer import ObservationCondition

        for field in ("condition_id", "transform_id", "source_sha256"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "lowercase|hex"):
                    ObservationCondition(**self._condition_kwargs(**{field: "A" * 64}))
                with self.assertRaisesRegex(ValueError, "lowercase|hex"):
                    ObservationCondition(**self._condition_kwargs(**{field: "g" * 64}))
        for class_id, class_name in ((0, "Pedestrian"), (1, "Car"), (2, "Pedestrian")):
            with self.subTest(class_id=class_id, class_name=class_name):
                with self.assertRaisesRegex(ValueError, "class_name"):
                    ObservationCondition(**self._condition_kwargs(class_id=class_id, class_name=class_name))

        controlled = self._condition_kwargs(
            intervention_kind="sampling",
            intervention_factor="sampling",
            intervention_severity=0.5,
            matched_background_bbox=(40.0, 10.0, 60.0, 30.0),
        )
        with self.assertRaisesRegex(ValueError, "lowercase|hex"):
            ObservationCondition(**{**controlled, "pair_id": "A" * 64})

        canonical_severity = 0.1 + 0.15
        canonical = ObservationCondition(
            **{
                **controlled,
                "intervention_severity": canonical_severity,
                "condition_id": self._condition_id(
                    ObservationCondition(**controlled),
                    intervention_severity=canonical_severity,
                ),
            }
        )
        self.assertEqual(canonical.intervention_severity, 0.25)
        with self.assertRaisesRegex(ValueError, "registered"):
            ObservationCondition(**{**controlled, "intervention_severity": 0.33})

    def test_direct_condition_schema_requires_natural_and_controlled_shapes(self) -> None:
        from ifdr_yolo.eval.factor_observer import ObservationCondition

        natural = self._condition_kwargs()
        for field, value in (
            ("intervention_factor", "sampling"),
            ("pair_id", "12" * 32),
            ("matched_background_bbox", (40.0, 10.0, 60.0, 30.0)),
            ("region_role", "background"),
            ("intervention_severity", 0.1),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    ObservationCondition(**{**natural, field: value})

        clean = self._condition_kwargs(
            intervention_kind="clean",
            intervention_factor="sampling",
            intervention_severity=0.0,
            matched_background_bbox=(40.0, 10.0, 60.0, 30.0),
        )
        ObservationCondition(**clean)
        for field, value in (
            ("intervention_severity", 0.25),
            ("matched_background_bbox", None),
            ("intervention_factor", None),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    ObservationCondition(**{**clean, field: value})

        for kind, factor, severity in (
            ("sampling", "visibility", 0.5),
            ("sampling", "sampling", 0.0),
            ("visibility", "visibility", 0.0),
        ):
            with self.subTest(kind=kind, factor=factor, severity=severity):
                with self.assertRaises(ValueError):
                    ObservationCondition(
                        **{
                            **clean,
                            "intervention_kind": kind,
                            "intervention_factor": factor,
                            "intervention_severity": severity,
                        }
                    )

        with self.assertRaises(ValueError):
            ObservationCondition(**{**clean, "region_role": "background", "bbox_xyxy": (41.0, 10.0, 61.0, 30.0)})

    def test_direct_manifest_and_plan_hashes_require_lowercase_hex(self) -> None:
        from ifdr_yolo.eval.factor_observer import FactorObservationManifest, ImageObservationPlan

        with tempfile.TemporaryDirectory() as directory:
            manifest = self._manifest(Path(directory))
            with self.assertRaisesRegex(ValueError, "lowercase|hex"):
                FactorObservationManifest(
                    plans=manifest.plans,
                    checkpoint_sha256="A" * 64,
                    seed=manifest.seed,
                    required_nodes=manifest.required_nodes,
                    input_size=manifest.input_size,
                )
            with self.assertRaisesRegex(ValueError, "lowercase|hex"):
                ImageObservationPlan(
                    image_id=manifest.plans[0].image_id,
                    image_path=manifest.plans[0].image_path,
                    width=manifest.plans[0].width,
                    height=manifest.plans[0].height,
                    source_sha256=manifest.plans[0].source_sha256,
                    conditions=manifest.plans[0].conditions,
                    expected_observation_ids=("A" * 64,) + manifest.plans[0].expected_observation_ids[1:],
                )

    def test_condition_height_and_plan_order_bounds_are_strict(self) -> None:
        from ifdr_yolo.eval.factor_observer import ImageObservationPlan, ObservationCondition

        with tempfile.TemporaryDirectory() as directory:
            manifest = self._manifest(Path(directory))
            plan = manifest.plans[0]
            natural = plan.conditions[0]
            with self.assertRaisesRegex(ValueError, "box_height"):
                ObservationCondition(**{**natural.to_dict(), "bbox_xyxy": list(natural.bbox_xyxy), "box_height": 19.0, "matched_background_bbox": None})
            with self.assertRaises(ValueError):
                ImageObservationPlan(
                    image_id=plan.image_id,
                    image_path=plan.image_path,
                    width=plan.width,
                    height=plan.height,
                    source_sha256=plan.source_sha256,
                    conditions=tuple(reversed(plan.conditions)),
                    expected_observation_ids=plan.expected_observation_ids,
                )
            outside_bbox = (-1.0, 10.0, 30.0, 30.0)
            outside = replace(
                natural,
                bbox_xyxy=outside_bbox,
                condition_id=self._condition_id(natural, bbox_xyxy=outside_bbox),
            )
            altered = (outside,) + plan.conditions[1:]
            with self.assertRaisesRegex(ValueError, "image bounds"):
                ImageObservationPlan(
                    image_id=plan.image_id,
                    image_path=plan.image_path,
                    width=plan.width,
                    height=plan.height,
                    source_sha256=plan.source_sha256,
                    conditions=altered,
                    expected_observation_ids=plan.expected_observation_ids,
                )

    def test_plan_rejects_controlled_pair_and_cross_condition_inconsistency(self) -> None:
        from ifdr_yolo.eval.factor_observer import ImageObservationPlan

        with tempfile.TemporaryDirectory() as directory:
            manifest = self._manifest(Path(directory))
            plan = manifest.plans[0]
            controlled_index = next(
                index for index, condition in enumerate(plan.conditions)
                if condition.intervention_kind == "sampling"
            )
            controlled = plan.conditions[controlled_index]
            wrong_background_bbox = plan.conditions[0].bbox_xyxy
            wrong_background = replace(
                controlled,
                bbox_xyxy=wrong_background_bbox,
                matched_background_bbox=wrong_background_bbox,
                condition_id=self._condition_id(controlled, bbox_xyxy=wrong_background_bbox),
            )
            altered = list(plan.conditions)
            altered[controlled_index] = wrong_background
            with self.assertRaisesRegex(ValueError, "IoU|background"):
                ImageObservationPlan(
                    image_id=plan.image_id,
                    image_path=plan.image_path,
                    width=plan.width,
                    height=plan.height,
                    source_sha256=plan.source_sha256,
                    conditions=tuple(altered),
                    expected_observation_ids=plan.expected_observation_ids,
                )
            missing = tuple(item for index, item in enumerate(plan.conditions) if index != controlled_index)
            with self.assertRaisesRegex(ValueError, "pair|condition"):
                ImageObservationPlan(
                    image_id=plan.image_id,
                    image_path=plan.image_path,
                    width=plan.width,
                    height=plan.height,
                    source_sha256=plan.source_sha256,
                    conditions=missing,
                    expected_observation_ids=plan.expected_observation_ids,
            )
            inconsistent_source = "cd" * 32
            inconsistent_pair = self._pair_id(controlled, source_sha256=inconsistent_source)
            inconsistent = replace(
                controlled,
                source_sha256=inconsistent_source,
                pair_id=inconsistent_pair,
                condition_id=self._condition_id(
                    controlled,
                    source_sha256=inconsistent_source,
                    pair_id=inconsistent_pair,
                ),
            )
            altered = list(plan.conditions)
            altered[controlled_index] = inconsistent
            with self.assertRaisesRegex(ValueError, "source|consistent"):
                ImageObservationPlan(
                    image_id=plan.image_id,
                    image_path=plan.image_path,
                    width=plan.width,
                    height=plan.height,
                    source_sha256=plan.source_sha256,
                    conditions=tuple(altered),
                    expected_observation_ids=plan.expected_observation_ids,
                )

    def _manifest(self, root: Path, *, nodes=(11, 14, 17, 20, 23, 26)):
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
                required_nodes=(11, 14, 17, 20, 23, 26),
                input_size=64,
            )
            self.assertEqual(first.hash(), second.hash())
            self.assertEqual(first.image_ids, ("a", "b"))
            self.assertEqual(first.to_dict()["registered_severities"], [0.25, 0.5, 0.75, 1.0])
            self.assertEqual(len(first.plans[0].conditions), 21)
            self.assertEqual(first.expected_observation_count, 132)
            self.assertEqual(
                set(first.plans[0].expected_observation_ids),
                set(first.expected_observation_ids[:126]),
            )
            self.assertEqual(len(first.plans[0].expected_observation_ids), len(first.plans[0].conditions) * 6)
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

    def test_transform_ids_reuse_source_and_distinguish_interventions(self) -> None:
        from ifdr_yolo.eval.factor_observer import build_factor_observation_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "a.png"
            encoded_ok, encoded = cv2.imencode(
                ".png", np.full((100, 140, 3), 77, dtype=np.uint8)
            )
            self.assertTrue(encoded_ok)
            path.write_bytes(encoded.tobytes())
            records = (
                _record("a", 1, (10.0, 10.0, 30.0, 30.0)),
                _record("a", 2, (60.0, 10.0, 80.0, 30.0)),
            )
            kwargs = {
                "records": records,
                "image_paths": {"a": path},
                "selected_intervention_objects": {("a", 1), ("a", 2)},
                "checkpoint_sha256": "ab" * 32,
                "seed": 17,
                "required_nodes": (11, 14, 17, 20, 23, 26),
                "input_size": 64,
            }
            manifest = build_factor_observation_manifest(**kwargs)
            repeated = build_factor_observation_manifest(
                **{**kwargs, "records": tuple(reversed(records))}
            )
            plan = manifest.plans[0]
            repeated_plan = repeated.plans[0]
            source_conditions = [
                condition
                for condition in plan.conditions
                if condition.intervention_kind in {"natural", "clean"}
            ]
            self.assertTrue(source_conditions)
            self.assertEqual(len({condition.transform_id for condition in source_conditions}), 1)

            def transform_map(current_plan):
                return {
                    (
                        condition.object_id,
                        condition.intervention_kind,
                        condition.intervention_factor,
                        condition.intervention_severity,
                        condition.region_role,
                    ): condition.transform_id
                    for condition in current_plan.conditions
                }

            transforms = transform_map(plan)
            self.assertEqual(transforms, transform_map(repeated_plan))
            nonzero = {
                key: value
                for key, value in transforms.items()
                if key[3] > 0.0
            }
            self.assertEqual(len(nonzero), len(set(nonzero.values())))
            self.assertNotEqual(
                transforms[(1, "sampling", "sampling", 0.25, "target")],
                transforms[(1, "visibility", "visibility", 0.25, "target")],
            )
            self.assertNotEqual(
                transforms[(1, "sampling", "sampling", 0.25, "target")],
                transforms[(1, "sampling", "sampling", 0.5, "target")],
            )
            self.assertNotEqual(
                transforms[(1, "sampling", "sampling", 0.25, "target")],
                transforms[(1, "sampling", "sampling", 0.25, "background")],
            )

    def test_plan_rejects_noncanonical_transform_identity(self) -> None:
        from ifdr_yolo.eval.factor_observer import ImageObservationPlan

        with tempfile.TemporaryDirectory() as directory:
            manifest = self._manifest(Path(directory))
            plan = manifest.plans[0]
            index = next(
                index
                for index, condition in enumerate(plan.conditions)
                if condition.intervention_kind == "sampling"
            )
            altered = list(plan.conditions)
            altered[index] = replace(altered[index], transform_id="12" * 32)
            with self.assertRaisesRegex(ValueError, "transform"):
                ImageObservationPlan(
                    image_id=plan.image_id,
                    image_path=plan.image_path,
                    width=plan.width,
                    height=plan.height,
                    source_sha256=plan.source_sha256,
                    conditions=tuple(altered),
                    expected_observation_ids=plan.expected_observation_ids,
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
                "required_nodes": (11, 14, 17, 20, 23, 26),
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
                    required_nodes=(11, 14, 17, 20, 23, 26),
                    input_size=64,
                )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                build_factor_observation_manifest(selected_intervention_objects=set(), checkpoint_sha256="bad", **{key: value for key, value in kwargs.items() if key != "checkpoint_sha256"})

    def test_required_nodes_are_the_registered_six_only(self) -> None:
        from ifdr_yolo.eval.factor_observer import FactorObservationManifest, build_factor_observation_manifest

        registered = (11, 14, 17, 20, 23, 26)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            image_paths = {plan.image_id: plan.image_path for plan in manifest.plans}
            records = (_record("a", 1), _record("b", 2))
            for nodes in ((11,), (14, 11, 17, 20, 23, 26), registered + (29,)):
                with self.subTest(nodes=nodes):
                    with self.assertRaises(ValueError):
                        build_factor_observation_manifest(
                            records,
                            image_paths,
                            {("a", 1)},
                            "ab" * 32,
                            17,
                            required_nodes=nodes,
                            input_size=64,
                        )
                    with self.assertRaises(ValueError):
                        FactorObservationManifest(
                            plans=manifest.plans,
                            checkpoint_sha256=manifest.checkpoint_sha256,
                            seed=manifest.seed,
                            required_nodes=nodes,
                            input_size=manifest.input_size,
                        )

    def test_manifest_seed_is_bound_to_every_condition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._manifest(Path(directory))
            with self.assertRaisesRegex(ValueError, "seed"):
                replace(manifest, seed=manifest.seed + 1)

    def test_condition_and_pair_ids_are_canonical_derived_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._manifest(Path(directory))
            plan = manifest.plans[0]
            natural = next(
                condition
                for condition in plan.conditions
                if condition.intervention_kind == "natural"
            )
            controlled = next(
                condition
                for condition in plan.conditions
                if condition.intervention_kind == "sampling"
            )
            with self.assertRaisesRegex(ValueError, "condition_id"):
                replace(natural, condition_id="12" * 32)
            with self.assertRaisesRegex(ValueError, "pair_id"):
                replace(controlled, pair_id="34" * 32)

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
            manifest = self._manifest(root)
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
            with self.assertRaisesRegex(ValueError, "unterminated|malformed|suffix"):
                FactorObservationJournal(manifest, output, progress)

    def test_journal_commit_uses_cached_prefix_without_scanning(self) -> None:
        from unittest.mock import patch
        from ifdr_yolo.eval.factor_observer import FactorObservationJournal

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            journal = FactorObservationJournal(manifest, root / "rows.jsonl", root / "progress.json")
            plan = manifest.plans[0]
            rows = [{"observation_id": item, "image_id": plan.image_id} for item in plan.expected_observation_ids]
            with patch.object(journal, "_scan_file", side_effect=AssertionError("commit must not rescan JSONL")) as scan:
                self.assertTrue(journal.commit_image(plan.image_id, rows))
            scan.assert_not_called()

    def test_journal_commit_rejects_external_output_suffix_without_mutation(self) -> None:
        from ifdr_yolo.eval.factor_observer import FactorObservationJournal

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            output, progress = root / "rows.jsonl", root / "progress.json"
            journal = FactorObservationJournal(manifest, output, progress)
            plan = manifest.plans[0]
            rows = [{"observation_id": item, "image_id": plan.image_id} for item in plan.expected_observation_ids]
            before_progress = progress.read_bytes()
            output.write_bytes(output.read_bytes() + b'{"external":true}\n')
            before_output = output.read_bytes()
            with self.assertRaisesRegex(ValueError, "output JSONL changed"):
                journal.commit_image(plan.image_id, rows)
            self.assertEqual(output.read_bytes(), before_output)
            self.assertEqual(progress.read_bytes(), before_progress)

    def test_journal_commit_rejects_external_progress_change_without_output_write(self) -> None:
        from ifdr_yolo.eval.factor_observer import FactorObservationJournal

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            output, progress = root / "rows.jsonl", root / "progress.json"
            journal = FactorObservationJournal(manifest, output, progress)
            plan = manifest.plans[0]
            rows = [{"observation_id": item, "image_id": plan.image_id} for item in plan.expected_observation_ids]
            forged = json.loads(progress.read_text(encoding="utf-8"))
            forged["status"] = "external-writer"
            progress.write_text(json.dumps(forged, sort_keys=True) + "\n", encoding="utf-8")
            before_output = output.read_bytes()
            before_progress = progress.read_bytes()
            with self.assertRaisesRegex(ValueError, "progress JSON changed"):
                journal.commit_image(plan.image_id, rows)
            self.assertEqual(output.read_bytes(), before_output)
            self.assertEqual(progress.read_bytes(), before_progress)

    def test_journal_rejects_hash_drift_identity_variants_and_missing_finalize(self) -> None:
        from ifdr_yolo.eval.factor_observer import FactorObservationJournal, build_factor_observation_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
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
                required_nodes=(11, 14, 17, 20, 23, 26),
                input_size=64,
            )
            with self.assertRaisesRegex(ValueError, "hash"):
                FactorObservationJournal(drift, output, progress)

    def test_journal_uses_fsync_for_progress_and_rows(self) -> None:
        from unittest.mock import patch
        from ifdr_yolo.eval.factor_observer import FactorObservationJournal

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            journal = FactorObservationJournal(manifest, root / "rows.jsonl", root / "progress.json")
            plan = manifest.plans[0]
            rows = [{"observation_id": item, "image_id": plan.image_id} for item in plan.expected_observation_ids]
            with patch("ifdr_yolo.eval.factor_observer.os.fsync", wraps=os_fsync) as fsync:
                journal.commit_image("a", rows)
            self.assertGreaterEqual(fsync.call_count, 2)

    def test_journal_rejects_forged_inflight_progress_without_mutation(self) -> None:
        from ifdr_yolo.eval.factor_observer import FactorObservationJournal

        cases = [
            ("extra field", lambda value: value.update(extra=True)),
            ("missing field", lambda value: value.pop("expected_hash")),
            ("unknown image", lambda value: value.update(image_id="unknown")),
            ("float offset", lambda value: value.update(start_offset=1.2)),
            ("bool offset", lambda value: value.update(start_offset=True)),
            ("offset beyond file", lambda value: value.update(start_offset=10**9)),
            ("uppercase hash", lambda value: value.update(expected_hash="A" * 64)),
            ("wrong hash", lambda value: value.update(expected_hash="0" * 64)),
            ("bool row count", lambda value: value.update(expected_row_count=True)),
            ("float row count", lambda value: value.update(expected_row_count=1.0)),
            ("wrong row count", lambda value: value.update(expected_row_count=0)),
        ]
        for name, mutate in cases:
            with self.subTest(case=name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    manifest = self._manifest(root)
                    output = root / "rows.jsonl"
                    progress = root / "progress.json"
                    journal = FactorObservationJournal(manifest, output, progress)
                    plan = manifest.plans[1]
                    rows = [{"observation_id": item, "image_id": plan.image_id} for item in plan.expected_observation_ids]

                    def crash(_phase: str) -> None:
                        raise RuntimeError("power loss")

                    with self.assertRaises(RuntimeError):
                        journal.commit_image(plan.image_id, rows, crash_hook=crash)
                    progress_payload = json.loads(progress.read_text())
                    original_output = output.read_bytes()
                    mutate(progress_payload["inflight"])
                    progress.write_text(json.dumps(progress_payload, sort_keys=True) + "\n")
                    forged_progress = progress.read_text()
                    with self.assertRaises(ValueError):
                        FactorObservationJournal(manifest, output, progress)
                    self.assertEqual(output.read_bytes(), original_output)
                    self.assertEqual(progress.read_text(), forged_progress)

        # A valid transaction for an already completed image is also forged by
        # changing only the inflight identity; it must not truncate/clear.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            output, progress = root / "rows.jsonl", root / "progress.json"
            journal = FactorObservationJournal(manifest, output, progress)
            first = manifest.plans[0]
            first_rows = [{"observation_id": item, "image_id": first.image_id} for item in first.expected_observation_ids]
            journal.commit_image(first.image_id, first_rows)
            second = manifest.plans[1]
            second_rows = [{"observation_id": item, "image_id": second.image_id} for item in second.expected_observation_ids]

            def crash(_phase: str) -> None:
                raise RuntimeError("power loss")

            with self.assertRaises(RuntimeError):
                journal.commit_image(second.image_id, second_rows, crash_hook=crash)
            payload = json.loads(progress.read_text())
            payload["inflight"]["image_id"] = first.image_id
            before_output = output.read_bytes()
            progress.write_text(json.dumps(payload, sort_keys=True) + "\n")
            forged_progress = progress.read_text()
            with self.assertRaises(ValueError):
                FactorObservationJournal(manifest, output, progress)
            self.assertEqual(output.read_bytes(), before_output)
            self.assertEqual(progress.read_text(), forged_progress)

    def test_journal_rejects_inflight_offset_before_completed_prefix(self) -> None:
        from ifdr_yolo.eval.factor_observer import FactorObservationJournal

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            output, progress = root / "rows.jsonl", root / "progress.json"
            journal = FactorObservationJournal(manifest, output, progress)
            first, second = manifest.plans
            first_rows = [{"observation_id": item, "image_id": first.image_id} for item in first.expected_observation_ids]
            second_rows = [{"observation_id": item, "image_id": second.image_id} for item in second.expected_observation_ids]
            journal.commit_image(first.image_id, first_rows)

            def crash(_phase: str) -> None:
                raise RuntimeError("power loss")

            with self.assertRaises(RuntimeError):
                journal.commit_image(second.image_id, second_rows, crash_hook=crash)
            payload = json.loads(progress.read_text())
            payload["inflight"]["start_offset"] = 0
            progress.write_text(json.dumps(payload, sort_keys=True) + "\n")
            forged_progress = progress.read_text()
            before_output = output.read_bytes()
            with self.assertRaises(ValueError):
                FactorObservationJournal(manifest, output, progress)
            self.assertEqual(output.read_bytes(), before_output)
            self.assertEqual(progress.read_text(), forged_progress)

    def test_journal_rejects_completed_prefix_corruption_before_any_recovery(self) -> None:
        from ifdr_yolo.eval.factor_observer import FactorObservationJournal

        for corruption in ("truncate", "byte", "offset"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self._manifest(root)
                output, progress = root / "rows.jsonl", root / "progress.json"
                journal = FactorObservationJournal(manifest, output, progress)
                plan = manifest.plans[0]
                rows = [{"observation_id": item, "image_id": plan.image_id} for item in plan.expected_observation_ids]
                journal.commit_image(plan.image_id, rows)
                if corruption == "truncate":
                    output.write_bytes(output.read_bytes()[:-1])
                elif corruption == "byte":
                    data = bytearray(output.read_bytes())
                    data[0] = ord("{") if data[0] != ord("{") else ord("[")
                    output.write_bytes(bytes(data))
                else:
                    payload = json.loads(progress.read_text())
                    payload["completed"][plan.image_id]["end_offset"] = 0
                    progress.write_text(json.dumps(payload, sort_keys=True) + "\n")
                forged_output = output.read_bytes()
                forged_progress = progress.read_text()
                with self.assertRaises(ValueError):
                    FactorObservationJournal(manifest, output, progress)
                self.assertEqual(output.read_bytes(), forged_output)
                self.assertEqual(progress.read_text(), forged_progress)


def _iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (area_left + area_right - intersection) if intersection else 0.0


if __name__ == "__main__":
    unittest.main()
