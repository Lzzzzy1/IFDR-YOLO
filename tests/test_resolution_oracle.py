from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import os
from unittest.mock import patch

from ifdr_yolo.data.kitti_types import BoundingBox, Detection, KittiObject
from ifdr_yolo.eval.resolution_oracle import (
    MAX_DETECTIONS,
    PROPOSAL_CONF_MAX,
    PROPOSAL_CONF_MIN,
    UtilityComponents,
    ImageSize,
    build_o1_candidates,
    build_o2_candidate_pool,
    classwise_nms,
    crop_to_full_box,
    fixed_crop_window,
    full_to_crop_box,
    moderate_utility,
    normalized_yolo_to_box,
    select_one_crop,
    box_to_normalized_yolo,
)

from scripts.run_resolution_oracle import (
    ActiveOracleRunError,
    OracleConfig,
    OracleIdentityError,
    OraclePaths,
    OracleRules,
    ResolutionOracleServices,
    load_oracle_config,
    run_resolution_oracle,
)
from PIL import Image


def obj(kind: str, box: BoundingBox) -> KittiObject:
    return KittiObject(
        kind=kind,
        truncated=0.0,
        occluded=0,
        alpha=0.0,
        bbox=box,
        dimensions_hwl=(1.0, 1.0, 1.0),
        location_xyz=(0.0, 0.0, 10.0),
        rotation_y=0.0,
    )


class ResolutionOracleTest(unittest.TestCase):
    def test_fixed_crop_is_2x_half_extent_and_clamped(self) -> None:
        size = ImageSize(100, 80)
        crop = fixed_crop_window(BoundingBox(5.0, 10.0, 25.0, 30.0), size)
        self.assertEqual(crop.as_xyxy(), (0.0, 0.0, 50.0, 40.0))
        self.assertEqual(crop.width, size.width / 2.0)
        self.assertEqual(crop.height, size.height / 2.0)
        self.assertGreaterEqual(crop.x1, 0.0)
        self.assertLessEqual(crop.x2, size.width)

    def test_odd_image_uses_integer_ceil_half_window_and_round_trips(self) -> None:
        size = ImageSize(1242, 375)
        target = BoundingBox(1220.0, 350.0, 1241.0, 374.0)
        crop = fixed_crop_window(target, size)
        self.assertEqual(crop.as_xyxy(), (621.0, 187.0, 1242.0, 375.0))
        self.assertEqual((crop.width, crop.height), (621.0, 188.0))
        self.assertTrue(all(float(value).is_integer() for value in crop.as_xyxy()))
        full_box = BoundingBox(1220.0, 350.0, 1241.0, 374.0)
        mapped = crop_to_full_box(
            full_to_crop_box(full_box, crop), crop, image_size=size
        )
        self.assertEqual(mapped.as_xyxy(), full_box.as_xyxy())

    def test_odd_image_grid_is_integer_fixed_size_and_in_bounds(self) -> None:
        size = ImageSize(1242, 375)
        candidates = build_o2_candidate_pool({"000001": size}, {"000001": ()})[
            "000001"
        ]
        grid = [candidate.window for candidate in candidates if candidate.source == "grid"]
        self.assertEqual(len(grid), 6)
        self.assertTrue(
            all(
                window.width == 621.0
                and window.height == 188.0
                and all(float(value).is_integer() for value in window.as_xyxy())
                and 0.0 <= window.x1 < window.x2 <= 1242.0
                and 0.0 <= window.y1 < window.y2 <= 375.0
                for window in grid
            )
        )

    def test_crop_mapping_and_normalized_yolo_round_trip(self) -> None:
        size = ImageSize(200, 100)
        crop = fixed_crop_window(BoundingBox(90.0, 40.0, 110.0, 60.0), size)
        full_box = BoundingBox(92.0, 42.0, 108.0, 58.0)
        crop_box = full_to_crop_box(full_box, crop)
        mapped = crop_to_full_box(crop_box, crop, image_size=size)
        self.assertEqual(mapped.as_xyxy(), full_box.as_xyxy())

        yolo = box_to_normalized_yolo(full_box, size)
        restored = normalized_yolo_to_box(yolo, size)
        for expected, actual in zip(full_box.as_xyxy(), restored.as_xyxy()):
            self.assertAlmostEqual(expected, actual)

    def test_classwise_nms_is_stable_base_first_and_keeps_cross_class(self) -> None:
        box = BoundingBox(10.0, 10.0, 50.0, 90.0)
        overlap = BoundingBox(11.0, 11.0, 51.0, 91.0)
        detections = (
            Detection("000001", "Pedestrian", 0.8, box),  # base, first tie
            Detection("000001", "Pedestrian", 0.8, overlap),  # crop
            Detection("000001", "Cyclist", 0.8, overlap),
        )
        kept = classwise_nms(detections, iou_threshold=0.70, max_det=MAX_DETECTIONS)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0].kind, "Pedestrian")
        self.assertEqual(kept[0].bbox, box)
        self.assertEqual(kept[1].kind, "Cyclist")

    def test_nms_has_global_max_det(self) -> None:
        detections = tuple(
            Detection(
                "000001",
                "Car",
                1.0 - index / 1000.0,
                BoundingBox(float(index * 20), 0.0, float(index * 20 + 10), 30.0),
            )
            for index in range(MAX_DETECTIONS + 7)
        )
        self.assertEqual(len(classwise_nms(detections)), MAX_DETECTIONS)

    def test_o1_uses_only_small_ground_truth_targets(self) -> None:
        gt = {
            "000001": (
                obj("Pedestrian", BoundingBox(10.0, 10.0, 20.0, 49.0)),
                obj("Cyclist", BoundingBox(30.0, 10.0, 40.0, 50.0)),
            )
        }
        candidates = build_o1_candidates(gt, {"000001": ImageSize(100, 80)})
        self.assertEqual(len(candidates["000001"]), 1)
        self.assertEqual(candidates["000001"][0].source, "gt")

    def test_o2_pool_is_gt_free_bounded_and_uses_fixed_windows(self) -> None:
        size = {"000001": ImageSize(120, 80)}
        predictions = {
            "000001": tuple(
                Detection(
                    "000001",
                    "Pedestrian",
                    PROPOSAL_CONF_MIN + index * 0.01,
                    BoundingBox(float(index * 3), 10.0, float(index * 3 + 10), 45.0),
                )
                for index in range(30)
            )
        }
        first = build_o2_candidate_pool(size, predictions)
        second = build_o2_candidate_pool(size, predictions)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first["000001"]), 24)
        self.assertEqual(sum(c.source == "grid" for c in first["000001"]), 6)
        self.assertLessEqual(sum(c.source == "proposal" for c in first["000001"]), 18)
        self.assertTrue(all(c.source != "gt" for c in first["000001"]))
        self.assertTrue(all(c.window.width == 60.0 for c in first["000001"]))
        self.assertTrue(all(c.window.height == 40.0 for c in first["000001"]))
        grid = [candidate for candidate in first["000001"] if candidate.source == "grid"]
        self.assertEqual(
            [candidate.window.x1 for candidate in grid],
            [0.0, 30.0, 60.0, 0.0, 30.0, 60.0],
        )
        self.assertEqual(
            [candidate.window.y1 for candidate in grid],
            [0.0, 0.0, 0.0, 40.0, 40.0, 40.0],
        )
        self.assertGreaterEqual(PROPOSAL_CONF_MAX, PROPOSAL_CONF_MIN)

    def test_o2_excludes_confidence_and_height_boundaries(self) -> None:
        predictions = {
            "000001": (
                Detection("000001", "Pedestrian", 0.0009, BoundingBox(0, 0, 10, 39)),
                Detection("000001", "Pedestrian", 0.001, BoundingBox(0, 0, 10, 39)),
                Detection("000001", "Pedestrian", 0.25, BoundingBox(0, 0, 10, 39)),
                Detection("000001", "Pedestrian", 0.2501, BoundingBox(0, 0, 10, 39)),
                Detection("000001", "Pedestrian", 0.1, BoundingBox(0, 0, 10, 40)),
            )
        }
        candidates = build_o2_candidate_pool({"000001": ImageSize(100, 80)}, predictions)
        proposals = [candidate for candidate in candidates["000001"] if candidate.source == "proposal"]
        self.assertEqual(len(proposals), 1)

    def test_o2_proposals_are_area_then_confidence_then_class_then_coordinates(self) -> None:
        predictions = {
            "000001": (
                Detection("000001", "Cyclist", 0.20, BoundingBox(40, 0, 50, 39)),
                Detection("000001", "Pedestrian", 0.10, BoundingBox(20, 0, 30, 39)),
                Detection("000001", "Car", 0.10, BoundingBox(60, 0, 70, 39)),
            )
        }
        candidates = build_o2_candidate_pool(
            {"000001": ImageSize(100, 80)}, predictions
        )["000001"]
        proposals = [candidate for candidate in candidates if candidate.source == "proposal"]
        self.assertEqual([candidate.proposal_score for candidate in proposals], [0.10, 0.10, 0.20])

    def test_one_crop_budget_and_non_positive_utility(self) -> None:
        candidates = build_o2_candidate_pool(
            {"000001": ImageSize(100, 80)}, {"000001": ()}
        )["000001"]
        utilities = {
            candidate: UtilityComponents(delta_tp=0.0, delta_mean_iou=0.0, delta_fp=0.0, delta_duplicates=0.0)
            for candidate in candidates
        }
        self.assertIsNone(select_one_crop(candidates, utilities))
        positive = dict(utilities)
        positive[candidates[2]] = UtilityComponents(delta_tp=1.0, delta_mean_iou=0.0, delta_fp=0.0, delta_duplicates=0.0)
        chosen = select_one_crop(candidates, positive)
        self.assertEqual(chosen, candidates[2])

    def test_utility_coefficients_are_fixed_and_moderate_matching_uses_class_iou(self) -> None:
        components = UtilityComponents(1.0, 0.8, 2.0, 3.0)
        self.assertAlmostEqual(components.utility, 1.0 + 0.25 * 0.8 - 0.25 * 2.0 - 0.10 * 3.0)
        truth = {
            "000001": (obj("Pedestrian", BoundingBox(0, 0, 50, 80)),)
        }
        base = {"000001": (Detection("000001", "Pedestrian", 0.9, BoundingBox(0, 0, 25, 80)),)}
        crop = {"000001": (Detection("000001", "Pedestrian", 0.9, BoundingBox(0, 0, 50, 80)),)}
        result = moderate_utility(base, crop, truth)
        self.assertEqual(result.delta_tp, 1.0)
        self.assertEqual(result.delta_fp, -1.0)

    def test_moderate_utility_ignores_dontcare_and_invalid_same_class(self) -> None:
        dontcare_truth = {
            "000001": (obj("DontCare", BoundingBox(0, 0, 100, 100)),)
        }
        dontcare_detection = {
            "000001": (
                Detection("000001", "Pedestrian", 0.9, BoundingBox(10, 10, 40, 80)),
            )
        }
        empty = {"000001": ()}
        dontcare_result = moderate_utility(dontcare_detection, empty, dontcare_truth)
        self.assertEqual(dontcare_result.delta_fp, 0.0)

        ignored_truth = {
            "000001": (obj("Pedestrian", BoundingBox(0, 0, 20, 20)),)
        }
        ignored_detection = {
            "000001": (
                Detection("000001", "Pedestrian", 0.9, BoundingBox(0, 0, 20, 20)),
            )
        }
        ignored_result = moderate_utility(ignored_detection, empty, ignored_truth)
        self.assertEqual(ignored_result.delta_fp, 0.0)

    def test_registered_oracle_config_freezes_route_and_paths(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_oracle_config(
            root / "configs/experiments/oracles/kitti_p2_resolution_oracle_s17.yaml",
            repository_root=root,
        )
        self.assertEqual(config.model_role, "plain_p2")
        self.assertEqual(config.checkpoint_role, "last.pt")
        self.assertEqual(config.rules.proposal_limit, 18)
        self.assertEqual(config.rules.max_crops_per_image, 1)
        self.assertEqual(config.rules.o1_min_delta_ap40, 3.0)

    def test_oracle_identity_mismatch_fails_closed_and_o1_failure_does_not_create_o2_pool(self) -> None:
        class EmptyAdapter:
            def predict(self, *, output_dir: Path, image_paths: tuple[Path, ...], **_: object) -> Path:
                labels = output_dir / "labels"
                labels.mkdir(parents=True, exist_ok=True)
                (labels / f"{image_paths[0].stem}.txt").write_text(
                    "0 0.5 0.5 0.2 0.2 0.5\n", encoding="utf-8"
                )
                return labels

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            for image_id in ("000000", "000001"):
                Image.new("RGB", (100, 80), color="black").save(images / f"{image_id}.png")
                label = "Car 0.0 0 0.0 10 10 30 40 1 1 1 0 0 10 0\n" if image_id == "000000" else ""
                (labels / f"{image_id}.txt").write_text(label, encoding="utf-8")
            split = root / "development_ids.txt"
            split.write_text("000000\n000001\n", encoding="utf-8")
            model = root / "kitti-p2-m.yaml"
            model.write_text("nc: 3\n", encoding="utf-8")
            checkpoint = root / "last.pt"
            checkpoint.write_bytes(b"checkpoint")
            fit = root / "fit_ids.txt"
            fit.write_text("000002\n", encoding="utf-8")
            model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            fit_sha = hashlib.sha256(fit.read_bytes()).hexdigest()
            dev_sha = hashlib.sha256(split.read_bytes()).hexdigest()
            reference_identity = "a" * 64
            audit = {
                "intersection_count": 0,
                "fit_manifest_sha256": fit_sha,
                "development_manifest_sha256": dev_sha,
                "identity_sha256": reference_identity,
            }
            audit_path = root / "post_training_leakage_audit.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            provenance = {
                "checkpoint_role": "last.pt",
                "primary_checkpoint_role": "last.pt",
                "checkpoint_sha256": checkpoint_sha,
                "last_pt_sha256": checkpoint_sha,
                "model_sha256": model_sha,
                "fit_manifest_sha256": fit_sha,
                "development_manifest_sha256": dev_sha,
                "post_training_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                "identity_sha256": reference_identity,
            }
            (root / "checkpoint_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
            config = OracleConfig(
                schema_version=1,
                dataset="kitti",
                model_role="plain_p2",
                seed=17,
                checkpoint_role="last.pt",
                paths=OraclePaths(
                    model=model,
                    checkpoint=checkpoint,
                    development_ids=split,
                    raw_images=images,
                    raw_labels=labels,
                    model_sha256=model_sha,
                    checkpoint_sha256=checkpoint_sha,
                    fit_ids=fit,
                    fit_manifest_sha256=fit_sha,
                ),
                rules=OracleRules(),
                source_path=root / "oracle.yaml",
            )
            config.source_path.write_text("schema_version: 1\n", encoding="utf-8")
            evaluations = []
            def fake_eval(**_: object) -> dict[str, object]:
                evaluations.append(True)
                score = 2.0 if len(evaluations) == 2 else 0.0
                return {"classes": {name: {"moderate": {"ap40": score}} for name in ("Car", "Pedestrian", "Cyclist")}}
            services = ResolutionOracleServices(adapter=EmptyAdapter(), evaluate=fake_eval)
            with patch("scripts.run_resolution_oracle.REGISTERED_DEVELOPMENT_COUNT", 2), patch(
                "scripts.run_resolution_oracle.REGISTERED_DEVELOPMENT_IDS_SHA256",
                hashlib.sha256(split.read_bytes()).hexdigest(),
            ):
                null_config = replace(
                    config,
                    paths=replace(config.paths, checkpoint_sha256=None),
                )
                with self.assertRaises(OracleIdentityError):
                    run_resolution_oracle(
                        null_config,
                        repository_root=root,
                        output_dir=root / "null-job",
                        mirror_dir=root / "null-mirror",
                        services=services,
                    )
                result = run_resolution_oracle(
                    config,
                    repository_root=root,
                    output_dir=root / "job",
                    mirror_dir=root / "mirror",
                    services=services,
                )
                self.assertEqual(result.state, "complete")
                self.assertFalse((root / "job" / "candidate_pool.jsonl").exists())
                decision = json.loads((root / "job" / "route_a_decision.json").read_text(encoding="utf-8"))
                self.assertNotEqual(decision["decision"], "ADVANCE")
                self.assertEqual(
                    json.loads((root / "job" / "stratified_no_harm.json").read_text(encoding="utf-8"))["status"],
                    "NOT_RUN",
                )
                candidate_journal = json.loads(
                    (root / "job" / "journals" / "o1_select" / "000000.json").read_text(encoding="utf-8")
                )
                self.assertTrue(candidate_journal["candidate_records"])
                self.assertTrue(candidate_journal["candidate_records"][0]["mapped_prediction"])
                status_path = root / "job" / "status.json"
                status = json.loads(status_path.read_text(encoding="utf-8"))
                status.update({"state": "running", "pid": os.getpid()})
                status_path.write_text(json.dumps(status), encoding="utf-8")
                with self.assertRaises(ActiveOracleRunError):
                    run_resolution_oracle(
                        config,
                        repository_root=root,
                        output_dir=root / "job",
                        mirror_dir=root / "mirror",
                        services=services,
                    )
                identity = json.loads((root / "job" / "run_identity.json").read_text(encoding="utf-8"))
                identity["identity"]["checkpoint_sha256"] = "0" * 64
                (root / "job" / "run_identity.json").write_text(json.dumps(identity), encoding="utf-8")
                with self.assertRaises(OracleIdentityError):
                    run_resolution_oracle(
                        config,
                        repository_root=root,
                        output_dir=root / "job",
                        mirror_dir=root / "mirror",
                        services=services,
                        resume=True,
                    )



if __name__ == "__main__":
    unittest.main()
