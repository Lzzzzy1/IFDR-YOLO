from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from unittest.mock import patch

from ifdr_yolo.experiments.ultralytics_runtime import bootstrap_ultralytics_config


ROOT = Path(__file__).resolve().parents[1]
bootstrap_ultralytics_config(ROOT)


class FactorSpecificityDataTest(unittest.TestCase):
    def test_specificity_rejects_overlapping_background(self):
        from ifdr_yolo.data.ifdr_dataset import build_specificity_pair

        labels = {"bboxes": torch.tensor([[0.1, 0.1, 0.3, 0.3],
                                           [0.6, 0.6, 0.8, 0.8]])}
        with self.assertRaisesRegex(ValueError, "background overlaps annotated object"):
            build_specificity_pair(
                labels, target_index=0, background_box=(0.6, 0.6, 0.8, 0.8),
                severity=0.5, transform_seed=7,
            )

    def test_collate_preserves_three_views_and_normalized_roi_identity(self):
        from ifdr_yolo.data.ifdr_dataset import (
            BACKGROUND_IMAGE_KEY,
            CLEAN_IMAGE_KEY,
            FACTOR_OBJECT_TARGETS_KEY,
            TARGET_IMAGE_KEY,
            collate_ifdr_batch,
        )

        view = torch.zeros(3, 8, 8, dtype=torch.uint8)
        sample = {
            "img": view,
            "bboxes": torch.tensor([[0.25, 0.25, 0.50, 0.50]]),
            "cls": torch.tensor([[2.0]]),
            CLEAN_IMAGE_KEY: view.clone(),
            TARGET_IMAGE_KEY: view.clone(),
            BACKGROUND_IMAGE_KEY: view.clone(),
            FACTOR_OBJECT_TARGETS_KEY: ({
                "class_id": 2,
                "box_xyxy_normalized": (0.25, 0.25, 0.50, 0.50),
                "target": (0.2, 0.3),
                "valid": (True, True),
            },),
        }
        batch = collate_ifdr_batch([sample])
        self.assertEqual(tuple(batch[CLEAN_IMAGE_KEY].shape), (1, 3, 8, 8))
        self.assertEqual(tuple(batch[TARGET_IMAGE_KEY].shape), (1, 3, 8, 8))
        self.assertEqual(tuple(batch[BACKGROUND_IMAGE_KEY].shape), (1, 3, 8, 8))
        target = batch[FACTOR_OBJECT_TARGETS_KEY][0]
        self.assertEqual(target.batch_index, 0)
        self.assertEqual(target.box_xyxy_normalized, (0.25, 0.25, 0.50, 0.50))

    def test_low_severity_gets_zero_specificity_weight(self):
        from ifdr_yolo.data.ifdr_dataset import build_specificity_pair

        pair = build_specificity_pair(
            {"bboxes": torch.tensor([[0.1, 0.1, 0.3, 0.3]])},
            target_index=0,
            background_box=(0.6, 0.6, 0.8, 0.8),
            severity=0.24,
            transform_seed=7,
        )
        self.assertEqual(pair.weight, 0.0)

    def test_target_background_share_severity_and_transform_seed(self):
        from ifdr_yolo.data.ifdr_dataset import build_specificity_pair

        pair = build_specificity_pair(
            {"bboxes": torch.tensor([[0.1, 0.1, 0.3, 0.3]])},
            target_index=0,
            background_box=(0.6, 0.6, 0.8, 0.8),
            severity=0.5,
            transform_seed=7,
            factor_kind="sampling",
        )
        self.assertEqual(pair.severity, 0.5)
        self.assertEqual(pair.transform_seed, 7)
        self.assertEqual(pair.factor_kind, "sampling")
        self.assertEqual(pair.factor_channel, 0)
        self.assertEqual(pair.background_max_iou, 0.0)

    def test_malformed_pair_counts_rejection(self):
        from ifdr_yolo.data.ifdr_dataset import (
            SpecificityRejectionCounter,
            build_specificity_pair,
        )

        counter = SpecificityRejectionCounter()
        with self.assertRaises(ValueError):
            build_specificity_pair(
                {"bboxes": torch.tensor([[0.1, 0.1, 0.3, 0.3]])},
                target_index=0,
                background_box=(0.6, 0.6, 0.8, 0.8),
                severity=float("nan"),
                transform_seed=7,
                rejection_counter=counter,
            )
        self.assertEqual(counter.total, 1)

    def test_single_sided_intervention_spec_fails_closed_in_collate(self):
        from ifdr_yolo.data.ifdr_dataset import (
            SPECIFICITY_PAIRS_KEY,
            collate_ifdr_batch,
        )
        from ifdr_yolo.data.interventions.schema import (
            InterventionKind,
            InterventionRole,
            InterventionSpec,
        )

        target_spec = InterventionSpec(
            image_id="000123",
            kind=InterventionKind.SAMPLING,
            role=InterventionRole.OBJECT,
            strength=0.5,
            seed=7,
            object_id=0,
            region_xyxy=(0.1, 0.1, 0.3, 0.3),
        )
        pair = {
            "target_index": 0,
            "target_box_xyxy_normalized": (0.1, 0.1, 0.3, 0.3),
            "background_box_xyxy_normalized": (0.6, 0.6, 0.8, 0.8),
            "factor_kind": "sampling",
            "factor_channel": 0,
            "severity": 0.5,
            "transform_seed": 7,
            "weight": 0.5,
            "background_max_iou": 0.0,
            "target_spec": target_spec,
        }
        sample = {
            "img": torch.zeros(3, 8, 8, dtype=torch.uint8),
            "bboxes": torch.zeros(0, 4),
            "cls": torch.zeros(0, 1),
            SPECIFICITY_PAIRS_KEY: (pair,),
        }
        with self.assertRaisesRegex(ValueError, "specs|simultaneously|together"):
            collate_ifdr_batch([sample])

    def test_empty_background_has_zero_iou_with_annotations(self):
        from ifdr_yolo.data.ifdr_dataset import build_specificity_pair

        pair = build_specificity_pair(
            {"bboxes": torch.tensor([[0.1, 0.1, 0.3, 0.3],
                                      [0.6, 0.6, 0.8, 0.8]])},
            target_index=0,
            background_box=(0.35, 0.35, 0.55, 0.55),
            severity=0.5,
            transform_seed=7,
        )
        self.assertEqual(pair.background_max_iou, 0.0)

    def _calibration_labels(self):
        from ultralytics.utils.instance import Instances

        image = np.full((64, 96, 3), 127, dtype=np.uint8)
        return {
            "img": image,
            "im_file": "/dataset/images/train/000123.png",
            "cls": np.array([[0.0]], dtype=np.float32),
            "instances": Instances(
                bboxes=np.array([[0.20, 0.20, 0.20, 0.20]], dtype=np.float32),
                bbox_format="xywh",
                normalized=True,
            ),
            "_ifdr_metadata_records": ({
                "object_id": "000123:000000",
                "class_id": 0,
                "class_name": "Car",
                "raw_label_index": 0,
                "raw_box_xyxy_normalized": (0.10, 0.10, 0.30, 0.30),
                "box_xyxy_normalized": (0.10, 0.10, 0.30, 0.30),
                "sampling": 0.2,
                "visibility": 0.3,
                "sampling_valid": True,
                "visibility_valid": True,
                "target": (0.2, 0.3),
                "valid": (True, True),
            },),
        }

    def test_calibration_emits_matched_three_views_and_background_zero_target(self):
        from ifdr_yolo.data.ifdr_dataset import (
            BACKGROUND_FACTOR_TARGET_KEY,
            BACKGROUND_IMAGE_KEY,
            CLEAN_IMAGE_KEY,
            FACTOR_TARGET_KEY,
            FACTOR_WEIGHT_KEY,
            IFDRInterventionTransform,
            SPECIFICITY_PAIRS_KEY,
            TARGET_IMAGE_KEY,
            SharedEpoch,
        )
        from ifdr_yolo.data.interventions.sampler import SamplingPolicy

        transform = IFDRInterventionTransform(
            base_seed=17,
            epoch_state=SharedEpoch(0),
            enabled=True,
            calibration_enabled=True,
            policy=SamplingPolicy(
                identity_probability=0.0,
                sampling_probability=1.0,
                visibility_probability=0.0,
                minimum_strength=0.5,
                maximum_strength=0.5,
            ),
        )
        result = transform(self._calibration_labels())
        self.assertEqual(result[CLEAN_IMAGE_KEY].shape, (3, 64, 96))
        self.assertEqual(result[TARGET_IMAGE_KEY].shape, (3, 64, 96))
        self.assertEqual(result[BACKGROUND_IMAGE_KEY].shape, (3, 64, 96))
        self.assertTrue(torch.equal(result[BACKGROUND_FACTOR_TARGET_KEY], torch.zeros_like(result[FACTOR_TARGET_KEY])))
        pair = result[SPECIFICITY_PAIRS_KEY][0]
        self.assertEqual(pair.factor_channel, 0)
        self.assertEqual(pair.target_spec.seed, pair.background_spec.seed)
        self.assertEqual(pair.target_spec.strength, pair.background_spec.strength)
        self.assertGreater(float(result[FACTOR_WEIGHT_KEY].sum()), 0.0)

    def test_calibration_identity_drift_fails_closed(self):
        from ifdr_yolo.data.ifdr_dataset import IFDRInterventionTransform, SharedEpoch, SpecificityRejectionCounter
        from ifdr_yolo.data.interventions.sampler import SamplingPolicy

        labels = self._calibration_labels()
        labels["cls"] = np.array([[1.0]], dtype=np.float32)
        counter = SpecificityRejectionCounter()
        transform = IFDRInterventionTransform(
            base_seed=17,
            epoch_state=SharedEpoch(0),
            enabled=True,
            calibration_enabled=True,
            rejection_counter=counter,
            policy=SamplingPolicy(identity_probability=0.0, sampling_probability=1.0, visibility_probability=0.0),
        )
        with self.assertRaisesRegex(ValueError, "identity"):
            transform(labels)
        self.assertGreaterEqual(counter.total, 1)

    def test_calibration_letterbox_coordinates_use_current_instances_box(self):
        from ultralytics.utils.instance import Instances
        from ifdr_yolo.data.ifdr_dataset import IFDRInterventionTransform, SharedEpoch
        from ifdr_yolo.data.interventions.sampler import SamplingPolicy

        labels = self._calibration_labels()
        # Simulate a legal identity-preserving letterbox/resize: normalized
        # coordinates are changed by the geometry transform, while the raw
        # object index and class remain stable.
        labels["instances"] = Instances(
            bboxes=np.array([[0.15, 0.20, 0.20, 0.20]], dtype=np.float32),
            bbox_format="xywh", normalized=True,
        )
        transform = IFDRInterventionTransform(
            base_seed=17,
            epoch_state=SharedEpoch(0),
            enabled=True,
            calibration_enabled=True,
            policy=SamplingPolicy(identity_probability=0.0, sampling_probability=1.0, visibility_probability=0.0),
        )
        result = transform(labels)
        self.assertTrue(np.allclose(result["ifdr_specificity_pairs"][0].target_box_xyxy_normalized, (0.05, 0.1, 0.25, 0.3)))

    def test_calibration_transform_policy_rejects_unknown_geometry(self):
        from ultralytics.data.augment import Compose
        from ultralytics.data.dataset import YOLODataset
        from ifdr_yolo.data.ifdr_dataset import IFDRYOLODataset

        class UnknownGeometry:
            pass

        dataset = object.__new__(IFDRYOLODataset)
        dataset.calibration_enabled = True
        with patch.object(YOLODataset, "build_transforms", return_value=Compose([UnknownGeometry()])):
            with self.assertRaisesRegex(ValueError, "unknown calibration transform"):
                dataset.build_transforms(None)

    def test_calibration_dataset_disables_ultralytics_augment(self):
        from ifdr_yolo.data.ifdr_dataset import build_ifdr_dataset

        cfg = SimpleNamespace(
            fraction=1.0, imgsz=64, rect=False, cache=False,
            single_cls=False, task="detect", classes=None,
        )
        with patch("ifdr_yolo.data.ifdr_dataset.IFDRYOLODataset") as dataset_ctor:
            build_ifdr_dataset(
                cfg, "images", 1, {}, mode="train", rect=False, stride=32,
                intervention_seed=17, interventions_enabled=True,
                calibration_enabled=True,
            )
            self.assertFalse(dataset_ctor.call_args.kwargs["augment"])
            dataset_ctor.reset_mock()
            build_ifdr_dataset(
                cfg, "images", 1, {}, mode="train", rect=False, stride=32,
                intervention_seed=17, interventions_enabled=True,
                calibration_enabled=False,
            )
            self.assertTrue(dataset_ctor.call_args.kwargs["augment"])

    def test_metadata_binding_uses_zero_point_nine_nine_iou(self):
        from ultralytics.utils.instance import Instances
        from ultralytics.data.dataset import YOLODataset
        from ifdr_yolo.data.ifdr_dataset import IFDRYOLODataset, SpecificityRejectionCounter
        from ifdr_yolo.data.metadata_index import FactorMetadataIndex, FactorObjectRecord

        record = FactorObjectRecord(
            image_id="000123", object_id="000123:000000", class_id=0,
            class_name="Car", bbox_xyxy=(19.2, 12.8, 28.8, 19.2), height=6.4,
            depth_m=20.0, occlusion=0, truncation=0.0, sampling=0.2,
            visibility=0.3, joint=0.44, sampling_valid=True, visibility_valid=True,
        )
        metadata = FactorMetadataIndex(
            by_image={"000123": (record,)}, source_sha256="a" * 64,
            split_sha256="b" * 64, label_source_sha256="c" * 64, sha256="d" * 64,
        )
        dataset = object.__new__(IFDRYOLODataset)
        dataset.calibration_enabled = True
        dataset.metadata_index = metadata
        dataset.specificity_rejection_counter = SpecificityRejectionCounter()
        labels = {
            "img": np.zeros((32, 48, 3), dtype=np.uint8),
            "ori_shape": (32, 48),
            "im_file": "/dataset/images/train/000123.png",
            "cls": np.array([[0.0]], dtype=np.float32),
            "instances": Instances(
                bboxes=np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
                bbox_format="xywh", normalized=True,
            ),
        }
        with patch.object(YOLODataset, "get_image_and_label", return_value=labels):
            bound = dataset.get_image_and_label(0)
        self.assertEqual(bound["ifdr_raw_label_indices"], (0,))
        self.assertEqual(bound["_ifdr_metadata_records"][0]["object_id"], "000123:000000")

    def test_metadata_binding_uses_original_shape_when_image_is_resized(self):
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.utils.instance import Instances
        from ifdr_yolo.data.ifdr_dataset import IFDRYOLODataset, SpecificityRejectionCounter
        from ifdr_yolo.data.metadata_index import FactorMetadataIndex, FactorObjectRecord

        original_shape = (375, 1242)
        raw_box = (685.93, 173.71, 728.07, 195.98)
        record = FactorObjectRecord(
            image_id="002190", object_id="002190:000000", class_id=0,
            class_name="Car", bbox_xyxy=raw_box, height=22.27,
            depth_m=20.0, occlusion=0, truncation=0.0, sampling=0.2,
            visibility=0.3, joint=0.44, sampling_valid=True, visibility_valid=True,
        )
        metadata = FactorMetadataIndex(
            by_image={"002190": (record,)}, source_sha256="a" * 64,
            split_sha256="b" * 64, label_source_sha256="c" * 64, sha256="d" * 64,
        )
        dataset = object.__new__(IFDRYOLODataset)
        dataset.calibration_enabled = True
        dataset.metadata_index = metadata
        dataset.specificity_rejection_counter = SpecificityRejectionCounter()
        normalized_box = np.array(
            [[raw_box[0] / original_shape[1], raw_box[1] / original_shape[0],
              raw_box[2] / original_shape[1], raw_box[3] / original_shape[0]]],
            dtype=np.float32,
        )
        labels = {
            "img": np.zeros((64, 256, 3), dtype=np.uint8),
            "ori_shape": original_shape,
            "im_file": "/dataset/images/train/002190.png",
            "cls": np.array([[0.0]], dtype=np.float32),
            "instances": Instances(
                bboxes=normalized_box, bbox_format="xyxy", normalized=True,
            ),
        }
        with patch.object(YOLODataset, "get_image_and_label", return_value=labels):
            bound = dataset.get_image_and_label(0)
        self.assertEqual(bound["ifdr_raw_label_indices"], (0,))
        self.assertTrue(np.allclose(
            bound["_ifdr_metadata_records"][0]["raw_box_xyxy_normalized"],
            normalized_box[0], atol=1e-6,
        ))

    def test_metadata_binding_rejects_missing_or_invalid_original_shape(self):
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.utils.instance import Instances
        from ifdr_yolo.data.ifdr_dataset import IFDRYOLODataset, SpecificityRejectionCounter
        from ifdr_yolo.data.metadata_index import FactorMetadataIndex, FactorObjectRecord

        record = FactorObjectRecord(
            image_id="002190", object_id="002190:000000", class_id=0,
            class_name="Car", bbox_xyxy=(10.0, 10.0, 20.0, 20.0), height=10.0,
            depth_m=20.0, occlusion=0, truncation=0.0, sampling=0.2,
            visibility=0.3, joint=0.44, sampling_valid=True, visibility_valid=True,
        )
        metadata = FactorMetadataIndex(
            by_image={"002190": (record,)}, source_sha256="a" * 64,
            split_sha256="b" * 64, label_source_sha256="c" * 64, sha256="d" * 64,
        )
        for invalid_shape in (None, (375,), (0, 1242), (375, "1242")):
            with self.subTest(invalid_shape=invalid_shape):
                dataset = object.__new__(IFDRYOLODataset)
                dataset.calibration_enabled = True
                dataset.metadata_index = metadata
                dataset.specificity_rejection_counter = SpecificityRejectionCounter()
                labels = {
                    "img": np.zeros((64, 256, 3), dtype=np.uint8),
                    "ori_shape": invalid_shape,
                    "im_file": "/dataset/images/train/002190.png",
                    "cls": np.array([[0.0]], dtype=np.float32),
                    "instances": Instances(
                        bboxes=np.array([[0.1, 0.1, 0.2, 0.2]], dtype=np.float32),
                        bbox_format="xyxy", normalized=True,
                    ),
                }
                with patch.object(YOLODataset, "get_image_and_label", return_value=labels):
                    with self.assertRaisesRegex(ValueError, "ori_shape"):
                        dataset.get_image_and_label(0)
                self.assertEqual(dataset.specificity_rejection_counter["missing_metadata"], 1)



if __name__ == "__main__":
    unittest.main()
