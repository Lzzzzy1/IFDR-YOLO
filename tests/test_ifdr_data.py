from pathlib import Path
import unittest

import numpy as np
import torch
from unittest.mock import patch

from ifdr_yolo.experiments.ultralytics_runtime import (
    bootstrap_ultralytics_config,
)


ROOT = Path(__file__).resolve().parents[1]
bootstrap_ultralytics_config(ROOT)


class IFDRInterventionTransformTest(unittest.TestCase):
    def _labels(self) -> dict:
        from ultralytics.utils.instance import Instances

        return {
            "img": np.full((64, 96, 3), 127, dtype=np.uint8),
            "im_file": "/dataset/images/train/000123.png",
            "cls": np.array([[0.0], [1.0]], dtype=np.float32),
            "instances": Instances(
                bboxes=np.array(
                    [[0.25, 0.35, 0.20, 0.30], [0.75, 0.65, 0.12, 0.20]],
                    dtype=np.float32,
                ),
                bbox_format="xywh",
                normalized=True,
            ),
        }

    def test_is_reproducible_within_epoch_and_preserves_instances(self) -> None:
        from ifdr_yolo.data.ifdr_dataset import (
            IFDRInterventionTransform,
            SharedEpoch,
        )

        epoch = SharedEpoch(3)
        transform = IFDRInterventionTransform(
            base_seed=17,
            epoch_state=epoch,
            enabled=True,
        )
        first_labels = self._labels()
        second_labels = self._labels()
        original_boxes = first_labels["instances"].bboxes.copy()

        first = transform(first_labels)
        second = transform(second_labels)

        self.assertTrue(torch.equal(first["ifdr_factor_target"], second["ifdr_factor_target"]))
        self.assertTrue(torch.equal(first["ifdr_factor_weight"], second["ifdr_factor_weight"]))
        self.assertTrue(np.array_equal(first["img"], second["img"]))
        self.assertTrue(np.array_equal(first["instances"].bboxes, original_boxes))
        self.assertEqual(first["ifdr_spec"], second["ifdr_spec"])

    def test_epoch_changes_deterministic_intervention(self) -> None:
        from ifdr_yolo.data.ifdr_dataset import (
            IFDRInterventionTransform,
            SharedEpoch,
        )

        epoch = SharedEpoch(0)
        transform = IFDRInterventionTransform(
            base_seed=17,
            epoch_state=epoch,
            enabled=True,
        )
        first = transform(self._labels())
        epoch.set(1)
        second = transform(self._labels())

        self.assertNotEqual(first["ifdr_spec"], second["ifdr_spec"])

    def test_emits_dense_two_factor_supervision(self) -> None:
        from ifdr_yolo.data.ifdr_dataset import (
            IFDRInterventionTransform,
            SharedEpoch,
        )

        transform = IFDRInterventionTransform(
            base_seed=23,
            epoch_state=SharedEpoch(2),
            enabled=True,
        )

        result = transform(self._labels())

        self.assertEqual(result["ifdr_factor_target"].shape, (2, 64, 96))
        self.assertEqual(result["ifdr_factor_weight"].shape, (2, 64, 96))
        self.assertEqual(result["ifdr_factor_target"].dtype, torch.float32)
        self.assertTrue((result["ifdr_factor_weight"] >= 0).all())
        self.assertGreater(float(result["ifdr_factor_weight"].sum()), 0.0)

    def test_disabled_validation_path_is_image_exact_and_unsupervised(self) -> None:
        from ifdr_yolo.data.ifdr_dataset import (
            IFDRInterventionTransform,
            SharedEpoch,
        )

        transform = IFDRInterventionTransform(
            base_seed=31,
            epoch_state=SharedEpoch(),
            enabled=False,
        )
        labels = self._labels()
        original = labels["img"].copy()

        result = transform(labels)

        self.assertTrue(np.array_equal(result["img"], original))
        self.assertEqual(float(result["ifdr_factor_weight"].sum()), 0.0)
        self.assertEqual(result["ifdr_spec"], "disabled")

    def test_sampling_intervention_emits_clean_counterfactual_delta_pair(
        self,
    ) -> None:
        from ifdr_yolo.data.ifdr_dataset import (
            COUNTERFACTUAL_DELTA_KEY,
            COUNTERFACTUAL_IMAGE_KEY,
            COUNTERFACTUAL_WEIGHT_KEY,
            IFDRInterventionTransform,
            SharedEpoch,
        )
        from ifdr_yolo.data.interventions.sampler import SamplingPolicy

        labels = self._labels()
        clean = labels["img"].copy()
        transform = IFDRInterventionTransform(
            base_seed=17,
            epoch_state=SharedEpoch(3),
            enabled=True,
            policy=SamplingPolicy(
                identity_probability=0.0,
                sampling_probability=1.0,
                visibility_probability=0.0,
                minimum_strength=0.5,
                maximum_strength=0.5,
            ),
        )

        result = transform(labels)

        expected_clean = torch.from_numpy(
            np.ascontiguousarray(clean[:, :, ::-1].transpose(2, 0, 1))
        )
        self.assertTrue(
            torch.equal(result[COUNTERFACTUAL_IMAGE_KEY], expected_clean)
        )
        delta = result[COUNTERFACTUAL_DELTA_KEY]
        weight = result[COUNTERFACTUAL_WEIGHT_KEY]
        self.assertEqual(delta.shape, (2, 64, 96))
        self.assertEqual(weight.shape, (2, 64, 96))
        support = weight[0] > 0
        self.assertTrue(support.any())
        self.assertGreater(float(delta[0][support].max()), 0.0)
        self.assertEqual(float(delta[1][support].abs().max()), 0.0)
        self.assertTrue(torch.equal(weight[0], weight[1]))

    def test_disabled_path_emits_zero_counterfactual_weight(self) -> None:
        from ifdr_yolo.data.ifdr_dataset import (
            COUNTERFACTUAL_IMAGE_KEY,
            COUNTERFACTUAL_WEIGHT_KEY,
            IFDRInterventionTransform,
            SharedEpoch,
        )

        labels = self._labels()
        transform = IFDRInterventionTransform(
            base_seed=17,
            epoch_state=SharedEpoch(),
            enabled=False,
        )

        result = transform(labels)

        self.assertEqual(
            tuple(result[COUNTERFACTUAL_IMAGE_KEY].shape),
            (3, 64, 96),
        )
        self.assertEqual(
            float(result[COUNTERFACTUAL_WEIGHT_KEY].sum()),
            0.0,
        )

    def test_shared_epoch_rejects_invalid_values(self) -> None:
        from ifdr_yolo.data.ifdr_dataset import SharedEpoch

        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                SharedEpoch(value)
        epoch = SharedEpoch()
        with self.assertRaises(ValueError):
            epoch.set(-1)


class IFDRCollateTest(unittest.TestCase):
    def test_stacks_dense_factor_maps(self) -> None:
        from ifdr_yolo.data.ifdr_dataset import (
            COUNTERFACTUAL_DELTA_KEY,
            COUNTERFACTUAL_IMAGE_KEY,
            COUNTERFACTUAL_WEIGHT_KEY,
            collate_ifdr_batch,
        )

        samples = []
        for value in (0.0, 1.0):
            samples.append(
                {
                    "img": torch.zeros(3, 8, 8, dtype=torch.uint8),
                    "bboxes": torch.zeros(0, 4),
                    "cls": torch.zeros(0, 1),
                    "batch_idx": torch.zeros(0),
                    "ifdr_factor_target": torch.full((2, 8, 8), value),
                    "ifdr_factor_weight": torch.ones(2, 8, 8),
                    COUNTERFACTUAL_IMAGE_KEY: torch.full(
                        (3, 8, 8),
                        int(value),
                        dtype=torch.uint8,
                    ),
                    COUNTERFACTUAL_DELTA_KEY: torch.full(
                        (2, 8, 8),
                        value,
                    ),
                    COUNTERFACTUAL_WEIGHT_KEY: torch.ones(2, 8, 8),
                    "ifdr_spec": str(value),
                }
            )

        batch = collate_ifdr_batch(samples)

        self.assertEqual(batch["ifdr_factor_target"].shape, (2, 2, 8, 8))
        self.assertEqual(batch["ifdr_factor_weight"].shape, (2, 2, 8, 8))
        self.assertEqual(batch[COUNTERFACTUAL_IMAGE_KEY].shape, (2, 3, 8, 8))
        self.assertEqual(batch[COUNTERFACTUAL_DELTA_KEY].shape, (2, 2, 8, 8))
        self.assertEqual(batch[COUNTERFACTUAL_WEIGHT_KEY].shape, (2, 2, 8, 8))
        self.assertEqual(batch["ifdr_spec"], ("0.0", "1.0"))


class IFDRYOLODatasetTest(unittest.TestCase):
    def test_inserts_intervention_immediately_before_format(self) -> None:
        from ultralytics.data.augment import Compose
        from ultralytics.data.dataset import YOLODataset

        from ifdr_yolo.data.ifdr_dataset import IFDRYOLODataset

        format_transform = object()
        intervention = object()
        dataset = object.__new__(IFDRYOLODataset)
        dataset.intervention_transform = intervention

        with patch.object(
            YOLODataset,
            "build_transforms",
            return_value=Compose([format_transform]),
        ):
            transforms = dataset.build_transforms(None)

        self.assertEqual(
            transforms.tolist(),
            [intervention, format_transform],
        )

    def test_set_epoch_updates_shared_worker_clock(self) -> None:
        from ifdr_yolo.data.ifdr_dataset import IFDRYOLODataset, SharedEpoch

        dataset = object.__new__(IFDRYOLODataset)
        dataset.epoch_state = SharedEpoch()

        dataset.set_epoch(12)

        self.assertEqual(dataset.epoch_state.get(), 12)


if __name__ == "__main__":
    unittest.main()
