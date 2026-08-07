from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

from ifdr_yolo.experiments.ultralytics_runtime import bootstrap_ultralytics_config


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/kitti-p2-m.yaml"
bootstrap_ultralytics_config(ROOT)


class FactorRepairSmokeTests(unittest.TestCase):
    """CPU-only integration gate for the registered F0/F3 calibration path."""

    @staticmethod
    def _synthetic_kitti_fixture(root: Path) -> tuple[torch.Tensor, dict[str, object]]:
        """Write two tiny KITTI images/labels and build a calibration batch."""

        image_dir = root / "images"
        label_dir = root / "labels"
        image_dir.mkdir()
        label_dir.mkdir()
        image_ids = ("000001", "000002")
        classes = ("Car", "Pedestrian", "Cyclist")
        arrays: list[np.ndarray] = []
        for image_index, image_id in enumerate(image_ids):
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[4:28, 4:28] = (40 + image_index, 80, 120)
            image[32:52, 8:24] = (80, 40 + image_index, 120)
            image[28:58, 36:58] = (120, 80, 40 + image_index)
            Image.fromarray(image, mode="RGB").save(image_dir / f"{image_id}.png")
            (label_dir / f"{image_id}.txt").write_text(
                "0 0.25 0.25 0.375 0.375\n"
                "1 0.25 0.65625 0.25 0.3125\n"
                "2 0.734375 0.671875 0.34375 0.46875\n",
                encoding="utf-8",
            )
            arrays.append(np.asarray(Image.open(image_dir / f"{image_id}.png")))

        # Keep the fixture visibly KITTI-like: every image has all three train classes.
        labels = tuple(
            tuple(int(line.split()[0]) for line in (label_dir / f"{image_id}.txt").read_text(encoding="utf-8").splitlines())
            for image_id in image_ids
        )
        if any(item != (0, 1, 2) for item in labels):
            raise AssertionError("synthetic KITTI fixture must contain Car/Pedestrian/Cyclist")

        clean = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float() / 255.0
        target = clean.clone()
        target[:, :, 4:28, 4:28] = (target[:, :, 4:28, 4:28] + 0.15).clamp_max(1.0)
        background = clean.clone()
        background[:, :, 32:58, 36:60] = (background[:, :, 32:58, 36:60] + 0.10).clamp_max(1.0)

        from ifdr_yolo.data.ifdr_dataset import (
            BACKGROUND_IMAGE_KEY,
            CLEAN_IMAGE_KEY,
            FACTOR_OBJECT_TARGETS_KEY,
            FACTOR_TARGET_KEY,
            FACTOR_WEIGHT_KEY,
            SPECIFICITY_PAIRS_KEY,
            TARGET_IMAGE_KEY,
            SpecificityPair,
        )
        from ifdr_yolo.losses.factor_alignment import ObjectFactorTarget

        boxes = (
            (0.0625, 0.0625, 0.4375, 0.4375),
            (0.125, 0.5, 0.375, 0.8125),
            (0.5625, 0.4375, 0.90625, 0.90625),
        )
        natural_targets = tuple(
            ObjectFactorTarget(
                batch_index=image_index,
                class_id=class_id,
                box_xyxy_normalized=box,
                target=(0.45 + 0.05 * class_id, 0.25 + 0.05 * class_id),
                valid=(True, True),
            )
            for image_index in range(2)
            for class_id, box in enumerate(boxes)
        )
        specificity_pairs = tuple(
            SpecificityPair(
                target_index=0,
                target_box_xyxy_normalized=boxes[0],
                background_box_xyxy_normalized=boxes[2],
                factor_kind="sampling",
                factor_channel=0,
                severity=0.75,
                transform_seed=20260807 + image_index,
                weight=0.75,
                background_max_iou=0.0,
                batch_index=image_index,
            )
            for image_index in range(2)
        )
        batch = {
            CLEAN_IMAGE_KEY: clean,
            TARGET_IMAGE_KEY: target,
            BACKGROUND_IMAGE_KEY: background,
            FACTOR_TARGET_KEY: torch.full((2, 2, 8, 8), 0.65),
            FACTOR_WEIGHT_KEY: torch.ones((2, 2, 8, 8)),
            FACTOR_OBJECT_TARGETS_KEY: natural_targets,
            SPECIFICITY_PAIRS_KEY: specificity_pairs,
        }
        return clean, batch

    def test_cpu_batch(self) -> None:
        """Run real F0/F3 forwards/backwards and enforce the calibration contract."""

        from ifdr_yolo.experiments.factor_repair import (
            run_calibration_validation,
            semantic_calibration_phase,
            semantic_state_sha256,
        )
        from ifdr_yolo.data.ifdr_dataset import (
            CLEAN_IMAGE_KEY,
            FACTOR_OBJECT_TARGETS_KEY,
            SPECIFICITY_PAIRS_KEY,
        )

        with tempfile.TemporaryDirectory() as directory:
            validation_input, batch = self._synthetic_kitti_fixture(Path(directory))
            expected_masks = {
                "F0": {"synthetic": 1.0, "natural": 0.0, "specificity": 0.0},
                "F3": {"synthetic": 1.0, "natural": 1.0, "specificity": 1.0},
            }
            self.assertEqual(len(batch[FACTOR_OBJECT_TARGETS_KEY]), 6)
            self.assertEqual(len(batch[SPECIFICITY_PAIRS_KEY]), 2)

            for variant, mask in expected_masks.items():
                torch.manual_seed(20260807)
                from ifdr_yolo.models.ifdr_model import IFDRDetectionModel

                model = IFDRDetectionModel(str(MODEL_PATH), verbose=False)
                # The Task5/6 trainer attaches this namespace before creating its criterion;
                # the direct CPU gate supplies the same minimal runtime boundary.
                model.args = SimpleNamespace()
                model.train()
                optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
                phase = semantic_calibration_phase(
                    model,
                    variant=variant,
                    epochs=30,
                    optimizer=optimizer,
                )
                self.assertEqual(dict(phase.loss_mask), mask)
                before_validation_hash = semantic_state_sha256(model)
                self.assertEqual(len(optimizer.state), 0)

                model.zero_grad(set_to_none=True)
                loss, components = model.loss(batch)
                self.assertIsInstance(loss, torch.Tensor)
                self.assertTrue(torch.isfinite(loss).all())
                self.assertEqual(tuple(components.shape), (3,))
                self.assertTrue(torch.isfinite(components).all())

                names = (
                    "synthetic_factor_loss",
                    "natural_factor_loss",
                    "specificity_loss",
                )
                component_mask = {
                    "synthetic_factor_loss": mask["synthetic"],
                    "natural_factor_loss": mask["natural"],
                    "specificity_loss": mask["specificity"],
                }
                for component_name, component in zip(names, components):
                    if component_mask[component_name] == 0.0:
                        self.assertEqual(
                            float(component),
                            0.0,
                            msg=f"{variant} must mask {component_name}",
                        )
                expected_total = (
                    float(components[0]) * component_mask["synthetic_factor_loss"]
                    + float(components[1]) * component_mask["natural_factor_loss"]
                    + 0.5 * float(components[2]) * component_mask["specificity_loss"]
                )
                expected_total *= 2.0
                self.assertAlmostEqual(float(loss.detach()), expected_total, places=5)

                loss.backward()
                parameters = dict(model.named_parameters())
                semantic_names = set(phase.trainable_parameter_names)
                for name, parameter in parameters.items():
                    if name in semantic_names:
                        self.assertIsNotNone(parameter.grad, msg=name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), msg=name)
                    else:
                        self.assertIsNone(parameter.grad, msg=f"task path leaked gradient: {name}")

                # Validation is a no-grad forward and must not mutate optimizer or semantic bytes.
                state_before = deepcopy(optimizer.state_dict())
                run_calibration_validation(model, validation_input, optimizer=optimizer)
                self.assertEqual(semantic_state_sha256(model), before_validation_hash)
                self.assertEqual(optimizer.state_dict(), state_before)


if __name__ == "__main__":
    unittest.main()
