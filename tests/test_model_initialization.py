from pathlib import Path
import unittest

import torch

from ifdr_yolo.experiments.ultralytics_runtime import (
    bootstrap_ultralytics_config,
)
from ifdr_yolo.models.initialization import (
    apply_semantic_prefix_initialization,
    select_semantic_prefix_state,
)


ROOT = Path(__file__).resolve().parents[1]


class SemanticInitializationTest(unittest.TestCase):
    def test_selector_requires_layer_limit_target_key_and_shape(self) -> None:
        source = {
            "model.0.weight": torch.ones(2, 2),
            "model.15.weight": torch.ones(1),
            "model.16.weight": torch.ones(1),
            "other.weight": torch.ones(1),
        }
        target = {
            "model.0.weight": torch.zeros(2, 2),
            "model.15.weight": torch.zeros(2),
            "model.16.weight": torch.zeros(1),
        }

        selected = select_semantic_prefix_state(
            source,
            target,
            max_layer=15,
        )

        self.assertEqual(tuple(selected), ("model.0.weight",))

    def test_selector_rejects_negative_layer_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_layer"):
            select_semantic_prefix_state({}, {}, max_layer=-1)

    def test_locked_models_transfer_exactly_306_items(self) -> None:
        bootstrap_ultralytics_config(ROOT)
        from ultralytics import YOLO
        from ultralytics.nn.tasks import DetectionModel

        target = DetectionModel(
            str(ROOT / "models/kitti-p2-m.yaml"),
            verbose=False,
        )
        source = YOLO(str(ROOT / "yolov8m.pt")).model

        report = apply_semantic_prefix_initialization(
            target,
            source,
            max_layer=15,
            expected_items=306,
        )

        self.assertEqual(report.transferred_items, 306)
        self.assertEqual(
            max(int(key.split(".")[1]) for key in report.transferred_keys),
            15,
        )
        self.assertEqual(
            sorted(
                {
                    int(key.split(".")[1])
                    for key in report.transferred_keys
                }
            ),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15],
        )


if __name__ == "__main__":
    unittest.main()
