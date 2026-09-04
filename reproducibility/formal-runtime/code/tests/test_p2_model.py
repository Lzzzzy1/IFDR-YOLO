from pathlib import Path
import unittest

import torch

from ifdr_yolo.data.splits import sha256_file
from ifdr_yolo.experiments.ultralytics_runtime import (
    bootstrap_ultralytics_config,
)
from ifdr_yolo.models.p2 import inspect_p2_model


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/kitti-p2-m.yaml"
MODEL_SHA256 = (
    "0d2cbd9215dc62c4c9920127964ce0ab61b5d10581861d8e83c33fc168aa302a"
)


class P2ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bootstrap_ultralytics_config(ROOT)
        from ultralytics.nn.tasks import DetectionModel

        cls.model = DetectionModel(str(MODEL_PATH), verbose=False)

    def test_project_model_hash_and_structure_are_fixed(self) -> None:
        self.assertEqual(sha256_file(MODEL_PATH), MODEL_SHA256)
        summary = inspect_p2_model(self.model)
        self.assertEqual(summary["strides"], [4.0, 8.0, 16.0, 32.0])
        self.assertEqual(summary["detect_inputs"], 4)
        self.assertEqual(summary["parameters"], 25_052_620)
        self.assertEqual(summary["state_items"], 581)

    def test_training_forward_produces_four_spatial_scales(self) -> None:
        self.model.train()
        with torch.no_grad():
            outputs = self.model(torch.zeros(1, 3, 320, 320))
        self.assertEqual(set(outputs), {"boxes", "scores", "feats"})
        features = outputs["feats"]
        self.assertEqual(len(features), 4)
        self.assertEqual(
            [tuple(feature.shape[-2:]) for feature in features],
            [(80, 80), (40, 40), (20, 20), (10, 10)],
        )


if __name__ == "__main__":
    unittest.main()
