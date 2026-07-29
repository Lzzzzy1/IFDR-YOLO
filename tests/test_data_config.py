from pathlib import Path
import unittest

import yaml


class DataConfigTest(unittest.TestCase):
    def test_kitti_v2_config_matches_export_layout(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (root / "configs/data/kitti_v2.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(config["path"], "data/processed/kitti_yolo_v2")
        self.assertEqual(config["train"], "images/train")
        self.assertEqual(config["val"], "images/val")
        self.assertEqual(
            config["names"],
            {0: "Car", 1: "Pedestrian", 2: "Cyclist"},
        )


if __name__ == "__main__":
    unittest.main()
