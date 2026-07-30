from pathlib import Path
import tempfile
import unittest

import yaml

from ifdr_yolo.experiments.ablation import (
    build_component_ablation_payloads,
    write_component_ablation_configs,
)
from ifdr_yolo.experiments.config import load_ifdr_config


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/experiments/kitti_ifdr_yolov8m_s17.yaml"


class IFDRAblationMatrixTest(unittest.TestCase):
    def test_builds_complete_two_by_two_component_factorial(self) -> None:
        base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))

        payloads = build_component_ablation_payloads(base)

        self.assertEqual(
            tuple(payloads),
            ("factor_control", "fusion_only", "dcli_only", "full"),
        )
        signatures = {
            name: (
                payload["ifdr"]["components"]["fusion_gate"],
                payload["ifdr"]["components"]["dcli"],
            )
            for name, payload in payloads.items()
        }
        self.assertEqual(
            signatures,
            {
                "factor_control": (False, False),
                "fusion_only": (True, False),
                "dcli_only": (False, True),
                "full": (True, True),
            },
        )
        self.assertTrue(
            all(
                payload["ifdr"]["components"]["factor_supervision"]
                and payload["ifdr"]["components"]["interventions"]
                for payload in payloads.values()
            )
        )

    def test_written_configs_are_strictly_loadable_and_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_component_ablation_configs(
                base_config=BASE_CONFIG,
                output_dir=Path(directory),
            )

            configs = [
                load_ifdr_config(path, repository_root=ROOT)
                for path in paths.values()
            ]

        self.assertEqual(len(configs), 4)
        self.assertEqual(
            len({config.experiment.variant for config in configs}),
            4,
        )
        self.assertTrue(
            all(
                config.experiment.variant.startswith("ifdr-")
                for config in configs
            )
        )


if __name__ == "__main__":
    unittest.main()
