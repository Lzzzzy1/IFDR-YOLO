import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ifdr_yolo.eval.bootstrap_visualization import (
    generate_bootstrap_forest_plots,
)


class BootstrapVisualizationTest(unittest.TestCase):
    def test_generates_one_nonempty_forest_plot_per_class(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "metric": (
                            "KITTI_PAIRED_BOOTSTRAP_CROSS_SEED_SUMMARY"
                        ),
                        "groups": [
                            {
                                "reference": "baseline",
                                "candidate": "p2",
                                "class_name": "Car",
                                "slice_name": "small_25_40",
                                "seed_summary": {
                                    "seed_results": [
                                        {
                                            "seed": 17,
                                            "difference_ap40": 2.0,
                                            "ci_lower": 0.5,
                                            "ci_upper": 3.5,
                                        },
                                        {
                                            "seed": 29,
                                            "difference_ap40": 1.0,
                                            "ci_lower": -0.2,
                                            "ci_upper": 2.2,
                                        },
                                    ]
                                },
                            },
                            {
                                "reference": "p2",
                                "candidate": "fusion",
                                "class_name": "Pedestrian",
                                "slice_name": "far_gt_40m",
                                "seed_summary": {
                                    "seed_results": [
                                        {
                                            "seed": 17,
                                            "difference_ap40": -0.2,
                                            "ci_lower": -0.5,
                                            "ci_upper": 0.1,
                                        }
                                    ]
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            outputs = generate_bootstrap_forest_plots(
                summary_path,
                root / "figures",
            )

            self.assertEqual({path.name for path in outputs}, {
                "car_paired_bootstrap_forest.png",
                "pedestrian_paired_bootstrap_forest.png",
            })
            for output in outputs:
                self.assertGreater(output.stat().st_size, 1000)
                self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
