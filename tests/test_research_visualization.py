import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ifdr_yolo.eval.research_visualization import (
    discover_canonical_runs,
    generate_research_figures,
    summarize_gradient_diagnostics,
)


def _write_run(
    root: Path,
    name: str,
    *,
    epochs: int,
    moderate: tuple[float, float, float],
    gradient_rows: list[dict] | None = None,
) -> Path:
    run = root / name
    run.mkdir(parents=True)
    with (run / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train/box_loss", "metrics/mAP50-95(B)"],
        )
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            writer.writerow(
                {
                    "epoch": epoch,
                    "train/box_loss": 1.0 / epoch,
                    "metrics/mAP50-95(B)": 0.2 + epoch / 100.0,
                }
            )
    classes = {}
    for class_name, value in zip(
        ("Car", "Pedestrian", "Cyclist"),
        moderate,
        strict=True,
    ):
        classes[class_name] = {
            difficulty: {"ap40": value + offset}
            for difficulty, offset in (("easy", 5.0), ("moderate", 0.0), ("hard", -5.0))
        }
    (run / "metrics_ap40.json").write_text(
        json.dumps({"classes": classes}),
        encoding="utf-8",
    )
    if gradient_rows is not None:
        (run / "gradient_diagnostics.jsonl").write_text(
            "\n".join(json.dumps(row) for row in gradient_rows) + "\n",
            encoding="utf-8",
        )
    return run


class ResearchVisualizationTest(unittest.TestCase):
    def test_cli_can_run_directly_from_project_root(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "generate_research_visualizations.py"),
                "--help",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_discovery_keeps_longest_run_for_same_experiment_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(
                root,
                "20260101T000000Z-kitti-yolov8m-baseline-s17-aaaaaaa",
                epochs=1,
                moderate=(90.0, 50.0, 30.0),
            )
            full = _write_run(
                root,
                "20260101T010000Z-kitti-yolov8m-baseline-s17-aaaaaaa",
                epochs=3,
                moderate=(91.0, 51.0, 31.0),
            )

            records = discover_canonical_runs(root)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].directory, full)
            self.assertEqual(records[0].experiment, "baseline")
            self.assertEqual(records[0].seed, 17)
            self.assertEqual(records[0].max_epoch, 3)
            self.assertAlmostEqual(records[0].moderate_mean, 57.6666667)

    def test_gradient_summary_reports_valid_conflicts_and_missing_detection(self) -> None:
        rows = [
            {
                "step": 10,
                "gradient_norms": {"detection": 0.0, "factor": 1.0, "counterfactual": 1.0},
                "pairs": {
                    "counterfactual::factor": {"cosine": -0.5, "conflict": True},
                    "detection::factor": {"cosine": None, "conflict": False},
                },
            },
            {
                "step": 20,
                "gradient_norms": {"detection": 0.0, "factor": 1.0, "counterfactual": 1.0},
                "pairs": {
                    "counterfactual::factor": {"cosine": 0.25, "conflict": False},
                    "detection::factor": {"cosine": None, "conflict": False},
                },
            },
        ]

        summary = summarize_gradient_diagnostics(rows)

        pair = summary["pairs"]["counterfactual::factor"]
        self.assertEqual(pair["valid"], 2)
        self.assertEqual(pair["conflicts"], 1)
        self.assertAlmostEqual(pair["conflict_rate"], 0.5)
        self.assertTrue(summary["detection_gradient_missing"])

    def test_gradient_summary_verifies_protected_group_routing(self) -> None:
        rows = [
            {
                "schema_version": 2,
                "step": 10,
                "parameter_groups": {
                    "semantic_anchor": {
                        "gradient_norms": {
                            "detection": 0.0,
                            "factor": 1.0,
                            "counterfactual": 1.0,
                        },
                        "pairs": {
                            "counterfactual::factor": {
                                "cosine": -0.5,
                                "conflict": True,
                            },
                        },
                    },
                    "fusion_adapters": {
                        "gradient_norms": {
                            "detection": 2.0,
                            "factor": 0.5,
                            "counterfactual": 0.25,
                        },
                        "pairs": {
                            "detection::factor": {
                                "cosine": -0.25,
                                "conflict": True,
                            },
                        },
                    },
                    "localization_adapter": {
                        "gradient_norms": {
                            "detection": 0.75,
                            "factor": 0.0,
                            "counterfactual": 0.0,
                        },
                        "pairs": {
                            "detection::factor": {
                                "cosine": None,
                                "conflict": False,
                            },
                        },
                    },
                },
            }
        ]

        summary = summarize_gradient_diagnostics(rows)

        self.assertTrue(summary["semantic_anchor_detection_blocked"])
        self.assertTrue(summary["protection_path_verified"])
        fusion_pair = summary["parameter_groups"]["fusion_adapters"][
            "pairs"
        ]["detection::factor"]
        self.assertEqual(fusion_pair["valid"], 1)
        self.assertEqual(fusion_pair["conflicts"], 1)

    def test_end_to_end_generation_writes_publication_pngs_and_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            output = Path(temporary) / "figures"
            for experiment in ("baseline", "p2", "ifdr-fusion-only"):
                for seed in (17, 29, 41):
                    _write_run(
                        root,
                        f"20260101T010000Z-kitti-yolov8m-{experiment}-s{seed}-aaaaaaa",
                        epochs=3,
                        moderate=(90.0, 50.0 + seed / 10.0, 30.0),
                    )
            gradient_rows = [
                {
                    "step": 10,
                    "gradient_norms": {"detection": 0.0, "factor": 1.0, "counterfactual": 1.0},
                    "pairs": {
                        "counterfactual::factor": {"cosine": -0.5, "conflict": True},
                        "detection::factor": {"cosine": None, "conflict": False},
                    },
                }
            ]
            _write_run(
                root,
                "20260102T010000Z-kitti-yolov8m-ifdr-protected-counterfactual-joint-e90-s17-bbbbbbb",
                epochs=3,
                moderate=(91.0, 52.0, 32.0),
                gradient_rows=gradient_rows,
            )

            generated = generate_research_figures(root, output)

            expected = {
                "01_method_ap40_overview.png",
                "02_classwise_moderate_ap40.png",
                "03_multiseed_stability.png",
                "04_mechanism_ablation.png",
                "05_training_curves.png",
                "06_auxiliary_gradient_conflict.png",
                "07_research_evidence_dashboard.png",
                "summary_metrics.csv",
                "gradient_summary.json",
                "visualization_manifest.json",
            }
            self.assertTrue(expected.issubset({path.name for path in generated}))
            for name in expected:
                self.assertGreater((output / name).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
