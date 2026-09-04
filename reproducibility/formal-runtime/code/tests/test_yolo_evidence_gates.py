from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.yolo_evidence_gates import (
    build_data_use_ledger,
    build_denominator_audit,
    build_independent_acceptance,
    build_official_reconciliation,
    build_protocol_matrix,
    build_storage_preflight,
)


class EvidenceGateTests(unittest.TestCase):
    def test_direct_cli_keeps_repo_imports_for_real_positive_image_counts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts" / "yolo_evidence_gates.py"),
                    "--config",
                    str(root / "configs" / "experiments" / "yolo_evidence_gates_20260811.json"),
                    "--output-dir",
                    str(output),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((output / "evidence_gates.json").read_text(encoding="utf-8"))
        first = report["gates"]["denominator_saturation"]["results"][0]
        self.assertEqual(first["classes"]["Pedestrian"]["moderate"]["positive_images"], 91)
        self.assertEqual(first["classes"]["Cyclist"]["moderate"]["positive_images"], 40)

    def test_ledger_permanently_marks_registered_development_and_blocks_confirmation(self) -> None:
        ledger = build_data_use_ledger(
            {
                "fit": {"count": 3341, "path": "fit_ids.txt", "role": "training"},
                "development": {"count": 371, "path": "development_ids.txt", "role": "development_route_selection"},
                "historical_exposed": {"count": 3769, "path": "kitti_val.txt", "role": "development_exposed"},
                "confirmation": None,
                "test": {"count": 7518, "role": "official_kitti_test_hidden_labels", "status": "BLOCKED"},
            }
        )
        self.assertEqual(ledger["status"], "BLOCKED")
        self.assertEqual(ledger["sets"]["development"]["role"], "development_route_selection_permanent")
        self.assertEqual(ledger["independent_confirmation"]["status"], "NONE")
        self.assertNotEqual(ledger["sets"]["development"]["role"], "confirmation")

    def test_denominator_audit_reports_gt_positive_images_counts_strata_and_ci(self) -> None:
        evidence = {
            "source_path": "diagnostics.json",
            "source_sha256": "a" * 64,
            "observed": {
                "moderate": {
                    "candidate": {
                        "Pedestrian": {"ap40": 93.0, "valid_gt": 205, "tp": 198, "fp": 345, "fn": 7},
                        "Cyclist": {"ap40": 97.0, "valid_gt": 55, "tp": 54, "fp": 63, "fn": 1},
                        "macro": 95.0,
                    }
                },
                "strata": {
                    "height": {"small_25_40": {"counts": {"candidate": {"Pedestrian": {"valid_gt": 37}, "Cyclist": {"valid_gt": 8}}}, "candidate": {"macro": 70.0}}},
                    "depth": {"far_gt_40m": {"counts": {"candidate": {"Pedestrian": {"valid_gt": 10}, "Cyclist": {"valid_gt": 2}}}, "candidate": {"macro": 60.0}}},
                },
            },
            "positive_images": {"Pedestrian": 200, "Cyclist": 50},
            "bootstrap": {
                "moderate": {
                    "macro": {
                        "delta": {"ci_lower": -1.0, "ci_upper": 2.0}
                    }
                }
            },
        }
        audit = build_denominator_audit([evidence])
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["results"][0]["classes"]["Cyclist"]["moderate"]["valid_gt"], 55)
        self.assertEqual(audit["results"][0]["classes"]["Cyclist"]["moderate"]["positive_images"], 50)
        self.assertIn("small_25_40", audit["results"][0]["strata"]["height"])
        self.assertTrue(audit["results"][0]["paired_ci"]["available"])
        self.assertEqual(audit["results"][0]["paired_ci"]["macro"], {"lower": -1.0, "upper": 2.0})

    def test_official_reconciliation_is_unresolved_without_runnable_devkit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            zip_path = Path(directory) / "devkit_object.zip"
            zip_path.write_bytes(b"official-devkit")
            reconciliation = build_official_reconciliation(
                {
                    "official_zip": str(zip_path),
                    "official_zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
                    "internal_evaluator": "ifdr_yolo.eval.kitti_ap40.evaluate_class",
                    "prediction_source": "fixed_predictions",
                    "official_tool": {"available": False, "compiler": None, "executable": None},
                }
            )
        self.assertEqual(reconciliation["status"], "UNRESOLVED")
        self.assertEqual(reconciliation["severity"], "P0")
        self.assertEqual(reconciliation["official_zip_sha256"], hashlib.sha256(b"official-devkit").hexdigest())
        self.assertIn("reproduce", reconciliation["reconciliation_entry"])

    def test_storage_preflight_fails_closed_and_retention_disables_periodic_archives(self) -> None:
        config = {
            "checkpoint_bytes": {"last": 100, "best": 100, "periodic": 0},
            "log_peak_bytes": 100,
            "prediction_peak_bytes": 100,
            "mirror_peak_bytes": 100,
            "headroom_bytes": 100,
            "retention": {"save_period": -1, "retain": ["last.pt", "best.pt"]},
        }
        blocked = build_storage_preflight(config, free_bytes=499)
        self.assertEqual(blocked["status"], "BLOCKED")
        passed = build_storage_preflight(config, free_bytes=600)
        self.assertEqual(passed["status"], "PASS")
        self.assertEqual(passed["retention"]["save_period"], -1)
        self.assertEqual(passed["forecast"]["periodic_checkpoint_bytes"], 0)

    def test_protocol_matrix_rejects_p2_vs_c_and_allows_matching_family(self) -> None:
        rows = [
            {"name": "p3p5", "model_role": "plain_p3p5", "data_use_role": "development", "epochs": 300, "imgsz": 640, "seed": 17, "checkpoint_role": "last.pt", "split_sha256": "s", "evaluator": "internal"},
            {"name": "p2", "model_role": "plain_p2", "data_use_role": "development", "epochs": 300, "imgsz": 640, "seed": 17, "checkpoint_role": "last.pt", "split_sha256": "s", "evaluator": "internal"},
            {"name": "c", "model_role": "ifdr_c", "data_use_role": "development", "epochs": 30, "imgsz": 640, "seed": 17, "checkpoint_role": "last.pt", "split_sha256": "s", "evaluator": "internal"},
        ]
        matrix = build_protocol_matrix(rows)
        self.assertEqual(matrix["status"], "PASS")
        self.assertTrue(any("p2" in reason and "c" in reason for reason in matrix["blocked_comparisons"]))
        self.assertIn("p3p5_vs_p2", matrix["allowed_families"])

    def test_acceptance_preserves_negative_zero_positive_and_blocks_unknown_seeds(self) -> None:
        result = build_independent_acceptance(
            {
                "preregistered": {"seeds": [17, 29, 41], "comparisons": ["B-C", "AB-C", "AB-B"], "minimum_meaningful_effect_ap": 1.0},
                "seed_results": {
                    "17": {"B-C": 0.7, "AB-C": 0.5, "AB-B": -0.1},
                    "29": {"B-C": -0.5, "AB-C": None, "AB-B": None},
                    "41": {"B-C": None, "AB-C": None, "AB-B": None},
                },
                "paired_ci": {"B-C": {"lower": None, "upper": None}, "AB-C": {"lower": None, "upper": None}, "AB-B": {"lower": None, "upper": None}},
            }
        )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["retention_rule"]["requires_all_three_seed_deltas_positive"], True)
        self.assertEqual(result["effects"]["17"]["AB-B"], -0.1)
        self.assertEqual(result["frozen_conclusions"]["neither_stable"], "stop_old_module_route")

    def test_acceptance_computes_registered_three_seed_paired_t_interval(self) -> None:
        result = build_independent_acceptance(
            {
                "preregistered": {"seeds": [17, 29, 41], "comparisons": ["B-C"], "minimum_meaningful_effect_ap": 1.0},
                "seed_results": {"17": {"B-C": 2.0}, "29": {"B-C": 2.1}, "41": {"B-C": 2.2}},
            }
        )
        summary = result["summaries"]["B-C"]
        self.assertEqual(summary["paired_ci"]["source"], "paired_seed_t_interval_df2")
        self.assertGreater(summary["paired_ci"]["lower"], 0.0)
        self.assertTrue(summary["stable_positive"])


if __name__ == "__main__":
    unittest.main()
