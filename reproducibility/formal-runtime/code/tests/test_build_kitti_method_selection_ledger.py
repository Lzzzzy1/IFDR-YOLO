import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_kitti_method_selection_ledger import (
    build_ledger,
    publish_ledger,
)


class KittiMethodSelectionLedgerTests(unittest.TestCase):
    def active(self):
        return {
            "fit_ids_sha256": "fit",
            "development_ids_sha256": "dev",
            "seed": 0,
            "epochs": 30,
            "imgsz": 640,
            "batch": 16,
            "workers": 8,
            "amp": True,
            "deterministic": True,
            "checkpoint_role": "last.pt",
            "evaluator": "ifdr_yolo.kitti_ap40",
            "prediction_args": {"conf": 0.001, "iou": 0.7, "max_det": 300},
        }

    def candidate(self, name, **overrides):
        candidate = {
            "name": name,
            "asset_exists": True,
            "audit_status": None,
            "run_identity_sha256": name.lower() + "-identity",
            "checkpoint_sha256": name.lower() + "-checkpoint",
            "code_sha256": name.lower() + "-code",
            "config_sha256": name.lower() + "-config",
            "model_sha256": name.lower() + "-model",
            "source_weight_sha256": name.lower() + "-source",
            "fit_ids_sha256": "fit",
            "development_ids_sha256": "dev",
            "actual_train_ids_sha256": "fit",
            "train_cache_sha256": name.lower() + "-cache",
            "data_content_manifest_sha256": name.lower() + "-data",
            "seed": 0,
            "epochs": 30,
            "imgsz": 640,
            "batch": 16,
            "workers": 8,
            "amp": True,
            "deterministic": True,
            "checkpoint_role": "last.pt",
            "initialization": "registered-source-plus-deterministic-unmatched",
            "augmentation": "registered-default",
            "runtime": "ultralytics-8.4.98",
            "evaluator": "ifdr_yolo.kitti_ap40",
            "evaluator_source_sha256": name.lower() + "-evaluator",
            "prediction_args": {"conf": 0.001, "iou": 0.7, "max_det": 300},
            "evidence": [name.lower() + ".json"],
        }
        candidate.update(overrides)
        return candidate

    def test_identity_field_omission_fails_closed(self):
        candidate = self.candidate("P3P5_CONTROL")
        del candidate["runtime"]
        with self.assertRaisesRegex(ValueError, "missing required identity field: runtime"):
            build_ledger(self.active(), [candidate])

    def test_mismatched_and_no_go_candidates_forbid_subtraction(self):
        control = self.candidate("P3P5_CONTROL", seed=17, epochs=300)
        method = self.candidate("DCLI", seed=17)
        repair = self.candidate(
            "R",
            asset_exists=False,
            audit_status="NO_GO_INSUFFICIENT_EVIDENCE",
            run_identity_sha256=None,
            checkpoint_sha256=None,
        )
        ledger, matrix = build_ledger(self.active(), [control, method, repair])
        self.assertEqual(ledger["candidates"]["P3P5_CONTROL"]["status"], "MISMATCHED")
        self.assertEqual(ledger["candidates"]["DCLI"]["status"], "MISMATCHED")
        self.assertEqual(ledger["candidates"]["R"]["status"], "NO_GO")
        self.assertEqual(ledger["decision"], "NO_GO_SEED0_FAIR_COMPARISON")
        self.assertTrue(all(row["status"] == "FORBIDDEN_SUBTRACTION" for row in matrix))

    def test_only_complete_matched_candidates_can_be_compared(self):
        control = self.candidate("P3P5_CONTROL")
        method = self.candidate("DCLI")
        ledger, matrix = build_ledger(self.active(), [control, method])
        self.assertEqual(ledger["candidates"]["P3P5_CONTROL"]["status"], "VALID_MATCHED")
        self.assertEqual(ledger["candidates"]["DCLI"]["status"], "VALID_MATCHED")
        self.assertEqual(matrix, [{
            "reference": "P3P5_CONTROL",
            "candidate": "DCLI",
            "status": "VALID_MAIN",
            "reason": "both candidates satisfy the active matched protocol",
        }])

    def test_publication_is_identical_and_manifest_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary"
            mirror = root / "mirror"
            ledger, matrix = build_ledger(
                self.active(),
                [self.candidate("P3P5_CONTROL"), self.candidate("DCLI")],
            )
            publish_ledger(primary, mirror, ledger, matrix)
            for name in ("DATA_EVALUATION_LEDGER.json", "protocol_matrix.csv", "manifest.json"):
                self.assertEqual((primary / name).read_bytes(), (mirror / name).read_bytes())
            manifest = json.loads((primary / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["files"]), {"DATA_EVALUATION_LEDGER.json", "protocol_matrix.csv"})
            with (primary / "protocol_matrix.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["status"], "VALID_MAIN")


if __name__ == "__main__":
    unittest.main()
