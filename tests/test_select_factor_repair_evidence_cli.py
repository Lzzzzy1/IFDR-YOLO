from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def _fake_evidence(condition: str, *, passed: bool = True):
    from types import SimpleNamespace

    gate = SimpleNamespace(
        passed=passed,
        failures=() if passed else ("absolute_gate_failed",),
        to_dict=lambda: {
            "passed": passed,
            "stage": "development",
            "failures": [] if passed else ["absolute_gate_failed"],
        },
    )
    return SimpleNamespace(
        condition=condition,
        stage="development",
        image_ids=("000001", "000002"),
        image_ids_hash="a" * 64,
        endpoints={
            "sampling_residual_spearman": 0.2,
            "visibility_residual_spearman": 0.3,
            "sampling_specificity_gap": 0.4,
            "visibility_specificity_gap": 0.5,
        },
        evidence_sha256=(condition + "0" * 64)[:64],
        absolute_gate_passed=passed,
        absolute_gate=gate,
        complete=True,
        raw_observations=({"image_id": "000001", "seed": 17},),
        endpoint_samples=None,
        recompute_endpoints=lambda _indices: {
            "sampling_residual_spearman": 0.2,
            "visibility_residual_spearman": 0.3,
            "sampling_specificity_gap": 0.4,
            "visibility_specificity_gap": 0.5,
        },
    )


class SelectFactorRepairEvidenceCliTest(unittest.TestCase):
    def test_parser_requires_four_conditions_and_output_directory(self) -> None:
        from scripts.select_factor_repair_evidence import build_parser

        args = build_parser().parse_args(
            [
                "--f0", "f0.json",
                "--f1", "f1.json",
                "--f2", "f2.json",
                "--f3", "f3.json",
                "--output-dir", "out",
            ]
        )
        self.assertEqual(args.f0, Path("f0.json"))
        self.assertEqual(args.f3, Path("f3.json"))
        self.assertEqual(args.output_dir, Path("out"))

    def test_run_writes_selection_and_mechanism_artifacts_without_overwrite(self) -> None:
        from types import SimpleNamespace

        from scripts import select_factor_repair_evidence as cli

        args = SimpleNamespace(
            f0=Path("f0.json"),
            f1=Path("f1.json"),
            f2=Path("f2.json"),
            f3=Path("f3.json"),
            output_dir=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = Path(directory)
            evidences = {condition: _fake_evidence(condition) for condition in ("F0", "F1", "F2", "F3")}
            selected = SimpleNamespace(
                selected_condition="F2",
                delta_s_point=0.11,
                delta_s_ci95=(0.02, 0.20),
                decision_sha256="d" * 64,
                to_dict=lambda: {
                    "reference_condition": "F0",
                    "selected_condition": "F2",
                    "delta_s_point": 0.11,
                    "delta_s_ci95": [0.02, 0.20],
                    "endpoint_table": {"F0": evidences["F0"].endpoints, "F2": evidences["F2"].endpoints},
                    "reference_evidence_sha256": evidences["F0"].evidence_sha256,
                    "selected_evidence_sha256": evidences["F2"].evidence_sha256,
                    "decision_sha256": "d" * 64,
                },
            )
            def paired(candidate, _f0):
                return SimpleNamespace(
                    point=0.11 if candidate.condition == "F2" else 0.01,
                    ci95=(0.02, 0.20) if candidate.condition == "F2" else (-0.01, 0.03),
                    candidate_endpoints=candidate.endpoints,
                    candidate_evidence_sha256=candidate.evidence_sha256,
                )

            with patch.object(cli, "load_factor_repair_evidence", side_effect=lambda path: evidences[path.stem.upper()]), patch.object(
                cli, "validate_shared_image_identity", return_value=(evidences["F0"].image_ids, evidences["F0"].image_ids_hash)
            ), patch.object(cli, "paired_image_cluster_delta", side_effect=paired), patch.object(
                cli, "select_repair_against_f0", return_value=selected
            ):
                evidence_path, table_path, csv_path = cli.run(args)

            self.assertEqual(evidence_path.name, "selection_decision.json")
            self.assertEqual(table_path.name, "mechanism_table.json")
            self.assertEqual(csv_path.name, "mechanism_table.csv")
            selection = json.loads(evidence_path.read_text(encoding="utf-8"))
            table = json.loads(table_path.read_text(encoding="utf-8"))
            self.assertEqual(selection["selected_condition"], "F2")
            self.assertEqual(table["conditions"]["F2"]["delta_s_ci95"], [0.02, 0.20])
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["condition"] for row in rows}, {"F0", "F1", "F2", "F3"})

            with self.assertRaisesRegex(ValueError, "overwrite"):
                cli.run(args)

    def test_point_sample_or_shared_identity_failure_is_propagated(self) -> None:
        from types import SimpleNamespace

        from scripts import select_factor_repair_evidence as cli

        args = SimpleNamespace(
            f0=Path("f0.json"), f1=Path("f1.json"), f2=Path("f2.json"), f3=Path("f3.json"), output_dir=Path("out")
        )
        with patch.object(cli, "load_factor_repair_evidence", side_effect=ValueError("point endpoint samples are not admissible evidence")):
            with self.assertRaisesRegex(ValueError, "point endpoint"):
                cli.run(args)

        evidences = {condition: _fake_evidence(condition) for condition in ("F0", "F1", "F2", "F3")}
        with patch.object(cli, "load_factor_repair_evidence", side_effect=lambda path: evidences[path.stem.upper()]), patch.object(
            cli, "validate_shared_image_identity", side_effect=ValueError("F0-F3 evidence image IDs mismatch")
        ):
            with self.assertRaisesRegex(ValueError, "image IDs mismatch"):
                cli.run(args)
