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
                "--mirror-dir", "mirror",
            ]
        )
        self.assertEqual(args.f0, Path("f0.json"))
        self.assertEqual(args.f3, Path("f3.json"))
        self.assertEqual(args.output_dir, Path("out"))
        self.assertEqual(args.mirror_dir, Path("mirror"))

    def test_run_writes_selection_and_mechanism_artifacts_without_overwrite(self) -> None:
        from types import SimpleNamespace

        from scripts import select_factor_repair_evidence as cli

        args = SimpleNamespace(
            f0=None,
            f1=None,
            f2=None,
            f3=None,
            output_dir=None,
            mirror_dir=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args.output_dir = root / "output"
            args.mirror_dir = root / "mirror"
            for condition in ("F0", "F1", "F2", "F3"):
                path = root / f"{condition.lower()}.json"
                path.write_text("{}", encoding="utf-8")
                setattr(args, condition.lower(), path)
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
            f0=Path("f0.json"), f1=Path("f1.json"), f2=Path("f2.json"), f3=Path("f3.json"), output_dir=Path("out"), mirror_dir=Path("mirror")
        )
        with patch.object(cli, "load_factor_repair_evidence", side_effect=ValueError("point endpoint samples are not admissible evidence")):
            with self.assertRaisesRegex(ValueError, "point endpoint"):
                cli.run(args)

        evidences = {condition: _fake_evidence(condition) for condition in ("F0", "F1", "F2", "F3")}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for condition in ("F0", "F1", "F2", "F3"):
                path = root / f"{condition.lower()}.json"
                path.write_text("{}", encoding="utf-8")
                setattr(args, condition.lower(), path)
            with patch.object(cli, "load_factor_repair_evidence", side_effect=lambda path: evidences[path.stem.upper()]), patch.object(
                cli, "validate_shared_image_identity", side_effect=ValueError("F0-F3 evidence image IDs mismatch")
            ):
                with self.assertRaisesRegex(ValueError, "image IDs mismatch"):
                    cli.run(args)

    def test_mirror_publish_writes_atomic_resume_manifest_set(self) -> None:
        from scripts import select_factor_repair_evidence as cli

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mirror = root / "mirror"
            cli._mirror_publish(
                mirror,
                stage="initialized",
                output_dir=root / "output",
                checkpoint_dir=root / "output" / "checkpoints",
                conditions={condition: "pending" for condition in ("F0", "F1", "F2", "F3")},
                resume=False,
                workers=1,
            )
            self.assertEqual(
                {path.name for path in mirror.iterdir()},
                {"checkpoint_index.json", "manifest.json", "progress.json", "summary.json", "resume.txt"},
            )
            manifest = json.loads((mirror / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["stage"], "initialized")
            self.assertTrue(manifest["generation"])
            for name, digest in manifest["files"].items():
                payload = json.loads((mirror / name).read_text(encoding="utf-8"))
                self.assertEqual(payload["generation"], manifest["generation"])
                self.assertEqual(__import__("hashlib").sha256((mirror / name).read_bytes()).hexdigest(), digest)
            self.assertIn("resume command", (mirror / "resume.txt").read_text(encoding="utf-8"))
            self.assertEqual(list(mirror.glob("*.tmp")), [])

    def test_partial_mirror_publish_is_repaired_on_next_generation(self) -> None:
        from scripts import select_factor_repair_evidence as cli

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mirror = root / "mirror"
            kwargs = dict(
                stage="initialized", output_dir=root / "output", checkpoint_dir=root / "checkpoints",
                conditions={condition: "pending" for condition in ("F0", "F1", "F2", "F3")},
                resume=False, workers=1,
            )
            original = cli._atomic_replace
            def fail_manifest(path, payload):
                if path.name == "manifest.json":
                    raise OSError("simulated mirror interruption")
                return original(path, payload)
            with patch.object(cli, "_atomic_replace", side_effect=fail_manifest):
                with self.assertRaisesRegex(OSError, "mirror interruption"):
                    cli._mirror_publish(mirror, **kwargs)
            self.assertFalse((mirror / "manifest.json").exists())
            cli._mirror_publish(mirror, **kwargs)
            cli._validate_mirror_commit(mirror)

    def test_complete_mirror_carries_exact_final_artifacts_and_manifest_hashes(self) -> None:
        from scripts import select_factor_repair_evidence as cli

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mirror = root / "mirror"
            artifacts = {
                "selection_decision.json": b'{"selected_condition":"F2"}\n',
                "mechanism_table.json": b'{"conditions":{}}\n',
                "mechanism_table.csv": b"condition\nF2\n",
            }
            with self.assertRaisesRegex(ValueError, "final_artifacts"):
                cli._mirror_publish(
                    mirror,
                    stage="complete",
                    output_dir=root / "output",
                    checkpoint_dir=root / "checkpoints",
                    conditions={condition: "complete" for condition in ("F0", "F1", "F2", "F3", "selection")},
                    selected_condition="F2",
                    resume=False,
                    workers=1,
                )
            cli._mirror_publish(
                mirror,
                stage="complete",
                output_dir=root / "output",
                checkpoint_dir=root / "checkpoints",
                conditions={condition: "complete" for condition in ("F0", "F1", "F2", "F3", "selection")},
                selected_condition="F2",
                resume=False,
                workers=1,
                final_artifacts=artifacts,
            )
            for name, expected in artifacts.items():
                self.assertEqual((mirror / name).read_bytes(), expected)
            manifest = json.loads((mirror / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["artifacts"]), set(artifacts))
            for name, expected in artifacts.items():
                self.assertEqual(manifest["artifacts"][name]["sha256"], __import__("hashlib").sha256(expected).hexdigest())
            cli._validate_mirror_commit(mirror)

    def test_mirror_ancestor_of_output_or_checkpoint_is_rejected(self) -> None:
        from types import SimpleNamespace

        from scripts import select_factor_repair_evidence as cli

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                f0=root / "f0.json", f1=root / "f1.json", f2=root / "f2.json", f3=root / "f3.json",
                output_dir=root / "output", checkpoint_dir=root / "output" / "checkpoints", mirror_dir=root,
                workers=1,
            )
            with self.assertRaisesRegex(ValueError, "mirror-dir"):
                cli.run(args)

    def test_formal_run_requires_external_mirror_and_worker_one(self) -> None:
        from types import SimpleNamespace

        from scripts import select_factor_repair_evidence as cli

        base = dict(f0=Path("f0.json"), f1=Path("f1.json"), f2=Path("f2.json"), f3=Path("f3.json"), output_dir=Path("out"))
        with self.assertRaisesRegex(ValueError, "mirror"):
            cli.run(SimpleNamespace(**base))
        with self.assertRaisesRegex(ValueError, "workers.*1"):
            cli.run(SimpleNamespace(**base, mirror_dir=Path("mirror"), workers=2))

    def test_finalization_journal_repairs_partial_outputs_on_resume(self) -> None:
        from types import SimpleNamespace

        from scripts import select_factor_repair_evidence as cli

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                f0=root / "f0.json", f1=root / "f1.json", f2=root / "f2.json", f3=root / "f3.json",
                output_dir=root / "output", mirror_dir=root / "mirror", resume=False,
            )
            for condition in ("F0", "F1", "F2", "F3"):
                getattr(args, condition.lower()).write_text("{}", encoding="utf-8")
            evidences = {condition: _fake_evidence(condition) for condition in ("F0", "F1", "F2", "F3")}
            deltas = {
                condition: SimpleNamespace(
                    point=0.11 if condition == "F2" else 0.01,
                    ci95=(0.02, 0.20) if condition == "F2" else (-0.01, 0.03),
                    candidate_endpoints=evidences[condition].endpoints,
                    candidate_evidence_sha256=evidences[condition].evidence_sha256,
                )
                for condition in ("F1", "F2", "F3")
            }
            selected = SimpleNamespace(
                selected_condition="F2", delta_s_point=0.11, delta_s_ci95=(0.02, 0.20),
                to_dict=lambda: {
                    "reference_condition": "F0", "selected_condition": "F2", "delta_s_point": 0.11,
                    "delta_s_ci95": [0.02, 0.20], "endpoint_table": {"F0": evidences["F0"].endpoints, "F2": evidences["F2"].endpoints},
                    "reference_evidence_sha256": evidences["F0"].evidence_sha256,
                    "selected_evidence_sha256": evidences["F2"].evidence_sha256, "decision_sha256": "d" * 64,
                },
            )
            def fake_run(candidate, *_args, **_kwargs):
                return deltas[candidate.condition]
            real_write = cli._write_final_artifact
            calls = {"count": 0}
            def fail_after_first(path, payload, digest):
                calls["count"] += 1
                if calls["count"] == 1:
                    return real_write(path, payload, digest)
                raise RuntimeError("simulated finalization interruption")
            patches = [
                patch.object(cli, "load_factor_repair_evidence", side_effect=lambda path: evidences[path.stem.upper()]),
                patch.object(cli, "validate_shared_image_identity", return_value=(evidences["F0"].image_ids, evidences["F0"].image_ids_hash)),
                patch.object(cli, "build_shared_reference_draws", return_value=object()),
                patch.object(cli, "run_resumable_factor_bootstrap", side_effect=fake_run),
                patch.object(cli, "select_repair_against_f0", return_value=selected),
                patch.object(cli, "_write_final_artifact", side_effect=fail_after_first),
            ]
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                with self.assertRaisesRegex(RuntimeError, "finalization interruption"):
                    cli.run(args)
            self.assertTrue((args.output_dir / "selection_decision.json").exists())
            self.assertFalse((args.output_dir / "mechanism_table.json").exists())
            self.assertEqual(json.loads((args.output_dir / "finalization.json").read_text(encoding="utf-8"))["state"], "pending")
            args.resume = True
            with patch.object(cli, "load_factor_repair_evidence", side_effect=lambda path: evidences[path.stem.upper()]), patch.object(
                cli, "validate_shared_image_identity", return_value=(evidences["F0"].image_ids, evidences["F0"].image_ids_hash)
            ), patch.object(cli, "build_shared_reference_draws", return_value=object()), patch.object(
                cli, "run_resumable_factor_bootstrap", side_effect=fake_run
            ), patch.object(cli, "select_repair_against_f0", return_value=selected):
                cli.run(args)
            self.assertTrue((args.output_dir / "mechanism_table.json").exists())
            self.assertTrue((args.output_dir / "mechanism_table.csv").exists())
            self.assertEqual(json.loads((args.output_dir / "finalization.json").read_text(encoding="utf-8"))["state"], "complete")
            for name in ("selection_decision.json", "mechanism_table.json", "mechanism_table.csv"):
                self.assertEqual((args.output_dir / name).read_bytes(), (args.mirror_dir / name).read_bytes())
