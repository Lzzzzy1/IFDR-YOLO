from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace


class FactorDevelopmentObserverCliTest(unittest.TestCase):
    def test_development_ids_are_exactly_371_unique_kitti_ids(self) -> None:
        from scripts.run_factor_development_observer import load_development_ids

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "development_ids.txt"
            ids = [f"{index:06d}" for index in range(371)]
            path.write_text("\n".join(ids) + "\n", encoding="utf-8")
            self.assertEqual(load_development_ids(path), tuple(ids))

            path.write_text("\n".join(ids[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "371"):
                load_development_ids(path)

            duplicate = ids[:-1] + [ids[0]]
            path.write_text("\n".join(duplicate) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_development_ids(path)

    def test_checkpoint_role_hash_requires_calibration_last(self) -> None:
        from scripts.run_factor_development_observer import resolve_calibration_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "last.pt"
            checkpoint.write_bytes(b"calibration-checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            roles = root / "checkpoint_roles.json"
            roles.write_text(
                json.dumps(
                    {
                        "calibration_last": {
                            "path": "last.pt",
                            "role": "primary",
                            "checkpoint_role": "calibration_last",
                            "sha256": digest,
                        }
                    }
                ),
                encoding="utf-8",
            )
            resolved, role_payload = resolve_calibration_checkpoint(checkpoint, roles)
            self.assertEqual(resolved["path"], str(checkpoint.resolve()))
            self.assertEqual(resolved["checkpoint_role"], "calibration_last")
            self.assertEqual(resolved["sha256"], digest)
            self.assertIn("calibration_last", role_payload)

            primary_roles = root / "primary_roles.json"
            primary_roles.write_text(
                json.dumps(
                    {
                        "diagnostic_checkpoint": {
                            "path": "best.pt",
                            "role": "diagnostic",
                            "sha256": "1" * 64,
                        },
                        "primary_checkpoint": {
                            "path": "last.pt",
                            "role": "primary",
                            "sha256": digest,
                        },
                    }
                ),
                encoding="utf-8",
            )
            resolved, _ = resolve_calibration_checkpoint(checkpoint, primary_roles)
            self.assertEqual(resolved["checkpoint_role"], "calibration_last")

            bad_roles = root / "bad_roles.json"
            bad_roles.write_text(
                json.dumps(
                    {
                        "calibration_last": {
                            "path": "last.pt",
                            "checkpoint_role": "calibration_last",
                            "sha256": "0" * 64,
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "hash"):
                resolve_calibration_checkpoint(checkpoint, bad_roles)

            best = root / "best.pt"
            best.write_bytes(checkpoint.read_bytes())
            with self.assertRaisesRegex(ValueError, "last.pt"):
                resolve_calibration_checkpoint(best, roles)

    def test_parser_freezes_seed17_and_three_view_protocol(self) -> None:
        from scripts.run_factor_development_observer import (
            DEVELOPMENT_SEED,
            build_parser,
        )

        args = build_parser().parse_args(
            [
                "--condition",
                "F2",
                "--checkpoint",
                "last.pt",
                "--checkpoint-roles",
                "checkpoint_roles.json",
                "--development-ids",
                "development_ids.txt",
                "--metadata-jsonl",
                "objects.jsonl",
                "--image-dir",
                "images",
                "--output-dir",
                "evidence/F2",
            ]
        )
        self.assertEqual(args.condition, "F2")
        self.assertEqual(args.seed, DEVELOPMENT_SEED)
        self.assertEqual(args.input_size, 640)
        self.assertEqual(args.transform_batch_size, 8)
        self.assertEqual(args.bootstrap_replicates, 2000)
        self.assertEqual(args.audit_seed, 20260805)
        self.assertEqual(tuple(args.views), ("target", "background", "natural"))
        self.assertEqual(
            build_parser().parse_args(
                [
                    "--condition", "F0", "--checkpoint", "17=last.pt",
                    "--checkpoint-roles", "roles.json", "--development-ids", "ids.txt",
                    "--metadata-jsonl", "m.jsonl", "--image-dir", "images", "--output-dir", "out",
                ]
            ).checkpoint,
            Path("last.pt"),
        )
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "--condition", "F0", "--checkpoint", "29=last.pt",
                    "--checkpoint-roles", "roles.json", "--development-ids", "ids.txt",
                    "--metadata-jsonl", "m.jsonl", "--image-dir", "images", "--output-dir", "out",
                ]
            )

    def test_complete_output_is_not_overwritten(self) -> None:
        from scripts.run_factor_development_observer import ensure_output_reusable

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "status.json").write_text(
                json.dumps({"schema_version": 1, "status": "complete"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "overwrite"):
                ensure_output_reusable(output, {"condition": "F0"})

    def test_audit_artifacts_include_shared_image_identity_and_summary(self) -> None:
        from scripts.run_factor_development_observer import write_audit_artifacts

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_ids = tuple(f"{index:06d}" for index in range(371))
            image_hash = hashlib.sha256(
                json.dumps(list(image_ids), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            decision = SimpleNamespace(
                passed=True,
                to_dict=lambda: {
                    "passed": True,
                    "reasons": [],
                    "required_seeds": [17],
                    "required_nodes": [11, 14, 17, 20, 23, 26],
                    "factors": {},
                },
            )
            manifest = SimpleNamespace(hash=lambda: "a" * 64, expected_observation_count=0)
            checkpoint = {"sha256": "b" * 64}
            write_audit_artifacts(
                root,
                condition="F0",
                rows=(),
                decision=decision,
                image_ids=image_ids,
                image_ids_hash=image_hash,
                checkpoint=checkpoint,
                manifest=manifest,
                selected_interventions=(),
            )
            audit = json.loads((root / "audit_decision.json").read_text(encoding="utf-8"))
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["image_ids"], list(image_ids))
            self.assertEqual(audit["image_ids_hash"], image_hash)
            self.assertEqual(summary["image_count"], 371)
            self.assertEqual(summary["condition"], "F0")
