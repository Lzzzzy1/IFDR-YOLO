from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np


class NaturalFactorAuditCliTest(unittest.TestCase):
    def test_parser_has_frozen_defaults_and_repeatable_checkpoints(self) -> None:
        from scripts.audit_natural_factors import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--metadata-jsonl",
                "metadata.jsonl",
                "--train-ids",
                "train.txt",
                "--val-ids",
                "val.txt",
                "--image-dir",
                "images",
                "--output-dir",
                "out",
                "--checkpoint",
                "17=a.pt",
                "--checkpoint",
                "29=b.pt",
                "--checkpoint",
                "41=c.pt",
            ]
        )
        self.assertEqual(args.input_size, 640)
        self.assertEqual(args.transform_batch_size, 8)
        self.assertEqual(args.bootstrap_replicates, 2000)
        self.assertEqual(args.audit_seed, 20260804)
        self.assertEqual(tuple(args.registered_severities), (0.25, 0.5, 0.75, 1.0))
        self.assertEqual(args.checkpoint, ["17=a.pt", "29=b.pt", "41=c.pt"])

    def test_train_selection_is_order_independent_and_never_val(self) -> None:
        from scripts.audit_natural_factors import select_audit_image_ids

        train = ("000004", "000001", "000003", "000002", "000005", "000006")
        val = ("000007",)
        first = select_audit_image_ids(train, val, audit_seed=20260804)
        second = select_audit_image_ids(tuple(reversed(train)), val, audit_seed=20260804)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertTrue(set(first).issubset(set(train)))
        self.assertTrue(set(first).isdisjoint(val))

    def test_intervention_selection_uses_joint_score_and_digest_tie_break(self) -> None:
        from ifdr_yolo.data.natural_degradation import NaturalDegradationRecord
        from scripts.audit_natural_factors import select_intervention_objects

        def record(image_id: str, object_id: int, class_name: str, sampling: float, visibility: float):
            return NaturalDegradationRecord(
                image_id=image_id,
                object_id=object_id,
                class_id={"Car": 0, "Pedestrian": 1, "Cyclist": 2}[class_name],
                class_name=class_name,
                bbox_xyxy=(10.0 + object_id, 10.0, 30.0 + object_id, 30.0),
                box_height=20.0,
                depth_m=20.0,
                depth_available=True,
                occlusion_level=0,
                truncation=0.0,
                sampling_score=sampling,
                visibility_score=visibility,
            )

        records = (
            record("000001", 0, "Car", 0.2, 0.2),
            record("000001", 1, "Car", 0.6, 0.6),
            record("000001", 2, "Pedestrian", 0.1, 0.1),
        )
        selected = select_intervention_objects(records, ("000001",), audit_seed=20260804)
        self.assertEqual(selected, (("000001", 1), ("000001", 2)))

    def test_strict_ids_reject_blank_duplicate_and_overlap(self) -> None:
        from scripts.audit_natural_factors import load_strict_ids, validate_split_ids

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ids.txt"
            path.write_text("000001\n\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_strict_ids(path)
            path.write_text("000001\n000001\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_strict_ids(path)
        with self.assertRaises(ValueError):
            validate_split_ids(("000001",), ("000001",))

    def test_resumable_run_rebuilds_root_without_duplicate_rows_and_writes_five_artifacts(self) -> None:
        from ifdr_yolo.eval.natural_factor_audit import NaturalFactorObservation
        import scripts.audit_natural_factors as cli

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            encoded_ok, encoded = cv2.imencode(".png", np.zeros((64, 64, 3), np.uint8))
            self.assertTrue(encoded_ok)
            (image_dir / "000001.png").write_bytes(encoded.tobytes())
            metadata = root / "metadata.jsonl"
            metadata.write_text(
                json.dumps(
                    {
                        "image_id": "000001",
                        "kind": "Car",
                        "bbox": {"x1": 10, "y1": 10, "x2": 30, "y2": 30},
                        "truncated": 0.0,
                        "occluded": 0,
                        "location_xyz": [0.0, 0.0, 20.0],
                        "object_id": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            train = root / "train.txt"
            val = root / "val.txt"
            train.write_text("000001\n", encoding="utf-8")
            val.write_text("000002\n", encoding="utf-8")
            checkpoints = []
            for seed in cli.REQUIRED_SEEDS:
                path = root / f"{seed}.pt"
                path.write_bytes(f"checkpoint-{seed}".encode())
                checkpoints.append(f"{seed}={path}")
            output = root / "audit"
            args = cli.build_parser().parse_args(
                [
                    *sum((["--checkpoint", item] for item in checkpoints), []),
                    "--metadata-jsonl",
                    str(metadata),
                    "--train-ids",
                    str(train),
                    "--val-ids",
                    str(val),
                    "--image-dir",
                    str(image_dir),
                    "--output-dir",
                    str(output),
                    "--bootstrap-replicates",
                    "2",
                ]
            )
            fixture = (
                NaturalFactorObservation(
                    seed=17,
                    node_id=11,
                    image_id="000001",
                    object_id=0,
                    class_id=0,
                    class_name="Car",
                    box_height=20.0,
                    region_role="target",
                    intervention_kind="natural",
                    intervention_severity=0.0,
                    pair_id=None,
                    natural_sampling=0.2,
                    natural_visibility=0.1,
                    predicted_sampling=0.3,
                    predicted_visibility=0.2,
                    branch_weights=(0.6, 0.4),
                ),
            )

            class FakeGate:
                def to_dict(self):
                    return {
                        "passed": True,
                        "factors": {
                            "sampling": {"passed": True},
                            "visibility": {"passed": True},
                        },
                    }

            def fake_loader(path, device):
                raw = Path(path).read_bytes()
                return type("Loaded", (), {"checkpoint_sha256": cli._sha256_bytes(raw)})()

            def fake_runner(loaded, manifest, journal, *, transform_batch_size):
                plan = manifest.plans[0]
                rows = [
                    {"observation_id": observation_id, "image_id": plan.image_id}
                    for observation_id in plan.expected_observation_ids
                ]
                journal.commit_image(plan.image_id, rows)
                journal.finalize()

            with patch.object(cli, "load_ifdr_checkpoint", side_effect=fake_loader), patch.object(
                cli, "run_factor_observer", side_effect=fake_runner
            ), patch.object(cli, "_load_observation_rows", return_value=fixture), patch.object(
                cli, "audit_natural_factors", return_value=FakeGate()
            ):
                self.assertEqual(cli._run(args), 0)
                first_root = (output / "observations.jsonl").read_bytes()
                self.assertGreater(len(first_root), 0)
                self.assertEqual(cli._run(args), 0)
                self.assertEqual((output / "observations.jsonl").read_bytes(), first_root)
            for name in ("observations.jsonl", "summary.json", "gate.json", "provenance.json", "status.json"):
                self.assertGreater((output / name).stat().st_size, 0)
            self.assertEqual(json.loads((output / "status.json").read_text())["status"], "complete")

    def test_failure_preserves_artifacts_and_records_exception_status(self) -> None:
        import scripts.audit_natural_factors as cli

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "out"
            args = cli.build_parser().parse_args(
                [
                    "--checkpoint",
                    "17=missing",
                    "--checkpoint",
                    "29=missing",
                    "--checkpoint",
                    "41=missing",
                    "--metadata-jsonl",
                    str(root / "metadata.jsonl"),
                    "--train-ids",
                    str(root / "train.txt"),
                    "--val-ids",
                    str(root / "val.txt"),
                    "--image-dir",
                    str(root / "images"),
                    "--output-dir",
                    str(output),
                ]
            )
            with self.assertRaises(ValueError):
                cli._run(args)
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["exception_type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
