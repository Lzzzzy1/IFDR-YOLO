from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import patch

import cv2
import numpy as np


class NaturalFactorAuditCliTest(unittest.TestCase):
    def test_git_commit_requires_valid_clean_worktree(self) -> None:
        import scripts.audit_natural_factors as cli

        clean_head = SimpleNamespace(returncode=0, stdout="a" * 40 + "\n")
        clean_status = SimpleNamespace(returncode=0, stdout="")
        with patch.object(cli.subprocess, "run", side_effect=[clean_head, clean_status]):
            self.assertEqual(cli._git_commit(), "a" * 40)

        dirty_head = SimpleNamespace(returncode=0, stdout="a" * 40 + "\n")
        dirty_status = SimpleNamespace(returncode=0, stdout=" M scripts/audit_natural_factors.py\n")
        with patch.object(cli.subprocess, "run", side_effect=[dirty_head, dirty_status]):
            with self.assertRaises(ValueError):
                cli._git_commit()

        invalid_head = SimpleNamespace(returncode=0, stdout="a" * 39 + "\n")
        with patch.object(cli.subprocess, "run", return_value=invalid_head):
            with self.assertRaises(ValueError):
                cli._git_commit()

        with patch.object(cli.subprocess, "run", side_effect=OSError("git missing")):
            with self.assertRaises(ValueError):
                cli._git_commit()

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

        for alias in ("--severity", "--registered-severities"):
            with self.subTest(alias=alias), self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "--metadata-jsonl", "metadata.jsonl", "--train-ids", "train.txt",
                        "--val-ids", "val.txt", "--image-dir", "images", "--output-dir", "out",
                        "--checkpoint", "17=a.pt", "--checkpoint", "29=b.pt", "--checkpoint", "41=c.pt",
                        alias, "0.25",
                    ]
                )

    def test_selection_uses_no_nul_golden_digest(self) -> None:
        from scripts.audit_natural_factors import select_audit_image_ids

        train = tuple(f"{index:06d}" for index in range(1, 11))
        self.assertEqual(
            select_audit_image_ids(train, (), audit_seed=20260804),
            ("000004", "000009"),
        )

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
        tie_records = (
            record("000001", 0, "Car", 0.2, 0.2),
            record("000001", 1, "Car", 0.2, 0.2),
        )
        self.assertEqual(
            select_intervention_objects(tie_records, ("000001",), audit_seed=20260804),
            (("000001", 0),),
        )

    def test_strict_ids_reject_blank_duplicate_and_overlap(self) -> None:
        from scripts.audit_natural_factors import load_strict_ids, validate_split_ids

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ids.txt"
            path.write_text("000001\n\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_strict_ids(path)
            path.write_text(" 000001\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_strict_ids(path)
            path.write_text("000001\n000001\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_strict_ids(path)
        with self.assertRaises(ValueError):
            validate_split_ids(("000001",), ("000001",))

    def test_checkpoint_helper_requires_exact_real_seed_set_and_severity_is_frozen(self) -> None:
        import scripts.audit_natural_factors as cli

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for seed in cli.REQUIRED_SEEDS:
                paths[seed] = root / f"{seed}.pt"
                paths[seed].write_bytes(f"seed-{seed}".encode("ascii"))
            valid = [f"{seed}={paths[seed]}" for seed in cli.REQUIRED_SEEDS]
            resolved, hashes = cli._checkpoint_hashes(valid)
            self.assertEqual(set(resolved), set(cli.REQUIRED_SEEDS))
            self.assertEqual(set(hashes), set(cli.REQUIRED_SEEDS))
            for specs in (
                valid[:2],
                valid + [f"53={paths[17]}"],
                [valid[0], valid[0], valid[1]],
            ):
                with self.subTest(specs=specs), self.assertRaises(ValueError):
                    cli._checkpoint_hashes(specs)
        with self.assertRaises(ValueError):
            cli._validate_severities((0.25, 0.5, 0.75, 0.9))

    def test_existing_provenance_directory_is_rejected(self) -> None:
        import scripts.audit_natural_factors as cli

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.json"
            path.mkdir()
            with self.assertRaises(ValueError):
                cli._validate_existing_provenance(path, {})

    def test_existing_provenance_identity_type_mismatch_is_rejected(self) -> None:
        import scripts.audit_natural_factors as cli

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.json"
            path.write_text(
                json.dumps({"scientific_identity": {"input_size": True}}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                cli._validate_existing_provenance(path, {"input_size": 1})

    def test_manifest_hash_helper_does_not_overwrite_existing_pair(self) -> None:
        import scripts.audit_natural_factors as cli

        class Manifest:
            def to_dict(self):
                return {"seed": 17}

            def hash(self):
                return cli._sha256_bytes(cli._canonical_json(self.to_dict()).encode("utf-8"))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            hash_path = path.with_name("manifest.sha256")
            path.write_bytes(b'{"seed": 99}\n')
            hash_path.write_bytes(b"b" * 64 + b"\n")
            before = (path.read_bytes(), hash_path.read_bytes())
            with self.assertRaises(ValueError):
                cli._manifest_hash_file(path, Manifest())
            self.assertEqual((path.read_bytes(), hash_path.read_bytes()), before)

            path.write_bytes(b'{"seed": 17.0}\n')
            hash_path.write_bytes(
                cli._sha256_bytes(cli._canonical_json(Manifest().to_dict()).encode("utf-8")).encode("ascii")
                + b"\n"
            )
            before = (path.read_bytes(), hash_path.read_bytes())
            with self.assertRaises(ValueError):
                cli._manifest_hash_file(path, Manifest())
            self.assertEqual((path.read_bytes(), hash_path.read_bytes()), before)

    def test_manifest_hash_helper_recovers_valid_partial_and_rejects_invalid(self) -> None:
        import scripts.audit_natural_factors as cli

        class Manifest:
            def to_dict(self):
                return {"seed": 17}

            def hash(self):
                return cli._sha256_bytes(cli._canonical_json(self.to_dict()).encode("utf-8"))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            hash_path = root / "manifest.sha256"
            manifest_path.write_bytes(b'{"seed": 17}\n')
            manifest_before = manifest_path.read_bytes()
            cli._manifest_hash_file(manifest_path, Manifest())
            self.assertEqual(manifest_path.read_bytes(), manifest_before)
            self.assertEqual(
                hash_path.read_bytes(),
                cli._sha256_bytes(cli._canonical_json(Manifest().to_dict()).encode("utf-8")).encode("ascii")
                + b"\n",
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            hash_path = root / "manifest.sha256"
            manifest_hash = cli._sha256_bytes(
                cli._canonical_json(Manifest().to_dict()).encode("utf-8")
            )
            hash_path.write_bytes(manifest_hash.encode("ascii") + b"\n")
            hash_before = hash_path.read_bytes()
            cli._manifest_hash_file(manifest_path, Manifest())
            self.assertEqual(
                manifest_path.read_bytes(),
                (cli._canonical_json(Manifest().to_dict()) + "\n").encode("utf-8"),
            )
            self.assertEqual(hash_path.read_bytes(), hash_before)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            hash_path = root / "manifest.sha256"
            manifest_path.write_bytes(b'{"seed": 99}\n')
            manifest_before = manifest_path.read_bytes()
            with self.assertRaises(ValueError):
                cli._manifest_hash_file(manifest_path, Manifest())
            self.assertEqual(manifest_path.read_bytes(), manifest_before)
            self.assertFalse(hash_path.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            hash_path = root / "manifest.sha256"
            hash_path.write_bytes(b"a" * 64 + b"\n")
            hash_before = hash_path.read_bytes()
            with self.assertRaises(ValueError):
                cli._manifest_hash_file(manifest_path, Manifest())
            self.assertFalse(manifest_path.exists())
            self.assertEqual(hash_path.read_bytes(), hash_before)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            hash_path = root / "manifest.sha256"
            manifest_path.mkdir()
            with self.assertRaises(ValueError):
                cli._manifest_hash_file(manifest_path, Manifest())
            self.assertTrue(manifest_path.is_dir())
            self.assertFalse(hash_path.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            hash_path = root / "manifest.sha256"
            manifest_path.write_bytes(b'{"seed": 17}\n')
            hash_path.write_bytes(b"\xff\n")
            with self.assertRaisesRegex(ValueError, "hash is malformed"):
                cli._manifest_hash_file(manifest_path, Manifest())
            self.assertEqual(manifest_path.read_bytes(), b'{"seed": 17}\n')
            self.assertEqual(hash_path.read_bytes(), b"\xff\n")

    def test_root_rebuild_streams_seed_files_without_path_read_bytes(self) -> None:
        import scripts.audit_natural_factors as cli

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in cli.REQUIRED_SEEDS:
                seed_dir = root / f"seed-{seed}"
                seed_dir.mkdir()
                (seed_dir / "observations.jsonl").write_bytes(
                    (json.dumps({"seed": seed}) + "\n").encode("utf-8")
                )
            with patch.object(Path, "read_bytes", side_effect=AssertionError("must stream")):
                output = cli._rebuild_root_observations(root, cli.REQUIRED_SEEDS)
            self.assertEqual(
                output.read_bytes(),
                b'{"seed": 17}\n{"seed": 29}\n{"seed": 41}\n',
            )

    def test_root_rebuild_keeps_existing_output_when_seed_copy_fails(self) -> None:
        import scripts.audit_natural_factors as cli

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in cli.REQUIRED_SEEDS:
                seed_dir = root / f"seed-{seed}"
                seed_dir.mkdir()
                suffix = b"\n" if seed != 29 else b""
                (seed_dir / "observations.jsonl").write_bytes(
                    json.dumps({"seed": seed}).encode("utf-8") + suffix
                )
            output = root / "observations.jsonl"
            output.write_bytes(b"old-root\n")
            with self.assertRaises(ValueError):
                cli._rebuild_root_observations(root, cli.REQUIRED_SEEDS)
            self.assertEqual(output.read_bytes(), b"old-root\n")

    def test_observation_loader_rejects_duplicate_observation_id(self) -> None:
        import scripts.audit_natural_factors as cli

        row = {
            "observation_id": "a" * 64,
            "seed": 17,
            "node_id": 11,
            "image_id": "000001",
            "object_id": 0,
            "class_id": 0,
            "class_name": "Car",
            "box_height": 20.0,
            "region_role": "target",
            "intervention_kind": "natural",
            "intervention_severity": 0.0,
            "pair_id": None,
            "natural_sampling": 0.2,
            "natural_visibility": 0.1,
            "predicted_sampling": 0.3,
            "predicted_visibility": 0.2,
            "branch_weights": [0.6, 0.4],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.jsonl"
            path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                cli._load_observation_rows(path)

    def test_manifest_keeps_natural_rows_for_all_selected_image_objects(self) -> None:
        from ifdr_yolo.data.natural_degradation import NaturalDegradationRecord
        from ifdr_yolo.eval.factor_observer import build_factor_observation_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoded_ok, encoded = cv2.imencode(".png", np.zeros((100, 100, 3), np.uint8))
            self.assertTrue(encoded_ok)
            image_path = root / "000001.png"
            image_path.write_bytes(encoded.tobytes())

            def record(object_id: int, x1: float) -> NaturalDegradationRecord:
                return NaturalDegradationRecord(
                    image_id="000001",
                    object_id=object_id,
                    class_id=0,
                    class_name="Car",
                    bbox_xyxy=(x1, 10.0, x1 + 20.0, 30.0),
                    box_height=20.0,
                    depth_m=20.0,
                    depth_available=True,
                    occlusion_level=0,
                    truncation=0.0,
                    sampling_score=0.2 + object_id * 0.1,
                    visibility_score=0.1,
                )

            manifest = build_factor_observation_manifest(
                (record(0, 10.0), record(1, 60.0)),
                {"000001": image_path},
                (("000001", 0),),
                "a" * 64,
                17,
            )
            natural_object_ids = {
                condition.object_id
                for condition in manifest.plans[0].conditions
                if condition.intervention_kind == "natural"
            }
            self.assertEqual(natural_object_ids, {0, 1})

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
                if loaded_refs:
                    self.assertIsNone(loaded_refs[-1]())
                loaded = type("Loaded", (), {"checkpoint_sha256": cli._sha256_bytes(raw)})()
                loaded_refs.append(weakref.ref(loaded))
                return loaded

            def fake_runner(loaded, manifest, journal, *, transform_batch_size):
                plan = manifest.plans[0]
                rows = [
                    {"observation_id": observation_id, "image_id": plan.image_id}
                    for observation_id in plan.expected_observation_ids
                ]
                journal.commit_image(plan.image_id, rows)
                journal.finalize()

            real_git_commit = cli._git_commit
            loaded_refs: list[weakref.ReferenceType[object]] = []
            with patch.object(cli, "load_ifdr_checkpoint", side_effect=fake_loader), patch.object(
                cli, "run_factor_observer", new=fake_runner
            ), patch.object(cli, "_load_observation_rows", return_value=fixture), patch.object(
                cli, "audit_natural_factors", return_value=FakeGate()
            ) as audit, patch.object(cli, "_git_commit", return_value="implementation-old") as git_commit:
                self.assertEqual(cli._run(args), 0)
                self.assertEqual(git_commit.call_count, 1)
                first_root = (output / "observations.jsonl").read_bytes()
                self.assertGreater(len(first_root), 0)
                provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
                scientific = provenance["scientific_identity"]
                self.assertEqual(scientific["confidence"], cli.AUDIT_CONFIDENCE)
                self.assertEqual(scientific["monotonic_threshold"], cli.MONOTONIC_THRESHOLD)
                self.assertEqual(scientific["implementation_git_commit"], "implementation-old")
                self.assertNotIn("transform_batch_size", scientific)
                for path_key in (
                    "metadata_path", "train_ids_path", "val_ids_path", "checkpoint_paths", "image_dir"
                ):
                    self.assertNotIn(path_key, scientific)
                self.assertEqual(provenance["runtime"]["transform_batch_size"], 8)
                self.assertIn("device", provenance["runtime"])
                self.assertEqual(
                    provenance["runtime"]["paths"]["metadata_jsonl"],
                    str(metadata.resolve()),
                )
                self.assertEqual(audit.call_args.kwargs["confidence"], cli.AUDIT_CONFIDENCE)
                self.assertEqual(
                    audit.call_args.kwargs["monotonic_threshold"], cli.MONOTONIC_THRESHOLD
                )
                valid_provenance_bytes = (output / "provenance.json").read_bytes()

                for git_run_side_effect in (
                    [
                        SimpleNamespace(returncode=0, stdout="a" * 40 + "\n"),
                        SimpleNamespace(returncode=0, stdout=" M dirty.txt\n"),
                    ],
                    OSError("git missing"),
                ):
                    protected_files = {
                        path: path.read_bytes()
                        for path in output.rglob("*")
                        if path.is_file()
                    }
                    with patch.object(cli, "_git_commit", side_effect=real_git_commit), patch.object(
                        cli.subprocess, "run", side_effect=git_run_side_effect
                    ), patch.object(cli, "load_ifdr_checkpoint") as loader, patch.object(
                        cli, "run_factor_observer"
                    ) as runner:
                        with self.assertRaises(ValueError):
                            cli._run(args)
                        loader.assert_not_called()
                        runner.assert_not_called()
                    self.assertEqual(
                        {
                            path: path.read_bytes()
                            for path in output.rglob("*")
                            if path.is_file()
                        },
                        protected_files,
                    )

                implementation_mismatch_files = {
                    path: path.read_bytes()
                    for path in output.rglob("*")
                    if path.is_file()
                }
                with patch.object(cli, "_git_commit", return_value="implementation-new"), patch.object(
                    cli, "load_ifdr_checkpoint"
                ) as loader, patch.object(cli, "run_factor_observer") as runner:
                    with self.assertRaises(ValueError):
                        cli._run(args)
                    loader.assert_not_called()
                    runner.assert_not_called()
                self.assertEqual(
                    {
                        path: path.read_bytes()
                        for path in output.rglob("*")
                        if path.is_file()
                    },
                    implementation_mismatch_files,
                )

                provenance["scientific_identity"]["audit_seed"] = 1
                (output / "provenance.json").write_text(
                    json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8"
                )
                frozen_files = {
                    path: path.read_bytes()
                    for path in output.rglob("*")
                    if path.is_file()
                }
                with patch.object(cli, "load_ifdr_checkpoint") as loader, patch.object(
                    cli, "run_factor_observer"
                ):
                    with self.assertRaises(ValueError):
                        cli._run(args)
                    loader.assert_not_called()
                changed_files = {
                    path: path.read_bytes()
                    for path in output.rglob("*")
                    if path.is_file()
                }
                self.assertEqual(changed_files, frozen_files)
                (output / "provenance.json").write_bytes(valid_provenance_bytes)

                args.transform_batch_size = 3
                args.device = "cpu:9"
                self.assertEqual(cli._run(args), 0)
                self.assertEqual((output / "observations.jsonl").read_bytes(), first_root)
                manifest_path = output / "seed-17" / "manifest.json"
                manifest_path.write_bytes(manifest_path.read_bytes() + b"\ncorrupt")
                manifest_files = {
                    path: path.read_bytes()
                    for path in output.rglob("*")
                    if path.is_file()
                }
                with patch.object(cli, "load_ifdr_checkpoint") as loader, patch.object(
                    cli, "run_factor_observer"
                ):
                    with self.assertRaises(ValueError):
                        cli._run(args)
                    loader.assert_not_called()
                self.assertEqual(
                    {
                        path: path.read_bytes()
                        for path in output.rglob("*")
                        if path.is_file()
                    },
                    manifest_files,
                )
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
            with patch.object(cli, "_git_commit", return_value="a" * 40):
                with self.assertRaises(ValueError):
                    cli._run(args)
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["exception_type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
