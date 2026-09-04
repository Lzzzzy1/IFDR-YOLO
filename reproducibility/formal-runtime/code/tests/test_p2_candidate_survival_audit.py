from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ifdr_yolo.experiments.p2_candidate_survival_audit import (
    AssignmentAuditObserver,
    AuditIdentity,
    PRE_REGISTERED_CONTRASTS,
    MODERATE_STRATA,
    build_cyclist_image_cluster_frame,
    bootstrap_cyclist_zero_p2_contrasts,
    derive_level_slices,
    validate_level_strides,
    run_fit_assignment_audit,
    _normalize_device,
    run_synthetic_audit,
    summarize_zero_p2_positive,
)


class P2CandidateSurvivalAuditTest(unittest.TestCase):
    @staticmethod
    def _identity(fit_ids: tuple[str, ...], development_ids: tuple[str, ...], *, fit_override: str | None = None) -> AuditIdentity:
        def digest(values: tuple[str, ...]) -> str:
            return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()

        return AuditIdentity.from_mapping({
            "fit_ids_sha256": fit_override or digest(fit_ids),
            "development_ids_sha256": digest(development_ids),
            "checkpoint_sha256": "c" * 64,
            "config_sha256": "d" * 64,
            "code_sha256": "e" * 64,
        })

    @staticmethod
    def _assignment_inputs() -> tuple[object, ...]:
        return (
            torch.tensor([[[0.95], [0.80], [0.70], [0.60]]]),
            torch.tensor(
                [
                    [
                        [0.0, 0.0, 4.0, 4.0],
                        [4.0, 0.0, 8.0, 4.0],
                        [8.0, 0.0, 12.0, 4.0],
                        [12.0, 0.0, 16.0, 4.0],
                    ]
                ]
            ),
            torch.tensor([[2.0, 2.0], [6.0, 2.0], [10.0, 2.0], [14.0, 2.0]]),
            torch.tensor([[[0], [0]]]),
            torch.tensor([[[0.0, 0.0, 8.0, 4.0], [8.0, 0.0, 16.0, 4.0]]]),
            torch.tensor([[[True], [True]]]),
        )

    def test_level_slices_are_dynamic_and_cover_flattened_anchors(self) -> None:
        features = [
            torch.zeros(1, 1, 2, 3),
            torch.zeros(1, 1, 1, 2),
            (1, 1, 1, 1),
            (1, 1, 1, 1),
        ]
        slices = derive_level_slices(features)
        self.assertEqual([(item.name, item.start, item.stop) for item in slices], [
            ("P2", 0, 6),
            ("P3", 6, 8),
            ("P4", 8, 9),
            ("P5", 9, 10),
        ])

    def test_observer_replays_original_assigner_and_returns_original_outputs(self) -> None:
        from ultralytics.utils.tal import TaskAlignedAssigner

        assigner = TaskAlignedAssigner(topk=2, num_classes=1, stride=[4, 8, 16, 32])
        inputs = self._assignment_inputs()
        slices = derive_level_slices([(1, 1, 1, 1)] * 4)
        observer = AssignmentAuditObserver(assigner, enabled=True, level_slices=slices)
        observer.set_metadata(image_ids=("000001",))
        before = tuple(value.clone() for value in inputs)
        rng_before = torch.get_rng_state()
        observer.attach()
        try:
            observed = assigner(*inputs)
        finally:
            observer.detach()
        rng_after = torch.get_rng_state()

        expected = assigner(*inputs)
        self.assertEqual(len(observed), len(expected))
        for left, right in zip(observed, expected):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.equal(rng_before, rng_after))
        for original, current in zip(before, inputs):
            self.assertTrue(torch.equal(original, current))
        self.assertEqual(len(observer.records), 8)
        sample = observer.records[0]
        self.assertEqual(sample["image_id"], "000001")
        self.assertIn("max_iou", sample)
        self.assertIn("max_tal_ciou", sample)
        self.assertIn("max_task_alignment", sample)
        self.assertIn("best_alignment_rank", sample)
        self.assertIn("rank_low", sample)
        self.assertIn("rank_high", sample)
        self.assertIn("tie_count", sample)

    def test_audit_disabled_has_no_records_and_keeps_assignments(self) -> None:
        from ultralytics.utils.tal import TaskAlignedAssigner

        assigner = TaskAlignedAssigner(topk=2, num_classes=1, stride=[4, 8, 16, 32])
        inputs = self._assignment_inputs()
        observer = AssignmentAuditObserver(
            assigner,
            enabled=False,
            level_slices=derive_level_slices([(1, 1, 1, 1)] * 4),
        )
        observer.attach()
        try:
            result = assigner(*inputs)
        finally:
            observer.detach()
        self.assertEqual(observer.records, [])
        self.assertIsInstance(result, tuple)

    def test_zero_alignment_records_an_explicit_rank_interval(self) -> None:
        from ultralytics.utils.tal import TaskAlignedAssigner

        assigner = TaskAlignedAssigner(topk=2, num_classes=1, stride=[4, 8, 16, 32])
        inputs = list(self._assignment_inputs())
        inputs[0] = torch.zeros_like(inputs[0])
        observer = AssignmentAuditObserver(
            assigner,
            enabled=True,
            level_slices=derive_level_slices([(1, 1, 1, 1)] * 4),
        )
        observer.attach()
        try:
            assigner(*inputs)
        finally:
            observer.detach()
        zero_rows = [row for row in observer.records if row["legal_anchor_count"] > 0]
        self.assertTrue(zero_rows)
        for row in zero_rows:
            self.assertIsNone(row["best_alignment_rank"])
            self.assertGreaterEqual(row["rank_low"], 1)
            self.assertGreaterEqual(row["rank_high"], row["rank_low"])
            self.assertGreater(row["tie_count"], 0)

    def test_empty_gt_batch_returns_empty_records_and_original_output(self) -> None:
        from ultralytics.utils.tal import TaskAlignedAssigner

        assigner = TaskAlignedAssigner(topk=2, num_classes=1, stride=[4, 8, 16, 32])
        inputs = list(self._assignment_inputs())
        inputs[3] = torch.empty((1, 0, 1), dtype=torch.long)
        inputs[4] = torch.empty((1, 0, 4), dtype=torch.float32)
        inputs[5] = torch.empty((1, 0, 1), dtype=torch.bool)
        observer = AssignmentAuditObserver(
            assigner,
            enabled=True,
            level_slices=derive_level_slices([(1, 1, 1, 1)] * 4),
        )
        observer.attach()
        try:
            output = assigner(*inputs)
        finally:
            observer.detach()
        self.assertEqual(observer.records, [])
        self.assertEqual(tuple(output[3].shape), (1, 4))
        self.assertEqual(int(output[3].sum().item()), 0)

    def test_strict_observer_fails_closed_without_metadata(self) -> None:
        from ultralytics.utils.tal import TaskAlignedAssigner

        assigner = TaskAlignedAssigner(topk=2, num_classes=1, stride=[4, 8, 16, 32])
        observer = AssignmentAuditObserver(
            assigner,
            enabled=True,
            strict=True,
            level_slices=derive_level_slices([(1, 1, 1, 1)] * 4),
            class_names={0: "Car"},
        )
        observer.attach()
        try:
            with self.assertRaisesRegex(ValueError, "image_ids"):
                assigner(*self._assignment_inputs())
        finally:
            observer.detach()

    def test_strict_observer_requires_complete_gt_metadata(self) -> None:
        from ultralytics.utils.tal import TaskAlignedAssigner

        assigner = TaskAlignedAssigner(topk=2, num_classes=1, stride=[4, 8, 16, 32])
        observer = AssignmentAuditObserver(
            assigner,
            enabled=True,
            strict=True,
            level_slices=derive_level_slices([(1, 1, 1, 1)] * 4),
            class_names={0: "Car"},
        )
        observer.set_metadata(image_ids=("000001",), gt_strata=[["moderate"]], gt_metadata=None)
        observer.attach()
        try:
            with self.assertRaisesRegex(ValueError, "complete GT metadata"):
                assigner(*self._assignment_inputs())
        finally:
            observer.detach()

    def test_level_strides_are_bound_and_wrong_order_rejected(self) -> None:
        self.assertEqual(validate_level_strides([4, 8, 16, 32]), (4, 8, 16, 32))
        with self.assertRaisesRegex(ValueError, "stride"):
            validate_level_strides([8, 4, 16, 32])

    def test_numeric_device_is_normalized_for_ultralytics(self) -> None:
        self.assertEqual(_normalize_device("0"), "cuda:0")
        self.assertEqual(_normalize_device("cpu"), "cpu")

    def test_formal_runner_uses_fit_only_identity_and_resume_artifacts(self) -> None:
        from ultralytics.utils.tal import TaskAlignedAssigner

        fit_ids = ("000001", "000002", "000003")
        development_ids = ("000004",)
        assignment_inputs = self._assignment_inputs()

        class FakeModel(torch.nn.Module):
            strides = (4, 8, 16, 32)

            def forward(self, images):
                return {"feats": [torch.zeros(images.shape[0], 1, 1, 1)] * 4}

        class FakeCriterion:
            def __init__(self):
                self.assigner = TaskAlignedAssigner(topk=2, num_classes=1, stride=[4, 8, 16, 32])

            def __call__(self, _preds, batch_data):
                size = int(batch_data["img"].shape[0])
                inputs = list(assignment_inputs)
                for index in (0, 1, 3, 4, 5):
                    inputs[index] = inputs[index].repeat((size,) + (1,) * (inputs[index].ndim - 1))
                self.assigner(*inputs)
                return torch.tensor([1.0]), torch.tensor([1.0])

        class FakeRuntime:
            def load_model(self, _checkpoint, _device):
                return FakeModel()

            def build_criterion(self, _model):
                return FakeCriterion()

            def build_loader(self, *, image_ids, **_kwargs):
                batches = []
                for image_id in image_ids:
                    batches.append(
                        {
                            "img": torch.zeros(1, 3, 4, 4),
                            "image_ids": [image_id],
                            "gt_strata": [["moderate", "moderate"]],
                            "gt_metadata": [[
                                {"moderate_valid": True, "height_px": 30.0, "depth_m": 50.0, "small_25_40": True, "large_gt_80": False, "far_gt_40m": True, "near_0_20m": False},
                                {"moderate_valid": True, "height_px": 100.0, "depth_m": 10.0, "small_25_40": False, "large_gt_80": True, "far_gt_40m": False, "near_0_20m": True},
                            ]],
                            "class_names": {0: "Car"},
                        }
                    )
                return batches

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            resolved_data = root / "resolved.yaml"
            fit_path = root / "fit.txt"
            development_path = root / "development.txt"
            checkpoint = root / "last.pt"
            labels = root / "labels"
            for path in (config_path, resolved_data, checkpoint):
                path.write_bytes(b"fixture")
            image_dir = root / "images" / "train"
            image_dir.mkdir(parents=True)
            for image_id in fit_ids:
                (image_dir / f"{image_id}.png").write_bytes(f"image-{image_id}".encode("ascii"))
            resolved_data.write_text(f"path: '{root.as_posix()}'\ntrain: images/train\n", encoding="utf-8")
            fit_path.write_text("\n".join(fit_ids) + "\n", encoding="utf-8")
            development_path.write_text("000004\n", encoding="utf-8")
            labels.mkdir()
            split = SimpleNamespace(fit_ids=fit_ids, development_ids=development_ids)
            config = SimpleNamespace(paths=SimpleNamespace(raw_images=root), training=SimpleNamespace(imgsz=4))
            with patch("ifdr_yolo.experiments.p2_candidate_survival_audit.load_baseline_config", return_value=config), patch(
                "ifdr_yolo.experiments.p2_candidate_survival_audit.validate_plain_p2_model"
            ), patch("ifdr_yolo.experiments.p2_candidate_survival_audit.validate_fit_development_split", return_value=split), patch(
                "ifdr_yolo.experiments.p2_candidate_survival_audit.validate_primary_checkpoint", return_value=checkpoint
            ), patch("ifdr_yolo.experiments.p2_candidate_survival_audit._fit_image_manifest_sha256", return_value="f" * 64), patch(
                "ifdr_yolo.experiments.p2_candidate_survival_audit._upstream_source_hashes",
                return_value=("u" * 64, {name: "a" * 64 for name in ("tal.py", "loss.py", "dataset.py", "augment.py")}),
            ):
                result = run_fit_assignment_audit(
                    config_path=config_path,
                    resolved_data_path=resolved_data,
                    fit_ids_path=fit_path,
                    development_ids_path=development_path,
                    checkpoint_path=checkpoint,
                    raw_label_dir=labels,
                    output_dir=root / "full",
                    mirror_dir=root / "full-mirror",
                    mode="smoke",
                    expected_checkpoint_sha256=hashlib.sha256(b"fixture").hexdigest(),
                    runtime=FakeRuntime(),
                )
            self.assertEqual(result["processed_fit_count"], 2)
            self.assertEqual(result["actual_loader_fit_count"], 2)
            self.assertEqual(result["intersection_count"], 0)
            self.assertNotIn("identity-json", json.dumps(result))
            self.assertEqual(result["identity"]["expected_checkpoint_sha256"], hashlib.sha256(b"fixture").hexdigest())
            self.assertIn("fit_image_manifest_sha256", result["identity"])
            self.assertIn("upstream_source_sha256", result["identity"])

    def test_formal_runner_rejects_wrong_checkpoint_sha256(self) -> None:
        fit_ids = ("000001", "000002", "000003")
        development_ids = ("000004",)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            resolved_data = root / "resolved.yaml"
            fit_path = root / "fit.txt"
            development_path = root / "development.txt"
            checkpoint = root / "last.pt"
            labels = root / "labels"
            for path in (config_path, checkpoint):
                path.write_bytes(b"fixture")
            resolved_data.write_text("path: .\ntrain: .\n", encoding="utf-8")
            fit_path.write_text("\n".join(fit_ids) + "\n", encoding="utf-8")
            development_path.write_text("000004\n", encoding="utf-8")
            labels.mkdir()
            split = SimpleNamespace(fit_ids=fit_ids, development_ids=development_ids)
            config = SimpleNamespace(paths=SimpleNamespace(raw_images=root), training=SimpleNamespace(imgsz=4))
            with patch("ifdr_yolo.experiments.p2_candidate_survival_audit.load_baseline_config", return_value=config), patch(
                "ifdr_yolo.experiments.p2_candidate_survival_audit.validate_plain_p2_model"
            ), patch("ifdr_yolo.experiments.p2_candidate_survival_audit.validate_fit_development_split", return_value=split), patch(
                "ifdr_yolo.experiments.p2_candidate_survival_audit.validate_primary_checkpoint", return_value=checkpoint
            ), patch("ifdr_yolo.experiments.p2_candidate_survival_audit._fit_image_manifest_sha256", return_value="f" * 64), patch(
                "ifdr_yolo.experiments.p2_candidate_survival_audit._upstream_source_hashes",
                return_value=("u" * 64, {name: "a" * 64 for name in ("tal.py", "loss.py", "dataset.py", "augment.py")}),
            ):
                with self.assertRaisesRegex(ValueError, "checkpoint SHA256 mismatch"):
                    run_fit_assignment_audit(
                        config_path=config_path,
                        resolved_data_path=resolved_data,
                        fit_ids_path=fit_path,
                        development_ids_path=development_path,
                        checkpoint_path=checkpoint,
                        expected_checkpoint_sha256="0" * 64,
                        raw_label_dir=labels,
                        output_dir=root / "out",
                        mirror_dir=root / "mirror",
                        mode="smoke",
                        runtime=object(),
                    )

    def test_formal_interrupted_resume_matches_uninterrupted_bytes(self) -> None:
        from ultralytics.utils.tal import TaskAlignedAssigner

        fit_ids = ("000001", "000002", "000003", "000004")
        development_ids = ("000005",)
        assignment_inputs = self._assignment_inputs()

        class FakeModel(torch.nn.Module):
            strides = (4, 8, 16, 32)

            def forward(self, images):
                return {"feats": [torch.zeros(images.shape[0], 1, 1, 1)] * 4}

        class FakeCriterion:
            def __init__(self):
                self.assigner = TaskAlignedAssigner(topk=2, num_classes=1, stride=[4, 8, 16, 32])

            def __call__(self, _preds, batch_data):
                size = int(batch_data["img"].shape[0])
                inputs = list(assignment_inputs)
                for index in (0, 1, 3, 4, 5):
                    inputs[index] = inputs[index].repeat((size,) + (1,) * (inputs[index].ndim - 1))
                self.assigner(*inputs)
                return torch.tensor([1.0]), torch.tensor([1.0])

        class FakeRuntime:
            def load_model(self, _checkpoint, _device):
                return FakeModel()

            def build_criterion(self, _model):
                return FakeCriterion()

            def build_loader(self, *, image_ids, batch_size=1, **_kwargs):
                return [
                    {
                        "img": torch.zeros(len(group), 3, 4, 4),
                        "image_ids": list(group),
                            "gt_strata": [["moderate", "moderate"] for _ in group],
                            "gt_metadata": [[
                                {"moderate_valid": True, "height_px": 30.0, "depth_m": 50.0, "small_25_40": True, "large_gt_80": False, "far_gt_40m": True, "near_0_20m": False},
                                {"moderate_valid": True, "height_px": 100.0, "depth_m": 10.0, "small_25_40": False, "large_gt_80": True, "far_gt_40m": False, "near_0_20m": True},
                            ] for _ in group],
                        "class_names": {0: "Car"},
                    }
                    for group in (tuple(image_ids[index : index + batch_size]) for index in range(0, len(image_ids), batch_size))
                ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            resolved_data = root / "resolved.yaml"
            fit_path = root / "fit.txt"
            development_path = root / "development.txt"
            checkpoint = root / "last.pt"
            labels = root / "labels"
            for path in (config_path, resolved_data, checkpoint):
                path.write_bytes(b"fixture")
            image_dir = root / "images" / "train"
            image_dir.mkdir(parents=True)
            for image_id in fit_ids:
                (image_dir / f"{image_id}.png").write_bytes(f"image-{image_id}".encode("ascii"))
            resolved_data.write_text(f"path: '{root.as_posix()}'\ntrain: images/train\n", encoding="utf-8")
            fit_path.write_text("\n".join(fit_ids) + "\n", encoding="utf-8")
            development_path.write_text("000005\n", encoding="utf-8")
            labels.mkdir()
            split = SimpleNamespace(fit_ids=fit_ids, development_ids=development_ids)
            config = SimpleNamespace(paths=SimpleNamespace(raw_images=root), training=SimpleNamespace(imgsz=4))
            common = dict(
                config_path=config_path,
                resolved_data_path=resolved_data,
                fit_ids_path=fit_path,
                development_ids_path=development_path,
                checkpoint_path=checkpoint,
                expected_checkpoint_sha256=hashlib.sha256(b"fixture").hexdigest(),
                raw_label_dir=labels,
                mode="full",
                batch=2,
                runtime=FakeRuntime(),
            )
            with patch("ifdr_yolo.experiments.p2_candidate_survival_audit.load_baseline_config", return_value=config), patch(
                "ifdr_yolo.experiments.p2_candidate_survival_audit.validate_plain_p2_model"
            ), patch("ifdr_yolo.experiments.p2_candidate_survival_audit.validate_fit_development_split", return_value=split), patch(
                "ifdr_yolo.experiments.p2_candidate_survival_audit.validate_primary_checkpoint", return_value=checkpoint
            ), patch("ifdr_yolo.experiments.p2_candidate_survival_audit._fit_image_manifest_sha256", return_value="f" * 64), patch(
                "ifdr_yolo.experiments.p2_candidate_survival_audit._upstream_source_hashes",
                return_value=("u" * 64, {name: "a" * 64 for name in ("tal.py", "loss.py", "dataset.py", "augment.py")}),
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    run_fit_assignment_audit(output_dir=root / "resumed", mirror_dir=root / "resumed-mirror", stop_after=2, **common)
                checkpoint_payload = json.loads((root / "resumed" / "checkpoint.json").read_text(encoding="utf-8"))
                self.assertEqual(checkpoint_payload["completed_image_ids"], ["000001", "000002"])
                with (root / "resumed" / "assignment_audit.jsonl").open("ab") as stream:
                    stream.write(b"legal post-checkpoint tail")
                run_fit_assignment_audit(output_dir=root / "resumed", mirror_dir=root / "resumed-mirror", resume=True, **common)
                run_fit_assignment_audit(output_dir=root / "uninterrupted", mirror_dir=root / "uninterrupted-mirror", **common)
            names = ("assignment_audit.jsonl", "checkpoint.json", "summary.json", "summary.csv", "manifest.json")
            for name in names:
                if name == "checkpoint.json":
                    resumed_checkpoint = json.loads((root / "resumed" / name).read_text(encoding="utf-8"))
                    uninterrupted_checkpoint = json.loads((root / "uninterrupted" / name).read_text(encoding="utf-8"))
                    self.assertGreaterEqual(float(resumed_checkpoint["elapsed_seconds"]), 0.0)
                    self.assertGreaterEqual(float(uninterrupted_checkpoint["elapsed_seconds"]), 0.0)
                    resumed_checkpoint.pop("elapsed_seconds", None)
                    uninterrupted_checkpoint.pop("elapsed_seconds", None)
                    self.assertEqual(resumed_checkpoint, uninterrupted_checkpoint, name)
                elif name == "manifest.json":
                    resumed_manifest = json.loads((root / "resumed" / name).read_text(encoding="utf-8"))
                    uninterrupted_manifest = json.loads((root / "uninterrupted" / name).read_text(encoding="utf-8"))
                    resumed_manifest["files"].pop("checkpoint.json", None)
                    uninterrupted_manifest["files"].pop("checkpoint.json", None)
                    self.assertEqual(resumed_manifest, uninterrupted_manifest, name)
                else:
                    self.assertEqual((root / "resumed" / name).read_bytes(), (root / "uninterrupted" / name).read_bytes(), name)

            with patch("ifdr_yolo.experiments.p2_candidate_survival_audit.load_baseline_config", return_value=config), patch(
                "ifdr_yolo.experiments.p2_candidate_survival_audit.validate_plain_p2_model"
            ), patch("ifdr_yolo.experiments.p2_candidate_survival_audit.validate_fit_development_split", return_value=split), patch(
                "ifdr_yolo.experiments.p2_candidate_survival_audit.validate_primary_checkpoint", return_value=checkpoint
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    run_fit_assignment_audit(output_dir=root / "window", mirror_dir=root / "window-mirror", stop_after=2, **common)
                latest = json.loads((root / "window" / "checkpoint.json").read_text(encoding="utf-8"))
                old = dict(latest)
                old["completed_image_ids"] = []
                old["next_position"] = 0
                old["journal_offset"] = 0
                old["journal_prefix_sha256"] = hashlib.sha256(b"").hexdigest()
                (root / "window-mirror" / "checkpoint.json").write_text(json.dumps(old, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                recovered = run_fit_assignment_audit(output_dir=root / "window", mirror_dir=root / "window-mirror", resume=True, **common)
                self.assertEqual(recovered["state"], "complete")
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    run_fit_assignment_audit(output_dir=root / "tampered", mirror_dir=root / "tampered-mirror", stop_after=2, **common)
                tampered_journal = root / "tampered" / "assignment_audit.jsonl"
                tampered_journal.write_bytes(tampered_journal.read_bytes().replace(b'"image_id":"000001"', b'"image_id":"999999"', 1))
                with self.assertRaisesRegex(ValueError, "prefix"):
                    run_fit_assignment_audit(output_dir=root / "tampered", mirror_dir=root / "tampered-mirror", resume=True, **common)

    def test_zero_p2_positive_summary_uses_legal_p2_denominator(self) -> None:
        records = [
            {
                "image_id": "a",
                "gt_index": 0,
                "class_name": "Cyclist",
                "small_25_40": True,
                "large_gt_80": False,
                "far_gt_40m": True,
                "near_0_20m": False,
                "level": "P2",
                "legal_anchor_count": 2,
                "assigned_positive_count": 0,
            },
            {
                "image_id": "b",
                "gt_index": 0,
                "class_name": "Cyclist",
                "small_25_40": False,
                "large_gt_80": True,
                "far_gt_40m": False,
                "near_0_20m": True,
                "level": "P2",
                "legal_anchor_count": 1,
                "assigned_positive_count": 1,
            },
            {
                "image_id": "c",
                "gt_index": 0,
                "class_name": "Pedestrian",
                "small_25_40": True,
                "large_gt_80": False,
                "far_gt_40m": False,
                "near_0_20m": False,
                "level": "P2",
                "legal_anchor_count": 0,
                "assigned_positive_count": 0,
            },
        ]
        summary = summarize_zero_p2_positive(records)
        self.assertEqual(summary["Cyclist"]["small_25_40"]["denominator"], 1)
        self.assertEqual(summary["Cyclist"]["small_25_40"]["zero_positive"], 1)
        self.assertEqual(summary["Cyclist"]["large_gt_80"]["denominator"], 1)
        self.assertEqual(summary["Pedestrian"]["small_25_40"]["denominator"], 0)
        self.assertEqual(set(summary["Cyclist"]), set(MODERATE_STRATA))

    def test_summary_declares_independent_strata_and_fixed_contrasts(self) -> None:
        records = [
            {
                "image_id": "a",
                "gt_index": 0,
                "class_name": "Pedestrian",
                "level": "P2",
                "legal_anchor_count": 1,
                "assigned_positive_count": 0,
                "small_25_40": True,
                "large_gt_80": False,
                "far_gt_40m": True,
                "near_0_20m": False,
            }
        ]
        summary = summarize_zero_p2_positive(records)
        self.assertEqual(summary["Pedestrian"]["small_25_40"]["rate"], 1.0)
        self.assertEqual(summary["Pedestrian"]["far_gt_40m"]["rate"], 1.0)
        self.assertEqual(summary["Pedestrian"]["large_gt_80"]["denominator"], 0)
        self.assertEqual(summary["Pedestrian"]["near_0_20m"]["denominator"], 0)
        self.assertEqual(
            {item["name"] for item in PRE_REGISTERED_CONTRASTS},
            {"small_25_40-vs-large_gt_80", "far_gt_40m-vs-near_0_20m"},
        )
        for contrast in PRE_REGISTERED_CONTRASTS:
            self.assertEqual(contrast["estimand"], "target_zero_p2_positive_rate - control_zero_p2_positive_rate")
            self.assertTrue(contrast["target_worse_positive"])

    def test_cyclist_frame_keeps_zero_eligible_fit_images_and_independent_flags(self) -> None:
        records = [
            {
                "image_id": "000001",
                "gt_index": 0,
                "class_name": "Cyclist",
                "moderate_valid": True,
                "level": "P2",
                "small_25_40": True,
                "large_gt_80": False,
                "far_gt_40m": True,
                "near_0_20m": False,
                "legal_anchor_count": 2,
                "assigned_positive_count": 0,
            },
            {
                "image_id": "000001",
                "gt_index": 0,
                "class_name": "Cyclist",
                "moderate_valid": True,
                "level": "P3",
                "small_25_40": True,
                "large_gt_80": False,
                "far_gt_40m": True,
                "near_0_20m": False,
                "legal_anchor_count": 1,
                "assigned_positive_count": 1,
            },
        ]
        frame = build_cyclist_image_cluster_frame(records, ("000001", "000002"))
        self.assertEqual([item["image_id"] for item in frame], ["000001", "000002"])
        self.assertEqual(frame[0]["strata"]["small_25_40"]["P2"], {"numerator": 1, "denominator": 1})
        self.assertEqual(frame[0]["strata"]["far_gt_40m"]["P2"], {"numerator": 1, "denominator": 1})
        self.assertEqual(frame[1]["strata"]["small_25_40"]["P2"], {"numerator": 0, "denominator": 0})

    def test_cyclist_bootstrap_is_ratio_of_sums_and_reports_bonferroni_ci(self) -> None:
        frame = [
            {
                "image_id": "a",
                "strata": {
                    name: {level: {"numerator": 0, "denominator": 0} for level in ("P2", "P3", "P4", "P5")}
                    for name in MODERATE_STRATA
                },
            },
            {
                "image_id": "b",
                "strata": {
                    name: {level: {"numerator": 0, "denominator": 0} for level in ("P2", "P3", "P4", "P5")}
                    for name in MODERATE_STRATA
                },
            },
        ]
        for image, target_num, control_num in ((frame[0], 1, 0), (frame[1], 0, 1)):
            for level in ("P2", "P3", "P4", "P5"):
                image["strata"]["small_25_40"][level] = {"numerator": target_num, "denominator": 1}
                image["strata"]["large_gt_80"][level] = {"numerator": control_num, "denominator": 1}
                image["strata"]["far_gt_40m"][level] = {"numerator": target_num, "denominator": 1}
                image["strata"]["near_0_20m"][level] = {"numerator": control_num, "denominator": 1}
        result = bootstrap_cyclist_zero_p2_contrasts(
            frame,
            reps=100,
            seed=20260812,
            journal_sha256="a" * 64,
            identity_sha256="b" * 64,
            manifest_sha256="c" * 64,
        )
        self.assertAlmostEqual(result["contrasts"]["small_25_40-vs-large_gt_80"]["observed"], 0.0)
        self.assertEqual(result["input_hashes"], {"journal_sha256": "a" * 64, "identity_sha256": "b" * 64, "manifest_sha256": "c" * 64})
        self.assertEqual(result["bootstrap_replicates"], 100)
        self.assertEqual(len(result["contrasts"]["far_gt_40m-vs-near_0_20m"]["ci95"]), 2)
        self.assertEqual(len(result["contrasts"]["far_gt_40m-vs-near_0_20m"]["ci97_5_bonferroni"]), 2)
        self.assertTrue(result["same_sampled_image_indices_for_contrasts"])

    def test_cyclist_bootstrap_fails_closed_when_a_draw_lacks_stratum_denominator(self) -> None:
        def empty_strata() -> dict[str, dict[str, dict[str, int]]]:
            return {
                name: {level: {"numerator": 0, "denominator": 0} for level in ("P2", "P3", "P4", "P5")}
                for name in MODERATE_STRATA
            }

        target_only = {"image_id": "target", "strata": empty_strata()}
        control_only = {"image_id": "control", "strata": empty_strata()}
        for level in ("P2", "P3", "P4", "P5"):
            target_only["strata"]["small_25_40"][level] = {"numerator": 1, "denominator": 1}
            control_only["strata"]["large_gt_80"][level] = {"numerator": 0, "denominator": 1}
            target_only["strata"]["far_gt_40m"][level] = {"numerator": 1, "denominator": 1}
            control_only["strata"]["near_0_20m"][level] = {"numerator": 0, "denominator": 1}
        with self.assertRaisesRegex(ValueError, "zero bootstrap denominator"):
            bootstrap_cyclist_zero_p2_contrasts(
                [target_only, control_only],
                reps=1,
                seed=20260812,
                journal_sha256="a" * 64,
                identity_sha256="b" * 64,
                manifest_sha256="c" * 64,
            )

    def test_formal_runner_rejects_unsupported_ultralytics_version(self) -> None:
        fit_ids = ("000001", "000002", "000003")
        development_ids = ("000004",)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            resolved_data = root / "resolved.yaml"
            fit_path = root / "fit.txt"
            development_path = root / "development.txt"
            checkpoint = root / "last.pt"
            labels = root / "labels"
            for path in (config_path, checkpoint):
                path.write_bytes(b"fixture")
            resolved_data.write_text("path: .\ntrain: .\n", encoding="utf-8")
            fit_path.write_text("\n".join(fit_ids) + "\n", encoding="utf-8")
            development_path.write_text("000004\n", encoding="utf-8")
            labels.mkdir()
            split = SimpleNamespace(fit_ids=fit_ids, development_ids=development_ids)
            config = SimpleNamespace(paths=SimpleNamespace(raw_images=root), training=SimpleNamespace(imgsz=4))
            with patch("ifdr_yolo.experiments.p2_candidate_survival_audit.load_baseline_config", return_value=config), patch(
                "ifdr_yolo.experiments.p2_candidate_survival_audit.validate_plain_p2_model"
            ), patch("ifdr_yolo.experiments.p2_candidate_survival_audit.validate_fit_development_split", return_value=split), patch(
                "ifdr_yolo.experiments.p2_candidate_survival_audit.validate_primary_checkpoint", return_value=checkpoint
            ), patch("ifdr_yolo.experiments.p2_candidate_survival_audit.importlib.metadata.version", return_value="8.4.99"):
                with self.assertRaisesRegex(ValueError, "Ultralytics version"):
                    run_fit_assignment_audit(
                        config_path=config_path,
                        resolved_data_path=resolved_data,
                        fit_ids_path=fit_path,
                        development_ids_path=development_path,
                        checkpoint_path=checkpoint,
                        expected_checkpoint_sha256=hashlib.sha256(b"fixture").hexdigest(),
                        raw_label_dir=labels,
                        output_dir=root / "out",
                        mirror_dir=root / "mirror",
                        mode="smoke",
                        runtime=object(),
                    )

    def test_cyclist_bootstrap_zero_denominator_fails_closed_and_smoke_is_not_evaluated(self) -> None:
        frame = [
            {
                "image_id": "a",
                "strata": {
                    name: {level: {"numerator": 0, "denominator": 0} for level in ("P2", "P3", "P4", "P5")}
                    for name in MODERATE_STRATA
                },
            }
        ]
        with self.assertRaisesRegex(ValueError, "zero aggregate denominator"):
            bootstrap_cyclist_zero_p2_contrasts(
                frame,
                reps=10,
                journal_sha256="a" * 64,
                identity_sha256="b" * 64,
                manifest_sha256="c" * 64,
            )
        smoke = bootstrap_cyclist_zero_p2_contrasts(
            frame,
            reps=10,
            full_dataset=False,
            journal_sha256="a" * 64,
            identity_sha256="b" * 64,
            manifest_sha256="c" * 64,
        )
        self.assertEqual(smoke["gate_state"], "not_evaluated_smoke")

    def test_cyclist_bootstrap_cross_level_specificity_is_not_collapsed(self) -> None:
        frame = [
            {
                "image_id": "a",
                "strata": {
                    name: {level: {"numerator": 0, "denominator": 0} for level in ("P2", "P3", "P4", "P5")}
                    for name in MODERATE_STRATA
                },
            }
        ]
        for name, numerator in (("small_25_40", 1), ("large_gt_80", 0), ("far_gt_40m", 1), ("near_0_20m", 0)):
            frame[0]["strata"][name]["P2"] = {"numerator": numerator, "denominator": 1}
            for level in ("P3", "P4", "P5"):
                frame[0]["strata"][name][level] = {"numerator": 0, "denominator": 1}
        result = bootstrap_cyclist_zero_p2_contrasts(
            frame,
            reps=50,
            journal_sha256="a" * 64,
            identity_sha256="b" * 64,
            manifest_sha256="c" * 64,
        )
        self.assertIn("P2", result["cross_level_specificity"])
        self.assertIn("P5", result["cross_level_specificity"])
        self.assertFalse(result["shared_all_levels"]["small_25_40-vs-large_gt_80"])

    def test_runner_rejects_fit_development_overlap(self) -> None:
        identity = self._identity(("a",), ("a",))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "overlap"):
                run_synthetic_audit(
                    fit_ids=("a",),
                    development_ids=("a",),
                    output_dir=Path(directory) / "job",
                    mirror_dir=Path(directory) / "mirror",
                    identity=identity,
                    expected_fit_count=1,
                    expected_development_count=1,
                )

    def test_runner_checkpoint_resume_and_mirror_are_atomic(self) -> None:
        fit_ids = ("000001", "000002", "000003")
        development_ids = ("000004",)
        identity = self._identity(fit_ids, development_ids)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "job"
            mirror = root / "mirror"
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_synthetic_audit(
                    fit_ids=fit_ids,
                    development_ids=development_ids,
                    output_dir=output,
                    mirror_dir=mirror,
                    identity=identity,
                    expected_fit_count=3,
                    expected_development_count=1,
                    stop_after=2,
                )
            checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["completed_image_ids"], ["000001", "000002"])
            self.assertTrue((mirror / "checkpoint.json").exists())
            self.assertEqual((output / "checkpoint.json").read_bytes(), (mirror / "checkpoint.json").read_bytes())
            self.assertEqual((output / "audit.jsonl").read_bytes(), (mirror / "audit.jsonl").read_bytes())
            tail = json.dumps({"identity_sha256": identity.sha256, "image_id": "tail"}, sort_keys=True) + "\n"
            with (output / "audit.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(tail)
            result = run_synthetic_audit(
                fit_ids=fit_ids,
                development_ids=development_ids,
                output_dir=output,
                mirror_dir=mirror,
                identity=identity,
                expected_fit_count=3,
                expected_development_count=1,
                resume=True,
            )
            self.assertEqual(result["state"], "complete")
            self.assertEqual(
                (output / "audit.jsonl").read_bytes(), (mirror / "audit.jsonl").read_bytes()
            )
            self.assertNotIn(b"tail", (output / "audit.jsonl").read_bytes())
            self.assertTrue((output / "manifest.json").exists())

    def test_resume_rejects_tampered_mirror_journal_prefix(self) -> None:
        fit_ids = ("000001", "000002")
        development_ids = ("000003",)
        identity = self._identity(fit_ids, development_ids)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                run_synthetic_audit(
                    fit_ids=fit_ids,
                    development_ids=development_ids,
                    output_dir=root / "job",
                    mirror_dir=root / "mirror",
                    identity=identity,
                    expected_fit_count=2,
                    expected_development_count=1,
                    stop_after=1,
                )
            mirror_journal = root / "mirror" / "audit.jsonl"
            mirror_journal.write_bytes(mirror_journal.read_bytes().replace(b"000001", b"tampered", 1))
            with self.assertRaisesRegex(ValueError, "journal|prefix"):
                run_synthetic_audit(
                    fit_ids=fit_ids,
                    development_ids=development_ids,
                    output_dir=root / "job",
                    mirror_dir=root / "mirror",
                    identity=identity,
                    expected_fit_count=2,
                    expected_development_count=1,
                    resume=True,
                )

    def test_resume_rejects_identity_mismatch(self) -> None:
        fit_ids = ("a", "b")
        development_ids = ("c",)
        identity = self._identity(fit_ids, development_ids)
        changed = self._identity(fit_ids, development_ids, fit_override="f" * 64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                run_synthetic_audit(
                    fit_ids=fit_ids,
                    development_ids=development_ids,
                    output_dir=root / "job",
                    mirror_dir=root / "mirror",
                    identity=identity,
                    expected_fit_count=2,
                    expected_development_count=1,
                    stop_after=1,
                )
            with self.assertRaisesRegex(ValueError, "identity"):
                run_synthetic_audit(
                    fit_ids=fit_ids,
                    development_ids=development_ids,
                    output_dir=root / "job",
                    mirror_dir=root / "mirror",
                    identity=changed,
                    expected_fit_count=2,
                    expected_development_count=1,
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
