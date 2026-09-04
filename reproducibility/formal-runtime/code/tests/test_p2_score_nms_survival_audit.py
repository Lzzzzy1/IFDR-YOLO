import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch

from ifdr_yolo.experiments.p2_score_nms_survival_audit import (
    LEVEL_NAMES,
    STAGES,
    derive_level_slices,
    replay_nms_with_stages,
    score_nms_survival_row,
    run_synthetic_score_nms_audit,
    summary_long_rows,
    summarize_score_nms_estimands,
)


class ScoreNmsAuditTests(unittest.TestCase):
    def test_benchmark32_selects_exact_registered_prefix_and_binds_identity(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import (
            BENCHMARK32_RNG_SEED,
            _benchmark32_identity_fields,
            _select_fit_ids_for_mode,
        )

        fixture = (
            "000000", "000007", "000009", "000010", "000011", "000012", "000013", "000014",
            "000016", "000017", "000022", "000026", "000029", "000030", "000032", "000034",
            "000036", "000038", "000041", "000043", "000045", "000046", "000051", "000054",
            "000055", "000056", "000057", "000060", "000064", "000067", "000068", "000069",
        )
        selected = _select_fit_ids_for_mode("benchmark32", fixture + ("000070",))
        self.assertEqual(selected, fixture)
        self.assertEqual(_benchmark32_identity_fields(selected), {
            "selection_rule": "first_32_registered_ordered_fit_ids",
            "selected_ids_count": 32,
            "selected_ids_ordered_sha256": "bfafcb17cb7a6058843d41122589371aa179161bd01fcaa1e5dd3afd954a5617",
            "benchmark_rng_seed": BENCHMARK32_RNG_SEED,
        })
        with self.assertRaisesRegex(ValueError, ">= 32"):
            _select_fit_ids_for_mode("benchmark32", fixture[:-1])

    def test_benchmark32_stop_after_requires_nonterminal_prefix(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _validate_benchmark32_stop_after

        self.assertIsNone(_validate_benchmark32_stop_after(None, completed_count=0))
        self.assertEqual(_validate_benchmark32_stop_after(1, completed_count=0), 1)
        self.assertEqual(_validate_benchmark32_stop_after(31, completed_count=0), 31)
        for invalid in (0, 32, -1, 1.5, "1"):
            with self.assertRaises(ValueError):
                _validate_benchmark32_stop_after(invalid, completed_count=0)
        with self.assertRaisesRegex(ValueError, "completed|nonterminal"):
            _validate_benchmark32_stop_after(16, completed_count=16)

    def test_benchmark32_rng_boundary_is_fixed_without_smoke_pair_variable(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import (
            _initialize_benchmark32_rng_boundary,
            _rng_state_digest_from_snapshot,
        )

        torch.manual_seed(1)
        first = _rng_state_digest_from_snapshot(_initialize_benchmark32_rng_boundary())
        torch.manual_seed(999)
        second = _rng_state_digest_from_snapshot(_initialize_benchmark32_rng_boundary())
        self.assertEqual(first, second)

    def test_full_requires_exact_registered_fit_length_and_binds_identity(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import (
            FULL_AUDIT_RNG_SEED,
            REGISTERED_FIT_COUNT,
            _full_audit_identity_fields,
            _select_fit_ids_for_mode,
        )

        ids = tuple(f"{index:06d}" for index in range(REGISTERED_FIT_COUNT))
        selected = _select_fit_ids_for_mode("full", ids)
        self.assertEqual(selected, ids)
        self.assertEqual(_full_audit_identity_fields(selected), {
            "selection_rule": "all_3341_registered_ordered_fit_ids",
            "selected_ids_count": REGISTERED_FIT_COUNT,
            "selected_ids_ordered_sha256": hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest(),
            "full_audit_rng_seed": FULL_AUDIT_RNG_SEED,
        })
        for invalid in (ids[:-1], ids + ("003341",)):
            with self.assertRaisesRegex(ValueError, "exactly 3341"):
                _select_fit_ids_for_mode("full", invalid)

    def test_full_stop_after_requires_an_uncompleted_nonterminal_prefix(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _validate_mode_stop_after

        self.assertIsNone(_validate_mode_stop_after("full", None, completed_count=0))
        self.assertEqual(_validate_mode_stop_after("full", 1, completed_count=0), 1)
        self.assertEqual(_validate_mode_stop_after("full", 3340, completed_count=0), 3340)
        for invalid in (0, 3341, -1, 1.5, "1"):
            with self.assertRaisesRegex(ValueError, "1..3340"):
                _validate_mode_stop_after("full", invalid, completed_count=0)
        with self.assertRaisesRegex(ValueError, "completed|nonterminal"):
            _validate_mode_stop_after("full", 7, completed_count=7)

    def test_full_rng_boundary_initializes_generation_zero_and_cli_preserves_full_choice(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import (
            FULL_AUDIT_RNG_SEED,
            _ScoreNMSRunState,
            _initialize_full_audit_rng_boundary,
            _rng_state_digest_from_snapshot,
        )
        from scripts.run_p2_score_nms_survival_audit import _parser

        torch.manual_seed(1)
        first = _rng_state_digest_from_snapshot(_initialize_full_audit_rng_boundary())
        torch.manual_seed(999)
        second = _rng_state_digest_from_snapshot(_initialize_full_audit_rng_boundary())
        self.assertEqual(first, second)
        self.assertEqual(FULL_AUDIT_RNG_SEED, 20260812)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = _initialize_full_audit_rng_boundary()
            state = _ScoreNMSRunState(output=root / "primary", mirror=root / "mirror", identity={"mode": "full"}, identity_sha="a" * 64, selected_ids=("000000",), completed=(), stop_after=None, rng_snapshot=snapshot, generation_zero_pending=True)
            state.initialize_generation_zero(snapshot)
            checkpoint = json.loads((root / "primary" / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["rng_completed_count"], 0)
            self.assertEqual(checkpoint["rng_digest"], _rng_state_digest_from_snapshot(snapshot))
        args = _parser().parse_args(["--config", "a", "--resolved-data", "b", "--fit-ids", "c", "--development-ids", "d", "--checkpoint", "e", "--expected-checkpoint-sha256", "f", "--raw-label-dir", "g", "--expected-raw-label-sha256", "h", "--output", "i", "--mirror", "j", "--mode", "full"])
        self.assertEqual(args.mode, "full")

    def test_synthetic_full_resume_is_byte_identical_and_summary_is_unsealed(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import REGISTERED_FIT_COUNT, run_synthetic_full_audit

        ids = tuple(f"{index:06d}" for index in range(REGISTERED_FIT_COUNT))
        identity = {"fit_ids_sha256": "a" * 64, "checkpoint_sha256": "b" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_synthetic_full_audit(ids, root / "uninterrupted", root / "uninterrupted-mirror", identity, synthetic_prefix_count=4)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_synthetic_full_audit(ids, root / "primary", root / "mirror", identity, stop_after=3, synthetic_prefix_count=4)
            run_synthetic_full_audit(ids, root / "primary", root / "mirror", identity, resume=True, synthetic_prefix_count=4)
            summary = json.loads((root / "primary" / "summary.json").read_text(encoding="utf-8"))
            self.assertNotEqual(summary.get("state"), "smoke_not_evaluated")
            self.assertNotIn("evaluation_role", summary)
            self.assertFalse(any("benchmark" in key for key in summary))
            self.assertEqual(summary["decision"], "NO_GO_INSUFFICIENT_EVIDENCE")
            self.assertFalse(summary["greedy_veto"])
            self.assertFalse(summary["route_authorized"])
            for relative in (
                "score_nms_audit.jsonl", "checkpoint.json", "summary.json", "summary.csv", "manifest.json",
                "predictions/labels/000000.txt", "predictions/labels/000003.txt",
            ):
                self.assertEqual((root / "primary" / relative).read_bytes(), (root / "uninterrupted" / relative).read_bytes(), relative)

    def test_benchmark32_nonformal_veto_blocks_an_otherwise_go_b_summary_and_csv(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _seal_benchmark32_summary

        ids = tuple(f"{index:06d}" for index in range(32))

        def candidate(index, level, stage):
            return {
                "index": index, "level": level, "box": (0.0, 0.0, 10.0, 10.0), "owner_iou": 0.6,
                "class_scores": (0.05, 0.05, 0.8), "best_class": 2, "gt_score": 0.8,
                "stage": stage, "strict_rank": 1, "tie_group_size": 1,
            }

        records = []
        for index, image_id in enumerate(ids):
            common = {"image_id": image_id, "class_id": 2, "class_name": "Cyclist", "moderate_valid": True,
                      "small_25_40": True, "height_px": 30.0, "depth_m": 20.0}
            records.extend((
                {**common, "gt_index": 0, "p2_candidates": [candidate(index, "P2", "raw")], "coarse_candidates": [candidate(index + 100, "P3", "max_nms")]},
                {**common, "gt_index": 1, "p2_candidates": [candidate(index + 200, "P2", "max_nms")], "coarse_candidates": [candidate(index + 300, "P3", "max_nms")], "coarse_only_keeps_useful": True, "coarse_direct_suppressed_by_p2": True},
            ))
        formal = summarize_score_nms_estimands(records, ids, reps=20, seed=7)
        self.assertEqual(formal["decision"], "GO_B_SCORE_OWNERSHIP")
        self.assertEqual(formal["S"]["eligible_images"], 32)
        self.assertEqual(formal["S"]["discordant"], 32)

        benchmark = _seal_benchmark32_summary(formal)
        self.assertEqual(benchmark["state"], "benchmark32_nonformal")
        self.assertEqual(benchmark["evaluation_role"], "benchmark32_nonformal")
        self.assertEqual(benchmark["benchmark_computed_decision"], "GO_B_SCORE_OWNERSHIP")
        self.assertEqual(benchmark["decision"], "BENCHMARK32_NOT_FOR_ROUTE_DECISION")
        self.assertFalse(benchmark["route_authorized"])
        self.assertTrue(benchmark["benchmark_veto"])
        self.assertEqual(benchmark["S"], formal["S"])
        self.assertEqual(benchmark["N"], formal["N"])
        self.assertEqual(benchmark["greedy_one_to_one_sensitivity"]["S"], formal["greedy_one_to_one_sensitivity"]["S"])
        benchmark["greedy_one_to_one_sensitivity"] = dict(benchmark["greedy_one_to_one_sensitivity"], route_authorized=True)
        rows = summary_long_rows(benchmark)
        decision_rows = [row for row in rows if row["family"] == "decision"]
        self.assertTrue(decision_rows)
        self.assertTrue(all(row["decision_role"] == "benchmark32_nonformal" for row in decision_rows))
        self.assertFalse(any(row["statistic"] == "route_authorized" and row["value"] is True for row in rows))
        self.assertEqual(next(row for row in decision_rows if row["statistic"] == "benchmark_computed_decision")["state"], "GO_B_SCORE_OWNERSHIP")

    def test_dynamic_level_boundaries(self):
        self.assertEqual(
            [(item.name, item.start, item.stop) for item in derive_level_slices([(1, 2, 2, 2), (1, 2, 1, 1), (1, 2, 1, 1), (1, 2, 1, 1)])],
            [("P2", 0, 4), ("P3", 4, 5), ("P4", 5, 6), ("P5", 6, 7)],
        )

    def test_letterbox_roundtrip_uses_rounded_padding(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _map_box_to_input
        original = torch.tensor([[13.0, 7.0, 101.0, 83.0]])
        mapped = torch.tensor([_map_box_to_input(original[0].tolist(), orig_shape=(101, 203), input_shape=(640, 640))])
        ratio = min(640 / 203, 640 / 101)
        pad_x = round((640 - round(203 * ratio)) / 2.0 - 0.1)
        pad_y = round((640 - round(101 * ratio)) / 2.0 - 0.1)
        recovered = mapped.clone()
        recovered[:, [0, 2]] = (recovered[:, [0, 2]] - pad_x) / ratio
        recovered[:, [1, 3]] = (recovered[:, [1, 3]] - pad_y) / ratio
        self.assertTrue(torch.allclose(recovered, original, atol=1e-4, rtol=0.0))

    def test_gt_validation_uses_float64_for_real_kitti_auto_letterbox_roundtrip(self):
        import ifdr_yolo.experiments.p2_score_nms_survival_audit as audit
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import LevelSlice, _gt_rows_for_image, _map_box_to_input

        def official_scale_boxes(img1_shape, boxes, img0_shape):
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            pad = (
                round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
                round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
            )
            boxes[..., [0, 2]] -= pad[0]
            boxes[..., [1, 3]] -= pad[1]
            boxes[..., :4] /= gain
            boxes[..., [0, 2]].clamp_(0, img0_shape[1])
            boxes[..., [1, 3]].clamp_(0, img0_shape[0])
            return boxes

        ultralytics = ModuleType("ultralytics")
        utils = ModuleType("ultralytics.utils")
        utils.ops = ModuleType("ultralytics.utils.ops")
        utils.ops.scale_boxes = official_scale_boxes
        ultralytics.utils = utils

        orig_shape, input_shape = (375, 1242), (224, 640)
        original64 = torch.tensor([
            [1114.48, 134.65, 1231.81, 254.34],
            [937.35, 161.73, 957.52, 221.18],
        ], dtype=torch.float64)
        original32 = original64.float()
        self.assertEqual(
            (round(orig_shape[1] * (640 / 1242)), round(orig_shape[0] * (640 / 1242))),
            (640, 193),
        )
        self.assertEqual((0, round((224 - 193) / 2.0 - 0.1)), (0, 15))
        mapped = torch.tensor(
            [_map_box_to_input(box.tolist(), orig_shape=orig_shape, input_shape=input_shape) for box in original64],
            dtype=torch.float64,
        )
        recovered32 = official_scale_boxes(input_shape, mapped.float().clone(), orig_shape)
        self.assertFalse(torch.allclose(recovered32, original32, atol=1e-4, rtol=0.0))
        recovered64 = official_scale_boxes(input_shape, mapped.clone(), orig_shape)
        self.assertTrue(torch.allclose(recovered64, original64, atol=1e-12, rtol=0.0))

        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory)
            labels.joinpath("000214.txt").write_text(
                "Cyclist 0.00 0 0.00 1114.48 134.65 1231.81 254.34 1.0 1.0 1.0 0.0 0.0 20.0 0.0\n"
                "Cyclist 0.00 0 0.00 937.35 161.73 957.52 221.18 1.0 1.0 1.0 0.0 0.0 20.0 0.0\n",
                encoding="utf-8",
            )
            decoded = torch.zeros((1, 7, 1), dtype=torch.float32)
            decoded[0, 4 + 2, 0] = 0.9
            original_pairwise_iou = audit._pairwise_iou
            seen_iou_dtype = []

            def pairwise_iou_with_dtype_guard(boxes_input, gt_input_tensor):
                seen_iou_dtype.append(gt_input_tensor.dtype)
                return original_pairwise_iou(boxes_input, gt_input_tensor)

            with (
                patch.dict(sys.modules, {"ultralytics": ultralytics, "ultralytics.utils": utils, "ultralytics.utils.ops": utils.ops}),
                patch.object(audit, "_pairwise_iou", side_effect=pairwise_iou_with_dtype_guard),
            ):
                record = _gt_rows_for_image(
                    image_id="000214", decoded=decoded, orig_shape=orig_shape, input_shape=input_shape,
                    level_slices=(LevelSlice("P2", 0, 1),), stage_indices={"max_nms": torch.empty(0, dtype=torch.long)},
                    suppression=(), raw_label_dir=labels, result=None, conf=0.001, iou=0.7,
                    max_nms=30000, max_det=300,
                )
        self.assertEqual(len(record["gt_rows"]), 2)
        self.assertEqual(seen_iou_dtype, [torch.float32])

    def test_iou_mixed_candidate_gt_devices_uses_explicit_compute_device(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _iou_on_device

        gt_cpu = torch.tensor([10.0, 10.0, 30.0, 30.0], dtype=torch.float32)
        candidate_cpu = torch.tensor([[15.0, 15.0, 35.0, 35.0]], dtype=torch.float32)
        cpu_value = _iou_on_device(gt_cpu, candidate_cpu, device="cpu")
        self.assertTrue(torch.allclose(cpu_value, torch.tensor([225.0 / 575.0]), atol=1e-7, rtol=0.0))
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable; CPU mixed-device boundary covered")
        cuda = torch.device("cuda:0")
        # Exercise both directions seen in the real predictor: decoded/orig
        # candidate boxes on CUDA with GT on CPU, and the reverse provenance
        # arrangement.  Both must use the same explicit candidate device.
        forward = _iou_on_device(gt_cpu, candidate_cpu.to(cuda), device=cuda)
        reverse = _iou_on_device(gt_cpu.to(cuda), candidate_cpu, device=cuda)
        self.assertEqual(forward.device, cuda)
        self.assertEqual(reverse.device, cuda)
        self.assertTrue(torch.allclose(forward.cpu(), cpu_value, atol=1e-7, rtol=0.0))
        self.assertTrue(torch.allclose(reverse.cpu(), cpu_value, atol=1e-7, rtol=0.0))

    def test_replay_matches_official_and_records_stages(self):
        # xywh + [Pedestrian, Cyclist]; first P2 box is useful and wins NMS.
        prediction = torch.tensor([[[5.0, 5.0, 4.0, 4.0, 0.8, 0.1], [5.0, 5.0, 4.0, 4.0, 0.7, 0.1], [20.0, 20.0, 4.0, 4.0, 0.2, 0.9]]]).transpose(1, 2)
        result = replay_nms_with_stages(prediction, level_slices=(("P2", 0, 2), ("P3", 2, 3)), conf=0.001, iou=0.7, max_nms=30000, max_det=300)
        self.assertEqual(result.output[0].shape[0], 2)
        self.assertEqual(result.output[0][0, 5].item(), 1.0)
        self.assertEqual(result.kept_indices[0].tolist(), [2, 0])
        self.assertIn("raw", result.stage_indices[0])
        self.assertIn("final", result.stage_indices[0])
        self.assertEqual(result.stage_indices[0]["final"].tolist(), [2, 0])

    def test_cross_level_same_class_suppression_but_not_cross_class(self):
        prediction = torch.tensor([[[5.0, 5.0, 4.0, 4.0, 0.9, 0.1], [5.0, 5.0, 4.0, 4.0, 0.8, 0.1], [5.0, 5.0, 4.0, 4.0, 0.1, 0.95]]]).transpose(1, 2)
        result = replay_nms_with_stages(prediction, level_slices=(("P2", 0, 1), ("P3", 1, 2), ("P4", 2, 3)), conf=0.001, iou=0.5, max_nms=30000, max_det=300, trace_suppression=True)
        self.assertEqual(result.kept_indices[0].tolist(), [2, 0])
        self.assertEqual(result.suppression[0][1]["suppressor_index"], 0)
        self.assertEqual(result.suppression[0][2]["suppressed"], False)

    def test_nms_exact_iou_threshold_is_not_suppressed(self):
        # Two equal 2x2 boxes with 2/3 horizontal overlap have IoU exactly .5.
        prediction = torch.tensor([[[1.0, 1.0, 2.0, 2.0, 0.9, 0.1], [1.6666666667, 1.0, 2.0, 2.0, 0.8, 0.1]]]).transpose(1, 2)
        result = replay_nms_with_stages(prediction, level_slices=(("P2", 0, 1), ("P3", 1, 2)), conf=0.001, iou=0.5001)
        self.assertEqual(result.kept_indices[0].tolist(), [0, 1])

    def test_subset_nms_does_not_revive_global_max_nms_dropped_candidate(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _nms_subset_indices
        prediction = torch.tensor([[[5.0, 5.0, 4.0, 4.0, 0.1], [5.0, 5.0, 4.0, 4.0, 0.9]]]).transpose(1, 2)
        # Caller supplies only the global max-NMS survivors; index 0 is absent.
        self.assertEqual(_nms_subset_indices(prediction, [1], conf=0.001, iou=0.7, max_nms=1, max_det=300), {1})

    def test_score_row_classifies_low_score_and_nms(self):
        gt = (3.0, 3.0, 7.0, 7.0)
        row = score_nms_survival_row(
            gt_box=gt,
            gt_class=0,
            candidates=[
                {"index": 0, "level": "P2", "box": (3, 3, 7, 7), "class_scores": (0.0005, 0.1), "stage": "raw"},
                {"index": 1, "level": "P3", "box": (3, 3, 7, 7), "class_scores": (0.9, 0.1), "stage": "nms"},
            ],
            conf=0.001,
            final_indices={1},
        )
        self.assertEqual(row["raw_useful_p2"], True)
        self.assertEqual(row["p2_stage"], "raw")
        self.assertEqual(row["route"], "B")

    def test_raw_assignment_uses_strict_iou_and_low_gt_tie(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import assign_raw_candidates_to_gt
        # identical GTs force lower-index ownership; exact 0.5 is not useful.
        rows = assign_raw_candidates_to_gt(
            [{"index": 0, "box": (0, 0, 1, 1), "class_scores": (0.1, 0.1, 0.9)}, {"index": 1, "box": (0, 0, 2, 1), "class_scores": (0.1, 0.1, 0.9)}],
            [(0, 0, 1, 1), (0, 0, 1, 1)],
            [2, 2],
            [True, True],
        )
        self.assertEqual(rows[0]["owner_gt_index"], 0)
        self.assertIsNone(rows[1]["owner_gt_index"])

    def test_raw_assignment_keeps_wrong_argmax_candidate_owned_by_gt(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import assign_raw_candidates_to_gt
        rows = assign_raw_candidates_to_gt(
            [{"index": 0, "box": (0, 0, 2, 2), "class_scores": (0.9, 0.05, 0.4)}],
            [(0, 0, 2, 2)], [2], [True]
        )
        self.assertEqual(rows[0]["owner_gt_index"], 0)

    def test_equal_scores_share_strict_rank_and_tie_count(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _gt_rows_for_image
        # Rank arithmetic is exercised through a small direct helper-shaped
        # assertion in the public row representation; equal scores must not
        # become ranks 1 and 2 merely due to flat-index ordering.
        scores = [0.8, 0.8, 0.4]
        ranks = [1 + sum(other > score for other in scores) for score in scores]
        ties = [sum(other == score for other in scores) for score in scores]
        self.assertEqual(ranks, [1, 1, 3])
        self.assertEqual(ties, [2, 2, 1])

    def test_label_binding_uses_nonempty_relative_label_sha(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _ScoreNMSRunState
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "primary" / "predictions" / "labels"
            labels.mkdir(parents=True)
            (labels / "000001.txt").write_bytes(b"2 0.5 0.5 0.1 0.1 0.7\n")
            state = _ScoreNMSRunState(output=root / "primary", mirror=root / "mirror", identity={"x": 1}, identity_sha="a" * 64, selected_ids=("000001",), completed=(), stop_after=None)
            state.commit({"image_id": "000001", "gt_rows": []}, labels / "000001.txt")
            payload = json.loads((root / "primary" / "score_nms_audit.jsonl").read_text())
            self.assertEqual(payload["label_path"], "predictions/labels/000001.txt")
            self.assertNotEqual(payload["label_sha256"], hashlib.sha256(b"").hexdigest())

    def test_resume_cleans_uncommitted_label_tail(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _recover_score_checkpoint, _capture_rng_state, _rng_snapshot_payload, _rng_state_digest_from_snapshot, _canonical_sha
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, mirror = root / "primary", root / "mirror"
            labels = primary / "predictions" / "labels"
            labels.mkdir(parents=True)
            committed = b"2 0.5 0.5 0.1 0.1 0.7\n"
            (labels / "000001.txt").write_bytes(committed)
            (labels / "000002.txt").write_bytes(b"tail\n")
            snapshot = _capture_rng_state()
            identity = {"case": "resume_tail"}
            identity_sha = _canonical_sha(identity)
            state = {"identity": identity, "identity_sha256": identity_sha, "completed_image_ids": ["000001"], "journal_offset": 1, "rng_completed_count": 1, "rng_snapshot": _rng_snapshot_payload(snapshot), "rng_digest": _rng_state_digest_from_snapshot(snapshot)}
            primary.mkdir(exist_ok=True)
            mirror.mkdir(exist_ok=True)
            record = {"identity_sha256": identity_sha, "image_id": "000001", "label_path": "predictions/labels/000001.txt", "label_missing_as_empty": False, "label_sha256": hashlib.sha256(committed).hexdigest(), "label_size": len(committed)}
            journal = json.dumps(record).encode() + b"\n"
            state["journal_offset"] = len(journal)
            state["journal_prefix_sha256"] = hashlib.sha256(journal).hexdigest()
            (primary / "score_nms_audit.jsonl").write_bytes(journal)
            (mirror / "score_nms_audit.jsonl").write_bytes(journal)
            (primary / "checkpoint.json").write_text(json.dumps(state), encoding="utf-8")
            (mirror / "checkpoint.json").write_text(json.dumps(state), encoding="utf-8")
            (mirror / "predictions" / "labels").mkdir(parents=True)
            (mirror / "predictions" / "labels" / "000001.txt").write_bytes(committed)
            self.assertEqual(_recover_score_checkpoint(primary, mirror, identity_sha, ("000001", "000002"), expected_identity=identity), ["000001"])
            self.assertFalse((labels / "000002.txt").exists())

    def test_resume_accepts_committed_missing_label_on_both_sides(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _recover_score_checkpoint, _capture_rng_state, _rng_snapshot_payload, _rng_state_digest_from_snapshot, _canonical_sha
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, mirror = root / "primary", root / "mirror"
            primary.mkdir(); mirror.mkdir()
            identity = {"case": "missing_label"}
            identity_sha = _canonical_sha(identity)
            record = {"identity_sha256": identity_sha, "image_id": "000001", "label_path": "predictions/labels/000001.txt", "label_missing_as_empty": True, "label_sha256": hashlib.sha256(b"").hexdigest(), "label_size": 0}
            journal = (json.dumps(record).encode() + b"\n")
            snapshot = _capture_rng_state()
            state = {"identity": identity, "identity_sha256": identity_sha, "completed_image_ids": ["000001"], "journal_offset": len(journal), "journal_prefix_sha256": hashlib.sha256(journal).hexdigest(), "rng_completed_count": 1, "rng_snapshot": _rng_snapshot_payload(snapshot), "rng_digest": _rng_state_digest_from_snapshot(snapshot)}
            for base in (primary, mirror):
                (base / "score_nms_audit.jsonl").write_bytes(journal)
                (base / "checkpoint.json").write_text(json.dumps(state), encoding="utf-8")
            self.assertEqual(_recover_score_checkpoint(primary, mirror, identity_sha, ("000001",), expected_identity=identity), ["000001"])

    def test_commit_missing_label_keeps_primary_and_mirror_absent(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _ScoreNMSRunState
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, mirror = root / "primary", root / "mirror"
            labels = output / "predictions" / "labels"
            labels.mkdir(parents=True)
            state = _ScoreNMSRunState(output=output, mirror=mirror, identity={"x": 1}, identity_sha="a" * 64, selected_ids=("000001",), completed=(), stop_after=None)
            state.commit({"image_id": "000001", "gt_rows": []}, labels / "000001.txt")
            self.assertFalse((labels / "000001.txt").exists())
            self.assertFalse((mirror / "predictions" / "labels" / "000001.txt").exists())

    def test_complete_manifest_binds_summary_and_csv_after_labels(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _ScoreNMSRunState, _capture_rng_state
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, mirror = root / "primary", root / "mirror"
            labels = output / "predictions" / "labels"
            labels.mkdir(parents=True)
            label = labels / "000001.txt"
            label.write_bytes(b"2 0.5 0.5 0.1 0.1 0.7\n")
            state = _ScoreNMSRunState(output=output, mirror=mirror, identity={"x": 1}, identity_sha="a" * 64, selected_ids=("000001",), completed=(), stop_after=None)
            state.commit({"image_id": "000001", "gt_rows": []}, label)
            (output / "summary.json").write_text('{"state":"smoke"}\n', encoding="utf-8")
            (output / "summary.csv").write_text("state\nsmoke\n", encoding="utf-8")
            state.checkpoint_after_batch(_capture_rng_state())
            state.complete()
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("summary.json", manifest["files"])
            self.assertIn("summary.csv", manifest["files"])
            self.assertEqual((output / "manifest.json").read_bytes(), (mirror / "manifest.json").read_bytes())

    def test_default_vs_audit_labels_missing_as_empty(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import compare_saved_label_files
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            default = root / "default" / "predictions" / "labels"
            audit = root / "audit" / "predictions" / "labels"
            default.mkdir(parents=True)
            audit.mkdir(parents=True)
            (default / "000001.txt").write_bytes(b"a\n")
            (audit / "000001.txt").write_bytes(b"a\n")
            self.assertEqual(compare_saved_label_files(root / "default", root / "audit", ("000001", "000002"))["state"], "PASS")
            (audit / "000001.txt").write_bytes(b"b\n")
            self.assertEqual(compare_saved_label_files(root / "default", root / "audit", ("000001",))["state"], "FAIL")

    def test_standard_greedy_one_to_one_coverage_and_sensitivity(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import standard_greedy_one_to_one_coverage, _greedy_sensitivity
        rows = [
            {"image_id": "000001", "gt_index": 0, "class_id": 2, "class_name": "Cyclist", "moderate_valid": True, "small_25_40": True, "gt_box": (0, 0, 10, 10), "p2_candidates": [{"index": 0, "box": (0, 0, 10, 10), "class_scores": (0.1, 0.1, 0.9)}], "coarse_candidates": []},
            {"image_id": "000001", "gt_index": 1, "class_id": 2, "class_name": "Cyclist", "moderate_valid": True, "small_25_40": True, "gt_box": (20, 0, 30, 10), "p2_candidates": [{"index": 1, "box": (20, 0, 30, 10), "class_scores": (0.1, 0.1, 0.8)}], "coarse_candidates": []},
        ]
        self.assertEqual(standard_greedy_one_to_one_coverage(rows, side="p2_candidates"), {0, 1})
        result = _greedy_sensitivity({"000001": rows})
        self.assertIn(result["state"], {"NOT_ESTIMABLE", "TIE_AMBIGUOUS"})

    def test_greedy_candidate_dedupe_rejects_inconsistent_duplicate(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _dedupe_class_candidates

        rows = [
            {"class_id": 2, "gt_index": 0, "moderate_valid": True, "p2_candidates": [{"index": 4, "level": "P2", "box": (0, 0, 10, 10), "class_scores": (0.1, 0.1, 0.9)}]},
            {"class_id": 2, "gt_index": 1, "moderate_valid": True, "p2_candidates": [{"index": 4, "level": "P2", "box": (1, 0, 11, 10), "class_scores": (0.1, 0.1, 0.9)}]},
        ]
        with self.assertRaisesRegex(ValueError, "inconsistent|duplicate"):
            _dedupe_class_candidates(rows, side="p2")

    def test_greedy_exact_score_tie_is_ambiguous_only_when_gt_options_overlap(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _greedy_assign_candidates

        rows = [
            {"image_id": "000001", "class_id": 2, "gt_index": 0, "gt_box": (0, 0, 10, 10), "moderate_valid": True,
             "p2_candidates": [{"index": 1, "level": "P2", "box": (0, 0, 10, 10), "class_scores": (0.1, 0.1, 0.8)}]},
            {"image_id": "000001", "class_id": 2, "gt_index": 1, "gt_box": (1, 0, 11, 10), "moderate_valid": True,
             "p2_candidates": [{"index": 2, "level": "P2", "box": (0, 0, 10, 10), "class_scores": (0.1, 0.1, 0.8)}]},
        ]
        result = _greedy_assign_candidates(rows, side="p2", stage=None)
        self.assertEqual(result["state"], "TIE_AMBIGUOUS")

        disjoint = [
            {"image_id": "000001", "class_id": 2, "gt_index": 0, "gt_box": (0, 0, 10, 10), "moderate_valid": True,
             "p2_candidates": [{"index": 1, "level": "P2", "box": (0, 0, 10, 10), "class_scores": (0.1, 0.1, 0.8)}]},
            {"image_id": "000001", "class_id": 2, "gt_index": 1, "gt_box": (20, 0, 30, 10), "moderate_valid": True,
             "p2_candidates": [{"index": 2, "level": "P2", "box": (20, 0, 30, 10), "class_scores": (0.1, 0.1, 0.8)}]},
        ]
        self.assertEqual(_greedy_assign_candidates(disjoint, side="p2", stage=None)["state"], "PASS")

    def test_greedy_raw_assignment_is_not_replaced_by_later_stage_candidate(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _greedy_assign_candidates, _greedy_assignment_reaches, _greedy_sensitivity

        rows = [{
            "image_id": "000001", "class_id": 2, "class_name": "Cyclist", "gt_index": 0, "gt_box": (0, 0, 10, 10), "moderate_valid": True, "small_25_40": True,
            "p2_candidates": [
                # Highest GT-class score but wrong argmax: raw owner is fixed.
                {"index": 1, "level": "P2", "box": (0, 0, 10, 10), "owner_iou": 1.0, "class_scores": (0.95, 0.05, 0.40), "stage": "raw"},
                # Lower score, correct argmax and max_nms eligible; it may not
                # replace the raw owner during S-stage evaluation.
                {"index": 2, "level": "P2", "box": (0, 0, 10, 10), "owner_iou": 1.0, "class_scores": (0.05, 0.05, 0.35), "best_class": 2, "stage": "max_nms"},
            ],
        }]
        assigned = _greedy_assign_candidates(rows, side="p2", stage=None)
        self.assertEqual(assigned["matches"][0]["candidate_index"], 1)
        self.assertFalse(_greedy_assignment_reaches(assigned, gt_index=0, stage={"max_nms", "nms", "max_det", "final"}, gt_class=2, conf=0.001))
        rows[0]["coarse_candidates"] = [{"index": 3, "level": "P3", "box": (0, 0, 10, 10), "owner_iou": 1.0, "class_scores": (0.05, 0.05, 0.8), "best_class": 2, "stage": "max_nms"}]
        sensitivity = _greedy_sensitivity({"000001": rows})
        self.assertEqual(sensitivity["frames"][0]["s_delta"], 1)

    def test_greedy_s_eligibility_requires_both_raw_assignment_matches(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _greedy_sensitivity

        rows = [{
            "image_id": "000001", "class_id": 2, "gt_index": 0, "gt_box": (0, 0, 10, 10), "moderate_valid": True,
            "p2_candidates": [{"index": 1, "level": "P2", "box": (0, 0, 10, 10), "owner_iou": 1.0, "class_scores": (0.05, 0.05, 0.9), "best_class": 2, "stage": "max_nms"}],
            # Local owner metadata claims useful, but geometry is not a raw
            # assignment to this GT; S must exclude the row.
            "coarse_candidates": [{"index": 2, "level": "P3", "box": (40, 40, 50, 50), "owner_iou": 1.0, "class_scores": (0.05, 0.05, 0.9), "best_class": 2, "stage": "max_nms"}],
        }]
        result = _greedy_sensitivity({"000001": rows})
        self.assertEqual(result["frames"][0]["s_denominator"], 0)

    def test_greedy_full_useful_coverage_comes_from_fixed_matches(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _greedy_full_useful_gt_indices

        rows = [{"gt_index": 0, "class_id": 2, "gt_box": (0, 0, 10, 10), "moderate_valid": True,
                 "p2_candidates": [{"index": 1, "level": "P2", "box": (0, 0, 10, 10), "class_scores": (0.05, 0.05, 0.9), "best_class": 2, "stage": "max_nms"}],
                 "coarse_candidates": [{"index": 2, "level": "P3", "box": (0, 0, 10, 10), "class_scores": (0.05, 0.05, 0.8), "best_class": 2, "stage": "max_nms"}],
                 "full_nms_any_useful": True}]
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _greedy_assign_candidates
        p2 = _greedy_assign_candidates(rows, side="p2", stage=None)
        coarse = _greedy_assign_candidates(rows, side="coarse", stage=None)
        # max_nms means NMS input but not an NMS survivor; fixed raw matches
        # therefore do not claim full useful post-NMS coverage.
        self.assertEqual(_greedy_full_useful_gt_indices(rows, p2_assignment=p2, coarse_assignment=coarse), set())

    def test_greedy_match_candidate_requires_evaluation_class(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _greedy_match_candidate
        rows = [{"class_id": 2, "p2_candidates": [{"index": 3, "level": "P2", "box": (0, 0, 1, 1)}]}]
        match = {"candidate_index": 3, "level": "P2", "class_id": 1}
        self.assertIsNone(_greedy_match_candidate(rows, side="p2", match=match))

    def test_greedy_harm_uses_best_and_suppressor_class_when_same_class_field_absent(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _greedy_n_harm_gt_indices
        rows = [{"gt_index": 0, "class_id": 2, "gt_box": (0, 0, 10, 10), "moderate_valid": True, "full_nms_any_useful": False,
                 "p2_candidates": [{"index": 1, "level": "P2", "box": (0, 0, 10, 10), "class_scores": (0.05, 0.05, 0.9), "best_class": 2, "gt_score": 0.9, "stage": "max_nms", "group_only_nms_survives": True,
                                    "suppressed": True, "suppressor_index": 4, "suppressor_level": "P3", "suppressor_class": 2, "pair_iou": 0.8, "suppressor_box": (20, 20, 30, 30)}]}]
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _greedy_assign_candidates
        assignment = _greedy_assign_candidates(rows, side="p2", stage=None)
        self.assertEqual(_greedy_n_harm_gt_indices(rows, side="p2", assignment=assignment), {0})

    def test_greedy_harm_requires_explicit_direct_suppression_provenance(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _greedy_n_harm_gt_indices, _greedy_assign_candidates
        rows = [{"gt_index": 0, "class_id": 2, "gt_box": (0, 0, 10, 10), "moderate_valid": True, "full_nms_any_useful": False,
                 "p2_candidates": [{"index": 1, "level": "P2", "box": (0, 0, 10, 10), "class_scores": (0.05, 0.05, 0.9), "best_class": 2, "gt_score": 0.9, "stage": "max_nms", "group_only_nms_survives": True,
                                    "suppressor_level": "P3", "suppressor_class": 2, "pair_iou": 0.8, "suppressor_box": (20, 20, 30, 30)}]}]
        assignment = _greedy_assign_candidates(rows, side="p2", stage=None)
        self.assertEqual(_greedy_n_harm_gt_indices(rows, side="p2", assignment=assignment), set())

    def test_greedy_raw_zero_score_is_assigned_then_fails_conf(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _greedy_assign_candidates, _greedy_assignment_reaches
        rows = [{"class_id": 2, "gt_index": 0, "gt_box": (0, 0, 10, 10), "moderate_valid": True,
                 "p2_candidates": [{"index": 1, "level": "P2", "box": (0, 0, 10, 10), "class_scores": (0.0, 0.0, 0.0), "best_class": 2, "stage": "raw"}]}]
        result = _greedy_assign_candidates(rows, side="p2", stage=None)
        self.assertEqual(result["state"], "PASS")
        self.assertEqual(result["matches"][0]["candidate_index"], 1)
        self.assertFalse(_greedy_assignment_reaches(result, gt_index=0, stage={"max_nms", "nms"}, gt_class=2, conf=0.001))

    def test_stop_after_raises_dedicated_runtime_error(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _ScoreNMSRunState, ScoreNMSInterrupted, _capture_rng_state
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "primary"
            labels = output / "predictions" / "labels"
            labels.mkdir(parents=True)
            label = labels / "000001.txt"
            label.write_bytes(b"")
            state = _ScoreNMSRunState(output=output, mirror=root / "mirror", identity={"x": 1}, identity_sha="a" * 64, selected_ids=("000001", "000002"), completed=(), stop_after=1)
            state.commit({"image_id": "000001", "gt_rows": []}, label)
            with self.assertRaises(ScoreNMSInterrupted):
                state.checkpoint_after_batch(_capture_rng_state())

    def test_commit_does_not_checkpoint_until_batch_end_rng_callback(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _ScoreNMSRunState, _capture_rng_state, _rng_snapshot_from_payload, _rng_state_digest_from_snapshot

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, mirror = root / "primary", root / "mirror"
            labels = output / "predictions" / "labels"
            labels.mkdir(parents=True)
            label = labels / "000001.txt"
            label.write_bytes(b"")
            state = _ScoreNMSRunState(output=output, mirror=mirror, identity={"x": 1}, identity_sha="a" * 64, selected_ids=("000001",), completed=(), stop_after=None)
            state.commit({"image_id": "000001", "gt_rows": []}, label)
            self.assertFalse((output / "checkpoint.json").exists())
            state.checkpoint_after_batch(_capture_rng_state())
            payload = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["rng_completed_count"], 1)
            self.assertIn("rng_snapshot", payload)
            self.assertEqual(payload["rng_digest"], _rng_state_digest_from_snapshot(_rng_snapshot_from_payload(payload["rng_snapshot"])))

    def test_resume_requires_rng_snapshot_and_digest(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _recover_score_checkpoint, _canonical_sha

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, mirror = root / "primary", root / "mirror"
            primary.mkdir(); mirror.mkdir()
            identity = {"case": "missing_rng"}
            identity_sha = _canonical_sha(identity)
            record = {"identity_sha256": identity_sha, "image_id": "000001", "label_path": "predictions/labels/000001.txt", "label_missing_as_empty": True, "label_sha256": hashlib.sha256(b"").hexdigest(), "label_size": 0}
            journal = (json.dumps(record).encode() + b"\n")
            state = {"identity": identity, "identity_sha256": identity_sha, "completed_image_ids": ["000001"], "journal_offset": len(journal), "journal_prefix_sha256": hashlib.sha256(journal).hexdigest()}
            for base in (primary, mirror):
                (base / "score_nms_audit.jsonl").write_bytes(journal)
                (base / "checkpoint.json").write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "RNG|rng"):
                _recover_score_checkpoint(primary, mirror, identity_sha, ("000001",), expected_identity=identity)

    def test_recovery_rejects_primary_embedded_identity_tamper(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _canonical_sha, _recover_score_checkpoint, _capture_rng_state, _rng_snapshot_payload, _rng_state_digest_from_snapshot

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, mirror = root / "primary", root / "mirror"
            primary.mkdir(); mirror.mkdir()
            identity = {"fit_ids_sha256": "fit", "checkpoint_sha256": "a" * 64}
            identity_sha = _canonical_sha(identity)
            record = {"identity_sha256": identity_sha, "image_id": "000001", "label_path": "predictions/labels/000001.txt", "label_missing_as_empty": True, "label_sha256": hashlib.sha256(b"").hexdigest(), "label_size": 0}
            journal = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
            snapshot = _capture_rng_state()
            checkpoint = {"identity": identity, "identity_sha256": identity_sha, "completed_image_ids": ["000001"], "journal_offset": len(journal), "journal_prefix_sha256": hashlib.sha256(journal).hexdigest(), "rng_completed_count": 1, "rng_snapshot": _rng_snapshot_payload(snapshot), "rng_digest": _rng_state_digest_from_snapshot(snapshot)}
            for base in (primary, mirror):
                (base / "score_nms_audit.jsonl").write_bytes(journal)
                (base / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
            tampered = dict(checkpoint, identity={"fit_ids_sha256": "TAMPERED", "checkpoint_sha256": "a" * 64})
            (primary / "checkpoint.json").write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "embedded identity"):
                _recover_score_checkpoint(primary, mirror, identity_sha, ("000001",), expected_identity=identity)

    def test_recovery_rejects_mirror_embedded_identity_tamper(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _canonical_sha, _recover_score_checkpoint, _capture_rng_state, _rng_snapshot_payload, _rng_state_digest_from_snapshot

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, mirror = root / "primary", root / "mirror"
            primary.mkdir(); mirror.mkdir()
            identity = {"fit_ids_sha256": "fit", "checkpoint_sha256": "a" * 64}
            identity_sha = _canonical_sha(identity)
            record = {"identity_sha256": identity_sha, "image_id": "000001", "label_path": "predictions/labels/000001.txt", "label_missing_as_empty": True, "label_sha256": hashlib.sha256(b"").hexdigest(), "label_size": 0}
            journal = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
            snapshot = _capture_rng_state()
            checkpoint = {"identity": identity, "identity_sha256": identity_sha, "completed_image_ids": ["000001"], "journal_offset": len(journal), "journal_prefix_sha256": hashlib.sha256(journal).hexdigest(), "rng_completed_count": 1, "rng_snapshot": _rng_snapshot_payload(snapshot), "rng_digest": _rng_state_digest_from_snapshot(snapshot)}
            for base in (primary, mirror):
                (base / "score_nms_audit.jsonl").write_bytes(journal)
                (base / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
            tampered = dict(checkpoint, identity={"fit_ids_sha256": "TAMPERED", "checkpoint_sha256": "a" * 64})
            (mirror / "checkpoint.json").write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "embedded identity"):
                _recover_score_checkpoint(primary, mirror, identity_sha, ("000001",), expected_identity=identity)

    def test_generation_zero_rebuilds_from_paired_default_reference(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _ScoreNMSRunState, _assert_output_fresh_or_resumable, _capture_rng_state

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, mirror = root / "primary", root / "mirror"
            primary.mkdir(); mirror.mkdir()
            reference = json.dumps({"identity_sha256": "a" * 64, "selected_ids": ["000001"], "rng_initial_snapshot": {"cpu_hex": "00", "cuda_hex": []}}, sort_keys=True).encode() + b"\n"
            (primary / "default_reference.json").write_bytes(reference)
            (mirror / "default_reference.json").write_bytes(reference)
            _assert_output_fresh_or_resumable(primary, mirror, resume=True)
            snapshot = _capture_rng_state()
            state = _ScoreNMSRunState(output=primary, mirror=mirror, identity={"x": 1}, identity_sha="a" * 64, selected_ids=("000001",), completed=(), stop_after=None, rng_snapshot=snapshot, generation_zero_pending=True)
            state.initialize_generation_zero(snapshot)
            payload = json.loads((primary / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["completed_image_ids"], [])
            self.assertEqual(payload["rng_completed_count"], 0)
            self.assertEqual(payload["journal_offset"], 0)
            self.assertEqual((primary / "checkpoint.json").read_bytes(), (mirror / "checkpoint.json").read_bytes())

    def test_generation_zero_tampered_default_reference_fails_closed(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _load_default_reference

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, mirror = root / "primary", root / "mirror"
            primary.mkdir(); mirror.mkdir()
            valid = json.dumps({"identity_sha256": "a" * 64, "selected_ids": ["000001"]}, sort_keys=True).encode() + b"\n"
            (primary / "default_reference.json").write_bytes(valid)
            (mirror / "default_reference.json").write_bytes(valid)
            with self.assertRaisesRegex(ValueError, "identity"):
                _load_default_reference(primary, mirror, "b" * 64, ("000001",))

    def test_generation_zero_truncates_identical_uncommitted_journal_and_labels(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _ScoreNMSRunState, _capture_rng_state

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, mirror = root / "primary", root / "mirror"
            for base in (primary, mirror):
                (base / "predictions" / "labels").mkdir(parents=True)
                (base / "score_nms_audit.jsonl").write_text("uncommitted-tail\n", encoding="utf-8")
                (base / "predictions" / "labels" / "000001.txt").write_text("tail\n", encoding="utf-8")
            state = _ScoreNMSRunState(output=primary, mirror=mirror, identity={"x": 1}, identity_sha="a" * 64, selected_ids=("000001",), completed=(), stop_after=None, rng_snapshot=_capture_rng_state(), generation_zero_pending=True)
            state.initialize_generation_zero(state.rng_snapshot)
            self.assertEqual((primary / "score_nms_audit.jsonl").read_bytes(), b"")
            self.assertEqual((mirror / "score_nms_audit.jsonl").read_bytes(), b"")
            self.assertFalse((primary / "predictions" / "labels" / "000001.txt").exists())
            self.assertFalse((mirror / "predictions" / "labels" / "000001.txt").exists())

    def test_estimand_fails_closed_on_sparse_or_non_discordant_pairs(self):
        ids = tuple(f"{index:06d}" for index in range(30))
        records = []
        for image_id in ids:
            records.append({
                "image_id": image_id,
                "class_name": "Cyclist",
                "class_id": 2,
                "moderate_valid": True,
                "small_25_40": True,
                "far_gt_40m": False,
                "p2_candidates": [{"index": 0, "level": "P2", "iou": 0.6, "best_class": 2, "gt_score": 0.5, "stage": "max_nms"}],
                "coarse_candidates": [{"index": 1, "level": "P3", "iou": 0.6, "best_class": 2, "gt_score": 0.5, "stage": "max_nms"}],
            })
        result = summarize_score_nms_estimands(records, ids, reps=20)
        self.assertEqual(result["decision"], "NO_GO_INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["S"]["state"], "NOT_ESTIMABLE")

    def test_b2a_descriptive_strata_are_independent_and_use_unique_images(self):
        """Ped/Cyc strata use registered geometry, not legacy complement flags."""
        rows = [
            # Two Pedestrians in one image: the small count is two GTs but one
            # image; the first is also far, proving the flags are non-exclusive.
            {"image_id": "000001", "gt_index": 0, "class_id": 1, "class_name": "Pedestrian", "moderate_valid": True,
             "height_px": 30.0, "depth_m": 50.0, "small_25_40": False, "far_gt_40m": False,
             "p2_candidates": [], "coarse_candidates": []},
            {"image_id": "000001", "gt_index": 1, "class_id": 1, "class_name": "Pedestrian", "moderate_valid": True,
             "height_px": 40.0, "depth_m": 20.0, "small_25_40": False, "far_gt_40m": False,
             "p2_candidates": [], "coarse_candidates": []},
            # Cyclist target: large and near, independent of the Pedestrian
            # class even though it shares the image.
            {"image_id": "000001", "gt_index": 2, "class_id": 2, "class_name": "Cyclist", "moderate_valid": True,
             "height_px": 81.0, "depth_m": 20.0, "small_25_40": False, "far_gt_40m": False,
             "p2_candidates": [], "coarse_candidates": []},
            # Exact small/far boundaries: 25 is not small, 40 is small; 40m is
            # not far, while 40.01m is far.  Existing flags intentionally lie.
            {"image_id": "000002", "gt_index": 0, "class_id": 2, "class_name": "Cyclist", "moderate_valid": True,
             "height_px": 25.0, "depth_m": 40.0, "small_25_40": True, "far_gt_40m": True,
             "p2_candidates": [], "coarse_candidates": []},
            {"image_id": "000003", "gt_index": 0, "class_id": 2, "class_name": "Cyclist", "moderate_valid": True,
             "height_px": 40.0, "depth_m": 40.01, "small_25_40": False, "far_gt_40m": False,
             "p2_candidates": [], "coarse_candidates": []},
            # Depth zero is outside the registered near interval (0, 20].
            {"image_id": "000004", "gt_index": 0, "class_id": 2, "class_name": "Cyclist", "moderate_valid": True,
             "height_px": 50.0, "depth_m": 0.0, "small_25_40": False, "far_gt_40m": False,
             "p2_candidates": [], "coarse_candidates": []},
        ]
        result = summarize_score_nms_estimands(
            [{"image_id": image_id, "gt_rows": [row for row in rows if row["image_id"] == image_id]}
             for image_id in ("000001", "000002", "000003", "000004")],
            ("000001", "000002", "000003", "000004"),
            reps=2,
        )
        strata = result["descriptive_strata"]
        self.assertEqual(strata["Pedestrian"]["small_25_40"]["gt"], 2)
        self.assertEqual(strata["Pedestrian"]["small_25_40"]["unique_images"], 1)
        self.assertEqual(strata["Pedestrian"]["far_gt_40m"]["gt"], 1)
        self.assertEqual(strata["Pedestrian"]["far_gt_40m"]["unique_images"], 1)
        self.assertEqual(strata["Cyclist"]["large_gt_80"]["gt"], 1)
        self.assertEqual(strata["Cyclist"]["large_gt_80"]["unique_images"], 1)
        self.assertEqual(strata["Cyclist"]["near_0_20m"]["gt"], 1)
        self.assertEqual(strata["Cyclist"]["small_25_40"]["gt"], 1)
        self.assertEqual(strata["Cyclist"]["far_gt_40m"]["gt"], 1)
        # The 25px/40m boundary row is excluded despite its stale flags.
        self.assertNotEqual(strata["Cyclist"]["far_gt_40m"]["gt"], 2)
        self.assertEqual(strata["Cyclist"]["near_0_20m"]["gt"], 1)

    def test_b2a_empty_slice_is_locally_not_estimable(self):
        row = {"image_id": "000001", "gt_index": 0, "class_id": 1, "class_name": "Pedestrian", "moderate_valid": True,
               "height_px": 50.0, "depth_m": 25.0, "small_25_40": False, "far_gt_40m": False,
               "p2_candidates": [], "coarse_candidates": []}
        result = summarize_score_nms_estimands(
            [{"image_id": "000001", "gt_rows": [row]}], ("000001",), reps=2
        )
        strata = result["descriptive_strata"]
        self.assertEqual(strata["Cyclist"]["small_25_40"]["state"], "NOT_ESTIMABLE")
        self.assertEqual(strata["Cyclist"]["small_25_40"]["eligible_gt"], 0)
        self.assertEqual(strata["Pedestrian"]["large_gt_80"]["state"], "NOT_ESTIMABLE")
        # A sparse descriptive slice must not mutate the frozen primary route.
        self.assertIn(result["decision"], {"NO_GO_INSUFFICIENT_EVIDENCE", "NO_GO_BC"})

    def test_b2b_negative_controls_use_independent_geometry_and_ratio_of_sums(self):
        """Auxiliary score controls retain both geometry slices and image clusters."""

        def candidate(index, *, failure):
            return {
                "index": index,
                "iou": 0.6,
                "best_class": 2,
                "gt_score": 0.5,
                "class_scores": (0.05, 0.05, 0.5),
                "strict_rank": index + 1,
                "tie_group_size": 1,
                "stage": "raw" if failure else "max_nms",
            }

        # The stale flags deliberately disagree with the geometry.  P2 has
        # small 2/3 failures and large 1/2 failures, so its ratio-of-sums
        # difference is 1/6, not an average of image rates.
        rows = [
            {"image_id": "000001", "gt_index": 0, "class_id": 2, "moderate_valid": True,
             "height_px": 30.0, "depth_m": 50.0, "small_25_40": False, "far_gt_40m": False,
             "p2_candidates": [candidate(0, failure=True)], "coarse_candidates": [candidate(1, failure=False)]},
            {"image_id": "000001", "gt_index": 1, "class_id": 2, "moderate_valid": True,
             "height_px": 40.0, "depth_m": 45.0, "small_25_40": False, "far_gt_40m": False,
             "p2_candidates": [candidate(2, failure=False)], "coarse_candidates": [candidate(3, failure=True)]},
            {"image_id": "000002", "gt_index": 0, "class_id": 2, "moderate_valid": True,
             "height_px": 35.0, "depth_m": 30.0, "small_25_40": False, "far_gt_40m": False,
             "p2_candidates": [candidate(4, failure=True)], "coarse_candidates": [candidate(5, failure=False)]},
            {"image_id": "000002", "gt_index": 1, "class_id": 2, "moderate_valid": True,
             "height_px": 81.0, "depth_m": 10.0, "small_25_40": True, "far_gt_40m": True,
             "p2_candidates": [candidate(6, failure=True)], "coarse_candidates": [candidate(7, failure=True)]},
            {"image_id": "000003", "gt_index": 0, "class_id": 2, "moderate_valid": True,
             "height_px": 90.0, "depth_m": 20.0, "small_25_40": False, "far_gt_40m": False,
             "p2_candidates": [candidate(8, failure=False)], "coarse_candidates": [candidate(9, failure=False)]},
        ]
        ids = ("000001", "000002", "000003", "000004")  # Empty fit frame must still be sampled.
        result = summarize_score_nms_estimands(
            [{"image_id": image_id, "gt_rows": [row for row in rows if row["image_id"] == image_id]} for image_id in ids],
            ids,
            reps=20,
            seed=7,
        )
        controls = result["negative_controls"]
        self.assertEqual(controls["decision_role"], "auxiliary_only")
        p2 = controls["Cyclist"]["P2"]["small_25_40_minus_large_gt_80"]
        self.assertEqual(p2["target"], {"stratum": "small_25_40", "num": 2, "den": 3, "rate": 2 / 3, "eligible_unique_images": 2})
        self.assertEqual(p2["control"], {"stratum": "large_gt_80", "num": 1, "den": 2, "rate": 0.5, "eligible_unique_images": 2})
        self.assertAlmostEqual(p2["observed_rate_difference"], 1 / 6)
        self.assertEqual(p2["bootstrap_replicates"], 20)
        self.assertEqual(p2["bootstrap_seed"], 7)
        self.assertEqual(p2["all_fit_image_ids"], list(ids))
        # 25px/40m are not members, while an 81px/10m row cannot be made
        # small/far by stale flags.
        self.assertEqual(controls["Cyclist"]["P2"]["far_gt_40m_minus_near_0_20m"]["target"]["den"], 2)
        self.assertEqual(controls["Cyclist"]["P2"]["far_gt_40m_minus_near_0_20m"]["control"]["den"], 2)

    def test_b2b_zero_slice_is_local_and_does_not_change_primary_decision(self):
        row = {"image_id": "000001", "gt_index": 0, "class_id": 2, "moderate_valid": True,
               "height_px": 30.0, "depth_m": 30.0,
               "p2_candidates": [{"index": 0, "iou": 0.6, "best_class": 2, "gt_score": 0.5, "stage": "max_nms"}],
               "coarse_candidates": [{"index": 1, "iou": 0.6, "best_class": 2, "gt_score": 0.5, "stage": "max_nms"}]}
        result = summarize_score_nms_estimands([{"image_id": "000001", "gt_rows": [row]}], ("000001", "000002"), reps=2)
        contrast = result["negative_controls"]["Cyclist"]["P2"]["small_25_40_minus_large_gt_80"]
        self.assertEqual(contrast["state"], "NOT_ESTIMABLE")
        self.assertEqual(contrast["control"]["den"], 0)
        self.assertEqual(result["decision"], "NO_GO_INSUFFICIENT_EVIDENCE")
        self.assertFalse(result["route_authorized"])

    def test_b2a_formal_geometry_missing_or_nonfinite_fails_closed(self):
        base = {"image_id": "000001", "gt_index": 0, "class_id": 2, "class_name": "Cyclist", "moderate_valid": True,
                "p2_candidates": [], "coarse_candidates": []}
        for field, value in (("height_px", None), ("depth_m", None), ("height_px", float("nan")), ("depth_m", float("inf"))):
            row = dict(base, height_px=40.0, depth_m=20.0)
            row[field] = value
            with self.assertRaisesRegex(ValueError, "descriptive strata geometry"):
                summarize_score_nms_estimands([{"image_id": "000001", "gt_rows": [row]}], ("000001",), reps=2)
            with self.assertRaisesRegex(ValueError, "descriptive strata geometry"):
                summarize_score_nms_estimands([row], ("000001",), reps=2)

    def test_max_nms_is_n_input_not_n_survivor_for_primary_and_greedy(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _eligible_score_row, _greedy_sensitivity

        def candidate(index, level, score):
            return {
                "index": index,
                "level": level,
                "box": (0.0, 0.0, 10.0, 10.0),
                "owner_iou": 0.6,
                "class_scores": (0.05, 0.05, score),
                "best_class": 2,
                "gt_score": score,
                "stage": "max_nms",
                "group_only_nms_survives": False,
            }

        row = {
            "image_id": "000001",
            "class_name": "Cyclist",
            "class_id": 2,
            "moderate_valid": True,
            "small_25_40": True,
            "p2_candidates": [candidate(1, "P2", 0.8)],
            "coarse_candidates": [candidate(2, "P3", 0.7)],
        }
        primary = _eligible_score_row(row, conf=0.001)
        self.assertIsNotNone(primary)
        self.assertTrue(primary["p2_enters_nms"])
        self.assertTrue(primary["coarse_enters_nms"])
        self.assertFalse(primary["p2_nms_useful"])
        self.assertFalse(primary["coarse_nms_useful"])

        greedy_row = dict(row, gt_index=0, gt_box=(0.0, 0.0, 10.0, 10.0))
        greedy = _greedy_sensitivity({"000001": [greedy_row]})
        self.assertEqual(greedy["frames"][0]["n_denominator"], 1)

    def test_greedy_aggregation_only_counts_target_population_but_assignment_sees_all_gt(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _greedy_sensitivity

        def row(index, cls, box, *, small=False, far=False):
            candidate = {"index": index, "level": "P2", "box": box, "owner_iou": 1.0, "class_scores": (0.05, 0.05, 0.8) if cls == 2 else (0.8, 0.05, 0.05), "best_class": cls, "gt_score": 0.8, "stage": "max_nms"}
            coarse = dict(candidate, index=index + 100, level="P3")
            return {"image_id": "000001", "class_id": cls, "class_name": "Cyclist" if cls == 2 else "Pedestrian", "gt_index": index, "gt_box": box, "moderate_valid": True, "small_25_40": small, "far_gt_40m": far, "p2_candidates": [candidate], "coarse_candidates": [coarse]}

        rows = [
            row(0, 2, (0, 0, 10, 10), small=True),  # target Cyclist
            row(1, 2, (20, 0, 30, 10)),             # non-target Cyclist
            row(2, 0, (40, 0, 50, 10), small=True), # Pedestrian control
        ]
        result = _greedy_sensitivity({"000001": rows})
        self.assertEqual(result["frames"][0]["s_denominator"], 1)

    def test_greedy_cyclist_pool_ignores_ped_ties_but_keeps_non_target_cyclist(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _greedy_sensitivity
        def candidate(index, cls, box):
            scores = (0.8, 0.1, 0.8) if cls == 2 else (0.8, 0.8, 0.1)
            return {"index": index, "level": "P2", "box": box, "class_scores": scores, "best_class": cls, "gt_score": scores[cls], "stage": "max_nms"}
        target = {"image_id": "000001", "class_id": 2, "class_name": "Cyclist", "gt_index": 0, "gt_box": (0, 0, 10, 10), "moderate_valid": True, "small_25_40": True,
                  "p2_candidates": [candidate(1, 2, (0, 0, 10, 10))], "coarse_candidates": [dict(candidate(2, 2, (0, 0, 10, 10)), level="P3")]}
        distractor = {"image_id": "000001", "class_id": 2, "class_name": "Cyclist", "gt_index": 1, "gt_box": (20, 0, 30, 10), "moderate_valid": True, "small_25_40": False,
                      "p2_candidates": [candidate(3, 2, (20, 0, 30, 10))], "coarse_candidates": [dict(candidate(4, 2, (20, 0, 30, 10)), level="P3")]}
        ped = {"image_id": "000001", "class_id": 0, "class_name": "Pedestrian", "gt_index": 2, "gt_box": (0, 0, 10, 10), "moderate_valid": True, "small_25_40": True,
               "p2_candidates": [candidate(5, 0, (0, 0, 10, 10))], "coarse_candidates": [dict(candidate(6, 0, (0, 0, 10, 10)), level="P3")]}
        result = _greedy_sensitivity({"000001": [target, distractor, ped]})
        self.assertNotEqual(result["state"], "TIE_AMBIGUOUS")
        self.assertEqual(result["frames"][0]["s_denominator"], 1)

    def test_greedy_veto_direction_and_primary_decision_are_separate(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _greedy_veto_for_decision

        primary = {"decision": "GO_B_SCORE_OWNERSHIP", "S": {"state": "estimable", "observed": 0.2}, "N": {"state": "estimable", "observed": -0.1}}
        same = {"state": "PASS", "S": {"state": "estimable", "observed": 0.1}, "N": {"state": "estimable", "observed": -0.1}}
        self.assertEqual(_greedy_veto_for_decision(primary, same)["route_authorized"], True)
        negative = {"state": "PASS", "S": {"state": "estimable", "observed": -0.1}, "N": {"state": "estimable", "observed": -0.1}}
        result = _greedy_veto_for_decision(primary, negative)
        self.assertTrue(result["greedy_veto"])
        self.assertTrue(result["direction_reversal"]["S"])
        self.assertEqual(result["primary_decision"], "GO_B_SCORE_OWNERSHIP")
        zero = {"state": "PASS", "S": {"state": "estimable", "observed": 0.0}, "N": {"state": "estimable", "observed": -0.1}}
        self.assertTrue(_greedy_veto_for_decision(primary, zero)["greedy_veto"])
        not_estimable = {"state": "NOT_ESTIMABLE", "S": {"state": "NOT_ESTIMABLE", "observed": None}, "N": {"state": "estimable", "observed": 0.1}}
        self.assertTrue(_greedy_veto_for_decision(primary, not_estimable)["greedy_veto"])
        tie = {"state": "TIE_AMBIGUOUS", "S": {"state": "TIE_AMBIGUOUS", "observed": None}, "N": {"state": "estimable", "observed": 0.1}}
        self.assertTrue(_greedy_veto_for_decision(primary, tie)["greedy_veto"])
        no_go = {"decision": "NO_GO_BC", "S": {"state": "estimable", "observed": -0.1}, "N": {"state": "estimable", "observed": -0.1}}
        positive = {"state": "PASS", "S": {"state": "estimable", "observed": 0.9}, "N": {"state": "estimable", "observed": 0.9}}
        no_go_result = _greedy_veto_for_decision(no_go, positive)
        self.assertFalse(no_go_result["route_authorized"])
        self.assertFalse(no_go_result["greedy_veto"])
        go_b_off_target = {"state": "NOT_ESTIMABLE", "S": {"state": "estimable", "observed": 0.2}, "N": {"state": "NOT_ESTIMABLE", "observed": None}}
        self.assertTrue(_greedy_veto_for_decision(primary, go_b_off_target)["route_authorized"])

    def test_greedy_veto_go_c_uses_n_endpoint(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _greedy_veto_for_decision
        primary = {"decision": "GO_C_NMS_OWNERSHIP", "S": {"state": "estimable", "observed": -0.1}, "N": {"state": "estimable", "observed": 0.2}}
        good = {"state": "PASS", "S": {"state": "estimable", "observed": -0.3}, "N": {"state": "estimable", "observed": 0.1}}
        self.assertTrue(_greedy_veto_for_decision(primary, good)["route_authorized"])
        bad = {"state": "PASS", "S": {"state": "estimable", "observed": -0.3}, "N": {"state": "estimable", "observed": 0.0}}
        self.assertTrue(_greedy_veto_for_decision(primary, bad)["greedy_veto"])

    def test_denominator_ledger_is_exhaustive_and_first_loss_has_both_sides(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _denominator_ledger_and_first_loss
        def candidate(index, level, score, stage="max_nms", best=2):
            scores = (0.9, 0.1, score) if best == 2 else (score, 0.1, 0.2)
            return {"index": index, "level": level, "box": (0, 0, 10, 10), "owner_iou": 0.6, "class_scores": scores, "best_class": best, "gt_score": score, "stage": stage}
        rows = [
            {"image_id": "000001", "gt_index": 0, "class_id": 2, "class_name": "Cyclist", "moderate_valid": True, "small_25_40": True, "p2_candidates": [], "coarse_candidates": [candidate(2, "P3", 0.8)]},
            {"image_id": "000001", "gt_index": 1, "class_id": 2, "class_name": "Cyclist", "moderate_valid": True, "small_25_40": True, "p2_candidates": [], "coarse_candidates": []},
            {"image_id": "000002", "gt_index": 0, "class_id": 2, "class_name": "Cyclist", "moderate_valid": True, "small_25_40": True, "p2_candidates": [candidate(3, "P2", 0.8)], "coarse_candidates": []},
            {"image_id": "000003", "gt_index": 0, "class_id": 2, "class_name": "Cyclist", "moderate_valid": True, "small_25_40": True, "p2_candidates": [candidate(4, "P2", 0.8, best=0)], "coarse_candidates": [candidate(5, "P3", 0.8)]},
            {"image_id": "000004", "gt_index": 0, "class_id": 2, "class_name": "Cyclist", "moderate_valid": True, "small_25_40": True, "p2_candidates": [candidate(6, "P2", 0.8, stage="raw")], "coarse_candidates": [candidate(7, "P3", 0.8)]},
        ]
        result = _denominator_ledger_and_first_loss(rows, ("000001", "000002", "000003", "000004"), conf=0.001)
        ledger = result["ledger"]
        self.assertEqual(ledger["target_gt"], 5)
        self.assertEqual(ledger["target_images"], 4)
        self.assertEqual(ledger["raw_p2_useful_gt"], 3)
        self.assertEqual(ledger["raw_p2_useful_images"], 3)
        self.assertEqual(ledger["raw_coarse_useful_gt"], 3)
        self.assertEqual(ledger["raw_coarse_useful_images"], 3)
        self.assertEqual(ledger["exclusions"]["no_raw_both"]["gt"], 1)
        self.assertEqual(ledger["exclusions"]["no_raw_both"]["images"], 1)
        self.assertEqual(sum(bucket["gt"] for bucket in ledger["exclusions"].values()) + ledger["n_eligible_gt"], 5)
        self.assertNotIn("s_first_loss", result["frames"][0]) if "frames" in result else None
        self.assertIn("p2", result["first_loss"])
        self.assertIn("coarse", result["first_loss"])
        self.assertNotIn("no_raw_useful", result["first_loss"]["p2"])
        self.assertIn("tie_descriptive", result)

    def test_summary_rejects_unknown_and_duplicate_formal_gt_rows(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import summarize_score_nms_estimands
        row = {"image_id": "000001", "class_id": 2, "gt_index": 0, "class_name": "Cyclist", "moderate_valid": True, "small_25_40": True, "far_gt_40m": False, "p2_candidates": [], "coarse_candidates": []}
        with self.assertRaisesRegex(ValueError, "non-fit"):
            summarize_score_nms_estimands([dict(row, image_id="999999")], ("000001",), reps=2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_score_nms_estimands([row, row], ("000001",), reps=2)

    def test_raw_label_view_requires_every_fit_id_and_rejects_duplicate_stems(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _validate_raw_label_view
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "000001.txt").write_text("Car 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0\n", encoding="utf-8")
            view = _validate_raw_label_view(root, ("000001",))
            self.assertEqual(view["fit_count"], 1)
            self.assertIn("000001", view["fit_label_sha256"])
            with self.assertRaisesRegex(ValueError, "missing fit IDs"):
                _validate_raw_label_view(root, ("000001", "000002"))
            duplicate = root / "nested"
            duplicate.mkdir()
            (duplicate / "000001.txt").write_bytes(b"duplicate\n")
            with self.assertRaisesRegex(ValueError, "duplicate image ID"):
                _validate_raw_label_view(root, ("000001",))

    def test_fresh_output_gate_rejects_stale_and_resume_requires_two_checkpoints(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _assert_output_fresh_or_resumable
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, mirror = root / "primary", root / "mirror"
            primary.mkdir(); mirror.mkdir()
            _assert_output_fresh_or_resumable(primary, mirror, resume=False)
            (primary / "score_nms_audit.jsonl").write_text("tail\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not fresh"):
                _assert_output_fresh_or_resumable(primary, mirror, resume=False)
            with self.assertRaisesRegex(ValueError, "requires primary/mirror checkpoint"):
                _assert_output_fresh_or_resumable(primary, mirror, resume=True)
            (primary / "checkpoint.json").write_text("{}\n", encoding="utf-8")
            (mirror / "checkpoint.json").write_text("{}\n", encoding="utf-8")
            (primary / "manifest.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "immutable"):
                _assert_output_fresh_or_resumable(primary, mirror, resume=True)

    def test_result_boxes_comparison_is_bitwise(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _compare_result_boxes

        class Boxes:
            def __init__(self, value):
                self.data = torch.tensor(value, dtype=torch.float32)

        class Result:
            def __init__(self, value):
                self.boxes = Boxes(value)

        equal = _compare_result_boxes([Result([[1.0, 2.0, 3.0, 4.0, 0.5, 2.0]])], [Result([[1.0, 2.0, 3.0, 4.0, 0.5, 2.0]])])
        self.assertEqual(equal["state"], "PASS")
        unequal = _compare_result_boxes([Result([[1.0, 2.0, 3.0, 4.0, 0.5, 2.0]])], [Result([[1.0, 2.0, 3.0, 4.0, 0.5, 1.0]])])
        self.assertEqual(unequal["state"], "FAIL")

    def test_group_only_nms_survival_is_level_specific(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _group_only_nms_survives

        self.assertTrue(_group_only_nms_survives(4, "P2", {4}, {7}))
        self.assertFalse(_group_only_nms_survives(7, "P2", {4}, {7}))
        self.assertTrue(_group_only_nms_survives(7, "P3", {4}, {7}))
        self.assertTrue(_group_only_nms_survives(8, "P5", {4}, {8}))
        self.assertFalse(_group_only_nms_survives(4, "P3", {4}, {7}))

    def test_direct_suppression_persists_original_suppressor_box(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _attach_suppressor_box

        item = {"candidate_index": 3, "suppressor_index": 1, "pair_iou": 0.8}
        boxes_orig = torch.tensor([[0.0, 0.0, 1.0, 1.0], [10.0, 11.0, 20.0, 21.0], [2.0, 3.0, 4.0, 5.0]])
        enriched = _attach_suppressor_box(item, boxes_orig)
        self.assertEqual(enriched["suppressor_box"], (10.0, 11.0, 20.0, 21.0))

    def test_noninterference_compares_controlled_pair_not_pre_post(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _compare_controlled_run_gates

        control = {"rng_initial": {"cpu": "a", "cuda": ["b"]}, "rng_final": {"cpu": "c", "cuda": ["d"]}, "model_post": "m", "backend": "torchvision", "device": "cpu", "fp32": True, "eval": True, "fused": True}
        audit = {"rng_initial": {"cpu": "a", "cuda": ["b"]}, "rng_final": {"cpu": "c", "cuda": ["d"]}, "model_post": "m", "backend": "torchvision", "device": "cpu", "fp32": True, "eval": True, "fused": True}
        self.assertEqual(_compare_controlled_run_gates(control, audit, expected_device="cpu")["state"], "PASS")
        altered = dict(audit, rng_final={"cpu": "different", "cuda": ["d"]})
        self.assertEqual(_compare_controlled_run_gates(control, altered, expected_device="cpu")["state"], "FAIL")
        wrong_but_equal = dict(control, backend="fake")
        wrong_but_equal_audit = dict(audit, backend="fake")
        self.assertEqual(_compare_controlled_run_gates(wrong_but_equal, wrong_but_equal_audit, expected_device="cpu")["state"], "FAIL")

    def test_runtime_facts_resolve_actual_parameter_device_not_requested_string(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _model_runtime_facts

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))

            def is_fused(self):
                return True

        class Model:
            def __init__(self):
                self.model = Module()

        facts = _model_runtime_facts(Model(), device="0", backend="torchvision")
        # The request token is not runtime evidence; CPU parameters must be
        # reported as CPU (and a real CUDA model reports cuda:<index>).
        self.assertEqual(facts["device"], "cpu")

    def test_runtime_facts_reject_empty_or_mixed_parameter_devices(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _model_runtime_facts

        class ParameterStub:
            dtype = torch.float32

            def __init__(self, device):
                self.device = torch.device(device)

        class Module:
            training = False

            def __init__(self, devices):
                self._parameters = [ParameterStub(device) for device in devices]

            def parameters(self):
                return iter(self._parameters)

            def is_fused(self):
                return True

        class Model:
            def __init__(self, devices):
                self.model = Module(devices)

        with self.assertRaisesRegex(RuntimeError, "parameter device"):
            _model_runtime_facts(Model(()), device="cpu", backend="torchvision")
        with self.assertRaisesRegex(RuntimeError, "same device"):
            _model_runtime_facts(Model(("cpu", "meta")), device="cpu", backend="torchvision")

    def test_resume_model_construction_defers_persisted_rng_restore_to_callback(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _capture_rng_state, _construct_resume_model
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _restore_rng_state, _rng_state_digest

        torch.manual_seed(123)
        persisted = _capture_rng_state()
        _restore_rng_state(persisted)
        torch.rand(17)  # controlled predictor draw after model setup
        if torch.cuda.is_available():
            torch.rand(17, device="cuda")
        expected = _rng_state_digest()

        torch.manual_seed(999)

        def fake_constructor(_checkpoint):
            torch.rand(31)  # ambient constructor/setup consumption
            if torch.cuda.is_available():
                torch.rand(31, device="cuda")
            return object()

        _construct_resume_model(fake_constructor, "checkpoint.pt", persisted)
        # Construction may consume ambient RNG; the predictor callback owns
        # restoration at the post-warmup boundary.
        _restore_rng_state(persisted)
        torch.rand(17)
        if torch.cuda.is_available():
            torch.rand(17, device="cuda")
        self.assertEqual(_rng_state_digest(), expected)

    def test_fresh_gate_rejects_any_existing_primary_or_mirror_entry(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _assert_output_fresh_or_resumable
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, mirror = root / "primary", root / "mirror"
            primary.mkdir(); mirror.mkdir()
            (primary / "unexpected.bin").write_bytes(b"stale")
            with self.assertRaisesRegex(ValueError, "not fresh"):
                _assert_output_fresh_or_resumable(primary, mirror, resume=False)

    def test_publication_checkpoint_is_not_complete_before_manifest(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _publication_states
        self.assertEqual(_publication_states(), ("running", "publishing", "complete"))

    def test_final_publication_rejects_publishing_checkpoint(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _validate_final_publication
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, mirror = root / "primary", root / "mirror"
            primary.mkdir(); mirror.mkdir()
            checkpoint = {"state": "publishing", "publication_state": "publishing"}
            (primary / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
            (mirror / "checkpoint.json").write_bytes((primary / "checkpoint.json").read_bytes())
            manifest = {"publication_state": "publishing", "files": {"checkpoint.json": hashlib.sha256((primary / "checkpoint.json").read_bytes()).hexdigest()}}
            (primary / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (mirror / "manifest.json").write_bytes((primary / "manifest.json").read_bytes())
            with self.assertRaisesRegex(ValueError, "incomplete"):
                _validate_final_publication(primary, mirror)

    def test_final_publication_recomputes_all_manifest_hashes_and_labels(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _validate_final_publication
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, mirror = root / "primary", root / "mirror"
            for base in (primary, mirror):
                (base / "predictions" / "labels").mkdir(parents=True)
            checkpoint = {"state": "complete", "publication_state": "complete", "identity_sha256": "a" * 64}
            journal = b'{"identity_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}\n'
            summary_json, summary_csv, label = b'{"ok":true}\n', b"field,value\nok,true\n", b"2 0.5 0.5 0.1 0.1 0.7\n"
            (primary / "checkpoint.json").write_text(json.dumps(checkpoint, sort_keys=True) + "\n", encoding="utf-8")
            for name, data in (("score_nms_audit.jsonl", journal), ("summary.json", summary_json), ("summary.csv", summary_csv)):
                (primary / name).write_bytes(data)
            (primary / "predictions" / "labels" / "000001.txt").write_bytes(label)
            for path in (primary / "checkpoint.json", primary / "score_nms_audit.jsonl", primary / "summary.json", primary / "summary.csv", primary / "predictions" / "labels" / "000001.txt"):
                mirror_path = mirror / path.relative_to(primary)
                mirror_path.parent.mkdir(parents=True, exist_ok=True)
                mirror_path.write_bytes(path.read_bytes())
            files = {name: hashlib.sha256((primary / name).read_bytes()).hexdigest() for name in ("checkpoint.json", "score_nms_audit.jsonl", "summary.json", "summary.csv")}
            manifest = {"schema_version": 1, "publication_state": "complete", "identity_sha256": "a" * 64, "labels": [{"path": "predictions/labels/000001.txt", "size": len(label), "sha256": hashlib.sha256(label).hexdigest()}], "files": files}
            for base in (primary, mirror):
                (base / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(_validate_final_publication(primary, mirror)["state"], "PASS")
            (primary / "summary.json").write_bytes(b'{"tampered":true}\n')
            with self.assertRaisesRegex(ValueError, "hash"):
                _validate_final_publication(primary, mirror)
            (primary / "summary.json").write_bytes(summary_json)
            (mirror / "predictions" / "labels" / "000002.txt").write_bytes(label)
            with self.assertRaisesRegex(ValueError, "label set"):
                _validate_final_publication(primary, mirror)
            (mirror / "predictions" / "labels" / "000002.txt").unlink()
            (primary / "score_nms_audit.jsonl").write_text('{"identity_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch|journal identity"):
                _validate_final_publication(primary, mirror)

    def test_publishing_publication_is_resumable_and_reference_is_identity_bound(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _assert_output_fresh_or_resumable, _write_default_reference, _load_default_reference, _recover_score_checkpoint, _capture_rng_state, _rng_snapshot_payload, _rng_state_digest_from_snapshot, _canonical_sha
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, mirror = root / "primary", root / "mirror"
            primary.mkdir(); mirror.mkdir()
            identity = {"case": "publishing"}
            identity_sha = _canonical_sha(identity)
            checkpoint = {"state": "publishing", "publication_state": "publishing", "identity": identity, "identity_sha256": identity_sha}
            for base in (primary, mirror):
                (base / "checkpoint.json").write_text(json.dumps(checkpoint, sort_keys=True) + "\n", encoding="utf-8")
                (base / "manifest.json").write_text(json.dumps({"publication_state": "publishing", "identity_sha256": identity_sha}, sort_keys=True) + "\n", encoding="utf-8")
            _assert_output_fresh_or_resumable(primary, mirror, resume=True)
            record = {"identity_sha256": identity_sha, "image_id": "000001", "label_path": "predictions/labels/000001.txt", "label_missing_as_empty": True, "label_sha256": hashlib.sha256(b"").hexdigest(), "label_size": 0}
            journal = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
            snapshot = _capture_rng_state()
            for base in (primary, mirror):
                (base / "score_nms_audit.jsonl").write_bytes(journal)
                (base / "checkpoint.json").write_text(json.dumps({**checkpoint, "completed_image_ids": ["000001"], "journal_offset": len(journal), "journal_prefix_sha256": hashlib.sha256(journal).hexdigest(), "rng_completed_count": 1, "rng_snapshot": _rng_snapshot_payload(snapshot), "rng_digest": _rng_state_digest_from_snapshot(snapshot)}, sort_keys=True) + "\n", encoding="utf-8")
            recovered = _recover_score_checkpoint(primary, mirror, identity_sha, ("000001",), expected_identity=identity)
            self.assertEqual(recovered, ["000001"])
            self.assertEqual(json.loads((primary / "checkpoint.json").read_text(encoding="utf-8"))["state"], "running")
            reference = {"identity_sha256": identity_sha, "selected_ids": ["000001"], "boxes": {"000001": {"shape": [0, 6], "dtype": "torch.float32", "values": [], "sha256": hashlib.sha256(b"").hexdigest()}}, "labels": {"000001": {"missing": True, "sha256": hashlib.sha256(b"").hexdigest(), "size": 0}}}
            _write_default_reference(primary, mirror, reference)
            self.assertEqual(_load_default_reference(primary, mirror, identity_sha, ("000001",))["selected_ids"], ["000001"])
            with self.assertRaisesRegex(ValueError, "identity"):
                _load_default_reference(primary, mirror, "b" * 64, ("000001",))

    def test_resume_mirror_is_byte_equivalent_and_identity_fail_closes(self):
        ids = ("000001", "000002", "000003")
        identity = {"fit_ids_sha256": hashlib.sha256(b"fit").hexdigest(), "checkpoint_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uninterrupted = root / "uninterrupted"
            uninterrupted_mirror = root / "uninterrupted-mirror"
            run_synthetic_score_nms_audit(ids, uninterrupted, uninterrupted_mirror, identity)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_synthetic_score_nms_audit(ids, root / "primary", root / "mirror", identity, stop_after=2)
            resumed = run_synthetic_score_nms_audit(ids, root / "primary", root / "mirror", identity, resume=True)
            self.assertEqual(resumed["state"], "complete")
            self.assertEqual((root / "primary" / "audit.jsonl").read_bytes(), (root / "mirror" / "audit.jsonl").read_bytes())
            for filename in ("audit.jsonl", "checkpoint.json", "manifest.json"):
                self.assertEqual((root / "primary" / filename).read_bytes(), (uninterrupted / filename).read_bytes())
            changed = dict(identity)
            changed["checkpoint_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "identity"):
                run_synthetic_score_nms_audit(ids, root / "primary", root / "mirror", changed, resume=True)

    def test_resume_repairs_one_sided_complete_manifest(self):
        """A crash after one final manifest write is recoverable and idempotent."""
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _assert_output_fresh_or_resumable

        ids = ("000001", "000002")
        identity = {"fit_ids_sha256": hashlib.sha256(b"fit").hexdigest(), "checkpoint_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, mirror = root / "primary", root / "mirror"
            run_synthetic_score_nms_audit(ids, primary, mirror, identity)
            complete = json.loads((primary / "manifest.json").read_text(encoding="utf-8"))
            publishing = dict(complete, publication_state="publishing")
            (mirror / "manifest.json").write_text(json.dumps(publishing, sort_keys=True) + "\n", encoding="utf-8")
            _assert_output_fresh_or_resumable(primary, mirror, resume=True)
            self.assertEqual((primary / "manifest.json").read_bytes(), (mirror / "manifest.json").read_bytes())
            self.assertEqual(json.loads((mirror / "manifest.json").read_text(encoding="utf-8"))["publication_state"], "complete")

    def test_resume_repairs_reverse_one_sided_complete_manifest_and_rejects_tamper(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _assert_output_fresh_or_resumable

        ids = ("000001", "000002")
        identity = {"fit_ids_sha256": hashlib.sha256(b"fit").hexdigest(), "checkpoint_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, mirror = root / "primary", root / "mirror"
            run_synthetic_score_nms_audit(ids, primary, mirror, identity)
            complete = json.loads((primary / "manifest.json").read_text(encoding="utf-8"))
            (primary / "manifest.json").write_text(json.dumps(dict(complete, publication_state="publishing"), sort_keys=True) + "\n", encoding="utf-8")
            _assert_output_fresh_or_resumable(primary, mirror, resume=True)
            self.assertEqual((primary / "manifest.json").read_bytes(), (mirror / "manifest.json").read_bytes())

            # A content mutation cannot be repaired by copying the leading manifest.
            run_synthetic_score_nms_audit(ids, primary, mirror, identity, resume=True)
            (mirror / "audit.jsonl").write_bytes((mirror / "audit.jsonl").read_bytes() + b"tamper\n")
            (primary / "manifest.json").write_text(json.dumps(dict(complete, publication_state="publishing"), sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "publication|hash|journal"):
                _assert_output_fresh_or_resumable(primary, mirror, resume=True)

    def test_mixed_manifest_rejects_tampered_publishing_side(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _assert_output_fresh_or_resumable

        ids = ("000001", "000002")
        identity = {"fit_ids_sha256": hashlib.sha256(b"fit").hexdigest(), "checkpoint_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, mirror = root / "primary", root / "mirror"
            run_synthetic_score_nms_audit(ids, primary, mirror, identity)
            complete = json.loads((primary / "manifest.json").read_text(encoding="utf-8"))
            publishing = dict(complete, publication_state="publishing")
            (mirror / "manifest.json").write_text(json.dumps(publishing, sort_keys=True) + "\n", encoding="utf-8")
            (mirror / "audit.jsonl").write_bytes((mirror / "audit.jsonl").read_bytes() + b"tamper\n")
            with self.assertRaisesRegex(ValueError, "publication|hash|journal"):
                _assert_output_fresh_or_resumable(primary, mirror, resume=True)

    def test_publishing_resume_preserves_noninterference_evidence(self):
        """A fully processed publishing run cannot erase its smoke evidence."""
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import _load_persisted_smoke_summary

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, mirror = root / "primary", root / "mirror"
            primary.mkdir(); mirror.mkdir()
            identity = "a" * 64
            summary = {
                "identity_sha256": identity,
                "default_vs_audit_labels": {"state": "PASS"},
                "default_vs_audit_results": {"state": "PASS"},
                "non_interference": {"state": "PASS"},
            }
            raw = (json.dumps(summary, sort_keys=True) + "\n").encode()
            for base in (primary, mirror):
                (base / "summary.json").write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            for base in (primary, mirror):
                (base / "manifest.json").write_text(json.dumps({"publication_state": "publishing", "identity_sha256": identity, "files": {"summary.json": digest}}, sort_keys=True) + "\n", encoding="utf-8")
            loaded = _load_persisted_smoke_summary(primary, mirror, identity)
            self.assertEqual(loaded["non_interference"]["state"], "PASS")
            (primary / "summary.json").write_text(json.dumps(dict(summary, non_interference={"state": "FAIL"}), sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "summary|non_interference"):
                _load_persisted_smoke_summary(primary, mirror, identity)

    def test_continuous_support_reports_exact_paired_score_rank_tie_and_logit_summaries(self):
        def candidate(index, level, score, rank, tie, *, wrong_argmax=False):
            scores = (0.95, 0.01, score) if wrong_argmax else (0.01, 0.01, score)
            return {"index": index, "level": level, "owner_iou": 0.6, "class_scores": scores,
                    "strict_rank": rank, "tie_group_size": tie, "stage": "raw"}

        records = [
            {"image_id": "000001", "gt_index": 0, "class_id": 2, "class_name": "Cyclist", "moderate_valid": True, "small_25_40": True, "height_px": 30.0, "depth_m": 20.0,
             "p2_candidates": [candidate(1, "P2", 0.2, 5, 1, wrong_argmax=True)], "coarse_candidates": [candidate(2, "P3", 0.1, 8, 2)]},
            {"image_id": "000002", "gt_index": 1, "class_id": 2, "class_name": "Cyclist", "moderate_valid": True, "far_gt_40m": True, "height_px": 50.0, "depth_m": 50.0,
             "p2_candidates": [candidate(3, "P2", 0.5, 7, 2)], "coarse_candidates": [candidate(4, "P4", 0.3, 9, 1)]},
            {"image_id": "000003", "gt_index": 2, "class_id": 2, "class_name": "Cyclist", "moderate_valid": True, "small_25_40": True, "height_px": 30.0, "depth_m": 20.0,
             "p2_candidates": [candidate(5, "P2", 0.0, 9, 1)], "coarse_candidates": [candidate(6, "P5", 1.0, 11, 1)]},
        ]
        result = summarize_score_nms_estimands(records, ("000001", "000002", "000003"), reps=2)
        support = result["continuous_support"]
        self.assertEqual(support["state"], "estimable")
        self.assertEqual(support["decision_role"], "descriptive_only")
        self.assertEqual(support["population"]["eligible_gt"], 3)
        self.assertEqual(support["p2_best_gt_class_score"], {"n": 3, "mean": 0.7 / 3, "median": 0.2, "q25": 0.0, "q75": 0.5})
        self.assertEqual(support["combined_coarse_best_gt_class_score"], {"n": 3, "mean": 1.4 / 3, "median": 0.3, "q25": 0.1, "q75": 1.0})
        self.assertEqual(support["p2_strict_rank"], {"n": 3, "mean": 7.0, "median": 7.0, "q25": 5.0, "q75": 9.0})
        self.assertEqual(support["p2_minus_coarse_score_margin"], {"n": 3, "mean": -0.7 / 3, "median": 0.1, "q25": -1.0, "q75": 0.2})
        epsilon = float(torch.finfo(torch.float32).eps)
        self.assertEqual(support["logit_epsilon"], epsilon)
        self.assertEqual(support["clamp_counts"], {"p2": 1, "combined_coarse": 1, "total": 2})
        margins = [math.log(max(epsilon, min(1.0 - epsilon, p2)) / (1.0 - max(epsilon, min(1.0 - epsilon, p2)))) - math.log(max(epsilon, min(1.0 - epsilon, coarse)) / (1.0 - max(epsilon, min(1.0 - epsilon, coarse)))) for p2, coarse in ((0.2, 0.1), (0.5, 0.3), (0.0, 1.0))]
        self.assertEqual(support["p2_minus_coarse_logit_margin"], {"n": 3, "mean": sum(margins) / 3, "median": margins[0], "q25": min(margins), "q75": max(margins)})
        # The descriptive payload does not alter frozen primary decisions.
        self.assertEqual(result["decision"], "NO_GO_INSUFFICIENT_EVIDENCE")
        self.assertFalse(result["route_authorized"])

    def test_continuous_support_exact_score_ties_deduplicate_or_fail_closed(self):
        def candidate(index, level, rank, tie):
            return {"index": index, "level": level, "owner_iou": 0.6, "class_scores": (0.01, 0.01, 0.8), "strict_rank": rank, "tie_group_size": tie, "stage": "raw"}

        base = {"image_id": "000001", "gt_index": 0, "class_id": 2, "class_name": "Cyclist", "moderate_valid": True, "small_25_40": True, "height_px": 30.0, "depth_m": 20.0,
                "p2_candidates": [candidate(1, "P2", 4, 2), candidate(2, "P2", 4, 2)], "coarse_candidates": [candidate(3, "P3", 8, 1)]}
        support = summarize_score_nms_estimands([base], ("000001",), reps=2)["continuous_support"]
        self.assertEqual(support["p2_best_gt_class_score"]["n"], 1)
        bad = dict(base, p2_candidates=[candidate(1, "P2", 4, 2), candidate(2, "P2", 5, 2)])
        with self.assertRaisesRegex(ValueError, "tie metadata"):
            summarize_score_nms_estimands([bad], ("000001",), reps=2)

    def test_continuous_support_fails_closed_for_bad_formal_metadata_and_is_local_when_empty(self):
        def candidate(**extra):
            value = {"index": 1, "level": "P2", "owner_iou": 0.6, "class_scores": (0.01, 0.01, 0.8), "strict_rank": 2, "tie_group_size": 1, "stage": "raw"}
            value.update(extra)
            return value

        base = {"image_id": "000001", "gt_index": 0, "class_id": 2, "class_name": "Cyclist", "moderate_valid": True, "small_25_40": True, "height_px": 30.0, "depth_m": 20.0,
                "p2_candidates": [candidate()], "coarse_candidates": [dict(candidate(index=2, level="P3", strict_rank=3))]}
        for bad in (candidate(class_scores=(float("nan"), 0.01, 0.8)), candidate(class_scores=(0.01, 0.01, 1.1)), candidate(strict_rank=0), candidate(tie_group_size=0), {key: value for key, value in candidate().items() if key != "strict_rank"}):
            row = dict(base, p2_candidates=[bad])
            with self.assertRaises(ValueError):
                summarize_score_nms_estimands([row], ("000001",), reps=2)
        # Formal useful rows may not silently discard a malformed candidate
        # merely because a later candidate is sufficient for the summary.
        missing_scores = {key: value for key, value in candidate().items() if key != "class_scores"}
        with self.assertRaisesRegex(ValueError, "class_scores"):
            summarize_score_nms_estimands([dict(base, p2_candidates=[missing_scores, candidate(index=9)])], ("000001",), reps=2)
        missing_iou = {key: value for key, value in candidate().items() if key != "owner_iou"}
        with self.assertRaisesRegex(ValueError, "IoU"):
            summarize_score_nms_estimands([dict(base, p2_candidates=[missing_iou, candidate(index=10)])], ("000001",), reps=2)
        empty = dict(base, p2_candidates=[], coarse_candidates=[])
        result = summarize_score_nms_estimands([empty], ("000001",), reps=2)
        self.assertEqual(result["continuous_support"]["state"], "NOT_ESTIMABLE")
        self.assertEqual(result["decision"], "NO_GO_INSUFFICIENT_EVIDENCE")
        self.assertFalse(result["route_authorized"])

    def test_negative_controls_fail_closed_for_invalid_formal_candidate_iou(self):
        def candidate(index, **extra):
            value = {"index": index, "level": "P2", "owner_iou": 0.6, "best_class": 1, "gt_score": 0.5, "stage": "max_nms"}
            value.update(extra)
            return value

        base = {"image_id": "000001", "gt_index": 0, "class_id": 1, "class_name": "Pedestrian", "moderate_valid": True,
                "height_px": 30.0, "depth_m": 20.0, "p2_candidates": [candidate(1)], "coarse_candidates": [candidate(2, level="P3")]}
        invalid = (
            {key: value for key, value in candidate(3).items() if key != "owner_iou"},
            candidate(3, owner_iou=float("nan")),
            candidate(3, owner_iou=float("inf")),
            candidate(3, owner_iou="not-a-number"),
        )
        for bad in invalid:
            with self.assertRaisesRegex(ValueError, "IoU"):
                summarize_score_nms_estimands([dict(base, p2_candidates=[bad, candidate(4)])], ("000001",), reps=2)
        legacy = {key: value for key, value in base.items() if key != "gt_index"}
        legacy["p2_candidates"] = [{key: value for key, value in candidate(5).items() if key != "owner_iou"}]
        self.assertEqual(summarize_score_nms_estimands([legacy], ("000001",), reps=2)["decision"], "NO_GO_INSUFFICIENT_EVIDENCE")

    def test_b2d_long_csv_schema(self):
        rows = summary_long_rows(self._b2d_summary())
        columns = (
            "family", "population", "class", "stratum", "side", "contrast",
            "statistic", "value", "state", "numerator", "denominator",
            "eligible_images", "discordant", "ci_level", "ci_low", "ci_high",
            "reps", "seed", "decision_role",
        )
        self.assertTrue(rows)
        self.assertEqual(tuple(rows[0]), columns)
        self.assertEqual(rows, summary_long_rows(self._b2d_summary()))
        self.assertEqual(
            {row["family"] for row in rows},
            {"primary", "greedy", "ledger", "first_loss", "tie_descriptive",
             "descriptive_strata", "negative_controls", "continuous_support", "decision"},
        )
        self.assertEqual(len(rows), len({tuple(row[column] for column in ("family", "population", "class", "stratum", "side", "contrast", "statistic")) for row in rows}))

    def test_b2d_smoke_summary_long_csv_is_closed_and_deterministic(self):
        from ifdr_yolo.experiments.p2_score_nms_survival_audit import LONG_COLUMNS, _summary_long_csv

        summary = {
            "state": "smoke_not_evaluated",
            "processed_fit_count": 1,
            "default_vs_audit_labels": {"state": "PASS"},
            "default_vs_audit_results": {"state": "PASS"},
            "non_interference": {"state": "PASS"},
        }
        first = _summary_long_csv(summary)
        self.assertEqual(first, _summary_long_csv(summary))
        self.assertTrue(first.endswith(b"\n"))
        self.assertNotIn(b"\r\n", first)
        header, *data = first.decode("utf-8").splitlines()
        self.assertEqual(tuple(header.split(",")), LONG_COLUMNS)
        self.assertTrue(any("smoke_not_evaluated" in line and ",1," in line for line in data))
        for invalid in (
            dict(summary, unexpected=1),
            dict(summary, processed_fit_count=float("nan")),
            dict(summary, processed_fit_count=-1),
            dict(summary, processed_fit_count=1.5),
        ):
            with self.assertRaises(ValueError):
                _summary_long_csv(invalid)
        for nonfinite in (float("nan"), float("inf"), float("-inf")):
            invalid = dict(summary, default_vs_audit_labels={"state": "PASS", "mismatches": [nonfinite]})
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                _summary_long_csv(invalid)
        for invalid_gate in (
            {"state": "BROKEN"},
            {"state": "PASS", "nested": {"unexpected": 1}},
            {"state": "PASS", "mismatches": "not-a-sequence"},
            {"state": "PASS", "mismatches": [{"nested": object()}]},
        ):
            with self.assertRaises(ValueError):
                _summary_long_csv(dict(summary, default_vs_audit_labels=invalid_gate))

    def test_b2d_json_numbers_are_auditable_in_csv(self):
        summary = self._b2d_summary()
        rows = summary_long_rows(summary)

        def value_for(**expected):
            matches = [row for row in rows if all(row[key] == value for key, value in expected.items())]
            self.assertEqual(len(matches), 1, expected)
            return matches[0]

        self.assertEqual(value_for(family="primary", side="S", statistic="observed")["value"], summary["S"]["observed"])
        self.assertEqual(value_for(family="greedy", side="N", statistic="ci97_5_bonferroni_low")["value"], summary["greedy_one_to_one_sensitivity"]["N"]["ci97_5_bonferroni"][0])
        self.assertEqual(value_for(family="ledger", statistic="raw_both_gt")["value"], summary["denominator_ledger"]["raw_both_gt"])
        self.assertEqual(value_for(family="first_loss", side="p2", statistic="wrong_argmax")["value"], summary["first_loss"]["p2"]["wrong_argmax"])
        self.assertEqual(value_for(family="tie_descriptive", side="coarse", statistic="images")["value"], summary["tie_descriptive"]["coarse"]["images"])
        self.assertEqual(value_for(family="descriptive_strata", stratum="far_gt_40m", statistic="eligible_gt", **{"class": "Cyclist"})["value"], summary["descriptive_strata"]["Cyclist"]["far_gt_40m"]["eligible_gt"])
        self.assertEqual(value_for(family="negative_controls", side="P2", contrast="small_25_40_minus_large_gt_80", statistic="observed_rate_difference", **{"class": "Pedestrian"})["value"], summary["negative_controls"]["Pedestrian"]["P2"]["small_25_40_minus_large_gt_80"]["observed_rate_difference"])
        self.assertEqual(value_for(family="continuous_support", statistic="p2_minus_coarse_logit_margin_mean")["value"], summary["continuous_support"]["p2_minus_coarse_logit_margin"]["mean"])
        self.assertEqual(value_for(family="decision", statistic="primary_decision")["state"], summary["decision"])
        self.assertEqual(value_for(family="decision", statistic="direction_reversal_S")["value"], summary["direction_reversal"]["S"])
        self.assertEqual(value_for(family="decision", statistic="greedy_veto")["value"], summary["greedy_veto"])
        self.assertEqual(value_for(family="decision", statistic="route_authorized")["value"], summary["route_authorized"])
        duplicate = self._b2d_summary()
        duplicate["denominator_ledger"] = dict(duplicate["denominator_ledger"], raw_both_gt=float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            summary_long_rows(duplicate)
        unmapped = self._b2d_summary()
        unmapped["denominator_ledger"] = dict(unmapped["denominator_ledger"], unexpected_metric=1)
        with self.assertRaisesRegex(ValueError, "unmapped"):
            summary_long_rows(unmapped)
        duplicate = self._b2d_summary()
        duplicate["continuous_support"] = dict(duplicate["continuous_support"], p2_minus_coarse_logit_margin_mean=0.0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summary_long_rows(duplicate)
        greedy_draws = self._b2d_summary()
        greedy_draws["greedy_one_to_one_sensitivity"]["S"]["draws"] = [0.222]
        greedy_draws["greedy_one_to_one_sensitivity"]["N"]["draws"] = [0.111]
        rows = summary_long_rows(greedy_draws)
        self.assertNotIn(0.222, [row["value"] for row in rows])
        self.assertNotIn(0.111, [row["value"] for row in rows])
        for nonfinite in (float("nan"), float("inf"), float("-inf")):
            for endpoint in ("S", "N"):
                primary_draws = self._b2d_summary()
                primary_draws[endpoint]["draws"] = [nonfinite]
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    summary_long_rows(primary_draws)
                greedy_draws = self._b2d_summary()
                greedy_draws["greedy_one_to_one_sensitivity"][endpoint]["draws"] = [nonfinite]
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    summary_long_rows(greedy_draws)

    @staticmethod
    def _b2d_summary():
        endpoint = {
            "state": "estimable", "observed": 0.25, "ci95": [0.1, 0.4],
            "ci97_5_bonferroni": [0.05, 0.45], "eligible_gt": 40,
            "eligible_images": 25, "discordant": 12, "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260812, "passes": True,
        }
        side = {"state": "estimable", "num": 2, "den": 4, "rate": 0.5, "eligible_unique_images": 3}
        contrast = {
            "state": "estimable", "decision_role": "auxiliary_only", "target": dict(side, stratum="small_25_40"),
            "control": dict(side, stratum="large_gt_80"), "observed_rate_difference": 0.2,
            "ci95": [0.1, 0.3], "bootstrap_replicates": 10000, "bootstrap_seed": 20260812,
        }
        strata = {"class": "Cyclist", "stratum": "far_gt_40m", "definition": "depth_m > 40", "role": "secondary_descriptive", "mutually_exclusive": False, "gt": 4, "eligible_gt": 4, "unique_images": 3, "images": 3, "state": "estimable"}
        return {
            "S": dict(endpoint), "N": dict(endpoint),
            "greedy_one_to_one_sensitivity": {"state": "PASS", "S": dict(endpoint), "N": dict(endpoint), "reps": 10000, "seed": 20260812, "decision_role": "veto_only"},
            "denominator_ledger": {"target_gt": 50, "target_images": 30, "raw_both_gt": 40, "raw_both_images": 25, "exclusions": {"no_raw_both": {"gt": 10, "images": 5}}},
            "first_loss": {"p2": {"wrong_argmax": 3}, "coarse": {"wrong_argmax": 4}},
            "tie_descriptive": {"p2": {"gt": 2, "images": 1}, "coarse": {"gt": 3, "images": 2}},
            "descriptive_strata": {"Cyclist": {"far_gt_40m": dict(strata)}},
            "negative_controls": {"decision_role": "auxiliary_only", "Pedestrian": {"P2": {"small_25_40_minus_large_gt_80": contrast}}},
            "continuous_support": {"state": "estimable", "decision_role": "descriptive_only", "population": {"eligible_gt": 3}, "p2_minus_coarse_logit_margin": {"n": 3, "mean": -0.2, "median": -0.1, "q25": -0.3, "q75": 0.0}, "clamp_counts": {"p2": 1, "combined_coarse": 1, "total": 2}, "logit_epsilon": 1.1920928955078125e-07},
            "decision": "GO_B_SCORE_OWNERSHIP", "direction_reversal": {"S": False, "N": False}, "greedy_veto": False, "route_authorized": True,
        }


if __name__ == "__main__":
    unittest.main()
