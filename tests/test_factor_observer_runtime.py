import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn

from ifdr_yolo.data.natural_degradation import NaturalDegradationRecord
from ifdr_yolo.eval.factor_observer import (
    FactorObservationJournal,
    build_factor_observation_manifest,
)
from ifdr_yolo.eval.factor_observer_runtime import (
    LoadedIFDRCheckpoint,
    _transform_seed_for_condition,
    run_factor_observer,
)


class _CheckpointModel(nn.Module):
    def __init__(self, *, context: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.context_enabled = context

    def consume_reliability_context(self):
        if not self.context_enabled:
            raise AssertionError("context should not be consumed during load")
        return {}


def _context(*, batch: int = 2, height: int = 4, width: int = 6):
    factors = torch.tensor(
        [[[[0.2] * width] * height, [[0.8] * width] * height]]
        * batch,
        dtype=torch.float32,
    )
    branches = torch.tensor(
        [[[[0.25] * width] * height, [[0.75] * width] * height]]
        * batch,
        dtype=torch.float32,
    )
    return SimpleNamespace(
        factors=factors,
        branch_weights=branches,
        gate_strength=0.5,
    )


def _contexts():
    return {
        node: _context(height=index + 2, width=index + 3)
        for index, node in enumerate((11, 14, 17, 20, 23, 26))
    }


class _RunnerModel(nn.Module):
    def __init__(self, *, missing_node: int | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.forward_calls = 0
        self.batch_sizes: list[int] = []
        self.consume_calls = 0
        self.inference_flags: list[bool] = []
        self.missing_node = missing_node
        self._contexts = None

    def forward(self, batch):
        self.forward_calls += 1
        self.batch_sizes.append(int(batch.shape[0]))
        self.inference_flags.append(torch.is_inference_mode_enabled())
        contexts = {}
        for index, node in enumerate((11, 14, 17, 20, 23, 26)):
            if node == self.missing_node:
                continue
            height, width = index + 2, index + 3
            value = batch[:, 0].mean(dim=(1, 2)).view(-1, 1, 1, 1).expand(-1, 2, height, width)
            factors = value.clamp(0.0, 1.0).to(dtype=torch.float32)
            branches = torch.cat((torch.full_like(factors[:, :1], 0.25), torch.full_like(factors[:, :1], 0.75)), dim=1)
            contexts[node] = SimpleNamespace(
                factors=factors,
                branch_weights=branches,
                gate_strength=0.5,
            )
        self._contexts = contexts
        return batch

    def consume_reliability_context(self):
        self.consume_calls += 1
        contexts = self._contexts
        self._contexts = None
        return contexts


def _runner_record() -> NaturalDegradationRecord:
    return NaturalDegradationRecord(
        image_id="runner-image",
        object_id=0,
        class_id=0,
        class_name="Car",
        bbox_xyxy=(4.0, 4.0, 12.0, 12.0),
        box_height=8.0,
        depth_m=20.0,
        depth_available=True,
        occlusion_level=0,
        truncation=0.0,
        sampling_score=0.1,
        visibility_score=0.2,
    )


def _runner_fixture(directory: str):
    root = Path(directory)
    image_path = root / "runner-image.png"
    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    image[4:12, 4:12] = (20, 30, 40)
    encoded_ok, encoded = cv2.imencode(".png", image)
    if not encoded_ok:
        raise AssertionError("failed to write fixture PNG")
    image_path.write_bytes(encoded.tobytes())
    checkpoint_sha256 = "ab" * 32
    manifest = build_factor_observation_manifest(
        [_runner_record()],
        {"runner-image": image_path},
        [("runner-image", 0)],
        checkpoint_sha256,
        seed=17,
        input_size=32,
    )
    return image_path, manifest


def _runner_fixture_two(directory: str):
    root = Path(directory)
    first_path, _ = _runner_fixture(directory)
    second_path = root / "runner-image-2.png"
    second_path.write_bytes(first_path.read_bytes())
    second_record = NaturalDegradationRecord(
        image_id="runner-image-2",
        object_id=0,
        class_id=0,
        class_name="Car",
        bbox_xyxy=(8.0, 8.0, 16.0, 16.0),
        box_height=8.0,
        depth_m=20.0,
        depth_available=True,
        occlusion_level=0,
        truncation=0.0,
        sampling_score=0.1,
        visibility_score=0.2,
    )
    manifest = build_factor_observation_manifest(
        [_runner_record(), second_record],
        {"runner-image": first_path, "runner-image-2": second_path},
        [("runner-image", 0), ("runner-image-2", 0)],
        "ab" * 32,
        seed=17,
        input_size=32,
    )
    return manifest


class LoadedIFDRCheckpointTest(unittest.TestCase):
    def test_loader_prefers_ema_and_records_hash_device_and_eval(self) -> None:
        from ifdr_yolo.eval.factor_observer_runtime import load_ifdr_checkpoint

        ema = _CheckpointModel()
        ema.role = "ema"
        ema.train()
        model = _CheckpointModel()
        model.role = "model"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"ema": ema, "model": model}, path)
            raw = path.read_bytes()
            loaded = load_ifdr_checkpoint(path, device="cpu")
        self.assertEqual(loaded.model.role, "ema")
        self.assertEqual(loaded.checkpoint_sha256, hashlib.sha256(raw).hexdigest())
        self.assertFalse(loaded.model.training)
        self.assertEqual(loaded.model.weight.device.type, "cpu")

    def test_loader_falls_back_to_model_when_ema_is_none(self) -> None:
        from ifdr_yolo.eval.factor_observer_runtime import load_ifdr_checkpoint

        model = _CheckpointModel()
        model.role = "model"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"ema": None, "model": model}, path)
            loaded = load_ifdr_checkpoint(path)
        self.assertEqual(loaded.model.role, "model")

    def test_loader_hashes_and_loads_the_same_bytes_buffer(self) -> None:
        from ifdr_yolo.eval.factor_observer_runtime import load_ifdr_checkpoint

        captured = []

        def fake_load(source, *, map_location, weights_only):
            captured.append((source, map_location, weights_only))
            self.assertIsInstance(source, io.BytesIO)
            return {"model": _CheckpointModel()}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            path.write_bytes(b"trusted-bytes")
            with patch("ifdr_yolo.eval.factor_observer_runtime.torch.load", side_effect=fake_load):
                loaded = load_ifdr_checkpoint(path, device="cpu")
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0].getvalue(), b"trusted-bytes")
        self.assertEqual(captured[0][1], "cpu")
        self.assertFalse(captured[0][2])
        self.assertEqual(loaded.checkpoint_sha256, hashlib.sha256(b"trusted-bytes").hexdigest())

    def test_loader_converts_half_precision_checkpoint_to_float32(self) -> None:
        from ifdr_yolo.eval.factor_observer_runtime import load_ifdr_checkpoint

        model = _CheckpointModel().half()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "half.pt"
            torch.save({"model": model}, path)
            loaded = load_ifdr_checkpoint(path, device="cpu")
        self.assertEqual(loaded.model.weight.dtype, torch.float32)

    def test_loader_rejects_empty_missing_and_non_ifdr_checkpoints(self) -> None:
        from ifdr_yolo.eval.factor_observer_runtime import load_ifdr_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "empty.pt"
            empty.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "empty"):
                load_ifdr_checkpoint(empty)
            missing = root / "missing.pt"
            torch.save({"epoch": 1}, missing)
            with self.assertRaisesRegex(ValueError, "ema|model"):
                load_ifdr_checkpoint(missing)
            no_ifdr = root / "no_ifdr.pt"
            torch.save({"model": nn.Linear(1, 1)}, no_ifdr)
            with self.assertRaisesRegex(ValueError, "consume_reliability_context"):
                load_ifdr_checkpoint(no_ifdr)


class PooledReliabilityTest(unittest.TestCase):
    def test_pooling_maps_each_node_roi_and_returns_means(self) -> None:
        from ifdr_yolo.eval.factor_observer import LetterboxGeometry
        from ifdr_yolo.eval.factor_observer_runtime import pool_reliability_contexts

        geometry = LetterboxGeometry(
            original_width=100,
            original_height=50,
            input_size=200,
            scale=2.0,
            resized_width=200,
            resized_height=100,
            pad_left=0,
            pad_top=50,
            pad_right=0,
            pad_bottom=50,
        )
        pooled = pool_reliability_contexts(
            _contexts(),
            batch_index=1,
            bbox_xyxy=(10.0, 5.0, 40.0, 25.0),
            geometry=geometry,
        )
        self.assertEqual(tuple(item.node for item in pooled), (11, 14, 17, 20, 23, 26))
        for index, item in enumerate(pooled):
            self.assertEqual(item.feature_shape, (index + 2, index + 3))
            self.assertGreater(item.roi_xyxy[2], item.roi_xyxy[0])
            self.assertGreater(item.roi_xyxy[3], item.roi_xyxy[1])
            self.assertAlmostEqual(item.sampling, 0.2)
            self.assertAlmostEqual(item.visibility, 0.8)
            self.assertAlmostEqual(item.branch_weights[0], 0.25)
            self.assertAlmostEqual(item.branch_weights[1], 0.75)
            self.assertAlmostEqual(item.gate_strength, 0.5)

    def test_pooling_rejects_bad_nodes_batch_shapes_values_and_weights(self) -> None:
        from ifdr_yolo.eval.factor_observer import LetterboxGeometry
        from ifdr_yolo.eval.factor_observer_runtime import pool_reliability_contexts

        geometry = LetterboxGeometry(
            original_width=20,
            original_height=20,
            input_size=20,
            scale=1.0,
            resized_width=20,
            resized_height=20,
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
        )
        kwargs = {
            "batch_index": 0,
            "bbox_xyxy": (1.0, 1.0, 10.0, 10.0),
            "geometry": geometry,
        }
        for contexts in (
            {node: _context() for node in (11, 14, 17, 20, 23)},
            {**_contexts(), 29: _context()},
        ):
            with self.assertRaisesRegex(ValueError, "nodes"):
                pool_reliability_contexts(contexts, **kwargs)
        with self.assertRaisesRegex(ValueError, "batch_index"):
            pool_reliability_contexts(_contexts(), batch_index=2, **{key: value for key, value in kwargs.items() if key != "batch_index"})
        malformed = _contexts()
        malformed[11] = type("Context", (), {"factors": torch.zeros(2, 3, 4, 6), "branch_weights": torch.zeros(2, 2, 4, 6), "gate_strength": 0.5})()
        with self.assertRaisesRegex(ValueError, "B2HW|shape"):
            pool_reliability_contexts(malformed, **kwargs)
        malformed = _contexts()
        malformed[11].factors[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            pool_reliability_contexts(malformed, **kwargs)
        malformed = _contexts()
        malformed[11].branch_weights[0, 0].fill_(0.7)
        with self.assertRaisesRegex(ValueError, "sum"):
            pool_reliability_contexts(malformed, **kwargs)
        malformed = _contexts()
        malformed[11].gate_strength = 1.5
        with self.assertRaisesRegex(ValueError, "gate_strength"):
            pool_reliability_contexts(malformed, **kwargs)
        malformed = _contexts()
        malformed[26] = _context(batch=3, height=7, width=8)
        with self.assertRaisesRegex(ValueError, "batch"):
            pool_reliability_contexts(malformed, **kwargs)

    def test_pooling_uses_exact_letterbox_roi_for_gradient_features(self) -> None:
        from ifdr_yolo.eval.factor_observer import LetterboxGeometry
        from ifdr_yolo.eval.factor_observer_runtime import pool_reliability_contexts

        geometry = LetterboxGeometry(
            original_width=100,
            original_height=50,
            input_size=200,
            scale=2.0,
            resized_width=200,
            resized_height=100,
            pad_left=0,
            pad_top=50,
            pad_right=0,
            pad_bottom=50,
        )
        contexts = _contexts()
        factors = torch.zeros_like(contexts[11].factors)
        factors[:, 0] = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        factors[:, 1] = torch.tensor([[0.6, 0.5, 0.4], [0.3, 0.2, 0.1]])
        branches = torch.zeros_like(contexts[11].branch_weights)
        branches[:, 0] = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        branches[:, 1] = 1.0 - branches[:, 0]
        contexts[11] = SimpleNamespace(
            factors=factors,
            branch_weights=branches,
            gate_strength=0.5,
        )
        pooled = pool_reliability_contexts(
            contexts,
            batch_index=1,
            bbox_xyxy=(10.0, 5.0, 40.0, 25.0),
            geometry=geometry,
        )
        first = pooled[0]
        self.assertEqual(first.roi_xyxy, (0, 0, 2, 1))
        self.assertAlmostEqual(first.sampling, 0.15)
        self.assertAlmostEqual(first.visibility, 0.55)
        self.assertAlmostEqual(first.branch_weights[0], 0.15)
        self.assertAlmostEqual(first.branch_weights[1], 0.85)


class FactorObserverRunnerTest(unittest.TestCase):
    def test_transform_seed_protocol_and_common_random_numbers(self) -> None:
        self.assertEqual(
            _transform_seed_for_condition(
                SimpleNamespace(intervention_kind="sampling", pair_id="ab" * 32)
            ),
            757744265707348135,
        )
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = _runner_fixture(directory)
            conditions = manifest.plans[0].conditions
            sampling = next(condition for condition in conditions if condition.intervention_kind == "sampling")
            expected = int.from_bytes(
                hashlib.sha256(
                    b"ifdr-observer-transform-v1\0" + sampling.pair_id.encode("ascii")
                ).digest()[:8],
                "big",
            ) & ((1 << 63) - 1)
            self.assertEqual(_transform_seed_for_condition(sampling), expected)
            self.assertEqual(
                {
                    _transform_seed_for_condition(condition)
                    for condition in conditions
                    if condition.pair_id == sampling.pair_id
                    and condition.intervention_kind == sampling.intervention_kind
                },
                {expected},
            )
            visibility = next(condition for condition in conditions if condition.intervention_kind == "visibility")
            self.assertNotEqual(_transform_seed_for_condition(sampling), _transform_seed_for_condition(visibility))

    def test_runner_groups_transforms_microbatches_and_emits_exact_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = _runner_fixture(directory)
            model = _RunnerModel()
            loaded = LoadedIFDRCheckpoint(model=model, checkpoint_sha256=manifest.checkpoint_sha256)
            journal = FactorObservationJournal(
                manifest,
                Path(directory) / "observations.jsonl",
                Path(directory) / "progress.json",
            )
            summary = run_factor_observer(loaded, manifest, journal, transform_batch_size=3)
            self.assertEqual(summary["status"], "complete")
            transform_count = len({condition.transform_id for condition in manifest.plans[0].conditions})
            self.assertEqual(sum(model.batch_sizes), transform_count)
            self.assertEqual(model.forward_calls, (transform_count + 2) // 3)
            self.assertEqual(model.consume_calls, model.forward_calls)
            self.assertTrue(all(model.inference_flags))
            rows = [json.loads(line) for line in journal.output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), manifest.expected_observation_count)
            self.assertEqual({row["observation_id"] for row in rows}, set(manifest.expected_observation_ids))
            self.assertTrue(all(row["schema_version"] == 1 for row in rows))
            self.assertEqual(set(rows[0]), {
                "schema_version", "manifest_sha256", "observation_id", "condition_id", "transform_id",
                "checkpoint_sha256", "source_sha256", "seed", "transform_seed", "node_id", "image_id",
                "object_id", "class_id", "class_name", "bbox_xyxy", "box_height", "natural_sampling",
                "natural_visibility", "region_xyxy", "region_role", "intervention_kind", "intervention_factor",
                "intervention_severity", "pair_id", "matched_background_bbox", "predicted_sampling",
                "predicted_visibility", "branch_weights", "gate_strength", "feature_roi_xyxy", "feature_shape",
                "input_shape",
            })
            self.assertTrue(all(isinstance(row["feature_roi_xyxy"], list) for row in rows))
            by_pair = {}
            for row in rows:
                if row["intervention_kind"] in {"sampling", "visibility"}:
                    by_pair.setdefault(row["pair_id"], set()).add(row["transform_seed"])
                else:
                    self.assertIsNone(row["transform_seed"])
            self.assertTrue(by_pair)
            self.assertTrue(all(len(seeds) == 1 for seeds in by_pair.values()))

    def test_runner_missing_context_does_not_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = _runner_fixture(directory)
            model = _RunnerModel(missing_node=26)
            loaded = LoadedIFDRCheckpoint(model=model, checkpoint_sha256=manifest.checkpoint_sha256)
            output = Path(directory) / "observations.jsonl"
            progress = Path(directory) / "progress.json"
            journal = FactorObservationJournal(manifest, output, progress)
            with self.assertRaisesRegex(ValueError, "nodes"):
                run_factor_observer(loaded, manifest, journal, transform_batch_size=2)
            self.assertEqual(output.read_bytes(), b"")
            self.assertEqual(journal.completed_image_ids, frozenset())

    def test_runner_resume_audits_rows_before_forward(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = _runner_fixture(directory)
            output = Path(directory) / "observations.jsonl"
            progress = Path(directory) / "progress.json"
            first_model = _RunnerModel()
            first_loaded = LoadedIFDRCheckpoint(first_model, manifest.checkpoint_sha256)
            first_journal = FactorObservationJournal(manifest, output, progress)
            run_factor_observer(first_loaded, manifest, first_journal)
            lines = output.read_text(encoding="utf-8").splitlines()
            tampered = json.loads(lines[0])
            tampered["predicted_sampling"] = 1.25
            lines[0] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
            output.write_text("\n".join(lines) + "\n", encoding="utf-8")
            state = json.loads(progress.read_text(encoding="utf-8"))
            entry = state["completed"][manifest.plans[0].image_id]
            entry["end_offset"] = output.stat().st_size
            entry["rows_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
            progress.write_text(json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
            second_model = _RunnerModel()
            second_loaded = LoadedIFDRCheckpoint(second_model, manifest.checkpoint_sha256)
            with self.assertRaisesRegex(ValueError, "predicted_sampling"):
                second_journal = FactorObservationJournal(manifest, output, progress)
                run_factor_observer(second_loaded, manifest, second_journal)
            self.assertEqual(second_model.forward_calls, 0)

    def test_runner_complete_rerun_is_zero_forward_and_zero_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = _runner_fixture(directory)
            output = Path(directory) / "observations.jsonl"
            progress = Path(directory) / "progress.json"
            first_model = _RunnerModel()
            run_factor_observer(
                LoadedIFDRCheckpoint(first_model, manifest.checkpoint_sha256),
                manifest,
                FactorObservationJournal(manifest, output, progress),
            )
            before = output.read_bytes()
            second_model = _RunnerModel()
            run_factor_observer(
                LoadedIFDRCheckpoint(second_model, manifest.checkpoint_sha256),
                manifest,
                FactorObservationJournal(manifest, output, progress),
            )
            self.assertEqual(second_model.forward_calls, 0)
            self.assertEqual(output.read_bytes(), before)

    def test_runner_interventions_use_exact_target_and_background_regions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = _runner_fixture(directory)
            output = Path(directory) / "observations.jsonl"
            progress = Path(directory) / "progress.json"
            model = _RunnerModel()
            from ifdr_yolo.data.interventions import apply_intervention as real_apply

            with patch(
                "ifdr_yolo.eval.factor_observer_runtime.apply_intervention",
                wraps=real_apply,
            ) as applied:
                run_factor_observer(
                    LoadedIFDRCheckpoint(model, manifest.checkpoint_sha256),
                    manifest,
                    FactorObservationJournal(manifest, output, progress),
                )
            self.assertEqual(applied.call_count, 16)
            for call in applied.call_args_list:
                spec = call.args[1]
                condition = next(
                    condition
                    for condition in manifest.plans[0].conditions
                    if condition.intervention_kind == spec.kind.value
                    and ("target" if spec.role.value == "object" else "background") == condition.region_role
                    and abs(condition.intervention_severity - spec.strength) < 1e-9
                    and condition.intervention_factor == spec.kind.value
                )
                expected_bbox = condition.bbox_xyxy
                self.assertEqual(
                    "target" if spec.role.value == "object" else "background",
                    condition.region_role,
                )
                self.assertEqual(
                    spec.region_xyxy,
                    tuple(
                        value / size
                        for value, size in zip(
                            expected_bbox,
                            (manifest.plans[0].width, manifest.plans[0].height) * 2,
                        )
                    ),
                )
                self.assertEqual(spec.seed, _transform_seed_for_condition(condition))

    def test_completed_image_rechecks_png_before_zero_forward_skip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path, manifest = _runner_fixture(directory)
            output = Path(directory) / "observations.jsonl"
            progress = Path(directory) / "progress.json"
            run_factor_observer(
                LoadedIFDRCheckpoint(_RunnerModel(), manifest.checkpoint_sha256),
                manifest,
                FactorObservationJournal(manifest, output, progress),
            )
            image_path.write_bytes(image_path.read_bytes() + b"trailing-png-bytes")
            second_model = _RunnerModel()
            with self.assertRaisesRegex(ValueError, "source hash"):
                run_factor_observer(
                    LoadedIFDRCheckpoint(second_model, manifest.checkpoint_sha256),
                    manifest,
                    FactorObservationJournal(manifest, output, progress),
                )
            self.assertEqual(second_model.forward_calls, 0)

    def test_manifest_hash_is_constant_per_run_not_per_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = _runner_fixture(directory)
            output = Path(directory) / "observations.jsonl"
            progress = Path(directory) / "progress.json"
            journal = FactorObservationJournal(manifest, output, progress)
            with patch.object(
                type(manifest),
                "hash",
                wraps=manifest.hash,
            ) as hash_method:
                run_factor_observer(
                    LoadedIFDRCheckpoint(_RunnerModel(), manifest.checkpoint_sha256),
                    manifest,
                    journal,
                )
            self.assertLessEqual(hash_method.call_count, 5)

    def test_resume_validates_each_completed_image_block_in_stream_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _runner_fixture_two(directory)
            output = Path(directory) / "observations.jsonl"
            progress = Path(directory) / "progress.json"
            run_factor_observer(
                LoadedIFDRCheckpoint(_RunnerModel(), manifest.checkpoint_sha256),
                manifest,
                FactorObservationJournal(manifest, output, progress),
            )
            from ifdr_yolo.eval import factor_observer_runtime as runtime

            with patch.object(
                runtime,
                "validate_observation_rows",
                wraps=runtime.validate_observation_rows,
            ) as validator:
                run_factor_observer(
                    LoadedIFDRCheckpoint(_RunnerModel(), manifest.checkpoint_sha256),
                    manifest,
                    FactorObservationJournal(manifest, output, progress),
                )
            self.assertEqual(
                [call.kwargs["plan"].image_id for call in validator.call_args_list],
                list(manifest.image_ids),
            )


if __name__ == "__main__":
    unittest.main()
