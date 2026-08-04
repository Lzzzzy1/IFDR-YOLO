import hashlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn


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


if __name__ == "__main__":
    unittest.main()
