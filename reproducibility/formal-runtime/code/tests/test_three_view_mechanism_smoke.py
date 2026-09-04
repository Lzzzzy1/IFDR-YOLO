from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn


class _FakeThreeViewModel(nn.Module):
    def __init__(self, *, incomplete: bool = False) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))
        self.forward_shapes: list[tuple[int, ...]] = []
        self._contexts = None
        self.incomplete = incomplete

    def forward(self, images: torch.Tensor):
        self.forward_shapes.append(tuple(images.shape))
        signal = images.mean(dim=(1, 2, 3)).view(-1, 1, 1, 1)
        nodes = (17, 20, 23, 26) if self.incomplete else (11, 14, 17, 20, 23, 26)
        from ifdr_yolo.models.gated_fusion import ReliabilityContext

        self._contexts = {
            node: ReliabilityContext(
                factors=torch.cat((signal + node / 100.0, signal + node / 100.0), dim=1).expand(-1, 2, 4, 4),
                branch_weights=torch.full((images.shape[0], 2, 4, 4), 0.5),
                gate_strength=1.0,
            )
            for node in nodes
        }
        return images

    def consume_reliability_context(self):
        contexts = self._contexts
        self._contexts = None
        if contexts is None:
            raise RuntimeError("context was not produced")
        return contexts


class ThreeViewMechanismSmokeTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        images = root / "images"
        labels = root / "labels"
        output = root / "out"
        images.mkdir()
        labels.mkdir()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[4:12, 4:12] = (40, 80, 120)
        self.assertTrue(cv2.imwrite(str(images / "000001.png"), image))
        (labels / "000001.txt").write_text("0 0.25 0.25 0.25 0.25\n", encoding="utf-8")
        return images, labels, output

    def test_source_commit_marks_linked_worktree_safe(self):
        from scripts.run_three_view_mechanism_smoke import _source_commit

        with patch(
            "scripts.run_three_view_mechanism_smoke.subprocess.run",
            return_value=SimpleNamespace(stdout="a" * 40),
        ) as run:
            self.assertEqual(_source_commit(), "a" * 40)
        repository_root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            run.call_args.args[0],
            [
                "git",
                "-c",
                f"safe.directory={repository_root}",
                "rev-parse",
                "HEAD",
            ],
        )

    def test_smoke_runs_one_ordered_3b_forward_and_writes_provenance(self):
        from scripts.run_three_view_mechanism_smoke import run_three_view_mechanism_smoke

        with tempfile.TemporaryDirectory() as directory:
            images, labels, output = self._fixture(Path(directory))
            model = _FakeThreeViewModel()
            loaded = SimpleNamespace(model=model, checkpoint_sha256="a" * 64)
            with patch("scripts.run_three_view_mechanism_smoke.load_ifdr_checkpoint", return_value=loaded):
                summary = run_three_view_mechanism_smoke(
                    checkpoint=Path(directory) / "model.pt",
                    images=images,
                    labels=labels,
                    output_dir=output,
                    device="cpu",
                    max_images=1,
                    input_size=32,
                    seed=17,
                )
            self.assertEqual(model.forward_shapes, [(3, 3, 32, 32)])
            self.assertEqual(summary["processed_images"], 1)
            self.assertEqual(summary["checkpoint_sha256"], "a" * 64)
            self.assertIn("source_commit", summary)
            self.assertIn("image_ids_sha256", summary)
            self.assertEqual(summary["rejection_count"], 0)
            self.assertIn("aggregates", summary)
            self.assertIn("mean_gap", summary["aggregates"])
            self.assertEqual(set(summary["aggregates"]["nodes"]), {"17", "20", "23", "26"})
            self.assertEqual(summary["aggregates"]["finite_count"], 4)
            rows = [json.loads(line) for line in (output / "three_view_observations.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual([item["factor_channel"] for item in rows[0]["mechanisms"]], [0])
            self.assertTrue(all(np.isfinite(node["gap"]) for item in rows[0]["mechanisms"] for node in item["nodes"]))

    def test_background_box_has_exact_zero_iou_and_evidence_is_recoverable(self):
        from scripts.run_three_view_mechanism_smoke import run_three_view_mechanism_smoke

        with tempfile.TemporaryDirectory() as directory:
            images, labels, output = self._fixture(Path(directory))
            (labels / "000001.txt").write_text("0 0.25 0.25 0.03 0.04\n", encoding="utf-8")
            with patch(
                "scripts.run_three_view_mechanism_smoke.load_ifdr_checkpoint",
                return_value=SimpleNamespace(model=_FakeThreeViewModel(), checkpoint_sha256="b" * 64),
            ):
                run_three_view_mechanism_smoke(
                    checkpoint=Path(directory) / "model.pt",
                    images=images,
                    labels=labels,
                    output_dir=output,
                    device="cpu",
                    max_images=1,
                    input_size=32,
                    seed=17,
                )
            row = json.loads((output / "three_view_observations.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(len(row["mechanisms"]), 1)
            for mechanism in row["mechanisms"]:
                background = mechanism["background_box"]
                target = mechanism["target_box"]
                self.assertEqual(mechanism["background_max_iou"], 0.0)
                self.assertLess(target[2] - target[0], 0.05)
                self.assertLess(target[3] - target[1], 0.05)
                self.assertAlmostEqual(background[2] - background[0], target[2] - target[0], delta=1e-6)
                self.assertAlmostEqual(background[3] - background[1], target[3] - target[1], delta=1e-6)
                self.assertNotEqual(background, target)
                self.assertGreaterEqual(mechanism["severity"], 0.0)

    def test_partial_jsonl_resumes_without_repeating_completed_three_view(self):
        from scripts.run_three_view_mechanism_smoke import run_three_view_mechanism_smoke

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, labels, output = self._fixture(root)
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[8:16, 16:24] = (80, 40, 120)
            self.assertTrue(cv2.imwrite(str(images / "000002.png"), image))
            (labels / "000002.txt").write_text("0 0.625 0.375 0.25 0.25\n", encoding="utf-8")
            first_model = _FakeThreeViewModel()
            with patch(
                "scripts.run_three_view_mechanism_smoke.load_ifdr_checkpoint",
                return_value=SimpleNamespace(model=first_model, checkpoint_sha256="d" * 64),
            ):
                run_three_view_mechanism_smoke(
                    checkpoint=root / "model.pt", images=images, labels=labels,
                    output_dir=output, device="cpu", max_images=2, input_size=32, seed=17,
                )
            observations = output / "three_view_observations.jsonl"
            lines = observations.read_text(encoding="utf-8").splitlines(keepends=True)
            observations.write_text(lines[0], encoding="utf-8")
            (output / "three_view_summary.json").unlink()
            resumed_model = _FakeThreeViewModel()
            with patch(
                "scripts.run_three_view_mechanism_smoke.load_ifdr_checkpoint",
                return_value=SimpleNamespace(model=resumed_model, checkpoint_sha256="d" * 64),
            ):
                summary = run_three_view_mechanism_smoke(
                    checkpoint=root / "model.pt", images=images, labels=labels,
                    output_dir=output, device="cpu", max_images=2, input_size=32, seed=17,
                )
            self.assertEqual(resumed_model.forward_shapes, [(3, 3, 32, 32)])
            self.assertEqual(summary["processed_images"], 2)

    def test_existing_summary_is_validated_and_never_overwritten(self):
        from scripts.run_three_view_mechanism_smoke import run_three_view_mechanism_smoke

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, labels, output = self._fixture(root)
            with patch(
                "scripts.run_three_view_mechanism_smoke.load_ifdr_checkpoint",
                return_value=SimpleNamespace(model=_FakeThreeViewModel(), checkpoint_sha256="e" * 64),
            ):
                run_three_view_mechanism_smoke(
                    checkpoint=root / "model.pt", images=images, labels=labels,
                    output_dir=output, device="cpu", max_images=1, input_size=32, seed=17,
                )
            summary_path = output / "three_view_summary.json"
            observations_path = output / "three_view_observations.jsonl"
            summary_before = summary_path.read_text(encoding="utf-8")
            observations_before = observations_path.read_text(encoding="utf-8")
            resumed_model = _FakeThreeViewModel()
            with patch(
                "scripts.run_three_view_mechanism_smoke.load_ifdr_checkpoint",
                return_value=SimpleNamespace(model=resumed_model, checkpoint_sha256="e" * 64),
            ):
                run_three_view_mechanism_smoke(
                    checkpoint=root / "model.pt", images=images, labels=labels,
                    output_dir=output, device="cpu", max_images=1, input_size=32, seed=17,
                )
            self.assertEqual(resumed_model.forward_shapes, [])
            self.assertEqual(summary_path.read_text(encoding="utf-8"), summary_before)
            self.assertEqual(observations_path.read_text(encoding="utf-8"), observations_before)

    def test_missing_or_incomplete_evidence_fails_closed(self):
        from scripts.run_three_view_mechanism_smoke import run_three_view_mechanism_smoke

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, labels, output = self._fixture(root)
            with patch(
                "scripts.run_three_view_mechanism_smoke.load_ifdr_checkpoint",
                return_value=SimpleNamespace(model=_FakeThreeViewModel(incomplete=True), checkpoint_sha256="c" * 64),
            ):
                with self.assertRaisesRegex(ValueError, "nodes|contexts|evidence"):
                    run_three_view_mechanism_smoke(
                        checkpoint=root / "model.pt",
                        images=images,
                        labels=labels,
                        output_dir=output,
                        device="cpu",
                        max_images=1,
                        input_size=32,
                        seed=17,
                    )
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                run_three_view_mechanism_smoke(
                    checkpoint=root / "missing.pt",
                    images=images,
                    labels=labels,
                    output_dir=output,
                    device="cpu",
                    max_images=1,
                    input_size=32,
                    seed=17,
                )


if __name__ == "__main__":
    unittest.main()
