from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from ifdr_yolo.eval.factor_repair_gate import PRIMARY_ENDPOINTS, FactorRepairEvidence


def _evidence(condition: str, value: float) -> FactorRepairEvidence:
    samples = {
        name: tuple(value + 0.01 * index for index in range(5))
        for name in PRIMARY_ENDPOINTS
    }
    return FactorRepairEvidence(
        condition=condition,
        image_ids_hash="a" * 64,
        image_ids=tuple(f"image-{index}" for index in range(5)),
        endpoints={name: value for name in PRIMARY_ENDPOINTS},
        evidence_sha256=(condition.lower() * 64)[:64],
        endpoint_samples=samples,
    )


class ResumableFactorBootstrapTest(unittest.TestCase):
    def test_non_boundary_interrupt_resumes_to_uninterrupted_ordered_vector(self) -> None:
        from ifdr_yolo.eval.resumable_factor_bootstrap import run_resumable_factor_bootstrap

        f0 = _evidence("F0", 0.10)
        f1 = _evidence("F1", 0.30)
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as clean:
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_resumable_factor_bootstrap(
                    f1,
                    f0,
                    Path(directory),
                    condition="F1",
                    total_replicates=11,
                    checkpoint_interval=3,
                    stop_after=4,
                )
            checkpoint = Path(directory) / "checkpoints" / "F1.json"
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(payload["state"], "running")
            self.assertEqual(payload["completed"], 3)
            resumed = run_resumable_factor_bootstrap(
                f1,
                f0,
                Path(directory),
                condition="F1",
                total_replicates=11,
                checkpoint_interval=3,
                resume=True,
            )
            uninterrupted = run_resumable_factor_bootstrap(
                f1,
                f0,
                Path(clean),
                condition="F1",
                total_replicates=11,
                checkpoint_interval=3,
            )
            self.assertEqual(resumed.replicates, uninterrupted.replicates)
            self.assertEqual(resumed.point, uninterrupted.point)
            self.assertEqual(resumed.ci95, uninterrupted.ci95)

    def test_existing_checkpoint_requires_explicit_resume(self) -> None:
        from ifdr_yolo.eval.resumable_factor_bootstrap import run_resumable_factor_bootstrap

        f0 = _evidence("F0", 0.10)
        f1 = _evidence("F1", 0.30)
        with tempfile.TemporaryDirectory() as directory:
            run_resumable_factor_bootstrap(
                f1, f0, Path(directory), condition="F1", total_replicates=4, checkpoint_interval=2
            )
            with self.assertRaisesRegex(ValueError, "resume"):
                run_resumable_factor_bootstrap(
                    f1, f0, Path(directory), condition="F1", total_replicates=4, checkpoint_interval=2
                )

    def test_resume_rejects_identity_mismatch_before_computation(self) -> None:
        from ifdr_yolo.eval.resumable_factor_bootstrap import run_resumable_factor_bootstrap

        f0 = _evidence("F0", 0.10)
        f1 = _evidence("F1", 0.30)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                run_resumable_factor_bootstrap(
                    f1,
                    f0,
                    Path(directory),
                    condition="F1",
                    total_replicates=8,
                    checkpoint_interval=2,
                    stop_after=1,
                )
            changed = _evidence("F1", 0.31)
            with self.assertRaisesRegex(ValueError, "identity|hash|mismatch"):
                run_resumable_factor_bootstrap(
                    changed,
                    f0,
                    Path(directory),
                    condition="F1",
                    total_replicates=8,
                    checkpoint_interval=2,
                    resume=True,
                )

    def test_checkpoint_records_wall_clock_resume_fields(self) -> None:
        from ifdr_yolo.eval.resumable_factor_bootstrap import run_resumable_factor_bootstrap

        f0 = _evidence("F0", 0.10)
        f1 = _evidence("F1", 0.30)
        with tempfile.TemporaryDirectory() as directory:
            run_resumable_factor_bootstrap(
                f1,
                f0,
                Path(directory),
                condition="F1",
                total_replicates=4,
                checkpoint_interval=4,
            )
            checkpoint = json.loads(
                (Path(directory) / "checkpoints" / "F1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["version"], 1)
            self.assertIn("last_saved_at", checkpoint)
            self.assertEqual(checkpoint["completed_range"], [0, 4])
            self.assertEqual(checkpoint["next_replicate_index"], 4)
            self.assertEqual(checkpoint["output_paths"]["checkpoint"], str(Path(directory).resolve() / "checkpoints" / "F1.json"))

    def test_wall_clock_forces_checkpoint_inside_large_interval(self) -> None:
        from ifdr_yolo.eval.resumable_factor_bootstrap import run_resumable_factor_bootstrap

        f0 = _evidence("F0", 0.10)
        f1 = _evidence("F1", 0.30)
        with tempfile.TemporaryDirectory() as directory:
            monotonic = iter((0.0, 301.0, 301.0, 301.0, 301.0, 301.0, 301.0))
            with patch(
                "ifdr_yolo.eval.resumable_factor_bootstrap.time.monotonic",
                side_effect=lambda: next(monotonic),
            ):
                run_resumable_factor_bootstrap(
                    f1,
                    f0,
                    Path(directory),
                    condition="F1",
                    total_replicates=3,
                    checkpoint_interval=100,
                    checkpoint_wall_time_seconds=300.0,
                )
            checkpoint = json.loads(
                (Path(directory) / "checkpoints" / "F1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["state"], "complete")
            self.assertEqual(checkpoint["completed"], 3)

    def test_shared_reference_draws_are_computed_once_for_three_candidates(self) -> None:
        from types import SimpleNamespace

        from ifdr_yolo.eval.factor_repair_gate import PRIMARY_ENDPOINTS
        from ifdr_yolo.eval.resumable_factor_bootstrap import (
            build_shared_reference_draws,
            run_resumable_factor_bootstrap,
        )

        calls = {"F0": 0, "F1": 0, "F2": 0, "F3": 0}

        def evidence(condition: str, value: float):
            def recompute(_indices):
                calls[condition] += 1
                return {name: value for name in PRIMARY_ENDPOINTS}

            return SimpleNamespace(
                condition=condition,
                image_ids=("a", "b", "c"),
                image_ids_hash="same",
                evidence_sha256=condition.lower() * 64,
                endpoints={name: value for name in PRIMARY_ENDPOINTS},
                complete=True,
                recompute_endpoints=recompute,
            )

        f0 = evidence("F0", 0.10)
        candidates = [evidence(condition, 0.20 + index * 0.01) for index, condition in enumerate(("F1", "F2", "F3"))]
        with tempfile.TemporaryDirectory() as directory:
            cache = build_shared_reference_draws(f0, Path(directory), total_replicates=5, checkpoint_interval=2)
            for candidate in candidates:
                run_resumable_factor_bootstrap(
                    candidate,
                    f0,
                    Path(directory),
                    condition=candidate.condition,
                    total_replicates=5,
                    checkpoint_interval=2,
                    reference_draws=cache,
                )
        self.assertEqual(calls["F0"], 5)
        self.assertEqual(calls["F1"], 5)
        self.assertEqual(calls["F2"], 5)
        self.assertEqual(calls["F3"], 5)

    def test_resume_elapsed_excludes_wall_clock_downtime(self) -> None:
        from ifdr_yolo.eval.resumable_factor_bootstrap import run_resumable_factor_bootstrap

        f0 = _evidence("F0", 0.10)
        f1 = _evidence("F1", 0.30)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_resumable_factor_bootstrap(
                    f1,
                    f0,
                    Path(directory),
                    condition="F1",
                    total_replicates=6,
                    checkpoint_interval=2,
                    stop_after=2,
                )
            checkpoint_path = Path(directory) / "checkpoints" / "F1.json"
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            payload["started_at"] = time.time() - 10_000.0
            payload["elapsed_seconds"] = 42.5
            checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")
            resumed = run_resumable_factor_bootstrap(
                f1,
                f0,
                Path(directory),
                condition="F1",
                total_replicates=6,
                checkpoint_interval=2,
                resume=True,
            )
            self.assertGreaterEqual(resumed.elapsed_seconds, 42.5)
            self.assertLess(resumed.elapsed_seconds, 60.0)

    def test_checkpoint_distinguishes_canonical_and_source_file_hashes(self) -> None:
        from ifdr_yolo.eval.resumable_factor_bootstrap import run_resumable_factor_bootstrap

        f0 = _evidence("F0", 0.10)
        f1 = _evidence("F1", 0.30)
        source_hashes = {"candidate": "b" * 64, "reference": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            run_resumable_factor_bootstrap(
                f1,
                f0,
                Path(directory),
                condition="F1",
                total_replicates=3,
                checkpoint_interval=2,
                source_file_sha256=source_hashes,
            )
            payload = json.loads((Path(directory) / "checkpoints" / "F1.json").read_text(encoding="utf-8"))
            identity = payload["identity"]
            self.assertEqual(identity["evidence_canonical_sha256"], {"candidate": f1.evidence_sha256, "reference": f0.evidence_sha256})
            self.assertEqual(identity["source_file_sha256"], source_hashes)
            with self.assertRaisesRegex(ValueError, "identity|source|mismatch"):
                run_resumable_factor_bootstrap(
                    f1,
                    f0,
                    Path(directory),
                    condition="F1",
                    total_replicates=3,
                    checkpoint_interval=2,
                    source_file_sha256={"candidate": "c" * 64, "reference": "a" * 64},
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
