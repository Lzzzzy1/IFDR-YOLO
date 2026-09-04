from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ifdr_yolo.eval.selection_gate import SelectionDecision
from ifdr_yolo.experiments.config import load_ifdr_config
from scripts.run_evidence_queue import EvidenceQueueResult
from scripts.run_protected_evidence_queue import (
    PRIMARY_SPEC,
    REPLICATION_SPECS,
    ProtectedEvidenceQueue,
)


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "abcdef1234567890abcdef1234567890abcdef12"


class FakeQueue:
    def __init__(
        self,
        *,
        specs,
        run_dir: Path,
        failed: tuple[str, ...] = (),
        **kwargs,
    ) -> None:
        del kwargs
        self.specs = tuple(specs)
        self.run_dir = run_dir
        self.failed = failed

    def run(self) -> EvidenceQueueResult:
        completed = () if self.failed else tuple(spec.key for spec in self.specs)
        return EvidenceQueueResult(completed=completed, failed=self.failed)

    def complete_run(self, spec):
        return self.run_dir if spec in self.specs and not self.failed else None


class QueueFactory:
    def __init__(self, run_dir: Path, *, primary_failure: bool = False) -> None:
        self.run_dir = run_dir
        self.primary_failure = primary_failure
        self.calls: list[tuple[int, ...]] = []

    def __call__(self, **kwargs):
        specs = tuple(kwargs["specs"])
        self.calls.append(tuple(spec.seed for spec in specs))
        failed = (specs[0].key,) if self.primary_failure and len(self.calls) == 1 else ()
        return FakeQueue(specs=specs, run_dir=self.run_dir, failed=failed)


def _decision(advance: bool) -> SelectionDecision:
    return SelectionDecision(
        advance=advance,
        candidate_mean_ap40=65.0,
        full_mean_ap40=63.5,
        fusion_mean_ap40=66.0,
        failed_checks=() if advance else ("performance",),
    )


class ProtectedEvidenceQueueTest(unittest.TestCase):
    def test_declares_locked_three_seed_300_epoch_configs(self) -> None:
        specs = (PRIMARY_SPEC, *REPLICATION_SPECS)

        self.assertEqual([spec.seed for spec in specs], [17, 29, 41])
        self.assertTrue(all(spec.expected_epochs == 300 for spec in specs))
        for spec in specs:
            config = load_ifdr_config(ROOT / spec.config, repository_root=ROOT)
            self.assertEqual(config.training.epochs, 300)
            self.assertEqual(config.experiment.seed, spec.seed)
            self.assertEqual(config.method.intervention.base_seed, spec.seed)
            self.assertEqual(config.experiment.variant, spec.variant)
            self.assertTrue(config.method.components.semantic_protection)
            self.assertTrue(config.method.components.counterfactual_consistency)

    def test_rejected_primary_does_not_spend_replication_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "candidate"
            run_dir.mkdir()
            factory = QueueFactory(run_dir)
            queue = ProtectedEvidenceQueue(
                repository_root=root,
                job_dir=root / "job",
                python_executable=Path("python"),
                device="0",
                commit=COMMIT,
                full_metrics=root / "full.json",
                fusion_metrics=root / "fusion.json",
                queue_factory=factory,
                gate_evaluator=lambda **kwargs: _decision(False),
            )

            result = queue.run()

            self.assertEqual(result.state, "rejected")
            self.assertEqual(factory.calls, [(17,)])
            status = json.loads(
                (root / "job" / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["state"], "rejected")
            self.assertEqual(status["failed_checks"], ["performance"])

    def test_accepted_primary_runs_two_replication_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "candidate"
            run_dir.mkdir()
            factory = QueueFactory(run_dir)
            queue = ProtectedEvidenceQueue(
                repository_root=root,
                job_dir=root / "job",
                python_executable=Path("python"),
                device="0",
                commit=COMMIT,
                full_metrics=root / "full.json",
                fusion_metrics=root / "fusion.json",
                queue_factory=factory,
                gate_evaluator=lambda **kwargs: _decision(True),
            )

            result = queue.run()

            self.assertEqual(result.state, "complete")
            self.assertEqual(factory.calls, [(17,), (29, 41)])
            self.assertEqual(
                result.completed,
                (PRIMARY_SPEC.key, *(spec.key for spec in REPLICATION_SPECS)),
            )

    def test_primary_failure_stops_before_selection_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factory = QueueFactory(root / "missing", primary_failure=True)
            called = False

            def gate(**kwargs):
                nonlocal called
                called = True
                return _decision(True)

            result = ProtectedEvidenceQueue(
                repository_root=root,
                job_dir=root / "job",
                python_executable=Path("python"),
                device="0",
                commit=COMMIT,
                full_metrics=root / "full.json",
                fusion_metrics=root / "fusion.json",
                queue_factory=factory,
                gate_evaluator=gate,
            ).run()

            self.assertEqual(result.state, "partial")
            self.assertFalse(called)
            self.assertEqual(factory.calls, [(17,)])

    def test_script_can_be_invoked_directly(self) -> None:
        completed = subprocess.run(
            (
                sys.executable,
                str(ROOT / "scripts/run_protected_evidence_queue.py"),
                "--help",
            ),
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
