from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.run_evidence_queue import (
    SPECS,
    EvidenceQueue,
    EvidenceSpec,
    process_lock,
)


COMMIT = "abcdef1234567890abcdef1234567890abcdef12"


def _run_dir(root: Path, spec: EvidenceSpec, suffix: str = "000000") -> Path:
    return (
        root
        / "runs"
        / (
            f"20260801T{suffix}Z-kitti-yolov8m-{spec.variant}-"
            f"s{spec.seed}-{COMMIT[:7]}"
        )
    )


def _write_epochs(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("epoch", "metrics/mAP50-95(B)"))
        for epoch in range(1, count + 1):
            writer.writerow((epoch, 0.5))


def _write_formal(
    root: Path,
    spec: EvidenceSpec,
    *,
    state: str,
    stage: str | None = None,
    epochs: int,
    suffix: str = "000000",
) -> Path:
    run_dir = _run_dir(root, spec, suffix)
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"best")
    (weights / "last.pt").write_bytes(b"last")
    payload: dict[str, object] = {"state": state}
    if stage is not None:
        payload["stage"] = stage
    (run_dir / "status.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    (run_dir / "config.resolved.yaml").write_text(
        "mode: full\n",
        encoding="utf-8",
    )
    _write_epochs(run_dir / "results.csv", epochs)
    if state == "complete":
        (run_dir / "metrics_ap40.json").write_text(
            json.dumps({"classes": {}}),
            encoding="utf-8",
        )
    return run_dir


class FakeRunner:
    def __init__(self, root: Path, specs: tuple[EvidenceSpec, ...]) -> None:
        self.root = root
        self.specs = specs
        self.calls: list[tuple[str, ...]] = []
        self.recovery_failures = 0

    def __call__(self, command: tuple[str, ...], cwd: Path) -> None:
        self.calls.append(command)
        self.assert_root(cwd)
        config_name = Path(command[command.index("--config") + 1]).name
        spec = next(item for item in self.specs if item.config.name == config_name)
        script = Path(command[1]).name
        if script.startswith("recover_"):
            if self.recovery_failures:
                self.recovery_failures -= 1
                raise RuntimeError("simulated recovery failure")
            run_dir = Path(command[command.index("--run-dir") + 1])
            _write_epochs(run_dir / "results.csv", 300)
            (run_dir / "metrics_ap40.json").write_text(
                json.dumps({"classes": {}}),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(
                json.dumps({"state": "complete"}),
                encoding="utf-8",
            )
        elif command[command.index("--mode") + 1] == "full":
            _write_formal(
                self.root,
                spec,
                state="complete",
                epochs=300,
                suffix=f"{len(self.calls):06d}",
            )

    def assert_root(self, cwd: Path) -> None:
        if cwd != self.root:
            raise AssertionError(f"unexpected cwd: {cwd}")


class EvidenceQueueTest(unittest.TestCase):
    def test_exposes_valid_complete_run_for_follow_up_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = SPECS[0]
            expected = _write_formal(
                root,
                spec,
                state="complete",
                epochs=spec.expected_epochs,
            )
            queue = EvidenceQueue(
                repository_root=root,
                job_dir=root / "job",
                python_executable=Path("python"),
                device="0",
                command_runner=FakeRunner(root, (spec,)),
                commit=COMMIT,
                specs=(spec,),
            )

            self.assertEqual(queue.complete_run(spec), expected)
    def test_condition_specific_epoch_budget_controls_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "screen.yaml"
            config.write_text("training:\n  epochs: 2\n", encoding="utf-8")
            spec = EvidenceSpec(
                "screen_s17",
                "ifdr",
                "ifdr-screen",
                17,
                config,
                expected_epochs=2,
            )
            _write_formal(root, spec, state="complete", epochs=2)
            runner = FakeRunner(root, (spec,))

            result = EvidenceQueue(
                repository_root=root,
                job_dir=root / "job",
                python_executable=Path("python"),
                device="0",
                command_runner=runner,
                commit=COMMIT,
                specs=(spec,),
            ).run()

            self.assertEqual(result.completed, (spec.key,))
            self.assertEqual(runner.calls, [])

    def test_runs_exact_locked_order_and_completes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = FakeRunner(root, SPECS)
            result = EvidenceQueue(
                repository_root=root,
                job_dir=root / "job",
                python_executable=Path("/venv/bin/python"),
                device="0",
                command_runner=runner,
                commit=COMMIT,
            ).run()

            full_configs = [
                Path(call[call.index("--config") + 1]).name
                for call in runner.calls
                if "--mode" in call
                and call[call.index("--mode") + 1] == "full"
            ]
            self.assertEqual(full_configs, [spec.config.name for spec in SPECS])
            self.assertEqual(result.completed, tuple(spec.key for spec in SPECS))
            self.assertEqual(result.failed, ())
            status = json.loads(
                (root / "job" / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["state"], "complete")

    def test_skips_already_complete_formal_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = SPECS[0]
            _write_formal(root, spec, state="complete", epochs=300)
            runner = FakeRunner(root, (spec,))

            result = EvidenceQueue(
                repository_root=root,
                job_dir=root / "job",
                python_executable=Path("python"),
                device="0",
                command_runner=runner,
                commit=COMMIT,
                specs=(spec,),
            ).run()

            self.assertEqual(result.completed, (spec.key,))
            self.assertEqual(runner.calls, [])

    def test_recovers_failed_training_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = SPECS[2]
            _write_formal(
                root,
                spec,
                state="failed",
                stage="training",
                epochs=17,
            )
            runner = FakeRunner(root, (spec,))

            result = EvidenceQueue(
                repository_root=root,
                job_dir=root / "job",
                python_executable=Path("python"),
                device="0",
                command_runner=runner,
                commit=COMMIT,
                specs=(spec,),
            ).run()

            self.assertEqual(result.completed, (spec.key,))
            self.assertTrue(
                any(Path(call[1]).name == "recover_baseline.py" for call in runner.calls)
            )

    def test_retries_failure_and_continues_next_condition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = SPECS[:2]
            _write_formal(
                root,
                first,
                state="failed",
                stage="training",
                epochs=17,
            )
            runner = FakeRunner(root, (first, second))
            runner.recovery_failures = 2

            result = EvidenceQueue(
                repository_root=root,
                job_dir=root / "job",
                python_executable=Path("python"),
                device="0",
                command_runner=runner,
                commit=COMMIT,
                specs=(first, second),
            ).run()

            recovery_calls = [
                call for call in runner.calls
                if Path(call[1]).name == "recover_baseline.py"
            ]
            self.assertEqual(len(recovery_calls), 2)
            self.assertEqual(result.failed, (first.key,))
            self.assertEqual(result.completed, (second.key,))
            status = json.loads(
                (root / "job" / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["state"], "partial")
            self.assertTrue((root / "job" / "failures.jsonl").is_file())

    def test_non_training_failure_starts_new_formal_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = SPECS[-1]
            _write_formal(
                root,
                spec,
                state="failed",
                stage="evaluation",
                epochs=300,
            )
            runner = FakeRunner(root, (spec,))

            result = EvidenceQueue(
                repository_root=root,
                job_dir=root / "job",
                python_executable=Path("python"),
                device="0",
                command_runner=runner,
                commit=COMMIT,
                specs=(spec,),
            ).run()

            self.assertEqual(result.completed, (spec.key,))
            self.assertFalse(
                any(Path(call[1]).name.startswith("recover_") for call in runner.calls)
            )

    def test_live_process_lock_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job_dir = Path(directory)
            (job_dir / "queue.pid").write_text("123\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "already running"):
                with process_lock(
                    job_dir,
                    pid=os.getpid(),
                    is_alive=lambda pid: pid == 123,
                ):
                    pass


if __name__ == "__main__":
    unittest.main()
