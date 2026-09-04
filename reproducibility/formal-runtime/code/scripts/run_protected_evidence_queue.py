from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Literal

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.eval.selection_gate import (
    SelectionDecision,
    evaluate_selection_gate,
)
from scripts.run_evidence_queue import (
    EvidenceQueue,
    EvidenceQueueResult,
    EvidenceSpec,
    process_lock,
)


VARIANT = "ifdr-protected-counterfactual-joint"
PRIMARY_SPEC = EvidenceSpec(
    "protected_joint_s17",
    "ifdr",
    VARIANT,
    17,
    Path(
        "configs/experiments/formal/"
        "kitti_ifdr_protected_counterfactual_joint_s17.yaml"
    ),
)
REPLICATION_SPECS = (
    EvidenceSpec(
        "protected_joint_s29",
        "ifdr",
        VARIANT,
        29,
        Path(
            "configs/experiments/formal/"
            "kitti_ifdr_protected_counterfactual_joint_s29.yaml"
        ),
    ),
    EvidenceSpec(
        "protected_joint_s41",
        "ifdr",
        VARIANT,
        41,
        Path(
            "configs/experiments/formal/"
            "kitti_ifdr_protected_counterfactual_joint_s41.yaml"
        ),
    ),
)


@dataclass(frozen=True)
class ProtectedQueueResult:
    state: Literal["complete", "partial", "rejected"]
    completed: tuple[str, ...]
    failed: tuple[str, ...]
    decision: SelectionDecision | None


QueueFactory = Callable[..., EvidenceQueue]
GateEvaluator = Callable[..., SelectionDecision]


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


class ProtectedEvidenceQueue:
    def __init__(
        self,
        *,
        repository_root: Path,
        job_dir: Path,
        python_executable: Path,
        device: str,
        commit: str,
        full_metrics: Path,
        fusion_metrics: Path,
        queue_factory: QueueFactory = EvidenceQueue,
        gate_evaluator: GateEvaluator = evaluate_selection_gate,
    ) -> None:
        self.root = repository_root.resolve()
        self.job_dir = job_dir.resolve()
        self.python = python_executable
        self.device = device
        self.commit = commit
        self.full_metrics = full_metrics.resolve()
        self.fusion_metrics = fusion_metrics.resolve()
        self.queue_factory = queue_factory
        self.gate_evaluator = gate_evaluator

    def _queue(
        self,
        specs: tuple[EvidenceSpec, ...],
        name: str,
    ) -> EvidenceQueue:
        return self.queue_factory(
            repository_root=self.root,
            job_dir=self.job_dir / name,
            python_executable=self.python,
            device=self.device,
            commit=self.commit,
            specs=specs,
        )

    def _write_status(
        self,
        result: ProtectedQueueResult,
    ) -> None:
        payload: dict[str, object] = {
            "state": result.state,
            "commit": self.commit,
            "completed": list(result.completed),
            "failed": list(result.failed),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if result.decision is not None:
            payload["failed_checks"] = list(result.decision.failed_checks)
        _atomic_json(self.job_dir / "status.json", payload)

    def run(self) -> ProtectedQueueResult:
        primary_queue = self._queue((PRIMARY_SPEC,), "primary")
        primary = primary_queue.run()
        if primary.failed:
            result = ProtectedQueueResult(
                state="partial",
                completed=primary.completed,
                failed=primary.failed,
                decision=None,
            )
            self._write_status(result)
            return result

        candidate_run = primary_queue.complete_run(PRIMARY_SPEC)
        if candidate_run is None:
            raise RuntimeError("primary queue completed without a valid run")
        decision = self.gate_evaluator(
            full_metrics=self.full_metrics,
            fusion_metrics=self.fusion_metrics,
            candidate_metrics=candidate_run / "metrics_ap40.json",
            gradient_diagnostics=(
                candidate_run / "gradient_diagnostics.jsonl"
            ),
        )
        _atomic_json(self.job_dir / "selection_gate.json", asdict(decision))
        if not decision.advance:
            result = ProtectedQueueResult(
                state="rejected",
                completed=primary.completed,
                failed=(),
                decision=decision,
            )
            self._write_status(result)
            return result

        replication_queue = self._queue(REPLICATION_SPECS, "replications")
        replications = replication_queue.run()
        result = ProtectedQueueResult(
            state="partial" if replications.failed else "complete",
            completed=primary.completed + replications.completed,
            failed=replications.failed,
            decision=decision,
        )
        self._write_status(result)
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run selection-gated formal protected IFDR evidence."
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--full-metrics", type=Path, required=True)
    parser.add_argument("--fusion-metrics", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--python", dest="python_executable", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repository_root.resolve()
    job_dir = args.job_dir.resolve()
    python_executable = args.python_executable or Path(os.sys.executable)
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        text=True,
    ).strip()
    queue = ProtectedEvidenceQueue(
        repository_root=root,
        job_dir=job_dir,
        python_executable=python_executable,
        device=args.device,
        commit=commit,
        full_metrics=args.full_metrics,
        fusion_metrics=args.fusion_metrics,
    )
    with process_lock(job_dir, pid=os.getpid()):
        result = queue.run()
    print(f"PROTECTED EVIDENCE QUEUE {result.state.upper()}")
    print(f"completed={','.join(result.completed)}")
    print(f"failed={','.join(result.failed)}")
    return 1 if result.state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
