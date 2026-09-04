from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_evidence_queue import (
    EvidenceQueue,
    EvidenceSpec,
    process_lock,
)


SPECS = (
    EvidenceSpec(
        "full_control_e90_s17",
        "ifdr",
        "ifdr-full-control-e90",
        17,
        Path(
            "configs/experiments/mechanisms/"
            "kitti_ifdr_full_control_e90_s17.yaml"
        ),
        expected_epochs=90,
    ),
    EvidenceSpec(
        "protected_only_e90_s17",
        "ifdr",
        "ifdr-protected-only-e90",
        17,
        Path(
            "configs/experiments/mechanisms/"
            "kitti_ifdr_protected_only_e90_s17.yaml"
        ),
        expected_epochs=90,
    ),
    EvidenceSpec(
        "counterfactual_only_e90_s17",
        "ifdr",
        "ifdr-counterfactual-only-e90",
        17,
        Path(
            "configs/experiments/mechanisms/"
            "kitti_ifdr_counterfactual_only_e90_s17.yaml"
        ),
        expected_epochs=90,
    ),
    EvidenceSpec(
        "protected_counterfactual_e90_s17",
        "ifdr",
        "ifdr-protected-counterfactual-e90",
        17,
        Path(
            "configs/experiments/mechanisms/"
            "kitti_ifdr_protected_counterfactual_e90_s17.yaml"
        ),
        expected_epochs=90,
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the recoverable 2x2 IFDR mechanism screen."
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
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
    queue = EvidenceQueue(
        repository_root=root,
        job_dir=job_dir,
        python_executable=python_executable,
        device=args.device,
        commit=commit,
        specs=SPECS,
    )
    with process_lock(job_dir, pid=os.getpid()):
        result = queue.run()
    state = "PARTIAL" if result.failed else "COMPLETE"
    print(f"MECHANISM SCREEN {state}")
    print(f"completed={','.join(result.completed)}")
    print(f"failed={','.join(result.failed)}")
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
