from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import os
import subprocess

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_evidence_queue import EvidenceQueue, EvidenceSpec, process_lock


VARIANT = "ifdr-unprotected-counterfactual-joint"
UNPROTECTED_SPECS = tuple(
    EvidenceSpec(
        f"unprotected_joint_s{seed}",
        "ifdr",
        VARIANT,
        seed,
        Path(
            "configs/experiments/formal/"
            f"kitti_ifdr_unprotected_counterfactual_joint_s{seed}.yaml"
        ),
    )
    for seed in (17, 29, 41)
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the equal-budget unprotected counterfactual control queue."
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--python", dest="python_executable", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repository_root.resolve()
    python_executable = args.python_executable or Path(os.sys.executable)
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        text=True,
    ).strip()
    queue = EvidenceQueue(
        repository_root=root,
        job_dir=args.job_dir.resolve(),
        python_executable=python_executable,
        device=args.device,
        commit=commit,
        specs=UNPROTECTED_SPECS,
    )
    with process_lock(args.job_dir.resolve(), pid=os.getpid()):
        result = queue.run()
    print(f"UNPROTECTED COUNTERFACTUAL QUEUE {('PARTIAL' if result.failed else 'COMPLETE')}")
    print(f"completed={','.join(result.completed)}")
    print(f"failed={','.join(result.failed)}")
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
