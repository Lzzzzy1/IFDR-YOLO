from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Literal

import yaml


@dataclass(frozen=True)
class EvidenceSpec:
    key: str
    kind: Literal["baseline", "ifdr"]
    variant: str
    seed: int
    config: Path
    expected_epochs: int = 300

    def __post_init__(self) -> None:
        if (
            isinstance(self.expected_epochs, bool)
            or not isinstance(self.expected_epochs, int)
            or self.expected_epochs <= 0
        ):
            raise ValueError("expected_epochs must be a positive integer")


SPECS = (
    EvidenceSpec(
        "baseline_s29",
        "baseline",
        "baseline",
        29,
        Path("configs/experiments/evidence/kitti_yolov8m_baseline_s29.yaml"),
    ),
    EvidenceSpec(
        "baseline_s41",
        "baseline",
        "baseline",
        41,
        Path("configs/experiments/evidence/kitti_yolov8m_baseline_s41.yaml"),
    ),
    EvidenceSpec(
        "p2_s29",
        "baseline",
        "p2",
        29,
        Path("configs/experiments/evidence/kitti_yolov8m_p2_s29.yaml"),
    ),
    EvidenceSpec(
        "p2_s41",
        "baseline",
        "p2",
        41,
        Path("configs/experiments/evidence/kitti_yolov8m_p2_s41.yaml"),
    ),
    EvidenceSpec(
        "fusion_only_s29",
        "ifdr",
        "ifdr-fusion-only",
        29,
        Path("configs/experiments/evidence/kitti_ifdr_fusion_only_s29.yaml"),
    ),
    EvidenceSpec(
        "fusion_only_s41",
        "ifdr",
        "ifdr-fusion-only",
        41,
        Path("configs/experiments/evidence/kitti_ifdr_fusion_only_s41.yaml"),
    ),
)


CommandRunner = Callable[[tuple[str, ...], Path], None]
ProcessCheck = Callable[[int], bool]


@dataclass(frozen=True)
class EvidenceQueueResult:
    completed: tuple[str, ...]
    failed: tuple[str, ...]


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def process_lock(
    job_dir: Path,
    *,
    pid: int,
    is_alive: ProcessCheck = _is_process_alive,
) -> Iterator[None]:
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / "queue.pid"
    if path.is_file():
        try:
            existing = int(path.read_text(encoding="utf-8").strip())
        except ValueError:
            existing = -1
        if existing > 0 and existing != pid and is_alive(existing):
            raise RuntimeError(f"evidence queue is already running as PID {existing}")
    temporary = path.with_suffix(".pid.tmp")
    temporary.write_text(f"{pid}\n", encoding="utf-8", newline="\n")
    temporary.replace(path)
    try:
        yield
    finally:
        try:
            owner = int(path.read_text(encoding="utf-8").strip())
        except (FileNotFoundError, ValueError):
            owner = -1
        if owner == pid:
            path.unlink()


def _run_command(command: tuple[str, ...], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _completed_epochs(path: Path) -> int | None:
    try:
        with path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        epochs = [int(float(row["epoch"])) for row in rows]
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None
    if not epochs or epochs[0] not in (0, 1):
        return None
    if epochs != list(range(epochs[0], epochs[0] + len(epochs))):
        return None
    return len(epochs)


def _is_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _formal_mode(run_dir: Path) -> bool:
    path = run_dir / "config.resolved.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(payload, dict) and payload.get("mode") == "full"


def _status(run_dir: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(
            (run_dir / "status.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


class EvidenceQueue:
    def __init__(
        self,
        *,
        repository_root: Path,
        job_dir: Path,
        python_executable: Path,
        device: str,
        command_runner: CommandRunner = _run_command,
        commit: str,
        specs: tuple[EvidenceSpec, ...] = SPECS,
    ) -> None:
        if len(commit) != 40:
            raise ValueError("commit must be a 40-character Git SHA")
        self.root = repository_root.resolve()
        self.job_dir = job_dir.resolve()
        self.python = str(python_executable)
        self.device = device
        self.command_runner = command_runner
        self.commit = commit
        self.specs = specs
        self.completed: list[str] = []
        self.failed: list[str] = []

    def _matching_runs(self, spec: EvidenceSpec) -> tuple[Path, ...]:
        pattern = (
            f"*-kitti-yolov8m-{spec.variant}-s{spec.seed}-{self.commit[:7]}"
        )
        return tuple(sorted((self.root / "runs").glob(pattern), reverse=True))

    def _complete_run(self, spec: EvidenceSpec) -> Path | None:
        for run_dir in self._matching_runs(spec):
            status = _status(run_dir)
            if (
                _formal_mode(run_dir)
                and status is not None
                and status.get("state") == "complete"
                and _completed_epochs(run_dir / "results.csv")
                == spec.expected_epochs
                and _is_nonempty(run_dir / "weights" / "best.pt")
                and _is_nonempty(run_dir / "weights" / "last.pt")
                and _is_nonempty(run_dir / "metrics_ap40.json")
            ):
                return run_dir
        return None

    def _recoverable_run(self, spec: EvidenceSpec) -> Path | None:
        for run_dir in self._matching_runs(spec):
            status = _status(run_dir)
            epochs = _completed_epochs(run_dir / "results.csv")
            if (
                _formal_mode(run_dir)
                and status is not None
                and status.get("state") == "failed"
                and status.get("stage") == "training"
                and epochs is not None
                and epochs < spec.expected_epochs
                and _is_nonempty(run_dir / "weights" / "best.pt")
                and _is_nonempty(run_dir / "weights" / "last.pt")
            ):
                return run_dir
        return None

    def _command(
        self,
        spec: EvidenceSpec,
        *,
        mode: Literal["dry-run", "smoke", "full"],
    ) -> tuple[str, ...]:
        script = "train_ifdr.py" if spec.kind == "ifdr" else "train_baseline.py"
        return (
            self.python,
            str(self.root / "scripts" / script),
            "--config",
            str(spec.config),
            "--mode",
            mode,
            "--device",
            self.device,
        )

    def _recovery_command(
        self,
        spec: EvidenceSpec,
        run_dir: Path,
    ) -> tuple[str, ...]:
        script = (
            "recover_ifdr.py" if spec.kind == "ifdr" else "recover_baseline.py"
        )
        return (
            self.python,
            str(self.root / "scripts" / script),
            "--config",
            str(spec.config),
            "--run-dir",
            str(run_dir),
            "--device",
            self.device,
        )

    def _write_status(
        self,
        *,
        state: str,
        current: str | None,
        attempt: int | None = None,
        error: str | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "state": state,
            "commit": self.commit,
            "current": current,
            "completed": self.completed,
            "failed": self.failed,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if attempt is not None:
            payload["attempt"] = attempt
        if error is not None:
            payload["error"] = error
        _atomic_json(self.job_dir / "status.json", payload)

    def _record_failure(
        self,
        spec: EvidenceSpec,
        error: BaseException,
    ) -> None:
        entry = {
            "key": spec.key,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with (self.job_dir / "failures.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        ) as output:
            output.write(json.dumps(entry, sort_keys=True) + "\n")

    def run(self) -> EvidenceQueueResult:
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self._write_status(state="running", current=None)
        for spec in self.specs:
            complete = self._complete_run(spec)
            if complete is not None:
                print(f"[Evidence] SKIP {spec.key}: {complete}", flush=True)
                self.completed.append(spec.key)
                self._write_status(state="running", current=spec.key)
                continue

            last_error: BaseException | None = None
            for attempt in (1, 2):
                self._write_status(
                    state="running",
                    current=spec.key,
                    attempt=attempt,
                )
                try:
                    print(
                        f"[Evidence] {spec.key} attempt {attempt}: preflight",
                        flush=True,
                    )
                    self.command_runner(
                        self._command(spec, mode="dry-run"),
                        self.root,
                    )
                    print(
                        f"[Evidence] {spec.key} attempt {attempt}: smoke",
                        flush=True,
                    )
                    self.command_runner(
                        self._command(spec, mode="smoke"),
                        self.root,
                    )
                    recoverable = self._recoverable_run(spec)
                    if recoverable is None:
                        print(
                            f"[Evidence] {spec.key} attempt {attempt}: full",
                            flush=True,
                        )
                        command = self._command(spec, mode="full")
                    else:
                        print(
                            f"[Evidence] {spec.key} attempt {attempt}: "
                            f"recover {recoverable}",
                            flush=True,
                        )
                        command = self._recovery_command(spec, recoverable)
                    self.command_runner(command, self.root)
                    complete = self._complete_run(spec)
                    if complete is None:
                        raise RuntimeError(
                            "command returned without a valid "
                            f"{spec.expected_epochs}-epoch AP40 run"
                        )
                    self.completed.append(spec.key)
                    self._write_status(state="running", current=spec.key)
                    print(f"[Evidence] COMPLETE {spec.key}: {complete}", flush=True)
                    last_error = None
                    break
                except BaseException as error:
                    last_error = error
                    self._write_status(
                        state="running",
                        current=spec.key,
                        attempt=attempt,
                        error=f"{type(error).__name__}: {error}",
                    )
                    print(
                        f"[Evidence] ERROR {spec.key} attempt {attempt}: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
            if last_error is not None:
                self.failed.append(spec.key)
                self._record_failure(spec, last_error)
                self._write_status(state="running", current=spec.key)

        final_state = "partial" if self.failed else "complete"
        self._write_status(state=final_state, current=None)
        return EvidenceQueueResult(
            completed=tuple(self.completed),
            failed=tuple(self.failed),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the recoverable six-condition KITTI evidence queue."
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
        job_dir=args.job_dir,
        python_executable=python_executable,
        device=args.device,
        commit=commit,
    )
    with process_lock(args.job_dir.resolve(), pid=os.getpid()):
        result = queue.run()
    print(f"EVIDENCE QUEUE {('PARTIAL' if result.failed else 'COMPLETE')}")
    print(f"completed={','.join(result.completed)}")
    print(f"failed={','.join(result.failed)}")
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
