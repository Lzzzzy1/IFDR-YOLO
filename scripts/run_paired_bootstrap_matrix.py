from __future__ import annotations

import argparse
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from time import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifdr_yolo.eval.bootstrap_matrix import (
    BootstrapTask,
    build_bootstrap_tasks,
    parse_comparison_spec,
    parse_run_spec,
    result_is_complete,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a resumable matrix of paired KITTI AP40 bootstraps.",
    )
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--comparison", action="append", required=True)
    parser.add_argument("--class-name", action="append", required=True)
    parser.add_argument("--slice", action="append", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _load_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _build_run_dirs(specifications: Sequence[str]) -> dict[str, dict[int, Path]]:
    run_dirs: dict[str, dict[int, Path]] = {}
    for specification in specifications:
        method, seed, prediction_dir = parse_run_spec(specification)
        runs = run_dirs.setdefault(method, {})
        if seed in runs:
            raise ValueError(f"duplicate run specification: {method}:{seed}")
        runs[seed] = prediction_dir.resolve()
    return run_dirs


def _task_command(
    task: BootstrapTask,
    *,
    args: argparse.Namespace,
    output: Path,
) -> list[str]:
    evaluator = Path(__file__).with_name("evaluate_paired_bootstrap.py")
    return [
        sys.executable,
        str(evaluator),
        "--reference-dir",
        str(task.reference_dir),
        "--candidate-dir",
        str(task.candidate_dir),
        "--reference-name",
        task.reference,
        "--candidate-name",
        task.candidate,
        "--class-name",
        task.class_name,
        "--slice",
        task.slice_name,
        "--iterations",
        str(args.iterations),
        "--seed",
        str(args.bootstrap_seed),
        "--label-dir",
        str(args.label_dir.resolve()),
        "--image-dir",
        str(args.image_dir.resolve()),
        "--split",
        str(args.split.resolve()),
        "--output",
        str(output),
    ]


def _execute_task(
    task: BootstrapTask,
    *,
    args: argparse.Namespace,
    output: Path,
    log_path: Path,
) -> tuple[str, int, str]:
    command = _task_command(task, args=args, output=output)
    with log_path.open("w", encoding="utf-8", newline="\n") as log_file:
        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return task.task_id, completed.returncode, " ".join(command)


def _is_complete_output(
    output: Path,
    task: BootstrapTask,
    args: argparse.Namespace,
) -> bool:
    return output.is_file() and result_is_complete(
        _load_json(output),
        task=task,
        iterations=args.iterations,
        bootstrap_seed=args.bootstrap_seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.iterations <= 0 or args.bootstrap_seed < 0 or args.workers <= 0:
        raise ValueError("iterations, bootstrap seed and workers must be positive")
    run_dirs = _build_run_dirs(args.run)
    comparisons = tuple(parse_comparison_spec(spec) for spec in args.comparison)
    tasks = build_bootstrap_tasks(
        run_dirs=run_dirs,
        comparisons=comparisons,
        class_names=tuple(args.class_name),
        slice_names=tuple(args.slice),
    )
    for method_runs in run_dirs.values():
        for prediction_dir in method_runs.values():
            if not prediction_dir.is_dir():
                raise FileNotFoundError(
                    f"prediction directory does not exist: {prediction_dir}"
                )

    output_dir = args.output_dir.resolve()
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    records: dict[str, dict[str, object]] = {}
    queued: list[BootstrapTask] = []
    for task in tasks:
        output = output_dir / f"{task.task_id}.json"
        if _is_complete_output(output, task, args):
            records[task.task_id] = {"state": "complete", "output": str(output)}
        else:
            records[task.task_id] = {"state": "pending", "output": str(output)}
            queued.append(task)

    started_at = time()

    def save_status(state: str) -> None:
        complete = sum(record["state"] == "complete" for record in records.values())
        failed = sum(record["state"] == "failed" for record in records.values())
        _atomic_json(
            status_path,
            {
                "schema_version": 1,
                "state": state,
                "started_at_unix": started_at,
                "updated_at_unix": time(),
                "total": len(tasks),
                "complete": complete,
                "failed": failed,
                "pending": len(tasks) - complete - failed,
                "iterations": args.iterations,
                "bootstrap_seed": args.bootstrap_seed,
                "workers": args.workers,
                "tasks": records,
            },
        )

    save_status("running")
    if not queued:
        save_status("complete")
        print(f"bootstrap_matrix=complete total={len(tasks)}")
        return 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for task in queued:
            output = output_dir / f"{task.task_id}.json"
            log_path = logs_dir / f"{task.task_id}.log"
            future = executor.submit(
                _execute_task,
                task,
                args=args,
                output=output,
                log_path=log_path,
            )
            futures[future] = task
        save_status("running")
        for future in as_completed(futures):
            task = futures[future]
            output = output_dir / f"{task.task_id}.json"
            try:
                _, return_code, command = future.result()
                if return_code == 0 and _is_complete_output(output, task, args):
                    records[task.task_id]["state"] = "complete"
                else:
                    records[task.task_id].update(
                        {"state": "failed", "return_code": return_code, "command": command}
                    )
            except Exception as error:
                records[task.task_id].update(
                    {"state": "failed", "error": repr(error)}
                )
            save_status("running")
            print(
                f"bootstrap_matrix complete={sum(r['state'] == 'complete' for r in records.values())} "
                f"failed={sum(r['state'] == 'failed' for r in records.values())}/{len(tasks)}",
                flush=True,
            )

    failed = any(record["state"] == "failed" for record in records.values())
    save_status("failed" if failed else "complete")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
