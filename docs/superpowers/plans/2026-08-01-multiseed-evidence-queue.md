# Multi-Seed Evidence Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, deploy, and start a recoverable six-run KITTI evidence queue for seeds 29 and 41 of the baseline, P2, and fusion-only conditions.

**Architecture:** A small evidence-config module deterministically derives six YAML files from the locked seed-17 sources. A repository-owned Python queue discovers prior runs by variant, seed, and commit; it skips complete runs, recovers training failures when possible, retries once, records `COMPLETE` or `PARTIAL`, and continues independent conditions. Existing training, prediction, AP40, checkpoint, and IFDR recovery code remains authoritative.

**Tech Stack:** Python 3.11/3.12, `unittest`, PyYAML, Ultralytics 8.4.98, PyTorch 2.8.0+cu128, Bash/nohup for detached server launch.

---

## File Structure

- Create `ifdr_yolo/experiments/evidence.py`: locked evidence matrix, seed-config generation, and run discovery/classification.
- Create `ifdr_yolo/experiments/baseline_recovery.py`: baseline/P2 checkpoint resume and AP40 finalization.
- Create `scripts/materialize_evidence_configs.py`: deterministic CLI for generating the six YAML files.
- Create `scripts/recover_baseline.py`: CLI wrapper for baseline/P2 recovery.
- Create `scripts/run_evidence_queue.py`: durable sequential queue and state file owner.
- Create `scripts/launch_evidence_queue_server.sh`: detached AutoDL launcher with stable logs and PID.
- Create `configs/experiments/evidence/*.yaml`: the six locked generated configs.
- Create `tests/test_evidence_matrix.py`: seed and configuration invariants.
- Create `tests/test_evidence_queue.py`: skip, retry, recovery, continuation, and status behavior.
- Create `tests/test_baseline_recovery.py`: resume and evaluation behavior without a GPU.

No existing model, loss, evaluator, or dataset file changes in this plan.

### Task 1: Deterministic Evidence Configurations

**Files:**
- Create: `tests/test_evidence_matrix.py`
- Create: `ifdr_yolo/experiments/evidence.py`
- Create: `scripts/materialize_evidence_configs.py`
- Create: `configs/experiments/evidence/kitti_yolov8m_baseline_s29.yaml`
- Create: `configs/experiments/evidence/kitti_yolov8m_baseline_s41.yaml`
- Create: `configs/experiments/evidence/kitti_yolov8m_p2_s29.yaml`
- Create: `configs/experiments/evidence/kitti_yolov8m_p2_s41.yaml`
- Create: `configs/experiments/evidence/kitti_ifdr_fusion_only_s29.yaml`
- Create: `configs/experiments/evidence/kitti_ifdr_fusion_only_s41.yaml`

- [ ] **Step 1: Write failing matrix tests**

```python
from pathlib import Path
import tempfile
import unittest

from ifdr_yolo.experiments.config import (
    load_baseline_config,
    load_ifdr_config,
)
from ifdr_yolo.experiments.evidence import write_evidence_configs


ROOT = Path(__file__).resolve().parents[1]


class EvidenceMatrixTest(unittest.TestCase):
    def test_writes_exact_locked_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_evidence_configs(
                repository_root=ROOT,
                output_dir=Path(directory),
            )
            self.assertEqual(
                tuple(paths),
                (
                    "baseline_s29", "baseline_s41",
                    "p2_s29", "p2_s41",
                    "fusion_only_s29", "fusion_only_s41",
                ),
            )
            for key, path in paths.items():
                seed = int(key.rsplit("s", 1)[1])
                if key.startswith("fusion_only"):
                    config = load_ifdr_config(path, repository_root=ROOT)
                    self.assertEqual(config.method.intervention.base_seed, seed)
                    self.assertTrue(config.method.components.fusion_gate)
                    self.assertFalse(config.method.components.dcli)
                else:
                    config = load_baseline_config(path, repository_root=ROOT)
                self.assertEqual(config.experiment.seed, seed)
                self.assertEqual(config.training.epochs, 300)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new test and verify the missing module failure**

Run:

```bash
python -m unittest tests.test_evidence_matrix -v
```

Expected: `ModuleNotFoundError: No module named 'ifdr_yolo.experiments.evidence'`.

- [ ] **Step 3: Implement the locked generator**

`ifdr_yolo/experiments/evidence.py` must expose this interface and behavior:

```python
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import yaml


EVIDENCE_SEEDS = (29, 41)
SOURCE_CONFIGS = OrderedDict(
    (
        ("baseline", "configs/experiments/kitti_yolov8m_baseline_s17.yaml"),
        ("p2", "configs/experiments/kitti_yolov8m_p2_s17.yaml"),
        (
            "fusion_only",
            "configs/experiments/ablations/"
            "kitti_ifdr_fusion_only_s17.yaml",
        ),
    )
)
OUTPUT_PREFIXES = {
    "baseline": "kitti_yolov8m_baseline",
    "p2": "kitti_yolov8m_p2",
    "fusion_only": "kitti_ifdr_fusion_only",
}


def _seed_payload(payload: object, seed: int) -> dict[str, object]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    result = deepcopy(payload)
    experiment = result.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("configuration experiment must be a mapping")
    experiment["seed"] = seed
    method = result.get("ifdr")
    if method is not None:
        if not isinstance(method, dict):
            raise ValueError("ifdr must be a mapping")
        intervention = method.get("intervention")
        if not isinstance(intervention, dict):
            raise ValueError("ifdr.intervention must be a mapping")
        intervention["base_seed"] = seed
    return result


def write_evidence_configs(
    *, repository_root: Path, output_dir: Path
) -> dict[str, Path]:
    root = repository_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for condition, relative_source in SOURCE_CONFIGS.items():
        source = root / relative_source
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        for seed in EVIDENCE_SEEDS:
            key = f"{condition}_s{seed}"
            path = output_dir / f"{OUTPUT_PREFIXES[condition]}_s{seed}.yaml"
            path.write_text(
                yaml.safe_dump(_seed_payload(payload, seed), sort_keys=False),
                encoding="utf-8",
                newline="\n",
            )
            written[key] = path
    return written
```

The CLI calls `write_evidence_configs()` with repository root and `configs/experiments/evidence`, then prints each generated path.

- [ ] **Step 4: Generate configs and run strict loaders**

Run:

```bash
python scripts/materialize_evidence_configs.py
python -m unittest tests.test_evidence_matrix tests.test_ifdr_config -v
```

Expected: six paths printed and all tests pass.

- [ ] **Step 5: Commit the matrix**

```bash
git add ifdr_yolo/experiments/evidence.py scripts/materialize_evidence_configs.py tests/test_evidence_matrix.py configs/experiments/evidence
git commit -m "experiment: add locked multi-seed evidence matrix"
```

### Task 2: Baseline and P2 Recovery

**Files:**
- Create: `tests/test_baseline_recovery.py`
- Create: `ifdr_yolo/experiments/baseline_recovery.py`
- Create: `scripts/recover_baseline.py`

- [ ] **Step 1: Write a failed-run recovery test**

The test creates a temporary failed training run containing contiguous `results.csv`, non-empty `best.pt` and `last.pt`, injects fake resume/predict/evaluate services, and asserts:

```python
result = recover_baseline_run(
    config,
    run_dir=run_dir,
    repository_root=ROOT,
    device="0",
    services=services,
)
self.assertEqual(result.completed_epochs, 300)
self.assertEqual(resume_calls, [(last, run_dir, "0", config.training.workers)])
self.assertEqual(
    json.loads((run_dir / "status.json").read_text())["state"],
    "complete",
)
self.assertTrue((run_dir / "metrics_ap40.json").is_file())
```

Add rejection tests for a complete run, a non-training failure, missing checkpoints, non-contiguous epochs, and a dirty tracked repository.

- [ ] **Step 2: Verify the new test fails**

Run:

```bash
python -m unittest tests.test_baseline_recovery -v
```

Expected: import failure for `baseline_recovery`.

- [ ] **Step 3: Implement baseline recovery**

`ifdr_yolo/experiments/baseline_recovery.py` must define:

```python
@dataclass(frozen=True)
class BaselineRecoveryServices:
    resume_training: Callable[[Path, Path, str, int], None]
    prediction_adapter: Any
    evaluate: Callable[..., dict[str, object]]
    collect_git: Callable[[Path], dict[str, object]]
    now: Callable[[], datetime]


@dataclass(frozen=True)
class BaselineRecoveryResult:
    run_dir: Path
    metrics_path: Path
    completed_epochs: int
```

The default resume callable must use the checkpoint as Ultralytics' resume source without rebuilding or reinitializing the model:

```python
def _resume_training(last: Path, run_dir: Path, device: str, workers: int) -> None:
    from ultralytics import YOLO

    model = YOLO(str(last))
    model.train(
        resume=True,
        device=device,
        workers=workers,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
    )
```

`recover_baseline_run()` validates `state=failed`, `stage=training`, checkpoints, contiguous epochs, clean Git state, and incomplete epoch count. It writes `status.before-recovery.json`, atomically records `recovery_status.json`, resumes in place, predicts from `best.pt`, evaluates AP40, and atomically changes `status.json` to `complete`. Reuse `_completed_epochs`, prediction argument construction, `ensure_prediction_files`, and `atomic_write_json`; do not duplicate their validation rules.

- [ ] **Step 4: Implement and verify the CLI**

`scripts/recover_baseline.py` accepts `--config`, `--run-dir`, and `--device`, loads `BaselineConfig`, invokes `recover_baseline_run()`, and prints `BASELINE RECOVERY COMPLETE`, epoch count, run directory, and metrics path.

Run:

```bash
python -m unittest tests.test_baseline_recovery tests.test_ifdr_recovery -v
```

Expected: all tests pass and existing IFDR recovery remains unchanged.

- [ ] **Step 5: Commit recovery**

```bash
git add ifdr_yolo/experiments/baseline_recovery.py scripts/recover_baseline.py tests/test_baseline_recovery.py
git commit -m "feat: recover interrupted baseline evidence runs"
```

### Task 3: Durable Evidence Queue

**Files:**
- Create: `tests/test_evidence_queue.py`
- Create: `scripts/run_evidence_queue.py`
- Create: `scripts/launch_evidence_queue_server.sh`

- [ ] **Step 1: Write queue-state tests**

Use a fake command runner and temporary run directories. Cover this exact sequence:

```python
queue = EvidenceQueue(
    repository_root=root,
    job_dir=job_dir,
    python_executable=Path("/venv/bin/python"),
    command_runner=fake_runner,
)
result = queue.run()
self.assertEqual(result.completed, tuple(EXPECTED_KEYS))
self.assertEqual(result.failed, ())
self.assertEqual(json.loads((job_dir / "status.json").read_text())["state"], "complete")
```

Additional tests must assert:

- complete 300-epoch runs are skipped;
- a failed training run calls the matching recovery CLI;
- a recovery failure is retried once, recorded, and the next condition runs;
- a non-training failure starts a new formal run rather than misusing recovery;
- any failed condition makes final queue state `partial`;
- duplicate live PID acquisition is rejected;
- configs run in locked order: baseline 29/41, P2 29/41, fusion-only 29/41.

- [ ] **Step 2: Verify queue tests fail**

Run:

```bash
python -m unittest tests.test_evidence_queue -v
```

Expected: import failure for `scripts.run_evidence_queue` or missing `EvidenceQueue`.

- [ ] **Step 3: Implement the queue with no training logic duplication**

The queue owns orchestration only. Define immutable specifications:

```python
@dataclass(frozen=True)
class EvidenceSpec:
    key: str
    kind: Literal["baseline", "ifdr"]
    variant: str
    seed: int
    config: Path


SPECS = (
    EvidenceSpec("baseline_s29", "baseline", "baseline", 29, Path("configs/experiments/evidence/kitti_yolov8m_baseline_s29.yaml")),
    EvidenceSpec("baseline_s41", "baseline", "baseline", 41, Path("configs/experiments/evidence/kitti_yolov8m_baseline_s41.yaml")),
    EvidenceSpec("p2_s29", "baseline", "p2", 29, Path("configs/experiments/evidence/kitti_yolov8m_p2_s29.yaml")),
    EvidenceSpec("p2_s41", "baseline", "p2", 41, Path("configs/experiments/evidence/kitti_yolov8m_p2_s41.yaml")),
    EvidenceSpec("fusion_only_s29", "ifdr", "ifdr-fusion-only", 29, Path("configs/experiments/evidence/kitti_ifdr_fusion_only_s29.yaml")),
    EvidenceSpec("fusion_only_s41", "ifdr", "ifdr-fusion-only", 41, Path("configs/experiments/evidence/kitti_ifdr_fusion_only_s41.yaml")),
)
```

For each spec, run dry-run, one-epoch smoke, then classify matching formal directories. A complete run requires `state=complete`, exactly 300 CSV rows, non-empty `best.pt`/`last.pt`, and `metrics_ap40.json`. A recoverable run requires `state=failed`, `stage=training`, fewer than 300 contiguous rows, and both checkpoints. Invoke `recover_baseline.py` or `recover_ifdr.py` as appropriate. If no reusable run exists, invoke `train_baseline.py` or `train_ifdr.py` in full mode.

Write `status.json` atomically after every state change with:

```json
{
  "state": "running",
  "commit": "<40-character sha>",
  "current": "p2_s29",
  "completed": ["baseline_s29", "baseline_s41"],
  "failed": [],
  "updated_at_utc": "<ISO-8601>"
}
```

On two failures for one condition, append its key and error to `failures.jsonl`, continue, and finish as `partial`. Never treat a partial queue as complete.

- [ ] **Step 4: Implement the detached launcher**

`scripts/launch_evidence_queue_server.sh` must use fixed AutoDL paths, refuse tracked changes, record the current commit, prevent a live duplicate PID, and launch:

```bash
nohup /root/autodl-tmp/venvs/kitti-yolo/bin/python3 \
  /root/autodl-tmp/kitti_project/scripts/run_evidence_queue.py \
  --repository-root /root/autodl-tmp/kitti_project \
  --job-dir /root/autodl-tmp/jobs/multiseed-evidence \
  --device 0 \
  >>/root/autodl-tmp/jobs/multiseed-evidence/nohup.log 2>&1 &
```

Write the child PID to `pid.txt` and print commands for `tail -f` and status inspection.

- [ ] **Step 5: Run queue tests and full regression**

Run:

```bash
python -m unittest tests.test_evidence_queue -v
python -m unittest discover -s tests -v
```

Expected: new queue tests pass and the full existing suite remains green.

- [ ] **Step 6: Commit the queue**

```bash
git add scripts/run_evidence_queue.py scripts/launch_evidence_queue_server.sh tests/test_evidence_queue.py
git commit -m "feat: add recoverable multi-seed evidence queue"
```

### Task 4: Server Deployment and Launch

**Files:**
- Create on server: `/root/autodl-tmp/evidence-<commit>.bundle`
- Create on server: `/root/autodl-tmp/jobs/multiseed-evidence/*`

- [ ] **Step 1: Verify the local release candidate**

Run:

```bash
git status --short
git log -1 --oneline
python -m unittest discover -s tests -v
```

Expected: only pre-existing unrelated untracked user files remain; all tests pass.

- [ ] **Step 2: Transfer the exact commit**

Create a Git bundle containing `master`, transfer it to `/root/autodl-tmp`, fetch it into the clean server repository, and fast-forward only. Verify local and server `git rev-parse HEAD` are identical. Do not copy the working tree directly.

- [ ] **Step 3: Run server preflight**

Run the project Phase 1 audit, the new matrix tests, and dry-run all six generated configurations with the server virtual environment.

Expected: `PHASE 1 ACCEPTED`, tests pass, and six preflights pass.

- [ ] **Step 4: Run one smoke condition and inspect artifacts**

Launch the first baseline smoke synchronously. Verify one CSV row, finite metrics, non-empty best/last checkpoints, and AP40 output. Do not start the long queue if smoke fails.

- [ ] **Step 5: Launch the detached queue**

Run `scripts/launch_evidence_queue_server.sh`, wait for the first formal epoch, and verify:

```bash
cat /root/autodl-tmp/jobs/multiseed-evidence/status.json
cat /root/autodl-tmp/jobs/multiseed-evidence/pid.txt
nvidia-smi
tail -n 40 /root/autodl-tmp/jobs/multiseed-evidence/nohup.log
```

Expected: queue state `running`, current condition `baseline_s29`, PID alive, GPU active, and epoch output present.

- [ ] **Step 6: Record handoff**

Report the exact commit, PID, current condition, estimated completion time, live-log path, status path, and recovery behavior. The Codex session may then close without stopping the server process.
