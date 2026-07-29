# Phase 2A Trusted YOLOv8m Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strict, reproducible YOLOv8m training pipeline that preflights KITTI data, records every experiment, runs smoke/full training, exports common-val predictions, and computes KITTI AP40.

**Architecture:** Keep stock Ultralytics 8.4.98 behind a delayed-import adapter. Pure project modules own strict configuration, provenance, run-state persistence, smoke dataset views, and AP40 orchestration; the CLI composes these units without modifying `site-packages`.

**Tech Stack:** Python 3.11, standard-library `dataclasses`/`unittest`, PyYAML 6.0.3, Pillow 12.3.0, PyTorch 2.5.1+cu121, Ultralytics 8.4.98, existing `ifdr_yolo.data` and `ifdr_yolo.eval`.

---

## File map

Create:

- `configs/experiments/kitti_yolov8m_baseline_s17.yaml`: versioned formal baseline input.
- `ifdr_yolo/experiments/__init__.py`: public experiment package.
- `ifdr_yolo/experiments/config.py`: immutable config types and strict YAML parsing.
- `ifdr_yolo/experiments/run_store.py`: collision-safe run layout and atomic state machine.
- `ifdr_yolo/experiments/provenance.py`: Git, environment, file-hash, and dataset preflight records.
- `ifdr_yolo/experiments/smoke_data.py`: deterministic 16/16 smoke view.
- `ifdr_yolo/experiments/ultralytics_runtime.py`: local config bootstrap and delayed framework adapter.
- `ifdr_yolo/experiments/baseline.py`: dry-run/smoke/full orchestration.
- `ifdr_yolo/eval/evaluate.py`: programmatic AP40 directory evaluation.
- `scripts/train_baseline.py`: user-facing CLI.
- `tests/test_experiment_config.py`
- `tests/test_run_store.py`
- `tests/test_provenance.py`
- `tests/test_smoke_data.py`
- `tests/test_ultralytics_runtime.py`
- `tests/test_evaluate_pipeline.py`
- `tests/test_baseline_pipeline.py`
- `tests/test_train_baseline_cli.py`

Modify:

- `scripts/evaluate_kitti.py`: delegate to the programmatic evaluator.
- `README.md`: document Phase 2A commands and artifact contract.

Do not modify or stage:

- historical root `train.py` and `kitti.yaml`;
- `KITTI_YOLOv8m_Results/`;
- raw/processed data, downloaded weights, `runs/`, or `tmp/`.

### Task 1: Strict baseline configuration

**Files:**

- Create: `ifdr_yolo/experiments/__init__.py`
- Create: `ifdr_yolo/experiments/config.py`
- Create: `tests/test_experiment_config.py`
- Create: `configs/experiments/kitti_yolov8m_baseline_s17.yaml`

- [ ] **Step 1: Write tests for valid parsing and repository-relative path resolution**

```python
class BaselineConfigTest(unittest.TestCase):
    def test_loads_valid_config_and_resolves_paths_from_repository(self) -> None:
        config = load_baseline_config(
            ROOT / "configs/experiments/kitti_yolov8m_baseline_s17.yaml",
            repository_root=ROOT,
        )
        self.assertEqual(config.schema_version, 1)
        self.assertEqual(config.experiment.seed, 17)
        self.assertEqual(config.training.epochs, 300)
        self.assertEqual(config.paths.data, ROOT / "configs/data/kitti_v2.yaml")
        self.assertEqual(
            config.paths.model_sha256,
            "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5",
        )
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_experiment_config -v
```

Expected: import failure for missing `ifdr_yolo.experiments.config`.

- [ ] **Step 3: Add immutable config types and a strict parser**

Expose exactly:

```python
@dataclass(frozen=True)
class BaselineConfig:
    schema_version: int
    experiment: ExperimentConfig
    paths: PathsConfig
    training: TrainingConfig
    prediction: PredictionConfig


def load_baseline_config(path: Path, *, repository_root: Path) -> BaselineConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("baseline config root must be a mapping")
    return _parse_baseline_payload(
        payload,
        repository_root=repository_root.resolve(),
    )
```

Parser behavior:

- require the four top-level sections and `schema_version`;
- reject unknown keys at every level;
- reject `bool` where an integer/float is required;
- require seed `>= 0`, epoch/batch/imgsz/workers positive except workers may be zero;
- require confidence in `[0, 1]`, NMS IoU in `(0, 1]`, `max_det > 0`;
- resolve all configured paths against `repository_root`;
- keep dataclasses frozen.

- [ ] **Step 4: Add failing validation cases before each validation implementation**

```python
def test_rejects_unknown_training_key(self) -> None:
    payload = valid_payload()
    payload["training"]["epochz"] = 300
    with self.assertRaisesRegex(ValueError, "unknown training fields"):
        load_payload(payload)


def test_rejects_boolean_batch(self) -> None:
    payload = valid_payload()
    payload["training"]["batch"] = True
    with self.assertRaisesRegex(ValueError, "training.batch"):
        load_payload(payload)
```

Run after each RED/GREEN cycle:

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_experiment_config -v
```

- [ ] **Step 5: Add the formal seed-17 YAML**

Use all values frozen in the design spec, including the exact model SHA256, 300 epochs, batch 16, SGD, seed 17, and prediction thresholds.

- [ ] **Step 6: Run Task 1 and regression tests**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_experiment_config tests.test_data_config tests.test_kitti_splits -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```powershell
git add configs/experiments/kitti_yolov8m_baseline_s17.yaml ifdr_yolo/experiments tests/test_experiment_config.py
git commit -m "feat: define strict baseline experiment config"
```

### Task 2: Collision-safe run store and state machine

**Files:**

- Create: `ifdr_yolo/experiments/run_store.py`
- Create: `tests/test_run_store.py`

- [ ] **Step 1: Write a failing deterministic-ID test**

```python
def test_build_run_id_contains_identity_seed_and_commit(self) -> None:
    value = build_run_id(
        timestamp=datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
        dataset="kitti",
        model="yolov8m",
        variant="baseline",
        seed=17,
        git_sha="034aee29d105",
    )
    self.assertEqual(
        value,
        "20260729T120000Z-kitti-yolov8m-baseline-s17-034aee2",
    )
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_run_store -v
```

Expected: missing module/function.

- [ ] **Step 3: Implement run ID and atomic JSON writes**

Expose:

```python
def build_run_id(
    *,
    timestamp: datetime,
    dataset: str,
    model: str,
    variant: str,
    seed: int,
    git_sha: str,
) -> str:
    utc = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"{utc}-{dataset}-{model}-{variant}-s{seed}-{git_sha[:7]}"
    )


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
```

- [ ] **Step 4: Write failing tests for collisions and state transitions**

```python
def test_create_refuses_existing_run_directory(self) -> None:
    path = self.root / "run"
    path.mkdir()
    with self.assertRaisesRegex(FileExistsError, "already exists"):
        RunStore.create(path)


def test_rejects_illegal_state_jump(self) -> None:
    store = RunStore.create(self.root / "run")
    with self.assertRaisesRegex(ValueError, "prepared -> trained"):
        store.transition("trained")
```

Allowed transitions:

```python
{
    "prepared": {"running", "failed"},
    "running": {"trained", "failed"},
    "trained": {"evaluating", "failed"},
    "evaluating": {"complete", "failed"},
    "complete": set(),
    "failed": set(),
}
```

- [ ] **Step 5: Implement `RunStore`**

Required API is a mutable `RunStore(root: Path, state: str)` dataclass with
`RunStore.create(root)`, `transition(state)`, and
`fail(stage=..., error=...)`. `create` uses `Path.mkdir(parents=True,
exist_ok=False)`, initializes `prepared`, and writes `status.json`.
`transition` checks the exact transition table before updating the JSON.

`fail()` records `state`, `stage`, exception class, message, and UTC timestamp without deleting artifacts.

- [ ] **Step 6: Run tests and commit**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_run_store -v
git add ifdr_yolo/experiments/run_store.py tests/test_run_store.py
git commit -m "feat: add atomic experiment run store"
```

### Task 3: Provenance and strict preflight

**Files:**

- Create: `ifdr_yolo/experiments/provenance.py`
- Create: `tests/test_provenance.py`

- [ ] **Step 1: Write a failing test that distinguishes byte and canonical split hashes**

```python
def test_canonical_id_hash_includes_one_newline_per_id(self) -> None:
    ids = ("000001", "000002")
    expected = sha256(b"000001\n000002\n").hexdigest()
    self.assertEqual(canonical_ids_sha256(ids), expected)
```

- [ ] **Step 2: Run RED and implement the canonical hash**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_provenance -v
```

Implementation:

```python
def canonical_ids_sha256(image_ids: tuple[str, ...]) -> str:
    content = "".join(f"{image_id}\n" for image_id in image_ids)
    return sha256(content.encode("utf-8")).hexdigest()
```

- [ ] **Step 3: Write failing tests for model hash mismatch and split-source mismatch**

Use temporary files and assert errors contain the exact path and expected/actual hash.

```python
with self.assertRaisesRegex(ValueError, "model SHA256 mismatch"):
    verify_file_sha256(weight, "0" * 64, label="model")
```

- [ ] **Step 4: Implement provenance collectors**

Expose `find_repository_root(start: Path) -> Path`,
`verify_file_sha256(path: Path, expected: str, label: str) -> str`,
`collect_git_provenance(root: Path) -> dict[str, object]`,
`collect_environment() -> dict[str, object]`, and
`verify_dataset(config: BaselineConfig, verify_all_hashes: bool) ->
dict[str, object]`.

`verify_dataset()` must:

- load split IDs;
- validate `source.json` byte hashes with `sha256_file`;
- validate generated manifest canonical hashes with `canonical_ids_sha256`;
- compare parsed IDs to metadata split membership;
- call `audit_generated_dataset`;
- return counts and both hash kinds.

- [ ] **Step 5: Add tracked-dirty policy tests**

Inject subprocess output into a small pure helper:

```python
def classify_porcelain_status(lines: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    tracked: list[str] = []
    untracked: list[str] = []
    for line in lines:
        target = untracked if line.startswith("?? ") else tracked
        target.append(line)
    return tuple(tracked), tuple(untracked)
```

Assert tracked modifications block `full`, while `?? data-or-history` is recorded but does not block.

- [ ] **Step 6: Run tests and commit**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_provenance tests.test_phase1_audit -v
git add ifdr_yolo/experiments/provenance.py tests/test_provenance.py
git commit -m "feat: verify baseline experiment provenance"
```

### Task 4: Deterministic smoke dataset view

**Files:**

- Create: `ifdr_yolo/experiments/smoke_data.py`
- Create: `tests/test_smoke_data.py`

- [ ] **Step 1: Write failing selection tests**

```python
def test_selects_first_sixteen_ids_without_overlap(self) -> None:
    train = tuple(f"{value:06d}" for value in range(20))
    val = tuple(f"{value:06d}" for value in range(100, 120))
    selection = select_smoke_ids(train, val, count=16)
    self.assertEqual(selection.train_ids, train[:16])
    self.assertEqual(selection.val_ids, val[:16])
```

- [ ] **Step 2: Run RED and implement immutable selection**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_smoke_data -v
```

Reject insufficient IDs and overlap.

- [ ] **Step 3: Write a failing smoke-view file test**

Assert:

- train/val text files contain absolute PNG paths;
- YAML points to those lists and preserves class names;
- selected images and labels are copied under the smoke root;
- `manifest.json` records selected IDs and source split hashes;
- repeated construction with identical content is idempotent;
- differing content at the same path fails rather than overwrites.

- [ ] **Step 4: Implement**

Expose:

```python
@dataclass(frozen=True)
class SmokeView:
    root: Path
    data_yaml: Path
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]


def build_smoke_view(
    *,
    output_dir: Path,
    generated_dir: Path,
    train_ids: tuple[str, ...],
    val_ids: tuple[str, ...],
    train_source_sha256: str,
    val_source_sha256: str,
    count: int = 16,
) -> SmokeView:
    selection = select_smoke_ids(train_ids, val_ids, count=count)
    return _write_smoke_view(
        output_dir=output_dir,
        generated_dir=generated_dir,
        selection=selection,
        train_source_sha256=train_source_sha256,
        val_source_sha256=val_source_sha256,
    )
```

Write YAML with PyYAML and UTF-8/LF. Copy only the selected 16/16 images and
labels into `tmp/smoke-kitti` so Ultralytics label caches stay outside the
formal generated dataset. Never modify the source files.

- [ ] **Step 5: Run tests and commit**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_smoke_data -v
git add ifdr_yolo/experiments/smoke_data.py tests/test_smoke_data.py
git commit -m "feat: build deterministic KITTI smoke view"
```

### Task 5: Programmatic KITTI AP40 evaluation

**Files:**

- Create: `ifdr_yolo/eval/evaluate.py`
- Create: `tests/test_evaluate_pipeline.py`
- Modify: `scripts/evaluate_kitti.py`

- [ ] **Step 1: Write a failing library API test**

Build a one-image temporary KITTI GT and a perfect six-field YOLO prediction.

```python
result = evaluate_prediction_directory(
    prediction_dir=prediction_dir,
    label_dir=label_dir,
    image_dir=image_dir,
    split_path=split_path,
)
self.assertEqual(result["classes"]["Car"]["easy"]["ap40"], 100.0)
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_evaluate_pipeline -v
```

- [ ] **Step 3: Move evaluation orchestration into the library**

Expose `evaluate_prediction_directory(prediction_dir=..., label_dir=...,
image_dir=..., split_path=...) -> dict[str, object]`. It loads the split,
collects actual image sizes, loads raw KITTI GT and YOLO predictions, evaluates
every `EVAL_CLASSES × Difficulty` pair, and returns the existing JSON payload.

Implement the writer exactly as:

```python
def write_evaluation_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
```

Reuse `load_ids`, Pillow image sizes, `load_kitti_ground_truth`, `load_yolo_predictions`, and `evaluate_class`. Keep result keys byte-for-byte compatible with the existing CLI JSON.

- [ ] **Step 4: Refactor the CLI to call the library**

`scripts/evaluate_kitti.py` should only parse arguments, call the library, print the class table, and write JSON.

- [ ] **Step 5: Run old and new tests**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_evaluate_pipeline tests.test_evaluate_cli tests.test_kitti_ap40 -v
```

- [ ] **Step 6: Commit**

```powershell
git add ifdr_yolo/eval/evaluate.py scripts/evaluate_kitti.py tests/test_evaluate_pipeline.py
git commit -m "refactor: expose programmatic KITTI AP40 evaluation"
```

### Task 6: Safe Ultralytics runtime adapter

**Files:**

- Create: `ifdr_yolo/experiments/ultralytics_runtime.py`
- Create: `tests/test_ultralytics_runtime.py`

- [ ] **Step 1: Write a failing bootstrap test**

```python
def test_bootstrap_sets_config_dir_before_import(self) -> None:
    path = bootstrap_ultralytics_config(self.root)
    self.assertEqual(
        os.environ["YOLO_CONFIG_DIR"],
        str(self.root / "tmp" / "yolo-config"),
    )
    self.assertTrue(path.is_dir())
```

- [ ] **Step 2: Run RED and implement bootstrap without importing Ultralytics at module import**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_ultralytics_runtime -v
```

The source file must contain no top-level Ultralytics import.

- [ ] **Step 3: Add version/device validation tests through dependency injection**

```python
validate_runtime(
    actual_ultralytics="8.4.98",
    expected_ultralytics="8.4.98",
    cuda_available=True,
    device_count=1,
    requested_device="0",
    require_cuda=True,
)
```

Test mismatch, unavailable CUDA, and out-of-range device.

- [ ] **Step 4: Implement `UltralyticsAdapter`**

Required methods are `runtime_info()`, `train(model_path=..., data_path=...,
run_dir=..., args=...)`, and `predict(weights=..., image_paths=...,
output_dir=..., args=...)`.

`train()` creates `YOLO(str(model_path))`, calls `.train()` with
`project=run_dir.parent`, `name=run_dir.name`, `exist_ok=True`, the resolved
data path, and every mapped training argument. It returns
`<run_dir>/weights/best.pt` only if that file exists.

`predict()` calls `.predict()` with `save_txt=True`, `save_conf=True`,
`project=output_dir.parent`, `name=output_dir.name`, `exist_ok=True`, and every
mapped prediction argument. It returns `<output_dir>/labels`.

- [ ] **Step 5: Run tests and commit**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_ultralytics_runtime -v
git add ifdr_yolo/experiments/ultralytics_runtime.py tests/test_ultralytics_runtime.py
git commit -m "feat: isolate Ultralytics baseline runtime"
```

### Task 7: Baseline orchestration

**Files:**

- Create: `ifdr_yolo/experiments/baseline.py`
- Create: `tests/test_baseline_pipeline.py`

- [ ] **Step 1: Write a failing dry-run test using a fake adapter**

```python
def test_dry_run_never_calls_train_or_predict(self) -> None:
    adapter = FakeAdapter()
    result = run_baseline(self.config, mode="dry-run", adapter=adapter)
    self.assertEqual(result.mode, "dry-run")
    self.assertEqual(adapter.calls, [])
```

Inject preflight/provenance functions so the test uses temporary data rather than the real 7481-image dataset.

- [ ] **Step 2: Run RED**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_baseline_pipeline -v
```

- [ ] **Step 3: Implement mode resolution**

Expose:

```python
@dataclass(frozen=True)
class BaselineResult:
    mode: str
    run_dir: Path | None
    metrics_path: Path | None


def run_baseline(
    config: BaselineConfig,
    *,
    mode: Literal["dry-run", "smoke", "full"],
    adapter: RuntimeAdapter | None = None,
    repository_root: Path,
    device_override: str | None = None,
) -> BaselineResult:
    runner = _BaselineRunner(
        config=config,
        repository_root=repository_root,
        adapter=adapter,
        device_override=device_override,
    )
    return runner.run(mode)
```

Mode changes:

- dry-run: no run directory and no adapter train/predict calls;
- smoke: smoke YAML, epoch 1, imgsz 320, batch 2, workers 0;
- full: formal YAML values unchanged.

- [ ] **Step 4: Write failing lifecycle and failure-state tests**

Fake adapter creates `weights/best.pt` and deterministic prediction files. Assert state order ends in `complete`. Make fake training raise and assert `failed` records `stage="training"`.

- [ ] **Step 5: Implement orchestration**

Order:

1. bootstrap runtime config;
2. collect preflight/provenance;
3. return for dry-run;
4. create collision-safe run;
5. snapshot input/resolved config, provenance, Git SHA;
6. transition running;
7. train;
8. transition trained;
9. transition evaluating;
10. predict all validation images;
11. create an empty text file for any validation ID omitted by Ultralytics;
12. validate every prediction row has six fields;
13. evaluate AP40 and write JSON;
14. transition complete;
15. on any exception call `fail()` and re-raise.

- [ ] **Step 6: Add prediction-completeness tests**

```python
ensure_prediction_files(labels_dir, ("000001", "000002"))
self.assertTrue((labels_dir / "000002.txt").exists())
```

Reject extra non-split IDs and five-field predictions.

- [ ] **Step 7: Run tests and commit**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_baseline_pipeline tests.test_evaluate_pipeline -v
git add ifdr_yolo/experiments/baseline.py tests/test_baseline_pipeline.py
git commit -m "feat: orchestrate trusted YOLOv8m baseline runs"
```

### Task 8: CLI, documentation, and dry-run acceptance

**Files:**

- Create: `scripts/train_baseline.py`
- Create: `tests/test_train_baseline_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write a failing CLI help test**

```python
completed = subprocess.run(
    [sys.executable, "scripts/train_baseline.py", "--help"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=False,
)
self.assertEqual(completed.returncode, 0, completed.stderr)
self.assertIn("--mode {dry-run,smoke,full}", completed.stdout)
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_train_baseline_cli -v
```

- [ ] **Step 3: Implement CLI**

Arguments:

```text
--config PATH   required
--mode {dry-run,smoke,full}   required
--device DEVICE optional explicit override
```

The script adds the repository root to `sys.path` when run directly, loads strict config, invokes `run_baseline`, prints a compact result, and exits nonzero on validation/training/evaluation failure.

- [ ] **Step 4: Update README**

Document:

- formal config location;
- dry-run command;
- smoke command;
- full AutoDL command;
- run-state/artifact contract;
- distinction between AP40 and Ultralytics mAP;
- refusal to overwrite runs;
- model/data files intentionally excluded from Git.

- [ ] **Step 5: Run all unit tests**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest discover -s tests -v
```

Expected: 64 existing tests plus all Phase 2A tests pass.

- [ ] **Step 6: Run real-data dry-run**

```powershell
& 'D:\ana\envs\yolo\python.exe' scripts/train_baseline.py `
  --config configs/experiments/kitti_yolov8m_baseline_s17.yaml `
  --mode dry-run
```

Expected final line:

```text
BASELINE PREFLIGHT PASSED
```

- [ ] **Step 7: Commit**

```powershell
git add scripts/train_baseline.py tests/test_train_baseline_cli.py README.md
git commit -m "feat: add trusted baseline command line workflow"
```

### Task 9: Real local smoke and Phase 2A acceptance report

**Files:**

- Create: `docs/reports/phase2a-baseline-infrastructure-acceptance.md`

- [ ] **Step 1: Run the complete regression suite immediately before smoke**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest discover -s tests -v
```

Do not start GPU smoke if any test fails.

- [ ] **Step 2: Run the real 16/16 smoke pipeline**

```powershell
& 'D:\ana\envs\yolo\python.exe' scripts/train_baseline.py `
  --config configs/experiments/kitti_yolov8m_baseline_s17.yaml `
  --mode smoke `
  --device 0
```

Required artifacts:

- `weights/best.pt`;
- 16 val prediction files including empty files;
- `metrics_ap40.json`;
- `config.resolved.yaml`;
- `environment.json`;
- `data_manifest.json`;
- terminal state `complete`.

- [ ] **Step 3: Run a post-smoke audit**

Check:

```powershell
Get-ChildItem -Recurse runs\<smoke-run-id>
Get-Content runs\<smoke-run-id>\status.json
Get-Content runs\<smoke-run-id>\metrics_ap40.json
```

Confirm no file under `kitti_raw/` or `data/processed/` changed by comparing recorded hashes.

- [ ] **Step 4: Write the acceptance report**

Record:

- Git commit;
- exact command;
- Python/PyTorch/CUDA/Ultralytics/GPU;
- data and split hashes;
- unit-test count;
- smoke run ID and elapsed time;
- artifact inventory;
- AP40 values explicitly labelled smoke-only and not research results;
- any warnings or constraints before AutoDL full training.

- [ ] **Step 5: Verify Git scope and commit**

```powershell
git status --short
git diff --check
git add docs/reports/phase2a-baseline-infrastructure-acceptance.md
git commit -m "docs: accept trusted baseline infrastructure"
```

- [ ] **Step 6: Merge, re-run tests, and synchronize GitHub**

Follow `finishing-a-development-branch`:

1. merge the feature branch into `master`;
2. rerun the complete unit test suite on merged `master`;
3. verify local and remote commit SHA;
4. push `master` to `origin`;
5. preserve ignored run artifacts locally;
6. remove only the owned `.worktrees/phase2a-trusted-baseline` worktree.

## Completion gate

Phase 2A implementation is complete only when all statements are true:

- strict config and preflight tests pass;
- byte and canonical split hashes are independently verified;
- framework import never depends on the roaming Ultralytics directory;
- dry-run passes against the real 7481-image dataset;
- real 16/16 smoke reaches `complete`;
- smoke exports confidence predictions and KITTI AP40;
- old Phase 1 tests remain green;
- no raw data, processed dataset, weights, runs, or historical files are committed;
- merged `master` and `origin/master` point to the same verified commit.
