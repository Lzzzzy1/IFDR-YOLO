# YOLOv8m-P2 Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a trusted, four-scale YOLOv8m-P2 experiment that reuses the accepted KITTI pipeline and transfers only semantically equivalent pretrained layers.

**Architecture:** Store an expanded, project-owned copy of the Ultralytics 8.4.98 P2 `m` graph with `nc=3`. Extend the strict experiment config with an optional initialization contract, implement a pure semantic-prefix state selector, prepare the model deterministically inside the runtime adapter, and preserve the existing dry-run/smoke/full orchestration and AP40 evaluation.

**Tech Stack:** Python 3.11, PyTorch 2.5.1+cu121, Ultralytics 8.4.98, PyYAML, `unittest`, KITTI AP40.

---

## Scope and file map

Create:

- `models/kitti-p2-m.yaml` — expanded Ultralytics YOLOv8-P2 `m` graph for KITTI.
- `ifdr_yolo/models/__init__.py` — public model-initialization exports.
- `ifdr_yolo/models/p2.py` — P2 structure constants and invariant validation.
- `ifdr_yolo/models/initialization.py` — semantic-prefix tensor selection and loading report.
- `configs/experiments/kitti_yolov8m_p2_s17.yaml` — formal P2 seed-17 experiment.
- `tests/test_p2_model.py` — real locked-version model structure tests.
- `tests/test_model_initialization.py` — pure and real initialization tests.
- `docs/reports/phase2b-p2-infrastructure-acceptance.md` — post-smoke acceptance evidence.

Modify:

- `ifdr_yolo/experiments/config.py` — optional strict `initialization` section.
- `ifdr_yolo/experiments/ultralytics_runtime.py` — deterministic model preparation.
- `ifdr_yolo/experiments/baseline.py` — prepare once, train the prepared model, save initialization manifest.
- `tests/test_experiment_config.py`
- `tests/test_provenance.py`
- `tests/test_ultralytics_runtime.py`
- `tests/test_baseline_pipeline.py`
- `README.md`

Do not modify `site-packages`, raw KITTI data, generated labels, the AP40 protocol,
or the accepted baseline experiment YAML.

### Task 1: Strict optional initialization configuration

**Files:**

- Modify: `ifdr_yolo/experiments/config.py`
- Modify: `tests/test_experiment_config.py`

- [ ] **Step 1: Write a failing valid-initialization test**

Add to `tests/test_experiment_config.py`:

```python
def test_loads_complete_semantic_prefix_initialization(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
        payload = valid_payload()
        payload["initialization"] = {
            "pretrained": "yolov8m.pt",
            "pretrained_sha256": MODEL_SHA256,
            "strategy": "semantic_prefix",
            "max_layer": 15,
            "expected_items": 306,
        }
        path = self.write_payload(directory, payload)

        config = load_baseline_config(path, repository_root=ROOT)

        assert config.initialization is not None
        self.assertEqual(config.initialization.pretrained, ROOT / "yolov8m.pt")
        self.assertEqual(config.initialization.max_layer, 15)
        self.assertEqual(config.initialization.expected_items, 306)
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest `
  tests.test_experiment_config.BaselineConfigTest.test_loads_complete_semantic_prefix_initialization -v
```

Expected: FAIL because `initialization` is an unknown top-level field.

- [ ] **Step 3: Implement the immutable config type and optional-field parser**

Add to `ifdr_yolo/experiments/config.py`:

```python
@dataclass(frozen=True)
class InitializationConfig:
    pretrained: Path
    pretrained_sha256: str
    strategy: str
    max_layer: int
    expected_items: int
```

Change `BaselineConfig` to include:

```python
initialization: InitializationConfig | None = None
source_path: Path | None = None
```

Change `_require_fields` to accept optional keys:

```python
def _require_fields(
    mapping: dict[str, Any],
    *,
    field: str,
    expected: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected - optional)
    if missing:
        raise ValueError(f"missing {field} fields: {missing}")
    if unknown:
        raise ValueError(f"unknown {field} fields: {unknown}")
```

Extract the existing model-hash validation into:

```python
def _require_sha256(value: object, field: str) -> str:
    result = _require_text(value, field).lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{field} must be 64 hexadecimal characters")
    return result
```

Add:

```python
def _parse_initialization(
    value: object,
    root: Path,
) -> InitializationConfig:
    mapping = _require_mapping(value, "initialization")
    expected = {
        "pretrained",
        "pretrained_sha256",
        "strategy",
        "max_layer",
        "expected_items",
    }
    _require_fields(mapping, field="initialization", expected=expected)
    strategy = _require_text(mapping["strategy"], "initialization.strategy")
    if strategy != "semantic_prefix":
        raise ValueError(
            "initialization.strategy must be 'semantic_prefix'"
        )
    return InitializationConfig(
        pretrained=_resolve_path(
            mapping["pretrained"],
            "initialization.pretrained",
            root,
        ),
        pretrained_sha256=_require_sha256(
            mapping["pretrained_sha256"],
            "initialization.pretrained_sha256",
        ),
        strategy=strategy,
        max_layer=_require_int(
            mapping["max_layer"],
            "initialization.max_layer",
            minimum=0,
        ),
        expected_items=_require_int(
            mapping["expected_items"],
            "initialization.expected_items",
            minimum=1,
        ),
    )
```

At the top level, require the five existing sections and allow only
`initialization` as optional. Parse it when present.

- [ ] **Step 4: Add failing invalid-contract cases**

Add table-driven subtests that reject:

```python
(
    {"strategy": "shape_only"},
    "initialization.strategy",
),
(
    {"max_layer": -1},
    "initialization.max_layer",
),
(
    {"expected_items": 0},
    "initialization.expected_items",
),
(
    {"pretrained_sha256": "not-a-hash"},
    "initialization.pretrained_sha256",
),
```

Also delete `expected_items` from one payload and assert
`missing initialization fields`.

- [ ] **Step 5: Keep fixture constructors backward compatible**

Leave `tests/test_provenance.py::make_config()` unchanged. The new dataclass
field defaults to `None`, so every existing fixture and baseline config must
continue working without an initialization section.

- [ ] **Step 6: Run GREEN and regression tests**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest `
  tests.test_experiment_config tests.test_provenance -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add ifdr_yolo/experiments/config.py tests/test_experiment_config.py
git commit -m "feat: define semantic model initialization config"
```

### Task 2: Project-owned expanded P2 model

**Files:**

- Create: `models/kitti-p2-m.yaml`
- Create: `ifdr_yolo/models/__init__.py`
- Create: `ifdr_yolo/models/p2.py`
- Create: `tests/test_p2_model.py`

- [ ] **Step 1: Write the missing-model RED test**

Create `tests/test_p2_model.py`:

```python
from pathlib import Path
import unittest

import torch

from ifdr_yolo.data.splits import sha256_file
from ifdr_yolo.experiments.ultralytics_runtime import (
    bootstrap_ultralytics_config,
)
from ifdr_yolo.models.p2 import inspect_p2_model


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/kitti-p2-m.yaml"
MODEL_SHA256 = (
    "0d2cbd9215dc62c4c9920127964ce0ab61b5d10581861d8e83c33fc168aa302a"
)


class P2ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bootstrap_ultralytics_config(ROOT)
        from ultralytics.nn.tasks import DetectionModel

        cls.model = DetectionModel(str(MODEL_PATH), verbose=False)

    def test_project_model_hash_and_structure_are_fixed(self) -> None:
        self.assertEqual(sha256_file(MODEL_PATH), MODEL_SHA256)
        summary = inspect_p2_model(self.model)
        self.assertEqual(summary["strides"], [4.0, 8.0, 16.0, 32.0])
        self.assertEqual(summary["detect_inputs"], 4)
        self.assertEqual(summary["parameters"], 25_052_620)
        self.assertEqual(summary["state_items"], 581)

    def test_training_forward_produces_four_spatial_scales(self) -> None:
        self.model.train()
        with torch.no_grad():
            outputs = self.model(torch.zeros(1, 3, 320, 320))
        self.assertEqual(set(outputs), {"boxes", "scores", "feats"})
        features = outputs["feats"]
        self.assertEqual(len(features), 4)
        self.assertEqual(
            [tuple(feature.shape[-2:]) for feature in features],
            [(80, 80), (40, 40), (20, 20), (10, 10)],
        )
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_p2_model -v
```

Expected: ERROR because `ifdr_yolo.models.p2` and the YAML do not exist.

- [ ] **Step 3: Add the exact expanded YAML**

Create `models/kitti-p2-m.yaml` with exactly:

```yaml
# Ultralytics YOLOv8 P2 model, expanded to m scale for KITTI
# Upstream: ultralytics 8.4.98 cfg/models/v8/yolov8-p2.yaml
# Upstream SHA256: ba54363e9f283e8f60b0fa0843eb37115093338d6f9273ccd4f74f60639754f4
# License: AGPL-3.0

nc: 3

backbone:
  - [-1, 1, Conv, [48, 3, 2]]
  - [-1, 1, Conv, [96, 3, 2]]
  - [-1, 2, C2f, [96, True]]
  - [-1, 1, Conv, [192, 3, 2]]
  - [-1, 4, C2f, [192, True]]
  - [-1, 1, Conv, [384, 3, 2]]
  - [-1, 4, C2f, [384, True]]
  - [-1, 1, Conv, [576, 3, 2]]
  - [-1, 2, C2f, [576, True]]
  - [-1, 1, SPPF, [576, 5]]

head:
  - [-1, 1, nn.Upsample, [None, 2, nearest]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C2f, [384]]
  - [-1, 1, nn.Upsample, [None, 2, nearest]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C2f, [192]]
  - [-1, 1, nn.Upsample, [None, 2, nearest]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C2f, [96]]
  - [-1, 1, Conv, [96, 3, 2]]
  - [[-1, 15], 1, Concat, [1]]
  - [-1, 2, C2f, [192]]
  - [-1, 1, Conv, [192, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 2, C2f, [384]]
  - [-1, 1, Conv, [384, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 2, C2f, [576]]
  - [[18, 21, 24, 27], 1, Detect, [nc]]
```

- [ ] **Step 4: Implement strict structure inspection**

Create `ifdr_yolo/models/p2.py`:

```python
from __future__ import annotations

from typing import Any


EXPECTED_STRIDES = [4.0, 8.0, 16.0, 32.0]
EXPECTED_PARAMETERS = 25_052_620
EXPECTED_STATE_ITEMS = 581


def inspect_p2_model(model: Any) -> dict[str, object]:
    strides = [float(value) for value in model.stride.tolist()]
    detect = model.model[-1]
    detect_inputs = len(detect.cv2)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    state_items = len(model.state_dict())
    if strides != EXPECTED_STRIDES:
        raise RuntimeError(f"P2 stride mismatch: {strides}")
    if detect_inputs != 4:
        raise RuntimeError(f"P2 Detect input mismatch: {detect_inputs}")
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"P2 parameter mismatch: {parameters}")
    if state_items != EXPECTED_STATE_ITEMS:
        raise RuntimeError(f"P2 state item mismatch: {state_items}")
    return {
        "strides": strides,
        "detect_inputs": detect_inputs,
        "parameters": parameters,
        "state_items": state_items,
    }
```

Export it from `ifdr_yolo/models/__init__.py`.

- [ ] **Step 5: Run GREEN**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_p2_model -v
```

Expected: 2 tests pass with no model-scale warning.

- [ ] **Step 6: Commit**

```powershell
git add models/kitti-p2-m.yaml ifdr_yolo/models tests/test_p2_model.py
git commit -m "feat: add fixed YOLOv8m P2 model graph"
```

### Task 3: Semantic-prefix weight transfer

**Files:**

- Create: `ifdr_yolo/models/initialization.py`
- Modify: `ifdr_yolo/models/__init__.py`
- Create: `tests/test_model_initialization.py`

- [ ] **Step 1: Write pure selector RED tests**

Create `tests/test_model_initialization.py` with:

```python
from pathlib import Path
import unittest

import torch

from ifdr_yolo.experiments.ultralytics_runtime import (
    bootstrap_ultralytics_config,
)
from ifdr_yolo.models.initialization import (
    apply_semantic_prefix_initialization,
    select_semantic_prefix_state,
)


ROOT = Path(__file__).resolve().parents[1]


class SemanticInitializationTest(unittest.TestCase):
    def test_selector_requires_layer_limit_target_key_and_shape(self) -> None:
        source = {
            "model.0.weight": torch.ones(2, 2),
            "model.15.weight": torch.ones(1),
            "model.16.weight": torch.ones(1),
            "other.weight": torch.ones(1),
        }
        target = {
            "model.0.weight": torch.zeros(2, 2),
            "model.15.weight": torch.zeros(2),
            "model.16.weight": torch.zeros(1),
        }

        selected = select_semantic_prefix_state(
            source,
            target,
            max_layer=15,
        )

        self.assertEqual(tuple(selected), ("model.0.weight",))

    def test_selector_rejects_negative_layer_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_layer"):
            select_semantic_prefix_state({}, {}, max_layer=-1)
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_model_initialization -v
```

Expected: ERROR because the initialization module is missing.

- [ ] **Step 3: Implement the pure selector and report**

Create `ifdr_yolo/models/initialization.py`:

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

import torch


_MODEL_KEY = re.compile(r"^model\.(\d+)\.")


@dataclass(frozen=True)
class InitializationReport:
    strategy: str
    max_layer: int
    expected_items: int
    transferred_items: int
    source_items: int
    target_items: int
    untransferred_items: int
    transferred_keys: tuple[str, ...]
    transferred_shapes: dict[str, list[int]]

    def to_payload(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "max_layer": self.max_layer,
            "expected_items": self.expected_items,
            "transferred_items": self.transferred_items,
            "source_items": self.source_items,
            "target_items": self.target_items,
            "untransferred_items": self.untransferred_items,
            "transferred_keys": list(self.transferred_keys),
            "transferred_shapes": self.transferred_shapes,
        }


def select_semantic_prefix_state(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    *,
    max_layer: int,
) -> dict[str, torch.Tensor]:
    if max_layer < 0:
        raise ValueError("max_layer must be >= 0")
    selected: dict[str, torch.Tensor] = {}
    for key in sorted(source):
        match = _MODEL_KEY.match(key)
        if match is None or int(match.group(1)) > max_layer:
            continue
        target_tensor = target.get(key)
        if target_tensor is None or source[key].shape != target_tensor.shape:
            continue
        selected[key] = source[key]
    return selected


def apply_semantic_prefix_initialization(
    target_model: Any,
    source_model: Any,
    *,
    max_layer: int,
    expected_items: int,
) -> InitializationReport:
    source_state = source_model.float().state_dict()
    target_state = target_model.state_dict()
    selected = select_semantic_prefix_state(
        source_state,
        target_state,
        max_layer=max_layer,
    )
    if len(selected) != expected_items:
        raise RuntimeError(
            "semantic initialization item mismatch: "
            f"expected={expected_items}, actual={len(selected)}"
        )
    target_model.load_state_dict(selected, strict=False)
    keys = tuple(selected)
    return InitializationReport(
        strategy="semantic_prefix",
        max_layer=max_layer,
        expected_items=expected_items,
        transferred_items=len(keys),
        source_items=len(source_state),
        target_items=len(target_state),
        untransferred_items=len(target_state) - len(keys),
        transferred_keys=keys,
        transferred_shapes={
            key: list(selected[key].shape)
            for key in keys
        },
    )
```

- [ ] **Step 4: Add the real 306-item RED/GREEN integration test**

Add:

```python
def test_locked_models_transfer_exactly_306_items(self) -> None:
    bootstrap_ultralytics_config(ROOT)
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    target = DetectionModel(
        str(ROOT / "models/kitti-p2-m.yaml"),
        verbose=False,
    )
    source = YOLO(str(ROOT / "yolov8m.pt")).model

    report = apply_semantic_prefix_initialization(
        target,
        source,
        max_layer=15,
        expected_items=306,
    )

    self.assertEqual(report.transferred_items, 306)
    self.assertEqual(
        max(int(key.split(".")[1]) for key in report.transferred_keys),
        15,
    )
    self.assertEqual(
        sorted(
            {
                int(key.split(".")[1])
                for key in report.transferred_keys
            }
        ),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15],
    )
```

Run the pure tests before implementation and the real test after implementation.

- [ ] **Step 5: Run GREEN**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest `
  tests.test_model_initialization tests.test_p2_model -v
```

Expected: all tests pass, exact count 306.

- [ ] **Step 6: Commit**

```powershell
git add ifdr_yolo/models tests/test_model_initialization.py
git commit -m "feat: initialize P2 from semantic model prefix"
```

### Task 4: Deterministic prepared-model runtime

**Files:**

- Modify: `ifdr_yolo/experiments/ultralytics_runtime.py`
- Modify: `tests/test_ultralytics_runtime.py`

- [ ] **Step 1: Write failing preparation tests**

Add a fake wrapper whose `.model` is a small `torch.nn.Module`. Add tests that:

```python
prepared = adapter.prepare_model(
    model_path=model_yaml,
    model_sha256="a" * 64,
    initialization=None,
    seed=17,
    deterministic=True,
)
self.assertIs(prepared.handle, FakeYOLO.instances[-1])
self.assertIsNone(prepared.initialization)
self.assertEqual(seed_calls, [(17, True)])
```

For an initialization case, patch
`apply_semantic_prefix_initialization` through constructor injection and assert
the adapter creates the target first, the pretrained source second, calls the
initializer with `max_layer=15` and `expected_items=306`, validates P2, and
returns the report payload.

- [ ] **Step 2: Run RED**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_ultralytics_runtime -v
```

Expected: FAIL because `prepare_model` and `PreparedModel` do not exist.

- [ ] **Step 3: Add the prepared-model API**

Add:

```python
@dataclass(frozen=True)
class PreparedModel:
    handle: Any
    initialization: dict[str, object] | None
```

Extend `UltralyticsAdapter.__init__` with injectable callables:

```python
seed_initializer: Callable[[int, bool], None] | None = None
model_initializer: Callable[..., Any] = apply_semantic_prefix_initialization
p2_inspector: Callable[[Any], dict[str, object]] = inspect_p2_model
```

Load the default seed helper lazily:

```python
def _initialize_seed(seed: int, deterministic: bool) -> None:
    from ultralytics.utils.torch_utils import init_seeds

    init_seeds(seed, deterministic=deterministic)
```

Implement:

```python
def prepare_model(
    self,
    *,
    model_path: Path,
    model_sha256: str,
    initialization: InitializationConfig | None,
    seed: int,
    deterministic: bool,
) -> PreparedModel:
    initializer = self._seed_initializer or _initialize_seed
    initializer(seed, deterministic)
    handle = self._factory()(str(model_path))
    payload = None
    if initialization is not None:
        source = self._factory()(str(initialization.pretrained))
        report = self._model_initializer(
            handle.model,
            source.model,
            max_layer=initialization.max_layer,
            expected_items=initialization.expected_items,
        )
        structure = self._p2_inspector(handle.model)
        payload = report.to_payload()
        payload["architecture"] = str(model_path)
        payload["architecture_sha256"] = model_sha256
        payload["pretrained"] = str(initialization.pretrained)
        payload["pretrained_sha256"] = (
            initialization.pretrained_sha256
        )
        payload["ultralytics"] = EXPECTED_ULTRALYTICS_VERSION
        payload["structure"] = structure
        payload["seed"] = seed
        payload["deterministic"] = deterministic
    return PreparedModel(handle=handle, initialization=payload)
```

Change `train()` to accept `prepared_model: PreparedModel` instead of
`model_path`, and call `prepared_model.handle.train(...)`.

- [ ] **Step 4: Update the existing adapter train test**

Prepare a `PreparedModel(handle=FakeYOLO(...), initialization=None)` and pass it
to `adapter.train`. Assert no second model instance is created. Prediction
continues loading `best.pt` normally.

- [ ] **Step 5: Run GREEN**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_ultralytics_runtime -v
```

Expected: all runtime tests pass.

- [ ] **Step 6: Commit**

```powershell
git add ifdr_yolo/experiments/ultralytics_runtime.py tests/test_ultralytics_runtime.py
git commit -m "feat: prepare research models deterministically"
```

### Task 5: Orchestration, provenance, and initialization manifest

**Files:**

- Modify: `ifdr_yolo/experiments/baseline.py`
- Modify: `tests/test_baseline_pipeline.py`

- [ ] **Step 1: Update the fake adapter and write a failing P2 lifecycle test**

The fake adapter must implement:

```python
def prepare_model(
    self,
    *,
    model_path,
    model_sha256,
    initialization,
    seed,
    deterministic,
):
    self.prepare_calls.append(
        {
            "model_path": model_path,
            "model_sha256": model_sha256,
            "initialization": initialization,
            "seed": seed,
            "deterministic": deterministic,
        }
    )
    payload = None
    if initialization is not None:
        payload = {
            "strategy": "semantic_prefix",
            "transferred_items": 306,
        }
    return PreparedModel(handle=object(), initialization=payload)
```

Write a test that creates an `InitializationConfig`, runs smoke with the fake
adapter, and asserts:

- the pretrained hash is verified with label `pretrained model`;
- `prepare_model` receives seed 17 and `deterministic=True`;
- `initialization.json` contains `transferred_items=306`;
- `train` receives the exact prepared object.

- [ ] **Step 2: Run RED**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_baseline_pipeline -v
```

Expected: FAIL because the protocol still passes a model path directly.

- [ ] **Step 3: Extend preflight for the initialization artifact**

In `_preflight`, when initialization is present:

```python
_require_file(
    config.initialization.pretrained,
    "pretrained model",
)
services.verify_file_sha256(
    config.initialization.pretrained,
    config.initialization.pretrained_sha256,
    label="pretrained model",
)
```

Keep `paths.model` hash validation; for P2 it now validates the architecture
YAML rather than a checkpoint.

- [ ] **Step 4: Prepare before dry-run completion and before training**

Extend `RuntimeAdapter` with `prepare_model`. In `run_baseline`:

```python
if mode == "dry-run":
    runtime.prepare_model(
        model_path=config.paths.model,
        model_sha256=config.paths.model_sha256,
        initialization=config.initialization,
        seed=config.experiment.seed,
        deterministic=config.training.deterministic,
    )
    return BaselineResult(
        mode="dry-run",
        run_dir=None,
        metrics_path=None,
    )
```

For smoke/full, prepare inside the run-store `try` block with
`stage="initialization"`. If a report exists, write:

```python
atomic_write_json(
    store.root / "initialization.json",
    prepared.initialization,
)
```

Then pass `prepared_model=prepared` to `runtime.train`.

- [ ] **Step 5: Preserve failure evidence**

Add a fake preparation failure and assert `status.json` records:

```json
{
  "state": "failed",
  "stage": "initialization",
  "error_type": "RuntimeError"
}
```

- [ ] **Step 6: Run GREEN and the full pipeline test group**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest `
  tests.test_baseline_pipeline `
  tests.test_provenance `
  tests.test_train_baseline_cli -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add ifdr_yolo/experiments/baseline.py tests/test_baseline_pipeline.py
git commit -m "feat: record P2 initialization provenance"
```

### Task 6: Formal P2 experiment configuration and dry-run

**Files:**

- Create: `configs/experiments/kitti_yolov8m_p2_s17.yaml`
- Modify: `README.md`
- Modify: `tests/test_experiment_config.py`

- [ ] **Step 1: Write a failing repository-config test**

Add:

```python
def test_repository_p2_config_is_a_controlled_baseline_derivative(self) -> None:
    baseline = load_baseline_config(
        ROOT / "configs/experiments/kitti_yolov8m_baseline_s17.yaml",
        repository_root=ROOT,
    )
    p2 = load_baseline_config(
        ROOT / "configs/experiments/kitti_yolov8m_p2_s17.yaml",
        repository_root=ROOT,
    )

    self.assertEqual(p2.experiment.variant, "p2")
    self.assertEqual(p2.paths.model, ROOT / "models/kitti-p2-m.yaml")
    self.assertEqual(p2.training, baseline.training)
    self.assertEqual(p2.prediction, baseline.prediction)
    self.assertEqual(p2.paths.data, baseline.paths.data)
    assert p2.initialization is not None
    self.assertEqual(p2.initialization.expected_items, 306)
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest `
  tests.test_experiment_config.BaselineConfigTest.test_repository_p2_config_is_a_controlled_baseline_derivative -v
```

Expected: FAIL because the P2 config does not exist.

- [ ] **Step 3: Add the exact seed-17 P2 config**

Create:

```yaml
schema_version: 1
experiment:
  dataset: kitti
  model: yolov8m
  variant: p2
  seed: 17
paths:
  model: models/kitti-p2-m.yaml
  model_sha256: 0d2cbd9215dc62c4c9920127964ce0ab61b5d10581861d8e83c33fc168aa302a
  data: configs/data/kitti_v2.yaml
  generated_data: data/processed/kitti_yolo_v2
  raw_images: kitti_raw/training/image_2/training/image_2
  raw_labels: kitti_raw/training/label_2/training/label_2
  train_ids: configs/splits/kitti_train.txt
  val_ids: configs/splits/kitti_val.txt
initialization:
  pretrained: yolov8m.pt
  pretrained_sha256: 5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5
  strategy: semantic_prefix
  max_layer: 15
  expected_items: 306
training:
  epochs: 300
  imgsz: 640
  batch: 16
  workers: 8
  device: "0"
  optimizer: SGD
  lr0: 0.01
  lrf: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3.0
  patience: 0
  amp: true
  deterministic: true
  cache: false
prediction:
  conf: 0.001
  iou: 0.7
  max_det: 300
  half: false
```

- [ ] **Step 4: Document the P2 commands and research boundary**

Add README commands for P2 dry-run, smoke, and full. State that P2 is an
ablation factor based on the locked upstream graph, not the claimed novel
contribution.

- [ ] **Step 5: Run GREEN and all 101+ unit tests**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest discover -s tests -v
```

Expected: all tests pass; count is greater than 101.

- [ ] **Step 6: Commit before real-data acceptance**

```powershell
git add configs/experiments/kitti_yolov8m_p2_s17.yaml README.md tests/test_experiment_config.py
git commit -m "feat: add controlled KITTI P2 experiment"
```

- [ ] **Step 7: Run real-data dry-run**

```powershell
& 'D:\ana\envs\yolo\python.exe' scripts/train_baseline.py `
  --config configs/experiments/kitti_yolov8m_p2_s17.yaml `
  --mode dry-run `
  --device 0
```

Expected:

```text
BASELINE PREFLIGHT PASSED
```

The command must verify all 7481 source pairs, the local YAML hash, the
pretrained checkpoint hash, the four strides, parameter count, and 306-item
transfer.

### Task 7: Real GPU smoke and Phase 2B acceptance

**Files:**

- Create: `docs/reports/phase2b-p2-infrastructure-acceptance.md`

- [ ] **Step 1: Run a fresh full regression**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest discover -s tests -v
```

Do not start GPU smoke if any test fails.

- [ ] **Step 2: Run the real 16/16 P2 smoke**

```powershell
& 'D:\ana\envs\yolo\python.exe' scripts/train_baseline.py `
  --config configs/experiments/kitti_yolov8m_p2_s17.yaml `
  --mode smoke `
  --device 0
```

Required result:

```text
BASELINE SMOKE COMPLETE
```

- [ ] **Step 3: Audit the run**

Confirm:

- `status.json` state is `complete`;
- `git_commit.txt` equals the pre-smoke commit;
- `initialization.json` records 306 keys and no key above layer 15;
- structure records four Detect inputs and strides 4/8/16/32;
- `weights/best.pt` and `weights/last.pt` exist;
- exactly 16 prediction files exist and match the smoke val ID set;
- `metrics_ap40.json` uses `ifdr_yolo.kitti_ap40` and split count 16;
- `data/processed/kitti_yolo_v2/labels/train.cache` and `val.cache` do not exist;
- temporary caches exist only below `tmp/smoke-kitti`.

- [ ] **Step 4: Write the acceptance report**

Record:

- Git SHA and exact commands;
- Python, PyTorch, CUDA, GPU, and Ultralytics;
- local model YAML and pretrained SHA256;
- stride, parameter count, state items, transfer count, and transferred layers;
- test count;
- smoke run ID and artifact inventory;
- AP40 explicitly labelled smoke-only and not a research result;
- the next gate: formal server baseline and P2 runs under the same effective batch.

- [ ] **Step 5: Verify and commit locally**

```powershell
git diff --check
& 'D:\ana\envs\yolo\python.exe' -m unittest discover -s tests
git add docs/reports/phase2b-p2-infrastructure-acceptance.md
git commit -m "docs: accept trusted P2 experiment pipeline"
```

- [ ] **Step 6: Finish without pushing**

Merge the local feature branch into local `master`, rerun the full test suite,
preserve ignored run artifacts under the main `runs/` directory, remove the
owned worktree, and verify `origin/master` remains at
`05a1b6c42459e8bdbf29315fd92d664c1ace9967`. Do not run `git push`.

## Completion gate

Phase 2B P2 infrastructure is complete only when:

- the expanded YAML hash is
  `0d2cbd9215dc62c4c9920127964ce0ab61b5d10581861d8e83c33fc168aa302a`;
- the real `nc=3` model has 25,052,620 parameters and 581 state items;
- Detect strides are exactly 4, 8, 16, 32;
- exactly 306 state items from layers 0–15 are transferred;
- all automated tests pass on the merged local `master`;
- a real GPU smoke reaches `complete`;
- no formal data or Phase 2A artifact changes;
- no GitHub push occurs.
