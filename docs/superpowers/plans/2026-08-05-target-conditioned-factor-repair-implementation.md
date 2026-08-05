# Target-Conditioned Factor Repair and Metadata Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement leakage-free metadata replay and target-conditioned factor repair for IFDR-YOLO, with deterministic recovery and hard evidence gates before any factor-guided claim or formal GPU run.

**Architecture:** Add an immutable KITTI object metadata index shared by Track M and Track F. Track M changes only the training sampler; Track F adds object-balanced natural ROI supervision, target/background specificity, an isolated semantic-calibration phase, and a mandatory post-adaptation audit. Existing P2, detection graph, DCLI inference behavior, KITTI validation annotations, and archived negative evidence remain unchanged.

**Tech Stack:** Python 3.12, PyTorch, Ultralytics YOLOv8, NumPy, PyYAML, `unittest`, existing IFDR run-store/provenance/audit utilities.

---

## Locked file map

New responsibilities are separated so the existing trainer and dataset do not become a second orchestration system.

- Create `ifdr_yolo/data/development_split.py`: deterministic leakage-free 90/10 fit/development split and digest.
- Create `ifdr_yolo/data/metadata_index.py`: immutable object identities, 0.99-IoU label binding, validity and provenance checks.
- Create `ifdr_yolo/data/replay_sampler.py`: M1/M2/M3 probabilities, registered eta schedule, deterministic draw journal and resume.
- Create `ifdr_yolo/losses/factor_alignment.py`: object-balanced ROI pooling, invalid-channel masking, and target/background specificity loss.
- Create `ifdr_yolo/experiments/factor_repair.py`: phase definitions, exact trainable parameter selection, equal-budget validation, and stage transition rules.
- Create `ifdr_yolo/eval/factor_repair_gate.py`: primary/diagnostic node gate and post-adaptation enforcement.
- Create `scripts/build_factor_metadata.py`: fail-closed metadata/split build CLI.
- Create `scripts/train_factor_repair.py`: development-only M/F runner using existing `RunStore` and IFDR trainer.
- Create `scripts/run_factor_repair_queue.py`: one-GPU resumable queue, with no automatic method advancement when a gate fails.
- Modify `ifdr_yolo/data/ifdr_dataset.py`: consume pre-matched object records and construct matched target/background intervention pairs.
- Modify `ifdr_yolo/experiments/ifdr_trainer.py`: accept a registered replay sampler and semantic-calibration phase without changing default behavior.
- Modify `ifdr_yolo/models/ifdr_model.py`: expose named factor-semantic parameters; do not change forward inference.
- Modify `ifdr_yolo/losses/ifdr_detection.py`: call the new alignment loss only when registered supervision is present.
- Modify `ifdr_yolo/experiments/config.py`: parse strict Track M/F configuration with unknown-field rejection.
- Add focused tests listed below; all existing 439 tests must stay green.

## Invariants before implementation

- Existing natural audit output at commit `10dc0374f0154068ebc9f49729eafea90abe83af` is immutable.
- The standard KITTI validation IDs must never enter recipe selection.
- F1/F2/F3 calibration must have fusion schedule `0.0` and DCLI schedule `0.0` for every batch.
- Track M must not load raw learned factors and must add no inference parameter.
- Track F cannot advance to factor-conditioned task adaptation until its development factor gate passes.
- A post-adaptation gate failure invalidates the factor-guided claim even if AP40 improves.
- Every output binds clean commit, split hash, metadata hash, checkpoint hash, resolved configuration and seed.

### Task 1: Deterministic leakage-free development split

**Files:**
- Create: `ifdr_yolo/data/development_split.py`
- Create: `scripts/build_factor_metadata.py`
- Test: `tests/test_development_split.py`

- [ ] **Step 1: Write the failing deterministic-split tests**

```python
import unittest

from ifdr_yolo.data.development_split import build_development_split


class DevelopmentSplitTest(unittest.TestCase):
    def test_split_is_stable_disjoint_and_stratified(self):
        rows = [
            {"image_id": f"{index:06d}", "cyclist": index % 3 == 0,
             "cyclist_joint": (index % 10) / 10.0}
            for index in range(120)
        ]
        first = build_development_split(rows, seed=20260805, fraction=0.10)
        second = build_development_split(list(reversed(rows)), seed=20260805, fraction=0.10)
        self.assertEqual(first, second)
        self.assertTrue(set(first.fit_ids).isdisjoint(first.development_ids))
        self.assertEqual(set(first.fit_ids) | set(first.development_ids),
                         {row["image_id"] for row in rows})
        self.assertEqual(len(first.development_ids), 12)
        self.assertEqual(len(first.sha256), 64)

    def test_split_rejects_duplicate_ids(self):
        rows = [{"image_id": "000001", "cyclist": False, "cyclist_joint": 0.0}] * 2
        with self.assertRaisesRegex(ValueError, "duplicate image_id"):
            build_development_split(rows, seed=20260805, fraction=0.10)
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_development_split -v`

Expected: import failure for `ifdr_yolo.data.development_split`.

- [ ] **Step 3: Implement the immutable split result and stable-hash selection**

```python
@dataclass(frozen=True)
class DevelopmentSplit:
    seed: int
    fit_ids: tuple[str, ...]
    development_ids: tuple[str, ...]
    strata: Mapping[str, tuple[str, ...]]
    sha256: str


def build_development_split(
    rows: Sequence[Mapping[str, object]], *, seed: int, fraction: float
) -> DevelopmentSplit:
    if seed != 20260805 or fraction != 0.10:
        raise ValueError("development split requires seed=20260805 and fraction=0.10")
    normalized = _validate_unique_rows(rows)
    strata = _cyclist_presence_and_tertile_strata(normalized)
    development = _stable_stratified_selection(strata, seed=seed, fraction=fraction)
    all_ids = tuple(sorted(row.image_id for row in normalized))
    fit = tuple(image_id for image_id in all_ids if image_id not in development)
    payload = {"seed": seed, "fit_ids": fit, "development_ids": tuple(sorted(development))}
    return DevelopmentSplit(seed, fit, tuple(sorted(development)), _freeze_strata(strata), _digest(payload))
```

Small strata with at least two images must contribute at least one fit and one development image. The builder CLI writes exact ID files and one atomic JSON manifest; it refuses to overwrite a non-identical manifest.

- [ ] **Step 4: Verify the split tests pass**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_development_split -v`

Expected: all tests pass with identical hashes after reversed input order.

- [ ] **Step 5: Commit**

```text
git add ifdr_yolo/data/development_split.py scripts/build_factor_metadata.py tests/test_development_split.py
git commit -m "feat: add leakage-free factor development split"
```

### Task 2: Immutable KITTI object metadata index

**Files:**
- Create: `ifdr_yolo/data/metadata_index.py`
- Modify: `scripts/build_factor_metadata.py`
- Test: `tests/test_factor_metadata_index.py`

- [ ] **Step 1: Write identity, score and rejection tests**

```python
import unittest

class FactorMetadataIndexTest(unittest.TestCase):
    def setUp(self):
        self.source = KittiMetadataObject(
            image_id="000001", object_index=2, class_id=2, class_name="Cyclist",
            bbox_xyxy=(10.0, 20.0, 30.0, 60.0), depth_m=40.0,
            occlusion=2, truncation=0.25,
        )

    def test_metadata_index_binds_exact_label_and_scores(self):
        index = build_metadata_index([self.source], labels={"000001": [self.source.as_label()]},
                                     split_sha256="a" * 64, source_sha256="b" * 64)
        record = index.by_image["000001"][0]
        self.assertEqual(record.object_id, "000001:000002")
        self.assertEqual(record.sampling, compute_sampling_score(40.0, 40.0))
        self.assertEqual(record.visibility, compute_visibility_score(2, 0.25))
        self.assertEqual(record.joint,
                         1.0 - (1.0 - record.sampling) * (1.0 - record.visibility))

    def test_metadata_index_rejects_ambiguous_label_match(self):
        same_label = self.source.as_label()
        with self.assertRaisesRegex(ValueError, "ambiguous metadata match"):
            build_metadata_index([self.source], labels={"000001": [same_label, same_label]},
                                 split_sha256="a" * 64, source_sha256="b" * 64)
```

Also test: IoU below `0.99`, class mismatch, duplicate object identity, non-finite boxes, negative depth handling, invalid occlusion/truncation, missing source/split hash, and serialization order.

- [ ] **Step 2: Run the tests and verify failure**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_metadata_index -v`

Expected: import failure for `ifdr_yolo.data.metadata_index`.

- [ ] **Step 3: Implement strict records and exact matching**

```python
@dataclass(frozen=True)
class FactorObjectRecord:
    image_id: str
    object_id: str
    class_id: int
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    height: float
    depth_m: float | None
    occlusion: int
    truncation: float
    sampling: float
    visibility: float
    joint: float
    sampling_valid: bool
    visibility_valid: bool


@dataclass(frozen=True)
class FactorMetadataIndex:
    by_image: Mapping[str, tuple[FactorObjectRecord, ...]]
    source_sha256: str
    split_sha256: str
    sha256: str


def match_metadata_object(record, candidates, *, minimum_iou: float = 0.99):
    matches = [candidate for candidate in candidates
               if candidate.class_id == record.class_id
               and box_iou(candidate.bbox_xyxy, record.bbox_xyxy) >= minimum_iou]
    if len(matches) != 1:
        reason = "missing" if not matches else "ambiguous"
        raise ValueError(f"{reason} metadata match for {record.object_id}")
    return matches[0]
```

Missing/invalid positive depth sets `depth_m=None` and `sampling_valid=True` because height remains valid; it is counted in provenance. Ambiguous identity never falls back to nearest IoU.

- [ ] **Step 4: Add deterministic CLI output and hash binding**

`scripts/build_factor_metadata.py` must atomically write `metadata_index.json`, `fit_ids.txt`, `development_ids.txt`, and `manifest.json`. Re-running with identical scientific identity is a no-op; any mismatch fails before writing.

- [ ] **Step 5: Run tests and commit**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_metadata_index tests.test_development_split -v`

```text
git add ifdr_yolo/data/metadata_index.py scripts/build_factor_metadata.py tests/test_factor_metadata_index.py
git commit -m "feat: build immutable KITTI factor metadata"
```

### Task 3: M1/M2/M3 conservative replay sampler

**Files:**
- Create: `ifdr_yolo/data/replay_sampler.py`
- Test: `tests/test_replay_sampler.py`

- [ ] **Step 1: Write schedule and probability tests**

```python
import unittest


class ReplaySamplerTest(unittest.TestCase):
    def test_registered_eta_schedule_boundaries(self):
        self.assertEqual(replay_eta(1), 0.0)
        self.assertEqual(replay_eta(5), 0.30)
        self.assertEqual(replay_eta(6), 0.30)
        self.assertEqual(replay_eta(40), 0.30)
        self.assertEqual(replay_eta(60), 0.0)

    def test_m3_priorities_are_clipped_and_only_cyclist_images_enter_focus(self):
        sampler = build_replay_distribution(
            image_ids=("a", "b", "c"),
            cyclist_joint={"a": 0.9, "b": 0.2},
            mode="M3", epoch=20,
        )
        self.assertEqual(sampler.focus_ids, ("a", "b"))
        self.assertNotIn("c", sampler.focus_probabilities)
        self.assertAlmostEqual(sum(sampler.probabilities.values()), 1.0)
        self.assertGreater(sampler.probabilities["c"], 0.0)
```

Also test M1 equals original distribution, M2 is uniform inside the Cyclist pool, M3 uses the fit-split 95th percentile and `0.05` focus-pool floor, and all epochs outside 1-60 fail.

- [ ] **Step 2: Run tests and verify failure**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_replay_sampler -v`

- [ ] **Step 3: Implement the registered schedule and distribution**

```python
def replay_eta(epoch: int) -> float:
    if not 1 <= epoch <= 60:
        raise ValueError("replay epoch must be in [1, 60]")
    if epoch <= 5:
        return 0.30 * (epoch - 1) / 4.0
    if epoch <= 40:
        return 0.30
    return 0.30 * (60 - epoch) / 20.0


def mixture_probability(original: float, focus: float, epoch: int) -> float:
    eta = replay_eta(epoch)
    return (1.0 - eta) * original + eta * focus
```

Use a frozen `ReplayDistribution` containing mode, epoch, eta, sorted IDs, original/focus/final probabilities, source hash, and distribution hash.

- [ ] **Step 4: Add exactly-once draw journaling tests and implementation**

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class ReplayDrawJournalTest(unittest.TestCase):
    def test_draw_journal_resume_is_exact(self):
        identity = {"seed": 17, "distribution_sha256": "a" * 64}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = ReplayDrawJournal.create(root, identity=identity)
            expected = [first.draw(epoch=3, draw_index=i) for i in range(10)]
            resumed = ReplayDrawJournal.open(root, identity=identity)
            self.assertEqual(
                [resumed.draw(epoch=3, draw_index=i) for i in range(10)], expected
            )
            with self.assertRaisesRegex(ValueError, "scientific identity mismatch"):
                ReplayDrawJournal.open(
                    root, identity={"seed": 29, "distribution_sha256": "a" * 64}
                )
```

The journal derives each draw from `(seed, epoch, draw_index, distribution_sha256)` so interrupted runs do not depend on mutable RNG state. Each committed draw records selected image ID and its probability; duplicate `(epoch, draw_index)` with different content fails closed.

- [ ] **Step 5: Run tests and commit**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_replay_sampler -v`

```text
git add ifdr_yolo/data/replay_sampler.py tests/test_replay_sampler.py
git commit -m "feat: add recoverable metadata replay sampler"
```

### Task 4: Object-balanced natural ROI alignment

**Files:**
- Create: `ifdr_yolo/losses/factor_alignment.py`
- Modify: `ifdr_yolo/losses/ifdr_detection.py`
- Test: `tests/test_factor_alignment_loss.py`

- [ ] **Step 1: Write area, class-frequency and channel-mask invariance tests**

```python
import unittest
import torch
from torch.nn import functional as F


class FactorAlignmentLossTest(unittest.TestCase):
    @staticmethod
    def factor_map(sampling, visibility, size=4):
        result = torch.empty(1, 2, size, size)
        result[:, 0].fill_(sampling)
        result[:, 1].fill_(visibility)
        return result

    def test_object_balanced_loss_is_invariant_to_roi_area(self):
        small = ObjectFactorTarget(0, 2, (0, 0, 1, 1), (0.8, 0.4), (True, True))
        large = ObjectFactorTarget(0, 2, (0, 0, 4, 4), (0.8, 0.4), (True, True))
        factor_map = self.factor_map(0.5, 0.5)
        self.assertTrue(torch.allclose(
            object_balanced_factor_loss([factor_map], [small]),
            object_balanced_factor_loss([factor_map], [large]),
        ))

    def test_invalid_sampling_keeps_visibility_channel(self):
        target = ObjectFactorTarget(0, 1, (0, 0, 2, 2), (0.0, 0.9), (False, True))
        loss = object_balanced_factor_loss([self.factor_map(0.7, 0.2)], [target])
        expected = F.smooth_l1_loss(torch.tensor(0.2), torch.tensor(0.9))
        self.assertTrue(torch.allclose(loss, expected))
```

Add a test proving 20 Cars plus one Cyclist equals the macro-average of the two present class losses rather than a 21-object mean.

- [ ] **Step 2: Run tests and verify failure**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_alignment_loss -v`

- [ ] **Step 3: Implement explicit object -> class -> node reduction**

```python
@dataclass(frozen=True)
class ObjectFactorTarget:
    batch_index: int
    class_id: int
    roi_xyxy: tuple[int, int, int, int]
    target: tuple[float, float]
    valid: tuple[bool, bool]


def object_balanced_factor_loss(
    node_maps: Sequence[torch.Tensor], targets: Sequence[ObjectFactorTarget]
) -> torch.Tensor:
    node_losses = []
    for factor_map in node_maps:
        class_losses: dict[int, list[torch.Tensor]] = {}
        for item in targets:
            pooled = pool_object_roi(factor_map[item.batch_index], item.roi_xyxy)
            mask = torch.as_tensor(item.valid, device=pooled.device, dtype=torch.bool)
            if mask.any():
                truth = pooled.new_tensor(item.target)
                class_losses.setdefault(item.class_id, []).append(
                    F.smooth_l1_loss(pooled[mask], truth[mask], reduction="mean")
                )
        if class_losses:
            node_losses.append(torch.stack([
                torch.stack(values).mean() for values in class_losses.values()
            ]).mean())
    if not node_losses:
        return node_maps[0].sum() * 0.0
    return torch.stack(node_losses).mean()
```

`pool_object_roi` clamps to the map, rejects empty/non-finite ROIs, and averages spatial values before any object/class reduction.

- [ ] **Step 4: Wire the loss without changing legacy batches**

In `IFDRDetectionLoss`, call the new function only when `batch["factor_object_targets"]` exists. Keep the current dense synthetic loss unchanged and expose separate scalar components: `synthetic_factor_loss`, `natural_factor_loss`, `specificity_loss`.

- [ ] **Step 5: Run focused and regression tests, then commit**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_alignment_loss tests.test_ifdr_detection_loss tests.test_ifdr_data -v`

```text
git add ifdr_yolo/losses/factor_alignment.py ifdr_yolo/losses/ifdr_detection.py tests/test_factor_alignment_loss.py
git commit -m "feat: add object-balanced natural factor alignment"
```

### Task 5: Matched target/background specificity supervision

**Files:**
- Modify: `ifdr_yolo/data/ifdr_dataset.py`
- Modify: `ifdr_yolo/losses/factor_alignment.py`
- Test: `tests/test_factor_specificity.py`

- [ ] **Step 1: Write common-randomness and margin-boundary tests**

```python
import unittest

import torch


class FactorSpecificityTest(unittest.TestCase):
    def test_specificity_margin_uses_delta_from_clean(self):
        loss = factor_specificity_loss(
            clean=torch.tensor([0.20]), target=torch.tensor([0.50]),
            background=torch.tensor([0.26]), margin=0.05,
        )
        self.assertTrue(torch.allclose(loss, torch.tensor(0.01)))

    def test_specificity_rejects_overlapping_background(self):
        labels = {"bboxes": torch.tensor([[0.1, 0.1, 0.3, 0.3],
                                           [0.6, 0.6, 0.8, 0.8]])}
        with self.assertRaisesRegex(ValueError, "background overlaps annotated object"):
            build_specificity_pair(
                labels, target_index=0, background_box=(0.6, 0.6, 0.8, 0.8),
                severity=0.5, transform_seed=7,
            )
```

Also test severity below `0.25` receives zero specificity weight, target and background carry identical severity/transform seed, malformed/missing pairs are counted, and empty background has zero IoU with every annotation.

- [ ] **Step 2: Run tests and verify failure**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_specificity -v`

- [ ] **Step 3: Implement delta ranking exactly**

```python
def factor_specificity_loss(clean, target, background, *, margin: float = 0.05):
    if margin != 0.05:
        raise ValueError("registered specificity margin is 0.05")
    target_delta = target - clean
    background_delta = background - clean
    return torch.relu(background_delta + margin - target_delta).mean()
```

Extend `IFDRInterventionTransform` to emit one immutable pair record with `pair_id`, target/background boxes, factor channel, severity, transform seed, validity, and rejection reason. Do not assign a positive degradation target to empty background.

- [ ] **Step 4: Verify legacy behavior and pair accounting**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_specificity tests.test_intervention_sampler tests.test_intervention_targets tests.test_ifdr_data -v`

- [ ] **Step 5: Commit**

```text
git add ifdr_yolo/data/ifdr_dataset.py ifdr_yolo/losses/factor_alignment.py tests/test_factor_specificity.py
git commit -m "feat: enforce target-specific factor response"
```

### Task 6: Exact semantic-calibration freeze contract

**Files:**
- Modify: `ifdr_yolo/models/ifdr_model.py`
- Create: `ifdr_yolo/experiments/factor_repair.py`
- Modify: `ifdr_yolo/experiments/ifdr_trainer.py`
- Test: `tests/test_factor_repair_phase.py`

- [ ] **Step 1: Write trainable-set and zero-schedule tests**

```python
from pathlib import Path
import unittest
import torch

from ifdr_yolo.experiments.ultralytics_runtime import bootstrap_ultralytics_config


ROOT = Path(__file__).resolve().parents[1]


class FactorRepairPhaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bootstrap_ultralytics_config(ROOT)
        from ifdr_yolo.models.ifdr_model import IFDRDetectionModel
        cls.model = IFDRDetectionModel(str(ROOT / "models/kitti-p2-m.yaml"), verbose=False)

    def test_semantic_calibration_trainable_names_are_exact(self):
        phase = semantic_calibration_phase(self.model, variant="F3", epochs=30)
        actual = {name for name, parameter in self.model.named_parameters()
                  if parameter.requires_grad}
        self.assertEqual(actual, set(phase.trainable_parameter_names))
        self.assertTrue(actual)
        self.assertTrue(all(
            "input_projection" in name or "shared_core" in name or "factor_head" in name
            for name in actual
        ))
        self.assertEqual(phase.fusion_schedule, 0.0)
        self.assertEqual(phase.dcli_schedule, 0.0)

    def test_calibration_rejects_detection_or_adapter_parameter(self):
        phase = semantic_calibration_phase(self.model, variant="F3", epochs=30)
        forbidden = ("detect", "router", "semantic_adapter", "factor_adapter", "localization")
        self.assertFalse(any(
            token in name for name in phase.trainable_parameter_names for token in forbidden
        ))
```

- [ ] **Step 2: Run tests and verify failure**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_repair_phase -v`

- [ ] **Step 3: Expose semantic parameter names and freeze by identity**

```python
def factor_semantic_named_parameters(self):
    allowed_modules = []
    seen = set()
    for module in self.model:
        if isinstance(module, ReliabilityGatedConcat):
            allowed_modules.extend((module.input_projection, module.shared_core, module.factor_head))
    for name, parameter in self.named_parameters():
        if any(parameter is owned for module in allowed_modules for owned in module.parameters()):
            if id(parameter) not in seen:
                seen.add(id(parameter))
                yield name, parameter
```

The implementation must account for shared reliability modules exactly once. `semantic_calibration_phase` first freezes all parameters, then enables only the yielded identities, resets optimizer state, sets all component schedules explicitly, and records sorted trainable names in provenance.

- [ ] **Step 4: Add validation no-step and optimizer-reset tests**

Snapshot parameter tensors before validation, run one validation batch, and assert byte-identical parameters and unchanged optimizer step. Assert a newly created calibration optimizer has empty state.

- [ ] **Step 5: Run tests and commit**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_repair_phase tests.test_ifdr_model tests.test_ifdr_trainer -v`

```text
git add ifdr_yolo/models/ifdr_model.py ifdr_yolo/experiments/factor_repair.py ifdr_yolo/experiments/ifdr_trainer.py tests/test_factor_repair_phase.py
git commit -m "feat: isolate semantic factor calibration"
```

### Task 7: Strict M/F configuration and equal-budget controls

**Files:**
- Modify: `ifdr_yolo/experiments/config.py`
- Create: `configs/experiments/kitti_ifdr_factor_repair_dev_s17.yaml`
- Create: `tests/test_factor_repair_config.py`

- [ ] **Step 1: Write strict parsing and budget tests**

```python
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory


class FactorRepairConfigTest(unittest.TestCase):
    def test_registered_conditions_have_equal_budgets(self):
        config = load_factor_repair_config(write_valid_config())
        self.assertEqual({config.conditions[name].epochs for name in ("M1", "M2", "M3")}, {60})
        self.assertEqual({config.conditions[name].epochs for name in ("F1", "F2", "F3")}, {30})
        self.assertEqual(config.development.seed, 20260805)
        self.assertEqual(config.development.fraction, 0.10)
        self.assertEqual(config.factor_loss.natural_gain, 1.0)
        self.assertEqual(config.factor_loss.specificity_gain, 0.5)
        self.assertEqual(config.factor_loss.specificity_margin, 0.05)

    def test_unknown_or_unregistered_threshold_fails(self):
        with TemporaryDirectory() as directory:
            path = write_valid_config(Path(directory),
                                      extra={"factor_gate": {"rho_threshold": 0.01}})
            with self.assertRaisesRegex(ValueError, "unknown factor_gate fields"):
                load_factor_repair_config(path)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_repair_config -v`

- [ ] **Step 3: Implement frozen configuration dataclasses**

Define `DevelopmentProtocolConfig`, `MetadataReplayConfig`, `FactorAlignmentConfig`, `FactorGateConfig`, `RepairConditionConfig`, and `FactorRepairConfig`. All fields are required; unknown fields fail. Hard-code validation of registered values from the approved spec rather than accepting alternative thresholds from the command line.

- [ ] **Step 4: Write the canonical seed-17 development YAML**

The YAML binds fit/development/metadata hashes, accepted initialization checkpoint hash, M/F budgets, image size, schedule, factor weights, nodes `(11, 14, 17, 20, 23, 26)`, primary nodes `(17, 20, 23, 26)`, and output root. `scripts/build_factor_metadata.py` must generate the file from completed manifests so every hash is real at first write; the command refuses to create the YAML until all required 64-hex hashes validate.

- [ ] **Step 5: Run tests and commit**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_repair_config tests.test_experiment_config -v`

```text
git add ifdr_yolo/experiments/config.py configs/experiments/kitti_ifdr_factor_repair_dev_s17.yaml tests/test_factor_repair_config.py
git commit -m "feat: register factor repair experiment controls"
```

### Task 8: Primary/diagnostic factor gate and post-adaptation enforcement

**Files:**
- Create: `ifdr_yolo/eval/factor_repair_gate.py`
- Test: `tests/test_factor_repair_gate.py`

- [ ] **Step 1: Write complete gate-boundary tests**

```python
import unittest

class FactorRepairGateTest(unittest.TestCase):
    def test_seed17_gate_requires_three_of_four_positive_primary_nodes(self):
        rows = candidate_rows(primary_positive=(17, 20, 23), diagnostic_reverse=())
        decision = evaluate_factor_repair_gate(rows, stage="development", expected_seeds=(17,))
        self.assertTrue(decision.passed)

    def test_significant_reverse_diagnostic_blocks_gate(self):
        rows = candidate_rows(primary_positive=(17, 20, 23, 26), diagnostic_reverse=(11,))
        decision = evaluate_factor_repair_gate(rows, stage="development", expected_seeds=(17,))
        self.assertFalse(decision.passed)
        self.assertIn("diagnostic_reverse_association", decision.failures)

    def test_post_adaptation_failure_blocks_factor_guided_claim(self):
        passing = gate_decision(passed=True)
        failing = gate_decision(passed=False)
        with self.assertRaisesRegex(ValueError, "post-adaptation factor gate failed"):
            require_factor_guided_advancement(pre=passing, post=failing)
```

Test the exact 80% severity ordering boundary, positive paired target response, target greater than background, 10/12 three-seed directions, CI lower bound above zero, malformed count zero, missing node/seed rejection, and no threshold mutation.

- [ ] **Step 2: Run tests and verify failure**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_repair_gate -v`

- [ ] **Step 3: Implement an immutable, fully enumerated decision**

```python
@dataclass(frozen=True)
class FactorRepairGateDecision:
    passed: bool
    stage: str
    primary_nodes: tuple[int, ...]
    diagnostic_nodes: tuple[int, ...]
    checks: Mapping[str, bool]
    failures: tuple[str, ...]
    evidence_sha256: str


def require_factor_guided_advancement(*, pre, post):
    if not pre.passed:
        raise ValueError("pre-adaptation factor gate failed")
    if not post.passed:
        raise ValueError("post-adaptation factor gate failed")
```

Reuse existing `natural_factor_audit` statistics; do not duplicate Spearman/bootstrap mathematics. The gate layer validates completeness and translates registered evidence into an explicit decision.

- [ ] **Step 4: Run tests and commit**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_repair_gate tests.test_natural_factor_audit -v`

```text
git add ifdr_yolo/eval/factor_repair_gate.py tests/test_factor_repair_gate.py
git commit -m "feat: gate factor repair evidence"
```

### Task 9: Development runner and one-GPU recoverable queue

**Files:**
- Create: `scripts/train_factor_repair.py`
- Create: `scripts/run_factor_repair_queue.py`
- Test: `tests/test_factor_repair_runner.py`
- Test: `tests/test_factor_repair_queue.py`

- [ ] **Step 1: Write fail-closed runner tests**

```python
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory


class FactorRepairRunnerTest(unittest.TestCase):
    def test_runner_rejects_development_id_in_fit_loader(self):
        config = registered_test_config(fit_ids=("fit-a",), development_ids=("dev-b",))
        with self.assertRaisesRegex(ValueError, "development leakage"):
            build_factor_repair_run(config, loader_ids=("fit-a", "dev-b"))

    def test_queue_does_not_advance_failed_f3_gate(self):
        with TemporaryDirectory() as directory:
            queue = FactorRepairQueue.create(
                Path(directory), jobs=("F3-calibration", "F3-adaptation")
            )
            queue.complete("F3-calibration", artifacts=failed_gate_artifacts())
            self.assertEqual(queue.status("F3-adaptation"), "blocked")
            self.assertFalse(queue.launchable("F3-adaptation"))
```

Also test dirty checkout, missing hash, mismatched initialization checkpoint, duplicate process, non-finite loss, empty `best.pt`/`last.pt`, and recovery from an interrupted epoch/draw journal.

- [ ] **Step 2: Run tests and verify failure**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_repair_runner tests.test_factor_repair_queue -v`

- [ ] **Step 3: Implement the runner around existing trusted primitives**

`train_factor_repair.py` must use existing `RunStore`, provenance hash validation, `IFDRDetectionTrainer`, and atomic status transitions. It accepts only a config path and registered condition name; it does not accept ad-hoc epochs, thresholds, seeds, or loss weights.

```python
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_factor_repair_config(args.config)
    condition = config.require_condition(args.condition)
    require_clean_commit(config.implementation_commit)
    verify_scientific_identity(config, condition)
    return run_registered_condition(config, condition)
```

- [ ] **Step 4: Implement the queue state machine**

Allowed transitions are `pending -> running -> complete`, `pending -> blocked`, and `running -> failed`. Resume changes `failed -> running` only when identity and existing artifacts validate. The queue executes one GPU process at a time and never auto-promotes a method after a failed factor or detection gate.

- [ ] **Step 5: Run tests and commit**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_repair_runner tests.test_factor_repair_queue tests.test_ifdr_recovery tests.test_repository_safety -v`

```text
git add scripts/train_factor_repair.py scripts/run_factor_repair_queue.py tests/test_factor_repair_runner.py tests/test_factor_repair_queue.py
git commit -m "feat: orchestrate recoverable factor repair screens"
```

### Task 10: Detection screen, paired uncertainty and claim labels

**Files:**
- Create: `ifdr_yolo/eval/factor_repair_report.py`
- Test: `tests/test_factor_repair_report.py`

- [ ] **Step 1: Write advancement and claim-boundary tests**

```python
import unittest


class FactorRepairReportTest(unittest.TestCase):
    def test_development_advancement_boundaries(self):
        result = evaluate_development_advancement(
            reference={"Car": 90.0, "Pedestrian": 65.0, "Cyclist": 42.0},
            candidate={"Car": 88.5, "Pedestrian": 63.5, "Cyclist": 47.0},
            slices={"small": 1.0, "far": 0.0, "occluded": 0.0},
            factor_pre=None, factor_post=None, method="M3",
        )
        self.assertTrue(result.passed)

    def test_factor_gain_with_failed_post_gate_is_not_factor_guided(self):
        result = evaluate_development_advancement(
            reference={"Car": 90.0, "Pedestrian": 65.0, "Cyclist": 42.0},
            candidate={"Car": 90.1, "Pedestrian": 65.1, "Cyclist": 50.0},
            slices={"small": 2.0, "far": 3.0, "occluded": 1.0},
            factor_pre=gate_decision(passed=True),
            factor_post=gate_decision(passed=False),
            method="F3",
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.allowed_claim,
                         "detection_gain_without_valid_factor_guidance")
```

Add formal three-seed tests for Cyclist paired-bootstrap CI, Car/Pedestrian no-harm `1.0` AP bound, mean/std, target slices, calibration, efficiency, and negative-result labels.

- [ ] **Step 2: Run tests and verify failure**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_repair_report -v`

- [ ] **Step 3: Implement report objects and reuse evaluators**

Reuse `paired_bootstrap_ap40`, `evaluate_target_slices`, detection reliability, and existing AP40 parsing. The report stores every failed check and never drops a class, seed, node or condition.

- [ ] **Step 4: Run tests and commit**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_repair_report tests.test_paired_bootstrap tests.test_stratified_ap40 tests.test_detection_reliability -v`

```text
git add ifdr_yolo/eval/factor_repair_report.py tests/test_factor_repair_report.py
git commit -m "feat: report factor repair advancement evidence"
```

### Task 11: CPU dry run, CUDA smoke, recovery and full regression gate

**Files:**
- Create: `tests/test_factor_repair_smoke.py`
- Modify only if a smoke exposes a defect in files owned by Tasks 1-10.

- [ ] **Step 1: Add one-batch CPU dry-run test**

The test builds two synthetic KITTI images with Car, Pedestrian and Cyclist objects, runs F3 calibration forward/backward, and asserts finite synthetic/natural/specificity losses, non-zero gradient only on semantic parameters, and byte-identical frozen parameters.

- [ ] **Step 2: Run the CPU dry run**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_repair_smoke.FactorRepairSmokeTests.test_cpu_batch -v`

Expected: one passing test, finite losses, no downloaded data.

- [ ] **Step 3: Run focused suites and full regression locally**

Run:

```text
D:\ana\envs\yolo\python.exe -m unittest tests.test_development_split tests.test_factor_metadata_index tests.test_replay_sampler tests.test_factor_alignment_loss tests.test_factor_specificity tests.test_factor_repair_phase tests.test_factor_repair_config tests.test_factor_repair_gate tests.test_factor_repair_runner tests.test_factor_repair_queue tests.test_factor_repair_report tests.test_factor_repair_smoke -v
D:\ana\envs\yolo\python.exe -m unittest discover -s tests -v
```

Expected: all new focused tests pass and all existing 439 tests plus new tests pass.

- [ ] **Step 4: Perform one-epoch CUDA smoke on the server**

Only after clean-commit deployment and server identity verification, run one registered seed-17 F3 calibration epoch on smoke data. Validate non-empty `status.json`, resolved config, provenance, draw journal, loss components, `best.pt`, and `last.pt`. Interrupt after a committed batch, resume, and verify exactly-once journal entries and identical scientific identity.

- [ ] **Step 5: Archive smoke evidence and commit test**

```text
git add tests/test_factor_repair_smoke.py
git commit -m "test: validate factor repair smoke and recovery"
```

### Task 12: Formal launch preflight (no training in this task)

**Files:**
- Create: `docs/reports/factor-repair-preflight.md`
- No model or data code changes.

- [ ] **Step 1: Record exact artifacts**

The preflight report records: clean commit, all source hashes, 90/10 IDs and hash, metadata index hash and counts, seed-17 initialization hash, trainable names, condition budgets, primary/diagnostic nodes, full-test result, CUDA smoke run path and archive SHA256.

- [ ] **Step 2: Enforce launch decision**

The report may say `READY` only if metadata preflight, leakage test, focused tests, full regression, CUDA smoke, recovery smoke, disk check and duplicate-process check all pass. Otherwise it says `BLOCKED` with the exact failed evidence; no formal GPU job is started.

- [ ] **Step 3: Commit**

```text
git add docs/reports/factor-repair-preflight.md
git commit -m "docs: record factor repair launch preflight"
```

## Formal experiment sequence after implementation acceptance

1. Generate and commit the immutable metadata index and 90/10 development split.
2. Retrain the matched seed-17 protected development reference on fit IDs only.
3. Run M1, M2 and M3 for 60 matched epochs.
4. Run F1, F2 and F3 calibration for 30 matched epochs.
5. Audit F0-F3; select at most one repair without changing thresholds.
6. If and only if the repair passes, run 60-epoch task adaptation and repeat the complete audit.
7. Freeze one valid recipe, then run formal seeds 17, 29 and 41 on all training IDs.
8. Produce paired bootstrap, stratified AP40, calibration, efficiency and failure-case evidence.
9. Archive all artifacts before any BDD100K transfer experiment.

## Self-review

- Spec coverage: target-conditioned definitions, object-balanced ROI loss, background specificity, leakage-free split, M1/M2/M3, F1/F2/F3, freeze rules, node gates, advancement, recovery and failure semantics all map to Tasks 1-12.
- Completeness scan: every code-changing step names concrete interfaces, commands and expected behavior. Unknown content hashes are generated from completed manifests before the canonical YAML can be written.
- Type consistency: `DevelopmentSplit`, `FactorMetadataIndex`, `ReplayDistribution`, `ObjectFactorTarget`, `FactorRepairGateDecision`, and phase/report interfaces keep the same names across producing and consuming tasks.
- Scope: no attention module, new backbone, new IoU variant, inference-graph change, validation-label modification, or BDD100K launch is included.

Plan completion condition: this document is approved, then Tasks 1-12 are executed with review after each task. No new GPU experiment begins before Task 12 reports `READY`.
