# Target-Conditioned Factor Repair and Metadata Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement leakage-free metadata replay and target-conditioned factor repair for IFDR-YOLO, with deterministic recovery and hard evidence gates before any factor-guided claim or formal GPU run.

**Architecture:** Add an immutable KITTI object metadata index shared by Track M and Track F. Track M changes only the training sampler; Track F adds object-balanced natural ROI supervision, target/background specificity, an isolated semantic-calibration phase, and a mandatory post-adaptation audit. Existing P2, detection graph, DCLI inference behavior, KITTI validation annotations, and archived negative evidence remain unchanged.

**Tech Stack:** Python 3.12, PyTorch, Ultralytics YOLOv8, NumPy, PyYAML, `unittest`, existing IFDR run-store/provenance/audit utilities.

The Chen KITTI train/validation split is a public development benchmark. Recipe
selection uses only the locked internal development split; external evidence is
the KITTI test server or a frozen BDD100K transfer result. Every registered
condition binds identical initialization checkpoint bytes and SHA256, resets
optimizer state, disables early stopping, and uses a fixed budget. `last.pt` is
the primary fixed-budget result and `best.pt` is an engineering diagnostic.

---

## Locked file map

New responsibilities are separated so the existing trainer and dataset do not become a second orchestration system.

- Create `ifdr_yolo/data/development_split.py`: deterministic leakage-free 90/10 fit/development split and digest.
- Create `ifdr_yolo/data/metadata_index.py`: immutable object identities, 0.99-IoU label binding, validity and provenance checks.
- Create `ifdr_yolo/data/replay_sampler.py`: M1/M2/M3 probabilities, registered eta schedule, deterministic draw journal and resume.
- Create `ifdr_yolo/data/learned_factor_manifest.py`: no-grad fit-image factor manifests, primary-node object aggregation, average-tie percentile ranking, and metadata/learned focus distribution.
- Create `ifdr_yolo/losses/factor_alignment.py`: object-balanced ROI pooling, invalid-channel masking, and target/background specificity loss.
- Create `ifdr_yolo/experiments/factor_repair.py`: semantic-calibration and task-adaptation phase definitions, exact trainable parameter selection, equal-budget validation, and stage transition rules.
- Create `ifdr_yolo/eval/factor_repair_gate.py`: primary/diagnostic node gate and post-adaptation enforcement.
- Create `scripts/build_factor_metadata.py`: fail-closed metadata/split build CLI.
- Create `scripts/train_factor_repair.py`: development-only M/F runner using existing `RunStore` and IFDR trainer.
- Create `scripts/run_factor_repair_queue.py`: one-GPU resumable queue, with no automatic method advancement when a gate fails.
- Modify `ifdr_yolo/data/ifdr_dataset.py`: bind raw labels before geometry, emit clean/target/background views, carry immutable ROI identities, and update `collate_ifdr_batch`.
- Modify `ifdr_yolo/experiments/ifdr_trainer.py`: accept a registered replay sampler and semantic-calibration phase without changing default behavior.
- Modify `ifdr_yolo/models/ifdr_model.py`: split one `3B` forward into clean/target/background contexts and expose named factor-semantic parameters; do not change inference behavior at zero schedule.
- Modify `ifdr_yolo/losses/ifdr_detection.py`: map normalized ROIs to P2-P5, use clean-only natural targets, target-view synthetic targets, target/background deltas relative to clean, and exclude detection loss during calibration.
- Modify `ifdr_yolo/experiments/config.py`: parse strict Track M/F configuration with unknown-field rejection.
- Add focused tests listed below; all existing 439 tests must stay green.

## Invariants before implementation

- Existing natural audit output at commit `10dc0374f0154068ebc9f49729eafea90abe83af` is immutable.
- The Chen public development benchmark IDs must never enter recipe selection;
  only internal development IDs may select a recipe.
- F0/F1/F2/F3 calibration must have fusion schedule `0.0` and DCLI schedule `0.0` for every batch.
- F0 is a compute-matched 30-epoch synthetic-only calibration control; F0-F3
  share the same three-view batches, optimizer/update schedule, batch/update
  count, freeze set, and calibration budget.
- F0 must have complete finite mechanism evidence on the same image-ID hash as
  every candidate. Selection uses four pooled primary-node endpoints, an
  equal-weight composite `S`, paired image-bootstrap `DeltaS`, and the fixed
  lower-bound/point/name tie-break; no manual condition string or default F3.
- Only F0 and at most one F1-F3 candidate that passes its development factor
  gate receive the matched 60-epoch task adaptation; both start only after that
  candidate is selected, and if none passes, no Track F adaptation starts.
- Track M must not load raw learned factors and must add no inference parameter.
- Track F cannot advance to factor-conditioned task adaptation until its development factor gate passes.
- A post-adaptation gate failure invalidates the factor-guided claim even if AP40 improves.
- Learned-factor replay is generated only from each frozen calibration
  checkpoint on fit clean images, uses the metadata/learned 0.5/0.5 safeguard,
  and cannot run for a candidate that fails its gate.
- Every output binds clean commit, split hash, metadata hash, checkpoint hash, resolved configuration and seed.
- Every fixed-budget report uses `last.pt` as the primary checkpoint and keeps
  `best.pt` only as an engineering diagnostic.
- Runner and evaluator write/consume `last.pt` primary metrics explicitly;
  best-checkpoint metrics remain diagnostic and cannot advance a method.

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
             "cyclist_joint": (index % 10) / 10.0 if index % 3 == 0 else 0.0}
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

Add these named tests with explicit outcomes: `test_joint_score_is_finite_and_bounded`
passes for every score in `[0, 1]`; `test_no_cyclist_joint_score_is_zero`
returns `0.0`; `test_rejects_unregistered_seed_and_fraction` raises
`ValueError`; `test_reversed_input_order_keeps_ids_and_sha256` returns identical
split objects; `test_round_half_up_count` uses `N=15` and expects two development
IDs; `test_fit_development_are_disjoint_and_complete` checks set equality;
`test_strata_are_immutable_after_build` rejects mutation; and
`test_hash_is_stable_for_same_rows` compares two SHA256 values. Use
`test_feasible_small_strata_have_both_seats` with `N=40`, four non-trivial
strata, and `dev_count=4`; use
`test_infeasible_small_strata_raise_quota_constraints` with `N=20`, the same
four strata, and `dev_count=2`, expecting `ValueError("quota constraints")`
before any manifest is written.

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
    development = _stable_stratified_selection(
        strata, seed=seed, fraction=fraction,
        total_count=round_half_up_tenth(len(normalized)),
    )
    all_ids = tuple(sorted(row.image_id for row in normalized))
    fit = tuple(image_id for image_id in all_ids if image_id not in development)
    payload = {"seed": seed, "fit_ids": fit, "development_ids": tuple(sorted(development))}
    return DevelopmentSplit(seed, fit, tuple(sorted(development)), _freeze_strata(strata), _digest(payload))
```

`round_half_up(0.10 * N)` is the strict development count. Tertiles use a
stable `(cyclist_joint, image_id)` ordering and lower/middle/upper groups whose
sizes differ by at most one; `no_cyclist` is separate. Largest-remainder
allocation honors a minimum development and fit seat for every stratum with at
least two images. If the minimum exceeds the development count or the fit
capacity, fail closed with `quota constraints`. The builder CLI writes exact ID
files and one atomic JSON manifest; it refuses to overwrite a non-identical
manifest.

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

Add named rejection tests with these exact fixtures: `test_iou_below_099_fails`
uses IoU `0.989`; `test_class_mismatch_fails` changes only class ID;
`test_duplicate_object_identity_fails` repeats one `(image_id, object_id)`;
`test_nonfinite_box_fails` uses `NaN`; `test_invalid_positive_depth_masks_only_depth`
asserts `depth_m=None`, `sampling_valid=True`, and a height-only score;
`test_invalid_occlusion_or_truncation_fails` supplies `occlusion=4` and
`truncation=1.1`; `test_missing_source_or_split_hash_fails` omits each hash in
turn; and `test_serialization_order_is_stable` compares byte-identical JSON.
`test_zero_multiple_and_ambiguous_matches_fail_closed` asserts that each
matching cardinality other than one raises before an index record is emitted.

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

Missing/invalid optional depth sets `depth_m=None`, masks only the depth
contribution while retaining the valid height score, and increments the
provenance counter. Zero, multiple, or ambiguous identity matches and duplicate
identities fail closed before serialization; they never continue with zero
alignment weight or nearest-IoU fallback.

- [ ] **Step 4: Add deterministic CLI output and hash binding**

Task 2 owns `scripts/build_factor_metadata.py`; it atomically writes
`metadata_index.json`, `fit_ids.txt`, `development_ids.txt`, `manifest.json`,
and the canonical seed-17 YAML only after all required hashes validate.
Re-running with identical scientific identity is a no-op; any mismatch fails
before writing.

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

Add named tests with fixed expected distributions:
`test_m1_is_uniform_over_fit_ids` checks every fit ID has probability
`1/len(fit_ids)`; `test_m2_is_uniform_over_cyclist_pool` checks equal focus
probabilities and zero non-Cyclist focus probability;
`test_m3_clips_at_fit_p95_and_adds_floor` checks the fit-only percentile and
`0.05` floor; `test_replay_eta_rejects_epoch_zero_and_61` expects `ValueError`;
and `test_sampler_draws_with_replacement_for_fit_count` consumes one epoch,
asserts exactly `len(fit_ids)` draws, and permits repeated IDs.
`test_factor_guided_distribution_has_full_provenance` checks mode,
manifest/checkpoint/metadata hashes, sorted IDs, all probability maps, and a
64-hex distribution digest.

- [ ] **Step 2: Run tests and verify failure**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_replay_sampler -v`

- [ ] **Step 3: Implement the registered schedule and distribution**

```python
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ReplayDistribution:
    mode: str
    epoch: int
    eta: float
    image_ids: tuple[str, ...]
    original_probabilities: Mapping[str, float]
    focus_probabilities: Mapping[str, float]
    probabilities: Mapping[str, float]
    source_sha256: str
    manifest_sha256: str | None
    calibration_checkpoint_sha256: str | None
    metadata_index_sha256: str | None
    distribution_sha256: str
    focus_scores: Mapping[str, float] = field(default_factory=dict)

    @property
    def focus_ids(self):
        return tuple(sorted(self.focus_probabilities))


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

Use this frozen `ReplayDistribution` for M1/M2/M3 and factor-guided replay.
For factor-guided mode, all four provenance hashes are required; for legacy
metadata-only modes, manifest/checkpoint hashes are `None` and the source hash
still binds the metadata/split identity. The canonical serializer covers every
field, including sorted IDs and all probability maps, when computing
`distribution_sha256`.

- [ ] **Step 4: Add exactly-once draw journaling tests and implementation**

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class ReplayDrawJournalTest(unittest.TestCase):
    def test_draw_journal_resume_is_exact(self):
        identity = {
            "seed": 17, "distribution_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "calibration_checkpoint_sha256": "c" * 64,
            "metadata_index_sha256": "d" * 64,
        }
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
                    root, identity={
                        "seed": 29, "distribution_sha256": "a" * 64,
                        "manifest_sha256": "b" * 64,
                        "calibration_checkpoint_sha256": "c" * 64,
                        "metadata_index_sha256": "d" * 64,
                    }
                )
```

Add `test_draw_key_changes_sequence` with one fixed distribution and change
each of `seed`, `epoch`, `draw_index`, and `distribution_sha256` separately;
each changed key must produce a different deterministic draw. Add
`test_draw_journal_records_realized_counts` and assert image/class count fields
for all `len(fit_ids)` records, plus
`test_duplicate_draw_content_fails_closed` for a conflicting duplicate key.
Add `test_distribution_hash_changes_with_epoch_and_manifest_identity`, which
expects different hashes for different eta epochs or any changed manifest,
checkpoint, or metadata hash, and
`test_draw_journal_rejects_missing_provenance_hashes`, which fails closed for a
factor-guided identity with any omitted hash.

The journal derives each draw from `(seed, epoch, draw_index,
distribution_sha256, manifest_sha256, calibration_checkpoint_sha256,
metadata_index_sha256)` so interrupted runs do not depend on mutable RNG state.
Each committed draw records selected image ID, probability, and realized image
and class counts; duplicate `(epoch, draw_index)` with different content fails
closed. Because eta changes by epoch, the queue persists and verifies a distinct
distribution hash for each epoch. The formal full-training sampler therefore
records exactly 3,712 draws per epoch on one GPU.

- [ ] **Step 5: Run tests and commit**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_replay_sampler -v`

```text
git add ifdr_yolo/data/replay_sampler.py tests/test_replay_sampler.py
git commit -m "feat: add recoverable metadata replay sampler"
```

### Task 3A: Immutable learned-factor manifest and safeguarded replay

**Files:**
- Create: `ifdr_yolo/data/learned_factor_manifest.py`
- Modify: `ifdr_yolo/data/replay_sampler.py`
- Test: `tests/test_learned_factor_manifest.py`

- [ ] **Step 1: Write manifest aggregation and safeguard tests**

```python
import unittest


class LearnedFactorManifestTest(unittest.TestCase):
    def test_primary_node_macro_average_and_joint(self):
        record = aggregate_primary_node_factors({
            17: (0.20, 0.40), 20: (0.40, 0.20),
            23: (0.60, 0.40), 26: (0.80, 0.60),
        })
        self.assertEqual(record.sampling, 0.50)
        self.assertEqual(record.visibility, 0.40)
        self.assertAlmostEqual(record.learned_joint, 0.70)

    def test_average_tie_percentile_keeps_equal_priorities_together(self):
        ranks = average_tie_percentile_rank({"a": 0.2, "b": 0.2, "c": 0.8})
        self.assertEqual(ranks["a"], ranks["b"])
        self.assertLess(ranks["a"], ranks["c"])

    def test_focus_score_uses_metadata_and_learned_half_weights(self):
        manifest = manifest_fixture(
            condition="F0", fit_ids=("a", "b"),
            object_records=(
                learned_object("a", "a:000001", 0.0, 0.0, 0.0),
                learned_object("b", "b:000001", 1.0, 1.0, 1.0),
            ),
        )
        distribution = build_learned_focus_distribution(
            manifest=manifest,
            metadata_index=metadata_index_fixture(fit_ids=("a", "b")),
            metadata_priorities=validated_metadata_priorities_fixture(
                values={"a": 1.0, "b": 0.0},
                metadata_index_sha256=manifest.metadata_index_sha256,
            ),
            epoch=20,
        )
        self.assertAlmostEqual(distribution.focus_scores["a"], 0.5)
        self.assertAlmostEqual(distribution.focus_scores["b"], 0.5)
        self.assertAlmostEqual(sum(distribution.focus_probabilities.values()), 1.0)

    def test_manifest_rejects_development_id_and_binds_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "development leakage"):
            build_manifest_from_records(
                condition="F0", checkpoint_path="/runs/f0/calibration/last.pt",
                checkpoint_role="calibration_last", records=("dev-1",),
                fit_ids=("fit-1",), checkpoint_sha256="a" * 64,
                fit_ids_sha256="c" * 64, metadata_index_sha256="d" * 64,
                expected_object_ids_sha256="e" * 64,
                primary_node_ids=(17, 20, 23, 26),
            )
        manifest = build_manifest_from_records(
            condition="F0", checkpoint_path="/runs/f0/calibration/last.pt",
            checkpoint_role="calibration_last", records=("fit-1",),
            fit_ids=("fit-1",), checkpoint_sha256="b" * 64,
            fit_ids_sha256="c" * 64, metadata_index_sha256="d" * 64,
            expected_object_ids_sha256="e" * 64,
            primary_node_ids=(17, 20, 23, 26),
        )
        self.assertEqual(manifest.schema_version, "factor-manifest-v1")
        self.assertEqual(manifest.condition, "F0")
        self.assertEqual(manifest.checkpoint_role, "calibration_last")
        self.assertEqual(manifest.checkpoint_path,
                         "/runs/f0/calibration/last.pt")
        self.assertEqual(manifest.checkpoint_sha256, "b" * 64)
        self.assertEqual(manifest.fit_ids, ("fit-1",))
        self.assertEqual(manifest.primary_node_ids, (17, 20, 23, 26))
        self.assertEqual(len(manifest.manifest_sha256), 64)
```

Add `test_image_priority_uses_max_eligible_cyclist_joint`, which supplies two
eligible Cyclist objects and one non-Cyclist object and expects the image
priority to equal only the larger eligible joint; and
`test_failed_candidate_gate_blocks_manifest_write`, which expects no manifest
file or digest when the candidate decision is failed.
`test_focus_rejects_unbound_priority_dict` passes a bare mapping or a priority
wrapper with the wrong metadata-index SHA256 and expects `ValueError` before a
distribution is returned.

Add these fail-closed builder tests with exact fixtures:
`test_manifest_generation_temporarily_evals_model_and_preserves_state` starts
the model in train mode with changed normalization buffers, compares the full
model-state hash before and after (including BN running buffers), and expects
every semantic module's training flag restored; `test_manifest_requires_every_fit_image` removes one fit image
and expects `ValueError("fit image coverage")`; and
`test_manifest_requires_exact_eligible_object_identity_set` removes one
eligible object, adds one development object, or duplicates one object and
expects `ValueError("object identity coverage")` in each case.
Add `test_manifest_digest_changes_for_checkpoint_fit_metadata_or_object_edit`,
which edits each bound field independently and expects a different
`manifest_sha256`, and
`test_distribution_rejects_manifest_or_metadata_hash_mismatch`, which expects
both distribution construction and queue resume to
fail closed after any such edit. `test_manifest_path_is_resolved_and_secret_free`
passes a credential-bearing path and expects the stored resolved provenance
path to reject the secret-bearing form.

- [ ] **Step 2: Run the tests and verify failure**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_learned_factor_manifest -v`

Expected: import failure for `ifdr_yolo.data.learned_factor_manifest`.

- [ ] **Step 3: Implement immutable manifest and focus distribution**

```python
from dataclasses import dataclass
from math import isfinite
from typing import Mapping


PRIMARY_NODE_IDS = (17, 20, 23, 26)


@dataclass(frozen=True)
class LearnedObjectFactor:
    image_id: str
    object_id: str
    sampling: float
    visibility: float
    learned_joint: float
    eligible_cyclist: bool


@dataclass(frozen=True)
class LearnedFactorManifest:
    schema_version: str
    condition: str
    checkpoint_path: str
    checkpoint_role: str
    checkpoint_sha256: str
    fit_ids_sha256: str
    fit_ids: tuple[str, ...]
    metadata_index_sha256: str
    primary_node_ids: tuple[int, ...]
    expected_object_ids_sha256: str
    objects: tuple[LearnedObjectFactor, ...]
    manifest_sha256: str


@dataclass(frozen=True)
class ValidatedMetadataPriorities:
    metadata_index_sha256: str
    values: Mapping[str, float]


def aggregate_primary_node_factors(node_values):
    if tuple(sorted(node_values)) != PRIMARY_NODE_IDS:
        raise ValueError("learned manifest requires primary P2-P5 nodes")
    sampling = sum(node_values[node][0] for node in PRIMARY_NODE_IDS) / 4.0
    visibility = sum(node_values[node][1] for node in PRIMARY_NODE_IDS) / 4.0
    if not all(isfinite(value) and 0.0 <= value <= 1.0
               for value in (sampling, visibility)):
        raise ValueError("learned factor output must be finite in [0, 1]")
    return LearnedObjectFactor(
        image_id="", object_id="", sampling=sampling, visibility=visibility,
        learned_joint=1.0 - (1.0 - sampling) * (1.0 - visibility),
        eligible_cyclist=True,
    )


def average_tie_percentile_rank(scores):
    ordered = sorted(scores.values())
    result = {}
    denominator = max(len(ordered) - 1, 1)
    for value in sorted(set(ordered)):
        indexes = [index for index, item in enumerate(ordered) if item == value]
        percentile = sum(indexes) / len(indexes) / denominator
        for image_id, score in scores.items():
            if score == value:
                result[image_id] = percentile
    return result


def build_learned_focus_distribution(*, manifest, metadata_index,
                                     metadata_priorities, epoch):
    verify_manifest_binding(manifest, metadata_index)
    if manifest.checkpoint_role != "calibration_last":
        raise ValueError("learned replay requires calibration_last")
    if metadata_priorities.metadata_index_sha256 != metadata_index.sha256:
        raise ValueError("metadata priority hash mismatch")
    metadata_priority = metadata_priorities.values
    if set(metadata_priority) != set(manifest.fit_ids):
        raise ValueError("metadata priority IDs differ from manifest fit IDs")
    learned_priority = image_max_eligible_cyclist_joint(manifest.objects)
    learned_percentile = average_tie_percentile_rank(learned_priority)
    focus_ids = tuple(sorted(learned_priority))
    focus_scores = {
        image_id: 0.5 * metadata_priority[image_id]
        + 0.5 * learned_percentile[image_id]
        for image_id in focus_ids
    }
    weighted = {image_id: score + 0.05 for image_id, score in focus_scores.items()}
    total = sum(weighted.values())
    if total <= 0.0:
        raise ValueError("learned focus distribution is empty")
    final_probabilities = mix_m3_probabilities(
        original=uniform_probabilities(manifest.fit_ids),
        focus={image_id: value / total for image_id, value in weighted.items()},
        epoch=epoch,
    )
    return ReplayDistribution(
        mode="factor_guided", epoch=epoch, eta=replay_eta(epoch),
        image_ids=tuple(sorted(manifest.fit_ids)),
        original_probabilities=final_probabilities.original,
        focus_probabilities=final_probabilities.focus,
        probabilities=final_probabilities.final,
        focus_scores=focus_scores,
        source_sha256=manifest.manifest_sha256,
        manifest_sha256=manifest.manifest_sha256,
        calibration_checkpoint_sha256=manifest.checkpoint_sha256,
        metadata_index_sha256=manifest.metadata_index_sha256,
        distribution_sha256=digest_distribution(
            "factor_guided", epoch, final_probabilities,
            manifest.manifest_sha256, manifest.checkpoint_sha256,
            manifest.metadata_index_sha256,
        ),
    )
```

```python
def build_learned_factor_manifest(*, condition, checkpoint_path, checkpoint_role,
                                  checkpoint_sha256, model, loader, fit_ids,
                                  metadata_index):
    if checkpoint_role != "calibration_last":
        raise ValueError("learned manifest requires calibration_last")
    resolved_path = resolve_provenance_path(checkpoint_path)
    if sha256_file(resolved_path) != checkpoint_sha256:
        raise ValueError("calibration checkpoint hash mismatch")
    load_validated_checkpoint(model, resolved_path, role=checkpoint_role)
    before_state = full_model_state_sha256(model)
    before_flags = capture_training_flags(model)
    expected_images = tuple(sorted(fit_ids))
    expected_objects = expected_eligible_cyclist_object_ids(
        metadata_index, expected_images,
    )
    observed_images = []
    observed_objects = []
    model.eval()
    try:
        with torch.no_grad():
            for batch in deterministic_no_augmentation_loader(loader, expected_images):
                observed_images.extend(batch.image_ids)
                observed_objects.extend(
                    evaluate_primary_nodes(model, batch, PRIMARY_NODE_IDS)
                )
    finally:
        restore_training_flags(model, before_flags)
    if tuple(sorted(observed_images)) != expected_images:
        raise ValueError("fit image coverage")
    observed_ids = tuple(sorted(
        (record.image_id, record.object_id) for record in observed_objects
    ))
    if observed_ids != tuple(sorted(expected_objects)):
        raise ValueError("object identity coverage")
    if full_model_state_sha256(model) != before_state:
        raise ValueError("manifest generation changed model state")
    return finalize_manifest(
        schema_version="factor-manifest-v1", condition=condition,
        checkpoint_path=resolved_path, checkpoint_role=checkpoint_role,
        checkpoint_sha256=checkpoint_sha256,
        fit_ids_sha256=digest_ids(expected_images),
        fit_ids=expected_images,
        metadata_index_sha256=metadata_index.sha256,
        primary_node_ids=PRIMARY_NODE_IDS,
        expected_object_ids_sha256=digest_ids(expected_objects),
        objects=tuple(sorted(observed_objects)),
    )
```

`build_manifest_from_records` rejects any image outside `fit_ids`, validates the
resolved provenance path, requires `checkpoint_role="calibration_last"`,
validates checkpoint/fit/metadata/object-set hashes, stores only eligible
Cyclist object records, and computes `manifest_sha256` from every manifest field
and sorted object payload. The builder uses a deterministic no-augmentation
loader whose observed image IDs must exactly equal `fit_ids`; it derives the
expected eligible object IDs from the bound metadata index and requires exact
set equality with observed `(image_id, object_id)` records. Before inference it
records every model parameter/buffer and training flag, switches the model to
`eval()`, evaluates under `torch.no_grad()`, and restores the flags; a complete
model-state hash mismatch before/after fails closed. The sampler uses the
validated manifest and typed metadata-priority record with
`build_learned_focus_distribution` output and the M3
replacement draw and eta rules; its distribution and draw-journal identities
include manifest, calibration-checkpoint, and metadata-index hashes. No
candidate manifest is created or consumed after a failed gate, and F0 is paired
only with the selected candidate.

- [ ] **Step 4: Verify manifest and replay integration**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_learned_factor_manifest tests.test_replay_sampler -v`

Expected: all manifest provenance/hash fields, temporary-eval state preservation,
exact fit/object coverage, tie ranks, focus probabilities, fit-ID leakage checks,
and M3 draw-key tests pass.

- [ ] **Step 5: Commit**

```text
git add ifdr_yolo/data/learned_factor_manifest.py ifdr_yolo/data/replay_sampler.py tests/test_learned_factor_manifest.py
git commit -m "feat: bind learned-factor replay manifests"
```

### Task 4: Object-balanced natural ROI alignment

**Files:**
- Create: `ifdr_yolo/losses/factor_alignment.py`
- Modify: `ifdr_yolo/losses/ifdr_detection.py`
- Test: `tests/test_factor_alignment_loss.py`

- [ ] **Step 1: Write area, class-frequency and channel-mask invariance tests**

```python
import unittest
import math
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
        small = ObjectFactorTarget(
            0, 2, (0.0, 0.0, 0.25, 0.25), (0.8, 0.4), (True, True)
        )
        large = ObjectFactorTarget(
            0, 2, (0.0, 0.0, 1.0, 1.0), (0.8, 0.4), (True, True)
        )
        factor_map = self.factor_map(0.5, 0.5)
        self.assertTrue(torch.allclose(
            object_balanced_factor_loss([factor_map], [small]),
            object_balanced_factor_loss([factor_map], [large]),
        ))

    def test_invalid_sampling_keeps_visibility_channel(self):
        target = ObjectFactorTarget(
            batch_index=0, class_id=1,
            box_xyxy_normalized=(0.0, 0.0, 0.5, 0.5),
            target=(0.0, 0.9), valid=(False, True),
        )
        loss = object_balanced_factor_loss([self.factor_map(0.7, 0.2)], [target])
        expected = F.smooth_l1_loss(torch.tensor(0.2), torch.tensor(0.9))
        self.assertTrue(torch.allclose(loss, expected))
```

Add `test_class_macro_average_ignores_object_frequency` with 20 Cars and one
Cyclist; assert the result equals `(car_loss + cyclist_loss) / 2`, not the
21-object mean. Add `test_normalized_box_maps_per_node_size` using the same
normalized box and P2/P3/P4/P5 sizes `(80, 120)`, `(40, 60)`, `(20, 30)`, and
`(10, 15)`; assert four distinct integer ROIs. Add
`test_clip_out_of_range_box_then_pool` for `(-0.01, 0.10, 1.01, 0.80)`;
`test_reverse_or_nonfinite_normalized_box_fails` for reversed and `NaN` input;
and `test_empty_roi_is_zero_weight_and_counted` with a clipped-to-empty box,
an output loss of zero, and `empty_roi_count == 1`.

- [ ] **Step 2: Run tests and verify failure**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_alignment_loss -v`

- [ ] **Step 3: Implement explicit object -> class -> node reduction**

```python
from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch.nn import functional as F

@dataclass(frozen=True)
class ObjectFactorTarget:
    batch_index: int
    class_id: int
    box_xyxy_normalized: tuple[float, float, float, float]
    target: tuple[float, float]
    valid: tuple[bool, bool]


def map_normalized_box_to_feature_roi(
    box_xyxy_normalized, height: int, width: int,
):
    x1, y1, x2, y2 = (float(value) for value in box_xyxy_normalized)
    if not all(torch.isfinite(torch.tensor(value)) for value in (x1, y1, x2, y2)):
        raise ValueError("normalized ROI must be finite")
    if x1 >= x2 or y1 >= y2:
        raise ValueError("normalized ROI must be ordered and non-empty")
    x1, x2 = max(0.0, x1), min(1.0, x2)
    y1, y2 = max(0.0, y1), min(1.0, y2)
    left, top = math.floor(x1 * width), math.floor(y1 * height)
    right, bottom = math.ceil(x2 * width), math.ceil(y2 * height)
    left, right = max(0, min(width, left)), max(0, min(width, right))
    top, bottom = max(0, min(height, top)), max(0, min(height, bottom))
    if left >= right or top >= bottom:
        return None
    return left, top, right, bottom


def object_balanced_factor_loss(
    node_maps: Sequence[torch.Tensor], targets: Sequence[ObjectFactorTarget],
    *, empty_roi_counter: list[int] | None = None,
) -> torch.Tensor:
    node_losses = []
    for factor_map in node_maps:
        class_losses: dict[int, list[torch.Tensor]] = {}
        for item in targets:
            roi = map_normalized_box_to_feature_roi(
                item.box_xyxy_normalized, factor_map.shape[-2], factor_map.shape[-1]
            )
            if roi is None:
                if empty_roi_counter is not None:
                    empty_roi_counter[0] += 1
                continue
            pooled = pool_object_roi(factor_map[item.batch_index], roi)
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

`map_normalized_box_to_feature_roi` validates finite coordinates and raw ordering,
clips explicitly out-of-range coordinates to `[0, 1]`, applies floor/ceil per
node map, and returns `None` for an empty clipped ROI. `pool_object_roi` then
averages spatial values; the loss increments `empty_roi_counter` and gives an
empty ROI zero weight before any object/class reduction. A fresh map call is
made for each node, so an integer ROI is never reused across P2-P5.

- [ ] **Step 4: Wire the loss with legacy/non-calibration compatibility**

In `IFDRDetectionLoss`, call the new function only when
`batch["factor_object_targets"]` exists. Preserve current behavior for every
legacy non-calibration batch. In a calibration three-view batch, route natural
ROI targets to clean contexts, synthetic dense targets to the target context,
and specificity to target/background deltas relative to clean; expose separate
scalar components `synthetic_factor_loss`, `natural_factor_loss`, and
`specificity_loss`, and never call the detection loss in calibration.

- [ ] **Step 5: Run focused and regression tests, then commit**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_alignment_loss tests.test_ifdr_detection_loss tests.test_ifdr_data -v`

```text
git add ifdr_yolo/losses/factor_alignment.py ifdr_yolo/losses/ifdr_detection.py tests/test_factor_alignment_loss.py
git commit -m "feat: add object-balanced natural factor alignment"
```

### Task 5: Matched target/background specificity supervision

**Files:**
- Modify: `ifdr_yolo/data/ifdr_dataset.py` (`IFDRInterventionTransform`, `collate_ifdr_batch`)
- Modify: `ifdr_yolo/models/ifdr_model.py` (three-view forward and calibration routing)
- Modify: `ifdr_yolo/losses/factor_alignment.py`
- Modify: `ifdr_yolo/losses/ifdr_detection.py` (calibration loss routing)
- Test: `tests/test_factor_specificity.py`
- Test: `tests/test_factor_three_view_loss.py`

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
        self.assertTrue(torch.allclose(loss, torch.tensor(0.0)))
        extra_target = factor_specificity_loss(
            clean=torch.tensor([0.20]), target=torch.tensor([0.30]),
            background=torch.tensor([0.26]), margin=0.05,
        )
        self.assertTrue(torch.allclose(extra_target, torch.tensor(0.01)))

    def test_specificity_rejects_overlapping_background(self):
        labels = {"bboxes": torch.tensor([[0.1, 0.1, 0.3, 0.3],
                                           [0.6, 0.6, 0.8, 0.8]])}
        with self.assertRaisesRegex(ValueError, "background overlaps annotated object"):
            build_specificity_pair(
                labels, target_index=0, background_box=(0.6, 0.6, 0.8, 0.8),
                severity=0.5, transform_seed=7,
            )

    def test_collate_preserves_three_views_and_normalized_roi_identity(self):
        from ifdr_yolo.data.ifdr_dataset import (
            BACKGROUND_IMAGE_KEY,
            CLEAN_IMAGE_KEY,
            FACTOR_OBJECT_TARGETS_KEY,
            TARGET_IMAGE_KEY,
            collate_ifdr_batch,
        )

        view = torch.zeros(3, 8, 8, dtype=torch.uint8)
        sample = {
            "img": view,
            "bboxes": torch.tensor([[0.25, 0.25, 0.50, 0.50]]),
            "cls": torch.tensor([[2.0]]),
            "batch_idx": torch.zeros(1),
            CLEAN_IMAGE_KEY: view.clone(),
            TARGET_IMAGE_KEY: view.clone(),
            BACKGROUND_IMAGE_KEY: view.clone(),
            FACTOR_OBJECT_TARGETS_KEY: ({
                "class_id": 2,
                "box_xyxy_normalized": (0.25, 0.25, 0.50, 0.50),
            },),
        }
        batch = collate_ifdr_batch([sample])
        self.assertEqual(tuple(batch[CLEAN_IMAGE_KEY].shape), (1, 3, 8, 8))
        self.assertEqual(tuple(batch[TARGET_IMAGE_KEY].shape), (1, 3, 8, 8))
        self.assertEqual(tuple(batch[BACKGROUND_IMAGE_KEY].shape), (1, 3, 8, 8))
        target = batch[FACTOR_OBJECT_TARGETS_KEY][0]
        self.assertEqual(target["batch_idx"], 0)
        self.assertEqual(target["box_xyxy_normalized"], (0.25, 0.25, 0.50, 0.50))
```

Add these named tests with explicit fixtures and outcomes: `test_low_severity_gets_zero_specificity_weight` passes a severity of `0.24` and expects a zero contribution; `test_target_background_share_severity_and_transform_seed` compares the emitted pair metadata and expects exact equality; `test_malformed_pair_counts_rejection` supplies missing and duplicate pair fields and expects the registered rejection counter; and `test_empty_background_has_zero_iou_with_annotations` checks IoU `0.0` against every annotated box.

`tests/test_factor_three_view_loss.py` must also define
`test_synthetic_loss_uses_target_context_only`, which mutates clean and
background factor tensors while holding the target tensors fixed and expects
identical synthetic loss, and `test_calibration_does_not_call_detection_loss`,
which spies on the detection criterion and expects zero calls during
calibration. These tests pin the ownership boundary: the dataset owns raw-stage
identity attachment and collated three-view records; the model owns the `3B`
forward and clean/target/background split; the loss module owns target-only
synthetic routing, clean-only natural routing, and target/background
specificity.

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

The calibration routing interface is explicit and target-only for synthetic
supervision:

```python
def route_calibration_losses(*, clean_context, target_context,
                             background_context, dense_target,
                             natural_object_targets, intervention):
    synthetic = synthetic_factor_loss_from_context(
        target_context, dense_target,
    )
    natural = object_balanced_factor_loss(
        clean_context, natural_object_targets,
    )
    specificity = factor_specificity_from_contexts(
        clean_context, target_context, background_context, intervention,
    )
    return {
        "synthetic_factor_loss": synthetic,
        "natural_factor_loss": natural,
        "specificity_loss": specificity,
    }
```

`IFDRDetectionModel.loss` calls this route only for a registered calibration
batch. It must not pass clean or background tensors into
`synthetic_factor_loss_from_context`, and it must not invoke the detection
criterion in that phase. Legacy non-calibration batches continue through their
existing detection and synthetic-loss path unchanged.

Extend `IFDRInterventionTransform.__call__` to bind raw `get_image_and_label`
output before geometry, then emit one clean/target/background tuple with
`CLEAN_IMAGE_KEY`, `TARGET_IMAGE_KEY`, and `BACKGROUND_IMAGE_KEY`. The target
and background use the same severity and transform seed; the clean view is
unmodified. Emit immutable object records under `FACTOR_OBJECT_TARGETS_KEY` with
`batch_idx`, `class_id`, and normalized `box_xyxy_normalized`. Update
`collate_ifdr_batch` to stack all three BCHW views, attach the collated
`batch_idx`, and clip each normalized ROI when mapping it to every P2-P5 node.
Update `IFDRDetectionModel.loss` to concatenate the three views as one `3B`
forward, consume and split all six node contexts into clean/target/background,
and pass clean-only ROI records plus intervention records to
`IFDRDetectionLoss`. Detection loss is frozen/excluded during calibration.
Do not assign a positive degradation target to empty background.

- [ ] **Step 4: Verify legacy behavior and pair accounting**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_specificity tests.test_factor_three_view_loss tests.test_intervention_sampler tests.test_intervention_targets tests.test_ifdr_data -v`

- [ ] **Step 5: Commit**

```text
git add ifdr_yolo/data/ifdr_dataset.py ifdr_yolo/models/ifdr_model.py ifdr_yolo/losses/ifdr_detection.py ifdr_yolo/losses/factor_alignment.py tests/test_factor_specificity.py tests/test_factor_three_view_loss.py
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
        actual = {
            name for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(actual, set(phase.trainable_parameter_names))
        self.assertTrue(actual)
        projection_modules = tuple(
            projection
            for index in self.model.fusion_node_indices
            for projection in self.model.model[index].projections
        )
        self.assertEqual(len(projection_modules), 12)
        expected_ids = {
            id(parameter)
            for projection in projection_modules
            for parameter in projection.parameters()
        }
        first_layer = self.model.model[self.model.fusion_node_indices[0]]
        expected_ids.update(
            id(parameter)
            for parameter in first_layer.reliability_estimator.shared_core.parameters()
        )
        expected_ids.update(
            id(parameter)
            for parameter in first_layer.reliability_estimator.factor_head.parameters()
        )
        actual_ids = {
            id(parameter)
            for name, parameter in self.model.named_parameters()
            if name in phase.trainable_parameter_names
        }
        self.assertEqual(actual_ids, expected_ids)
        self.assertTrue(all(
            ".projections." in name
            or ".reliability_estimator.shared_core." in name
            or ".reliability_estimator.factor_head." in name
            for name in phase.trainable_parameter_names
        ))
        self.assertEqual(phase.fusion_schedule, 0.0)
        self.assertEqual(phase.dcli_schedule, 0.0)

    def test_calibration_rejects_detection_or_adapter_parameter(self):
        phase = semantic_calibration_phase(self.model, variant="F3", epochs=30)
        forbidden = (
            "detect", "router", "fusion_adapter", "localization_adapter",
            "gate_logit",
        )
        self.assertFalse(any(
            token in name for name in phase.trainable_parameter_names for token in forbidden
        ))

    def test_f0_masks_only_the_new_repair_terms(self):
        expected = {
            "F0": {"synthetic": 1.0, "natural": 0.0, "specificity": 0.0},
            "F1": {"synthetic": 1.0, "natural": 1.0, "specificity": 0.0},
            "F2": {"synthetic": 1.0, "natural": 0.0, "specificity": 1.0},
            "F3": {"synthetic": 1.0, "natural": 1.0, "specificity": 1.0},
        }
        for variant, mask in expected.items():
            phase = semantic_calibration_phase(self.model, variant=variant, epochs=30)
            self.assertEqual(phase.loss_mask, mask)

    def test_three_view_forward_splits_each_node_context(self):
        from ifdr_yolo.experiments.factor_repair import split_three_view_contexts

        clean = torch.zeros(1, 3, 128, 128)
        target = torch.ones(1, 3, 128, 128)
        background = torch.full((1, 3, 128, 128), 2.0)
        self.model.set_component_schedules(
            fusion=0.0, dcli=0.0, factor_supervision=1.0
        )
        with torch.no_grad():
            self.model(torch.cat((clean, target, background), dim=0))
        raw_contexts = self.model.consume_reliability_context()
        split = split_three_view_contexts(raw_contexts, batch_size=1)
        self.assertEqual(set(split), {"clean", "target", "background"})
        for view in split.values():
            self.assertEqual(tuple(view), self.model.fusion_node_indices)
            self.assertTrue(all(
                context.factors.shape[0] == 1 for context in view.values()
            ))

    def test_semantic_gradient_diagnostics_cover_all_paths(self):
        groups = self.model.gradient_diagnostic_parameter_groups()
        self.assertEqual(
            set(groups),
            {f"projection_{index:02d}" for index in range(12)}
            | {"shared_core", "factor_head"},
        )
```

- [ ] **Step 2: Run tests and verify failure**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_repair_phase -v`

- [ ] **Step 3: Expose three-view context splitting and semantic parameter names**

```python
from ifdr_yolo.models.gated_fusion import ReliabilityContext, ReliabilityGatedConcat


def split_three_view_contexts(contexts, batch_size):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    result = {"clean": {}, "target": {}, "background": {}}
    for index, context in contexts.items():
        expected = 3 * batch_size
        if context.factors.shape[0] != expected or context.branch_weights.shape[0] != expected:
            raise RuntimeError("three-view reliability contexts must have leading dimension 3B")
        for name, start in (("clean", 0), ("target", batch_size), ("background", 2 * batch_size)):
            stop = start + batch_size
            result[name][index] = ReliabilityContext(
                factors=context.factors[start:stop],
                branch_weights=context.branch_weights[start:stop],
                gate_strength=context.gate_strength,
            )
    return result


def factor_semantic_named_parameters(self):
    allowed_parameter_ids: set[int] = set()
    projection_modules = []
    for index in self.fusion_node_indices:
        layer = self.model[index]
        if not isinstance(layer, ReliabilityGatedConcat):
            raise TypeError(f"fusion node {index} is not ReliabilityGatedConcat")
        if len(layer.projections) != 2:
            raise ValueError(f"fusion node {index} must expose two projections")
        projection_modules.extend(layer.projections)
        for projection in layer.projections:
            allowed_parameter_ids.update(
                id(parameter) for parameter in projection.parameters()
            )
        allowed_parameter_ids.update(
            id(parameter)
            for parameter in layer.reliability_estimator.shared_core.parameters()
        )
        allowed_parameter_ids.update(
            id(parameter)
            for parameter in layer.reliability_estimator.factor_head.parameters()
        )
    if len(projection_modules) != 12:
        raise ValueError("semantic calibration requires 12 projection modules")
    seen_parameter_ids: set[int] = set()
    for name, parameter in self.named_parameters():
        parameter_id = id(parameter)
        if parameter_id in allowed_parameter_ids and parameter_id not in seen_parameter_ids:
            seen_parameter_ids.add(parameter_id)
            yield name, parameter
```

The implementation must account for shared reliability modules exactly once. The
provenance must retain complete semantic paths: all 12 per-node projection
submodules, the shared `reliability_estimator.shared_core`, and the shared
`reliability_estimator.factor_head`; report group membership by parameter
identity so the shared modules are not counted once per node. `semantic_calibration_phase`
first freezes all parameters, then enables only the yielded identities, resets
optimizer state, sets all component schedules explicitly, and records sorted
trainable names in provenance. It also validates the exact loss masks above,
rejects any calibration epoch other than 30, and disables early stopping for
all F0-F3 phases. `gradient_diagnostic_parameter_groups` emits
`projection_00` through `projection_11`, `shared_core`, and `factor_head`; each
group is built from the same identity-deduplicated parameter set used by the
phase, so diagnostics cannot silently omit a projection or count shared
parameters once per node.

- [ ] **Step 4: Add validation no-step and optimizer-reset tests**

Snapshot parameter tensors before validation, run one validation batch, and assert byte-identical parameters and unchanged optimizer step. Assert a newly created calibration optimizer has empty state.

- [ ] **Step 5: Run tests and commit**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_repair_phase tests.test_ifdr_model tests.test_ifdr_trainer -v`

```text
git add ifdr_yolo/models/ifdr_model.py ifdr_yolo/experiments/factor_repair.py ifdr_yolo/experiments/ifdr_trainer.py tests/test_factor_repair_phase.py
git commit -m "feat: isolate semantic factor calibration"
```

### Task 6A: Exact 60-epoch task-adaptation phase

**Files:**
- Modify: `ifdr_yolo/experiments/factor_repair.py`
- Modify: `ifdr_yolo/experiments/ifdr_trainer.py`
- Test: `tests/test_factor_repair_phase.py`

- [ ] **Step 1: Write the matched-adaptation equality test**

Add `test_task_adaptation_phase_matches_f0_and_candidate` to
`tests/test_factor_repair_phase.py`. Construct independent F0 and selected
candidate models from the same initialization, then deliberately perturb their
calibration semantic tensors so their starting semantic hashes differ. Load
each condition from its own `calibration_last` checkpoint, call the registered
phase factory, run at least two adaptation epochs/optimizer steps, interrupt
and resume once, and assert all of the following:

- the frozen semantic names are exactly the twelve projections plus shared
  `reliability_estimator.shared_core` and `factor_head`;
- each condition records a complete semantic-state hash over those parameters
  and buffers before adaptation, and F0/candidate starting hashes are allowed
  (and expected in this fixture) to differ;
- the F0 and candidate trainable named-parameter sets are equal and contain
  only the enumerated task path (`backbone`, `C2f`, routers, fusion adapters,
  localization adapter, detection head, and explicitly registered gate
  parameters);
- optimizer class and every hyperparameter are equal after a fresh reset;
- `epochs == 60`, update count and eta schedule are equal, and early stopping
  is disabled;
- after every committed epoch, after the resume, and at the final checkpoint,
  each condition's semantic-state hash equals its own pre-adaptation hash;
- the trainer calls `model.train()` for task-path updates but forces every
  semantic submodule to `eval()`, so all semantic training flags are false and
  normalization buffers remain unchanged;
- both primary checkpoint paths are `last.pt` and each condition resumes only
  from its own checkpoint/provenance identity.

- [ ] **Step 2: Run the equality test and verify failure**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_repair_phase.FactorRepairPhaseTest.test_task_adaptation_phase_matches_f0_and_candidate -v`

Expected: import or assertion failure until the independent adaptation phase
contract is implemented.

- [ ] **Step 3: Implement an independent task-adaptation phase factory**

```python
def task_adaptation_phase(model, *, condition, calibration_checkpoint_path,
                          calibration_checkpoint_role="calibration_last",
                          epochs=60, optimizer_name, optimizer_hparams,
                          eta_schedule, primary_checkpoint="last.pt"):
    if (calibration_checkpoint_role != "calibration_last"
            or epochs != 60 or primary_checkpoint != "last.pt"):
        raise ValueError(
            "registered adaptation requires calibration_last, 60 epochs, and last.pt"
        )
    load_validated_checkpoint(model, calibration_checkpoint_path,
                             role=calibration_checkpoint_role)
    semantic = {
        name for name, _ in model.factor_semantic_named_parameters()
    }
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name not in semantic
    trainable = tuple(sorted(
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ))
    _require_registered_task_path(trainable)
    optimizer = build_optimizer(
        optimizer_name, model, trainable, optimizer_hparams,
    )
    return TaskAdaptationPhase(
        condition=condition, calibration_checkpoint_path=resolve_provenance_path(
            calibration_checkpoint_path,
        ), calibration_checkpoint_role=calibration_checkpoint_role,
        semantic_state_sha256=semantic_state_sha256(model, semantic_module_ids(model)),
        epochs=epochs, trainable_parameter_names=trainable,
        frozen_parameter_names=tuple(sorted(semantic)), optimizer=optimizer,
        optimizer_hparams=dict(optimizer_hparams), eta_schedule=tuple(eta_schedule),
        update_count=registered_update_count(epochs), early_stopping=False,
        primary_checkpoint=primary_checkpoint,
    )
```

The factory freezes the semantic parameters by identity, validates the exact
task-path allowlist, creates a new optimizer with no inherited state, and uses
the same optimizer hyperparameters, update count, eta schedule, and no-stop
policy for F0 and the selected candidate. Each phase loads its own validated
`calibration_last` checkpoint and snapshots a canonical semantic state hash
over parameters and buffers; no cross-condition byte-equality assertion is
allowed. The trainer receives the phase object rather than a free-form epoch
count. At each epoch boundary and resume it recomputes the condition-local
hash, fails closed on any change, and runs:

```python
def enforce_semantic_eval_mode(model, semantic_module_ids):
    model.train()
    for module in model.modules():
        if id(module) in semantic_module_ids:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False
```

`semantic_state_sha256(model, semantic_module_ids)` serializes sorted semantic
module names, every `named_parameter()` tensor, and every `named_buffer()`
tensor with dtype, shape, and raw bytes; it excludes task-path state and
returns a 64-hex digest. The epoch journal records this digest for
`epoch_commit`, `resume_check`, and `final_checkpoint`, and rejects any
missing or changed record.

The trainer writes the fixed-budget `last.pt` primary checkpoint and cannot
substitute `best.pt` for evaluation. A resume must verify the saved
condition-local semantic hash before continuing and after the resumed epoch.

- [ ] **Step 4: Verify adaptation and calibration contracts together**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_repair_phase tests.test_ifdr_trainer -v`

Expected: both phase suites pass, including equal trainable names, optimizer
reset, schedule, update count, condition-local semantic hash preservation across
epochs/resume/final checkpoint, forced semantic eval mode, and no early stop.

- [ ] **Step 5: Commit**

```text
git add ifdr_yolo/experiments/factor_repair.py ifdr_yolo/experiments/ifdr_trainer.py tests/test_factor_repair_phase.py
git commit -m "feat: add matched task adaptation phase"
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
        self.assertEqual({config.conditions[name].epochs for name in ("F0", "F1", "F2", "F3")}, {30})
        self.assertEqual(config.task_adaptation_epochs, 60)
        self.assertEqual(config.max_selected_factor_repairs, 1)
        self.assertFalse(config.early_stopping)
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

    def test_bootstrap_seed_or_replicate_override_fails(self):
        for extra in (
            {"factor_gate": {"bootstrap_replicates": 10}},
            {"factor_gate": {"bootstrap_seed": 17}},
        ):
            with TemporaryDirectory() as directory:
                path = write_valid_config(Path(directory), extra=extra)
                with self.assertRaisesRegex(ValueError, "unknown factor_gate fields"):
                    load_factor_repair_config(path)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_repair_config -v`

- [ ] **Step 3: Implement frozen configuration dataclasses**

Define `DevelopmentProtocolConfig`, `MetadataReplayConfig`, `FactorAlignmentConfig`, `FactorGateConfig`, `RepairConditionConfig`, and `FactorRepairConfig`. All fields are required; unknown fields fail. Register M1/M2/M3 at 60 epochs, F0/F1/F2/F3 calibration at 30 epochs, one 60-epoch task-adaptation budget shared by F0 and the single selected F1-F3 candidate, and `max_selected_factor_repairs=1`. F0 masks natural and specificity terms but retains synthetic supervision. Hard-code validation of registered values from the approved spec rather than accepting alternative thresholds from the command line.
`FactorGateConfig` intentionally has no bootstrap seed or replicate-count
fields: the gate module's fixed `FACTOR_GATE_BOOTSTRAP_REPLICATES=10000` and
`FACTOR_GATE_BOOTSTRAP_SEED=20260805` are not configurable.

- [ ] **Step 4: Write the canonical seed-17 development YAML**

The YAML binds fit/development/metadata hashes, accepted initialization checkpoint hash, M/F budgets, image size, schedule, factor weights, nodes `(11, 14, 17, 20, 23, 26)`, primary nodes `(17, 20, 23, 26)`, and output root. Task 2 owns canonical YAML generation through `scripts/build_factor_metadata.py`; Task 7 consumes that generated file and validates that every required 64-hex hash is real before loading it. No second YAML writer or ad-hoc config generator is introduced here.

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
import numpy

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

    def test_bootstrap_quantiles_are_fixed_and_ci_is_finite(self):
        self.assertEqual(FACTOR_GATE_BOOTSTRAP_PERCENTILES, (0.025, 0.975))
        delta = numpy.asarray((-0.20, -0.05, 0.10, 0.25), dtype=float)
        ci = numpy.quantile(
            delta, FACTOR_GATE_BOOTSTRAP_PERCENTILES, method="linear"
        )
        self.assertEqual(tuple(ci.shape), (2,))
        self.assertTrue(numpy.isfinite(ci).all())
```

Add named tests with concrete fixtures and expected decisions:
`test_severity_ordering_boundary_at_80_percent` uses exactly 80% valid
severity ordering and expects the registered check to pass;
`test_positive_paired_target_response_and_background_gap` expects both
specificity checks to pass; `test_twelve_primary_directions_requires_ten`
passes exactly 10/12 three-seed directions and expects success;
`test_ci_lower_bound_above_zero_is_required` compares intervals whose lower
bound is `0.0` and `1e-12` and expects only the latter to pass;
`test_incomplete_f0_evidence_fails_closed` supplies one missing endpoint and
expects no selection; and `test_missing_node_seed_or_malformed_count_fails`
expects each omitted node/seed or nonzero malformed count to fail without
mutating thresholds.

Add the relative-selection tests in the same module. A
`candidate_evidence(condition, image_ids_hash, endpoint_values, *,
absolute_gate_passed=True, complete=True)` fixture returns finite endpoint
rows and an absolute-gate decision. Then assert:
`test_candidate_requires_complete_f0_and_paired_delta_ci` rejects incomplete
F0 and a candidate whose paired `DeltaS` lower bound is `0.0`;
`test_multiple_eligible_candidates_use_lower_point_and_name_ties` selects the
largest lower bound, then point estimate, then `F1 < F2 < F3` when differences
are at most `1e-12`; `test_gate_decision_carries_reference_delta_and_hash`
checks `reference_condition`, point/CI, selected condition, endpoint table,
and both evidence hashes; and `test_manual_condition_string_is_rejected`
passes `"F3"` to the queue consumer and expects `ValueError`.
Add `test_bootstrap_resamples_are_byte_identical_across_repeated_calls`,
`test_bootstrap_resample_key_is_shared_across_candidate_names`, and
`test_bootstrap_seed_or_replicate_override_is_rejected`; these assert exactly
10,000 replicates, seed `20260805`, q tuple `(0.025, 0.975)` (the
2.5th/97.5th percentiles), NumPy `method="linear"`, candidate-name-independent
keys, and a selector API with no replicate/seed override parameters.

- [ ] **Step 2: Run tests and verify failure**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_repair_gate -v`

- [ ] **Step 3: Implement an immutable, fully enumerated decision**

```python
from dataclasses import dataclass
import math
from typing import Mapping

import numpy

@dataclass(frozen=True)
class FactorRepairGateDecision:
    passed: bool
    stage: str
    primary_nodes: tuple[int, ...]
    diagnostic_nodes: tuple[int, ...]
    checks: Mapping[str, bool]
    failures: tuple[str, ...]
    evidence_sha256: str


@dataclass(frozen=True)
class FactorRepairSelectionDecision:
    reference_condition: str
    selected_condition: str
    delta_s_point: float
    delta_s_ci95: tuple[float, float]
    endpoint_table: Mapping[str, Mapping[str, float]]
    reference_evidence_sha256: str
    selected_evidence_sha256: str
    decision_sha256: str


PRIMARY_ENDPOINTS = (
    "sampling_residual_spearman",
    "visibility_residual_spearman",
    "sampling_specificity_gap",
    "visibility_specificity_gap",
)

FACTOR_GATE_BOOTSTRAP_REPLICATES = 10_000
FACTOR_GATE_BOOTSTRAP_SEED = 20260805
FACTOR_GATE_BOOTSTRAP_PERCENTILES = (0.025, 0.975)


def composite_mechanism_score(evidence):
    values = [float(evidence[name]) for name in PRIMARY_ENDPOINTS]
    if not all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in values):
        raise ValueError("factor endpoint must be finite and bounded")
    return sum(values) / 4.0


def paired_resample_indices(*, stage, image_ids_hash, image_count,
                            replicate_index):
    key = (FACTOR_GATE_BOOTSTRAP_SEED, stage, image_ids_hash, replicate_index)
    # image_count is only the output length derived from the paired image IDs;
    # it is not an additional random-key component.
    return common_random_number_indices(key, image_count=image_count)


def paired_image_cluster_delta(candidate, f0):
    if candidate.image_ids_hash != f0.image_ids_hash:
        raise ValueError("candidate/F0 evidence image IDs mismatch")
    replicates = []
    for replicate_index in range(FACTOR_GATE_BOOTSTRAP_REPLICATES):
        indices = paired_resample_indices(
            stage="development", image_ids_hash=f0.image_ids_hash,
            image_count=len(f0.image_ids),
            replicate_index=replicate_index,
        )
        candidate_endpoints = recompute_endpoints(candidate, indices)
        f0_endpoints = recompute_endpoints(f0, indices)
        replicates.append(
            composite_mechanism_score(candidate_endpoints)
            - composite_mechanism_score(f0_endpoints)
        )
    point = composite_mechanism_score(candidate.endpoints) \
        - composite_mechanism_score(f0.endpoints)
    ci95 = tuple(numpy.quantile(
        replicates, FACTOR_GATE_BOOTSTRAP_PERCENTILES,
        method="linear",
    ))
    return PairedDelta(
        point=point, ci95=ci95,
        candidate_endpoints=candidate.endpoints,
        candidate_evidence_sha256=candidate.evidence_sha256,
    )


def select_repair_against_f0(f0, candidates):
    if not f0.complete:
        raise ValueError("incomplete F0 evidence")
    f0_score = composite_mechanism_score(f0.endpoints)
    if not math.isfinite(f0_score):
        raise ValueError("non-finite F0 evidence")
    selected = []
    for candidate in candidates:
        if candidate.condition not in ("F1", "F2", "F3"):
            raise ValueError("selection candidates must be F1, F2, or F3")
        if (tuple(candidate.image_ids) != tuple(f0.image_ids)
                or candidate.image_ids_hash != f0.image_ids_hash):
            raise ValueError("candidate/F0 evidence image IDs mismatch")
        if not candidate.complete:
            continue
        if not candidate.absolute_gate_passed:
            continue
        paired = paired_image_cluster_delta(candidate, f0)
        if paired.ci95[0] > 0.0:
            selected.append((paired.ci95[0], paired.point, candidate.condition, paired))
    if not selected:
        return None
    best_lower = max(item[0] for item in selected)
    lower_tied = [item for item in selected
                  if best_lower - item[0] <= 1e-12]
    best_point = max(item[1] for item in lower_tied)
    point_tied = [item for item in lower_tied
                  if best_point - item[1] <= 1e-12]
    lower, point, condition, paired = min(point_tied, key=lambda item: item[2])
    return FactorRepairSelectionDecision(
        reference_condition="F0", selected_condition=condition,
        delta_s_point=point, delta_s_ci95=paired.ci95,
        endpoint_table={"F0": f0.endpoints,
                        condition: paired.candidate_endpoints},
        reference_evidence_sha256=f0.evidence_sha256,
        selected_evidence_sha256=paired.candidate_evidence_sha256,
        decision_sha256=digest_selection_decision(
            "F0", condition, point, paired.ci95,
            {"F0": f0.endpoints, condition: paired.candidate_endpoints},
            f0.evidence_sha256, paired.candidate_evidence_sha256,
        ),
    )


def require_factor_guided_advancement(*, pre, post):
    if not pre.passed:
        raise ValueError("pre-adaptation factor gate failed")
    if not post.passed:
        raise ValueError("post-adaptation factor gate failed")
```

`paired_image_cluster_delta` receives the shared sorted image IDs and resamples
the same IDs for F0 and each candidate in every replicate; it recomputes all
four endpoints before subtracting the composite. Reuse existing
`natural_factor_audit` statistics; do not duplicate Spearman/bootstrap
mathematics. The gate layer validates completeness and translates registered
evidence into an explicit decision. The pooled primary-node statistic with the
formal paired image bootstrap is the only mechanism-selection statistic;
per-node confidence intervals are diagnostic, and no uncorrected per-node or
per-seed significance may select a repair.

`FactorRepairSelectionDecision` deep-freezes the endpoint table into sorted
tuples before computing `decision_sha256`; the queue reserializes and verifies
that digest before accepting it, so a caller cannot mutate a nested mapping or
replace the selected condition after the gate has run.

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

    def test_queue_runs_f0_control_only_with_selected_repair(self):
        with TemporaryDirectory() as directory:
            queue = FactorRepairQueue.create(
                Path(directory), jobs=(
                    "F0-calibration", "F1-calibration", "F2-calibration",
                    "F3-calibration", "F0-adaptation",
                    "selected-repair-adaptation",
                )
            )
            queue.complete("F0-calibration", artifacts=passing_control_artifacts())
            self.assertFalse(queue.launchable("F0-adaptation"))
            queue.complete("F3-calibration", artifacts=passing_repair_artifacts())
            decision = passing_selection_decision()
            queue.consume_selection_decision(decision)
            self.assertTrue(queue.launchable("F0-adaptation"))
            self.assertTrue(queue.launchable("selected-repair-adaptation"))

    def test_queue_rejects_manual_condition_string(self):
        with TemporaryDirectory() as directory:
            queue = FactorRepairQueue.create(Path(directory), jobs=("F0-calibration",))
            with self.assertRaisesRegex(ValueError, "selection decision"):
                queue.consume_selection_decision("F3")

    def test_no_factor_candidate_blocks_track_f_adaptation(self):
        with TemporaryDirectory() as directory:
            queue = FactorRepairQueue.create(
                Path(directory), jobs=("F0-calibration", "F1-calibration",
                                       "F2-calibration", "F3-calibration")
            )
            queue.complete("F0-calibration", artifacts=passing_control_artifacts())
            for name in ("F1-calibration", "F2-calibration", "F3-calibration"):
                queue.complete(name, artifacts=failed_gate_artifacts())
            self.assertEqual(queue.track_f_adaptation_status(), "blocked")
```

The passing fixture is not allowed to hard-code digest strings:

```python
def passing_selection_decision():
    endpoint_table = passing_repair_endpoint_table()
    reference_evidence = canonical_evidence_payload(
        condition="F0", image_ids=passing_image_ids(),
        endpoints=passing_f0_endpoint_table(),
    )
    selected_evidence = canonical_evidence_payload(
        condition="F3", image_ids=passing_image_ids(),
        endpoints=endpoint_table,
    )
    reference_sha256 = sha256_canonical(reference_evidence)
    selected_sha256 = sha256_canonical(selected_evidence)
    decision_payload = canonical_selection_payload(
        reference_condition="F0", selected_condition="F3",
        delta_s_point=0.12, delta_s_ci95=(0.04, 0.20),
        endpoint_table={"F0": passing_f0_endpoint_table(),
                        "F3": endpoint_table},
        reference_evidence_sha256=reference_sha256,
        selected_evidence_sha256=selected_sha256,
    )
    return FactorRepairSelectionDecision(
        reference_condition="F0", selected_condition="F3",
        delta_s_point=0.12, delta_s_ci95=(0.04, 0.20),
        endpoint_table=deep_freeze({
            "F0": passing_f0_endpoint_table(), "F3": endpoint_table,
        }),
        reference_evidence_sha256=reference_sha256,
        selected_evidence_sha256=selected_sha256,
        decision_sha256=sha256_canonical(decision_payload),
    )
```

`queue.consume_selection_decision` recomputes the same canonical evidence and
decision payloads, rejects any tampered SHA256 or mutable nested table, and
accepts this fixture only when all F0/selected-condition and manifest hashes
match the completed calibration jobs. `passing_control_artifacts()` and
`passing_repair_artifacts()` each bind their own non-identical
`calibration_last` checkpoint path/hash and condition-local semantic-state hash;
the queue never substitutes `best.pt`.

Add named runner tests with isolated temporary artifacts:
`test_dirty_checkout_fails_closed`, `test_missing_or_nonhex_hash_fails_closed`,
`test_initialization_checkpoint_mismatch_fails_closed`,
`test_duplicate_process_lock_fails_closed`,
`test_nonfinite_loss_marks_run_failed`,
`test_empty_last_or_best_checkpoint_fails_closed`, and
`test_interrupted_epoch_draw_journal_resumes_exactly_once`. Each test names
the expected exception/status and verifies that no queue job is promoted.
Add `test_primary_metrics_use_last_not_best`, which writes different non-empty
`last.pt` and `best.pt` bytes, mocks the evaluator, and expects the evaluator
input to be `last.pt`, output `metrics_ap40_primary_last.json`, and the
diagnostic best metric to remain separate. Add
`test_missing_empty_or_mismatched_last_hash_fails_closed`, which expects a
failure for each missing path, empty file, or SHA256 mismatch before metrics
are written.
Add `test_queue_rejects_best_manifest_or_hash_mismatch`, which submits a
`best.pt` manifest or a condition/checkpoint/metadata hash mismatch and expects
no adaptation launch; `test_queue_rejects_tampered_selection_sha` mutates the
canonical decision digest and expects `ValueError`; and
`test_factor_guided_draw_identity_binds_manifest`, which changes each
provenance hash and expects resume rejection.

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

The runner writes structured checkpoint roles after every fixed-budget job:
`primary_checkpoint={"path": "last.pt", "sha256": ...}` and
`diagnostic_checkpoint={"path": "best.pt", "sha256": ...}`. It refuses missing,
empty, or mismatched files. The evaluator is called once with the verified
`last.pt` and writes `metrics_ap40_primary_last.json`; any best-checkpoint
metrics are diagnostic-only and cannot satisfy a gate or advance the queue.

- [ ] **Step 4: Implement the queue state machine**

Allowed transitions are `pending -> running -> complete`, `pending -> blocked`, and `running -> failed`. Resume changes `failed -> running` only when identity and existing artifacts validate. The queue executes one GPU process at a time, permits the matched F0 control and at most one selected F1-F3 adaptation only after a candidate passes its gate, and blocks Track F adaptation entirely when no candidate passes. Before either adaptation starts, the runner evaluates that condition's validated `calibration_last` checkpoint on fit clean images and persists its immutable learned-factor manifest; a failed candidate gate creates no candidate manifest. `consume_selection_decision` accepts only an immutable `FactorRepairSelectionDecision`, verifies its F0/candidate evidence hashes and digest, and rejects strings, defaults, `best.pt` manifests, or any manifest whose condition/checkpoint/metadata hash does not match the queue job. It never auto-promotes a method after a failed factor or detection gate.

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

Add named tests with fixed inputs and expected labels:
`test_three_seed_cyclist_paired_bootstrap_ci` checks the registered CI;
`test_car_pedestrian_no_harm_bound_is_one_ap_point` rejects any bound above
`1.0`; `test_mean_and_std_are_reported_for_each_class` checks both values;
`test_target_slice_metrics_include_small_far_occluded` checks all slice keys;
`test_calibration_and_efficiency_fields_are_machine_bound` rejects missing
fields; `test_negative_result_gets_explicit_label` expects the registered
negative label; and `test_advancement_reads_primary_last_metrics_only`
provides different `metrics_ap40_primary_last.json` and best diagnostics and
expects advancement to use only the former.

- [ ] **Step 2: Run tests and verify failure**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_repair_report -v`

- [ ] **Step 3: Implement report objects and reuse evaluators**

Reuse `paired_bootstrap_ap40`, `evaluate_target_slices`, detection reliability,
and existing AP40 parsing. The report reads only
`metrics_ap40_primary_last.json` for primary advancement, binds that file's
SHA256 and the `last.pt` SHA256, and stores best-checkpoint metrics under a
separate diagnostic role. The report stores every failed check and never drops
a class, seed, node or condition.

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

The test builds two synthetic KITTI images with Car, Pedestrian and Cyclist objects, runs the F0 and F3 calibration forward/backward paths, and asserts finite synthetic/natural/specificity losses, zero natural/specificity contribution for F0, non-zero gradient only on semantic parameters, and condition-local semantic state hashes unchanged across the calibration validation step.

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

Only after clean-commit deployment and server identity verification, run one registered seed-17 F0 calibration epoch on smoke data. Validate non-empty `status.json`, resolved config, provenance, draw journal, loss components, non-empty `last.pt` primary plus its SHA256, non-empty `best.pt` diagnostic plus its SHA256, and `metrics_ap40_primary_last.json` bound to the last hash. Interrupt after a committed batch, resume, and verify exactly-once journal entries and identical scientific identity; do not launch task adaptation from this smoke job.

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

### Task 13: Independent review and reproducibility package (follow-up, no training)

**Files:**
- Create: `docs/reports/factor-repair-authoritative-index.json`
- Create: `docs/reports/factor-repair-results-evidence.jsonl`
- Create: `docs/reports/factor-repair-figure-status.md`
- Create: `docs/reports/factor-repair-reproducibility-package.md`
- Create: `requirements-factor-repair.txt`
- Create: `LICENSE`
- Create: `NOTICE`
- Create: `scripts/verify_factor_repair_package.py`
- Test: `tests/test_factor_repair_repro_package.py`

This follow-up is performed by an independent teacher/reviewer who did not
implement Tasks 1-12. It consumes only frozen artifacts and performs no
training, server submission, or dataset download.

- [ ] **Step 1: Write the package-contract test**

```python
import json
import unittest
from pathlib import Path

from scripts.verify_factor_repair_package import load_and_verify_package


class FactorRepairReproPackageTest(unittest.TestCase):
    def test_authoritative_index_requires_machine_bound_evidence(self):
        root = Path("docs/reports")
        index = json.loads((root / "factor-repair-authoritative-index.json").read_text())
        required = {
            "implementation_commit", "source_files", "config_files",
            "split_sha256", "metadata_sha256", "initialization_checkpoint_sha256",
            "condition_budgets", "checkpoint_policy", "results_evidence_path",
            "results_evidence_records",
            "environment_requirements", "license_path", "notice_path",
            "figure_status_path", "primary_metric_file",
            "primary_checkpoint_sha256",
        }
        self.assertTrue(required.issubset(index))
        self.assertEqual(index["checkpoint_policy"], {
            "primary": "last.pt", "diagnostic": "best.pt",
        })
        self.assertTrue(index["results_evidence_path"].endswith(".jsonl"))
        self.assertTrue(index["results_evidence_records"])
        for record in index["results_evidence_records"]:
            self.assertTrue(record["hostname"])
            self.assertTrue(record["gpu"])
            self.assertEqual(len(record["artifact_sha256"]), 64)
            self.assertEqual(record["primary_metric_file"],
                             "metrics_ap40_primary_last.json")
            self.assertEqual(len(record["primary_checkpoint_sha256"]), 64)
        load_and_verify_package(root)

    def test_transfer_datasets_are_not_marked_verified_without_evidence(self):
        index = json.loads(Path(
            "docs/reports/factor-repair-authoritative-index.json"
        ).read_text())
        for name in ("BDD100K", "CityPersons"):
            self.assertNotEqual(index.get("external_evidence", {}).get(name), "verified")
```

- [ ] **Step 2: Run the test and verify the package is not silently accepted when incomplete**

Run: `D:\ana\envs\yolo\python.exe -m unittest tests.test_factor_repair_repro_package -v`

Expected: the test fails until every required index field, machine-bound result
record, and 64-hex artifact hash is present; it must never infer verification
from a filename or a non-empty directory.

- [ ] **Step 3: Write the authoritative index and machine-bound evidence**

`factor-repair-authoritative-index.json` must enumerate the implementation
commit, every source/config path and SHA256, split and metadata hashes,
initialization checkpoint hash, all M/F budgets, the fixed checkpoint policy
(`last.pt` primary and `best.pt` diagnostic), the exact
`results_evidence_path` JSONL path and its `results_evidence_records`,
environment/requirements path, LICENSE/NOTICE paths, and figure-status path.
Each line of `factor-repair-results-evidence.jsonl` records condition,
seed, hostname, GPU model, CUDA/PyTorch versions, run directory, resolved
configuration hash, checkpoint filename and role, artifact SHA256, and clean
implementation commit. It also records
`primary_metric_file=metrics_ap40_primary_last.json` and the matching
`primary_checkpoint_sha256`;
the authoritative index repeats those fields. Missing, empty, or non-64-hex
hashes fail closed.

- [ ] **Step 4: Record figure status without overstating evidence**

`factor-repair-figure-status.md` labels the archived natural-factor negative
audit as `completed`, labels repaired calibration, replay, adaptation,
KITTI-test-server, BDD100K, and CityPersons figures as `planned` until their
machine-bound evidence records exist, and links each completed figure to its
authoritative index entry. The file must state explicitly that BDD100K and
CityPersons have not been validated by this package.

- [ ] **Step 5: Freeze environment and legal notices**

Run: `D:\ana\envs\yolo\python.exe -m pip freeze --all > requirements-factor-repair.txt`

The upstream canonical YAML is labeled AGPL-3.0. Task 13 therefore requires
an AGPL-3.0-compatible license path and a repository-owner confirmation before
writing a root `LICENSE`. If `license_owner_confirmation=true` is absent, the
verifier fails closed and no final reproducibility package is published; this
gate does not block Tasks 1-12. `NOTICE` is always written with the Ultralytics
8.4.98 version, each upstream file/path/hash/license, and the repository
attribution. Add `test_license_confirmation_gate`, which omits the confirmation
and expects a fail-closed verification result, then supplies it and expects the
license path to be accepted. `scripts/verify_factor_repair_package.py`
must resolve every path relative to the repository root, recompute all hashes,
validate the checkpoint policy and budget table, reject dirty or missing
evidence, and return exit code 0 only when the complete package is internally
consistent.

- [ ] **Step 6: Independent review and commit**

Run: `D:\ana\envs\yolo\python.exe scripts/verify_factor_repair_package.py --index docs/reports/factor-repair-authoritative-index.json`

Expected: `PACKAGE_OK` with the index SHA256, result-record count, and explicit
`BDD100K=planned CityPersons=planned` status. The independent reviewer signs
the report only after the command exits 0; no transfer-dataset validation claim
is added by this task.

```text
git add docs/reports/factor-repair-authoritative-index.json docs/reports/factor-repair-results-evidence.jsonl docs/reports/factor-repair-figure-status.md docs/reports/factor-repair-reproducibility-package.md requirements-factor-repair.txt LICENSE NOTICE scripts/verify_factor_repair_package.py tests/test_factor_repair_repro_package.py
git commit -m "docs: add independently verifiable factor repair package"
```

## Formal experiment sequence after implementation acceptance

1. Generate and commit the immutable metadata index and 90/10 development split.
2. Retrain the matched seed-17 protected development reference on fit IDs only.
3. Run M1, M2 and M3 for 60 matched epochs.
4. Run F0, F1, F2 and F3 calibration for 30 matched epochs.
5. Audit F0-F3; compute the registered four-endpoint composite and paired
   image-cluster `DeltaS` against F0, then select at most one F1-F3 repair
   without changing thresholds.
6. For F0 and the selected repair only, generate the immutable fit-image
   learned-factor manifests after calibration and before adaptation; run the
   matched 60-epoch task adaptation, then repeat the complete audit. If no
   repair passes, do not generate a candidate manifest or start Track F
   adaptation.
7. Freeze one valid recipe, then run formal seeds 17, 29 and 41 on all training IDs.
8. Produce paired bootstrap, stratified AP40, calibration, efficiency and failure-case evidence.
9. Archive all artifacts before any BDD100K transfer experiment.

## Self-review

- Spec coverage: target-conditioned definitions, object-balanced ROI loss, background specificity, leakage-free split, M1/M2/M3, F0-F3, relative selection, immutable learned replay, freeze rules, exact task adaptation, node gates, advancement, recovery, failure semantics, and independent reproducibility evidence all map to Tasks 1-13 (with Task 6A owning the matched adaptation phase).
- Completeness scan: every code-changing step names concrete interfaces, commands and expected behavior. Unknown content hashes are generated from completed manifests before the canonical YAML can be written, and Task 13 rejects missing machine-bound evidence.
- Type consistency: `DevelopmentSplit`, `FactorMetadataIndex`, `ReplayDistribution`, `LearnedFactorManifest`, `ValidatedMetadataPriorities`, `ObjectFactorTarget`, `FactorRepairGateDecision`, `FactorRepairSelectionDecision`, and phase/report interfaces keep the same names across producing and consuming tasks.
- Scope: no attention module, new backbone, new IoU variant, inference-graph change, validation-label modification, or BDD100K launch is included.

Plan completion condition: this document is approved, then Tasks 1-12 are executed with review after each task and Task 13 is completed by an independent reviewer. No new GPU experiment begins before Task 12 reports `READY`.
