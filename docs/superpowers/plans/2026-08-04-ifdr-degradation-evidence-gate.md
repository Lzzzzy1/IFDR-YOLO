# IFDR Degradation Evidence Gate Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task by task. Do not start a new GPU experiment until the evidence gates below pass and the current protected/unprotected queue is complete.

**Goal:** Establish whether IFDR's learned factors reflect natural KITTI degradation and whether hard Cyclist instances receive insufficient positive assignments, before changing replay or label assignment.

**Architecture:** Add a read-only natural-degradation audit around the existing reliability context, then add opt-in assignment instrumentation after Ultralytics' parent assigner. Both paths must preserve training outputs exactly when diagnostics are disabled. Their reports become the gate for the later degradation-conditioned conservative replay experiment; dynamic assignment remains prohibited unless its gate passes.

**Tech stack:** Python 3.12, PyTorch, Ultralytics 8.4.98, NumPy, unittest, JSON/JSONL, existing KITTI metadata and IFDR checkpoints. Implement average-tie ranks and Spearman correlation in the project to avoid adding SciPy solely for this audit.

**Success criteria:**

- Natural-factor audit is deterministic, resumable, validation-leakage free, covers seeds 17/29/41 and all six reliability nodes, and produces correlations with image-bootstrap 95% confidence intervals.
- Assignment diagnostics are opt-in and produce bitwise-identical loss/assignment outputs when observation is enabled.
- Dynamic assignment is authorized only if at least 100 matched Cyclist objects show a significant hard-vs-easy positive-assignment deficit.
- Existing tests and new tests all pass; no running server process or existing result is modified.

**Out of scope for this plan:** implementing replay, changing sampling probability, changing assigner behavior, BDD100K training, or launching GPU work. Those are separate stages gated by this evidence.

---

## Task 1: Define natural degradation targets from KITTI metadata

**Files:**

- Create: `ifdr_yolo/data/natural_degradation.py`
- Test: `tests/test_natural_degradation.py`

**Step 1: Write failing tests**

Cover the following exact behavior:

- A 64-pixel, 15-metre, fully visible object maps to sampling `0.0` and visibility `0.0`.
- A 4-pixel, 60-metre object maps to sampling `1.0`.
- KITTI occlusion level 3 and truncation 1.0 map to visibility `1.0`.
- Intermediate scores are monotonic and clipped to `[0, 1]`.
- JSONL parsing validates every metadata field consumed by this audit and skips canonical non-training KITTI classes with explicit counts; KITTI sentinel values such as `DontCare` truncation/occlusion `-1` and location `-1000` remain valid only for skipped classes. It rejects unknown class names, non-positive or non-finite derived boxes, non-finite consumed metadata, and invalid training-class occlusion/truncation. A finite non-positive training depth is retained as unavailable and counted, because its height/visibility metadata remain usable.

Run:

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_natural_degradation -v
```

Expected: import failure because the module does not exist.

**Step 2: Implement the minimal module**

Add frozen `NaturalDegradationRecord` with image id, object id, class id/name, bounding box, height, optional depth, occlusion, truncation, sampling score, and visibility score.

Use the preregistered equations:

```python
height_score = clip((64.0 - box_height) / 60.0, 0.0, 1.0)
depth_score = clip((depth_m - 15.0) / 45.0, 0.0, 1.0)
sampling = 1.0 - (1.0 - height_score) * (1.0 - depth_score)
occlusion_score = occlusion_level / 3.0
visibility = 1.0 - (1.0 - occlusion_score) * (1.0 - truncation)
```

If depth is absent or finite but non-positive, set `depth_m=None`, `depth_score=0.0`, and `depth_available=False`; increment `invalid_depth_count` for the latter and never take an absolute value or silently impute distance. Support only Car, Pedestrian, and Cyclist in the audit output. Canonical KITTI non-training classes (`Van`, `Truck`, `Person_sitting`, `Tram`, `Misc`, and `DontCare`) receive structural and finite-value validation but are skipped before training-class range checks; any other class name fails validation. Derive a stable per-image object id from original metadata row order when no explicit object id is present, and reject any explicit/implicit identifier collision within an image.

**Step 3: Verify and commit**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_natural_degradation -v
git add ifdr_yolo/data/natural_degradation.py tests/test_natural_degradation.py
git commit -m "feat: define natural KITTI degradation targets"
```

---

## Task 2: Compute factor alignment and controlled monotonicity

**Files:**

- Create: `ifdr_yolo/eval/natural_factor_audit.py`
- Test: `tests/test_natural_factor_audit.py`

**Step 1: Write failing tests with explicit fixtures**

Construct explicit observations for seeds 17/29/41 and six node ids where predicted sampling increases with natural sampling, predicted visibility increases with natural visibility, and the two natural targets are not identical. Include matched object/background interventions at several severities. Test:

- Spearman correlations have the expected direction.
- Sampling residual correlation with depth controls box height and class; visibility residual correlation controls box height and class.
- Controlled monotonicity compares upper and lower target tertiles within bins of the control factor.
- Intervention response is ordered by severity and is stronger in the target ROI than in its matched background ROI.
- Image-level bootstrap is deterministic for a fixed seed and resamples image ids rather than individual boxes.
- A reversed factor fails; a correct factor whose confidence interval crosses zero fails; a factor that passes only one seed or one reliability node fails the stability gate.

Run the new test and confirm it fails on import.

**Step 2: Implement the audit core**

Add frozen `NaturalFactorObservation` and `NaturalFactorGateDecision`. Each observation records seed, reliability node, image/object id, class, region role, intervention kind/severity, natural targets, predicted factors, and branch weights. Implement ROI mean pooling from factor maps using the letterboxed bounding box. Compute:

- raw Spearman `rho` for sampling and visibility;
- residual Spearman using class indicators and box height as registered controls;
- controlled monotonicity success rate across control-factor quartiles;
- paired object-minus-background response and intervention-severity monotonicity;
- 2,000 image-cluster bootstrap replicates with seed `20260804` and percentile 95% intervals. Average-tie ranks are frozen on the original audit sample before resampling. Raw Spearman intervals use per-image rank cross-moment sufficient statistics; residual intervals use per-image ranked-control cross-products and refit the registered weighted least-squares controls in every replicate. Multinomial image weights reproduce sampling source images with replacement while keeping every object, seed, and reliability node from the same source image in one cluster. This avoids rebuilding and sorting hundreds of thousands of rows 2,000 times without changing the registered image-level sampling unit.

Clean target/background rows are the internal intervention manifest and declare whether a pair audits sampling or visibility. Registered severities are `(0.25, 0.50, 0.75, 1.0)`; missing, extra, or duplicated rows make the pair malformed and block the gate. Task 3/4 own detection of an entirely absent pair, including its clean manifest, through the observer plan and expected-row progress manifest.

The gate passes a factor only when the expected direction holds for every seed, the pooled raw and residual confidence-interval lower bounds exceed zero, target response exceeds matched background response, and controlled intervention severity is ordered for at least `0.80` of eligible pairs. Report all six nodes separately and a pooled result; a single unstable seed blocks the pooled pass. Do not tune these thresholds after observing validation performance.

**Step 3: Verify and commit**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_natural_factor_audit -v
git add ifdr_yolo/eval/natural_factor_audit.py tests/test_natural_factor_audit.py
git commit -m "feat: add natural factor alignment gate"
```

---

## Task 3: Observe learned factors from an IFDR checkpoint

**Files:**

- Create: `ifdr_yolo/eval/factor_observer.py`
- Test: `tests/test_factor_observer.py`

**Step 1: Write failing tests**

Use a tiny fake model exposing six entries from `consume_reliability_context()` and a temporary two-image dataset. Verify:

- checkpoint loading accepts project checkpoints containing `ema` or `model` and rejects checkpoints without an IFDR reliability context;
- deterministic letterbox maps boxes to factor-map coordinates correctly;
- every image is forwarded exactly once;
- output JSONL contains seed, reliability node, image/object ids, class, natural targets, region role, intervention metadata, predicted factor means, branch weights, source hashes, and checkpoint hash;
- natural images and the existing deterministic matched object/background intervention pairs use the same letterbox and ROI mapping path;
- restarting skips already committed image ids and never duplicates rows;
- progress and JSONL are flushed and `fsync`-ed after each image.

**Step 2: Implement the observer**

Load the checkpoint with `torch.load`, prefer `ema` over `model`, set evaluation mode, run direct model inference, and immediately consume its six-node reliability context. Pool the sampling and visibility maps over each natural object box. In intervention mode, use the existing deterministic sampler/transform to create matched object and background interventions at registered severities and pool the geometrically corresponding ROIs. Write one JSONL row per object/node/condition and an atomic progress JSON containing completed `(seed, image_id, condition)` keys plus input/checkpoint hashes.

Do not use predictor post-processing because this audit measures latent factors, not detections.

**Step 3: Verify and commit**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_factor_observer -v
git add ifdr_yolo/eval/factor_observer.py tests/test_factor_observer.py
git commit -m "feat: observe IFDR factors on natural objects"
```

---

## Task 4: Add a reproducible natural-factor audit CLI

**Files:**

- Create: `scripts/audit_natural_factors.py`
- Test: `tests/test_audit_natural_factors_cli.py`

**Step 1: Write failing CLI tests**

Build temporary train/validation id files and three fake checkpoints and assert that the CLI helper:

- selects `floor(0.20 * N)` training ids by sorting `sha256(seed + image_id)`, with a minimum of one id for non-empty input;
- selects no validation id and raises on any train/validation overlap;
- reproduces the same selection independent of input ordering;
- requires exactly the registered seeds 17, 29, and 41 and refuses duplicate/missing seeds;
- refuses an output directory whose provenance names different checkpoints, intervention manifest, or split hash;
- writes non-empty `observations.jsonl`, `summary.json`, `gate.json`, `provenance.json`, and `status.json`.

**Step 2: Implement orchestration**

The CLI must perform these operations in order: validate all paths and hashes; construct or resume the deterministic audit split; record `running` status; invoke `factor_observer`; invoke `natural_factor_audit`; atomically write summary, gate, and provenance; verify all five output files are non-empty; then record `complete`. On exception, preserve outputs and record `failed` plus the exception type and message.

Expose repeatable `--checkpoint SEED=PATH` arguments, KITTI metadata JSONL, train ids, validation ids, output directory, image size, bootstrap replicates, audit seed, and the registered intervention severities. Defaults are image size 640, 2,000 replicates, audit seed 20260804, and severities `0.25,0.50,0.75,1.00` for both sampling and visibility interventions.

**Step 3: Verify and commit**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_audit_natural_factors_cli -v
git add scripts/audit_natural_factors.py tests/test_audit_natural_factors_cli.py
git commit -m "feat: add resumable natural factor audit"
```

---

## Task 5: Instrument assignment without changing training

**Files:**

- Create: `ifdr_yolo/experiments/assignment_diagnostics.py`
- Modify: `ifdr_yolo/config.py`
- Modify: `ifdr_yolo/runtime.py`
- Modify: `ifdr_yolo/models/ifdr_model.py`
- Modify: `ifdr_yolo/losses/ifdr_loss.py`
- Modify: `ifdr_yolo/trainers/ifdr_trainer.py`
- Test: `tests/test_assignment_diagnostics.py`
- Test: `tests/test_ifdr_loss.py`
- Test: `tests/test_ifdr_trainer.py`

**Step 1: Write failing tests**

Add `assignment_diagnostic_interval: int = 0` to the strict method config. Test rejection of negative values. With fixed tensors and RNG state, compare diagnostics disabled (`0`) and enabled (`1`) and require identical returned scalar loss, detached loss items, `fg_mask`, target indices, and model gradients. Also verify the diagnostic record includes step, image indices, feature-map shapes, positive-anchor counts, matched GT indices, class ids, and detached matching quality when available.

Test append-safe flushing: two flushes append two valid JSONL rows, and a reconstructed trainer continues rather than overwrites.

**Step 2: Implement post-assignment observation**

Call the existing Ultralytics parent assignment first. Pass detached copies of `fg_mask`, `target_gt_idx`, target boxes, anchor points, stride tensor, class ids, and feature shapes to a recorder only on configured steps. The recorder may count and serialize; it must not return replacement tensors or participate in autograd.

Mirror the existing gradient-diagnostics buffer/flush lifecycle in the trainer and store records under each run directory as `assignment_diagnostics.jsonl`. Keep the default interval at zero, so all historical configurations remain unchanged.

**Step 3: Run focused and regression tests**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_assignment_diagnostics tests.test_ifdr_loss tests.test_ifdr_trainer -v
```

Expected: all pass, including the exact invariance assertions.

**Step 4: Commit**

```powershell
git add ifdr_yolo/experiments/assignment_diagnostics.py ifdr_yolo/config.py ifdr_yolo/runtime.py ifdr_yolo/models/ifdr_model.py ifdr_yolo/losses/ifdr_loss.py ifdr_yolo/trainers/ifdr_trainer.py tests/test_assignment_diagnostics.py tests/test_ifdr_loss.py tests/test_ifdr_trainer.py
git commit -m "feat: add transparent assignment diagnostics"
```

---

## Task 6: Gate any dynamic-assignment experiment

**Files:**

- Create: `ifdr_yolo/eval/assignment_coverage.py`
- Create: `scripts/summarize_assignment_coverage.py`
- Test: `tests/test_assignment_coverage.py`

**Step 1: Write failing tests**

Create explicit easy/hard Car, Pedestrian, and Cyclist records. Verify:

- diagnostic GT rows join metadata only when image id and class match and box IoU is at least `0.90`;
- multiple qualifying metadata boxes are marked ambiguous and excluded;
- hardness strata use preregistered natural sampling/visibility scores, not model confidence;
- image-bootstrap intervals are deterministic;
- the gate fails below 100 matched Cyclists;
- the gate passes only when hard-Cyclist mean positive-anchor count is lower than easy-Cyclist count and the 95% interval of their difference excludes zero;
- Car/Pedestrian counts are reported as safeguards rather than hidden.

**Step 2: Implement summary and gate**

Produce `assignment_coverage.json` and `assignment_gate.json` containing match/exclusion counts, per-class/per-stratum positive anchors, branch/scale distribution, bootstrap intervals, thresholds, and a machine-readable decision. The only pass decision is `allow_dynamic_assignment`; all failures explicitly return `keep_standard_assignment` with reasons.

**Step 3: Verify and commit**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_assignment_coverage -v
git add ifdr_yolo/eval/assignment_coverage.py scripts/summarize_assignment_coverage.py tests/test_assignment_coverage.py
git commit -m "feat: gate dynamic assignment with coverage evidence"
```

---

## Task 7: Full verification and handoff gate

**Files:**

- Modify only if required by new public modules: `ifdr_yolo/data/__init__.py`, `ifdr_yolo/eval/__init__.py`, `ifdr_yolo/experiments/__init__.py`

**Step 1: Run the full local suite**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest discover -s tests -v
```

Expected: all existing and new tests pass.

**Step 2: Run static import and CLI checks**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m compileall -q ifdr_yolo scripts
& 'D:\ana\envs\yolo\python.exe' scripts/audit_natural_factors.py --help
& 'D:\ana\envs\yolo\python.exe' scripts/summarize_assignment_coverage.py --help
```

Expected: exit code 0 for all three commands.

**Step 3: Review scope and repository state**

```powershell
git diff --check
git status --short
git log --oneline -8
```

Confirm that `KITTI_YOLOv8m_Results/` remains untouched and that no server run path, checkpoint, or result archive is modified.

**Step 4: Evidence-based next decision**

- If natural-factor gate passes: authorize a separate plan for degradation-conditioned conservative replay with a standard-replay matched control.
- If it fails: first calibrate natural alignment; do not call the current factors degradation-aware.
- If assignment gate passes: permit one controlled dynamic-assignment experiment after replay testing.
- If it fails: retain the standard assigner; do not add a label-assignment module.

No GPU experiment starts until the existing queue is complete, its artifacts are verified, and the applicable gate is recorded in a non-empty JSON result.
