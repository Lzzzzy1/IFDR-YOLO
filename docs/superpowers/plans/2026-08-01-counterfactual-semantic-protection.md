# Counterfactual Semantic Protection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add intervention-paired factor-delta learning and protect shared degradation semantics from fusion and localization gradients.

**Architecture:** The dataset emits a clean counterfactual view plus dense factor-delta supervision for each existing controlled intervention. A shared semantic anchor is consumed through detached, zero-residual fusion and localization adapters, while absolute and counterfactual factor losses remain able to train the anchor.

**Tech Stack:** Python 3.11, PyTorch, Ultralytics 8.4.98, NumPy, OpenCV, unittest/pytest, YAML.

---

### Task 1: Counterfactual supervision contract

**Files:**
- Modify: `ifdr_yolo/data/ifdr_dataset.py`
- Test: `tests/test_ifdr_data.py`

- [ ] **Step 1: Write failing transform tests**

Assert that an enabled non-identity intervention emits a clean CHW counterfactual image, a two-channel delta target, and a two-channel weight. Assert that the affected factor has a positive delta, the other factor has zero delta, and disabled/no-object paths emit zero weights.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_ifdr_data.py -q`
Expected: FAIL because the counterfactual keys do not exist.

- [ ] **Step 3: Implement the minimal data contract**

Add constants for `ifdr_counterfactual_img`, `ifdr_counterfactual_delta`, and `ifdr_counterfactual_weight`. Preserve the pre-intervention image as RGB CHW uint8, derive dense delta targets from the known intervention, and stack all three fields in `collate_ifdr_batch`.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_ifdr_data.py tests/test_intervention_transforms.py -q`
Expected: PASS.

### Task 2: Protected semantic adapters

**Files:**
- Modify: `ifdr_yolo/models/gated_fusion.py`
- Modify: `ifdr_yolo/models/ifdr_model.py`
- Test: `tests/test_reliability_gated_concat.py`
- Test: `tests/test_ifdr_model.py`

- [ ] **Step 1: Write failing adapter and gradient tests**

Assert zero-residual adapters preserve the initial router input, routed detection updates the fusion adapter but not the shared semantic anchor, localization adaptation updates its adapter but not input factor tensors, and the unprotected path preserves legacy gradient flow.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_reliability_gated_concat.py tests/test_ifdr_model.py -q`
Expected: FAIL because protection and adapters are not implemented.

- [ ] **Step 3: Implement minimal adapters**

Add a zero-initialized residual 1x1 map adapter for fusion and a zero-initialized residual factor adapter for localization. Detach anchor tensors only when semantic protection is enabled. Register the localization adapter on `IFDRDetectionModel` and expose `adapt_localization_factors`.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_reliability_gated_concat.py tests/test_ifdr_model.py -q`
Expected: PASS.

### Task 3: Counterfactual loss and protected localization

**Files:**
- Modify: `ifdr_yolo/losses/ifdr_detection.py`
- Test: `tests/test_ifdr_detection_loss.py`

- [ ] **Step 1: Write failing loss tests**

Assert multiscale delta loss is zero for exact predicted deltas, positive for incorrect deltas, rejects misaligned contexts, and sends gradients to both semantic views. Assert protected localization calibration updates the localization adapter without updating raw anchor factors.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_ifdr_detection_loss.py -q`
Expected: FAIL because counterfactual loss is missing.

- [ ] **Step 3: Implement and integrate the loss**

Consume main contexts, run one tensor-only forward on the clean paired view when counterfactual weight is nonzero, consume paired contexts, compute the six-node delta loss, and add it with its configured gain. Pass flattened main factors through `model.adapt_localization_factors` before deriving DCLI uncertainty.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_ifdr_detection_loss.py -q`
Expected: PASS.

### Task 4: Strict configuration and trainer preprocessing

**Files:**
- Modify: `ifdr_yolo/experiments/config.py`
- Modify: `ifdr_yolo/experiments/ifdr_runtime.py`
- Modify: `ifdr_yolo/experiments/ifdr_trainer.py`
- Create: `configs/experiments/mechanisms/kitti_ifdr_protected_cf_s17.yaml`
- Test: `tests/test_ifdr_config.py`
- Test: `tests/test_ifdr_runtime.py`
- Test: `tests/test_ifdr_trainer.py`

- [ ] **Step 1: Write failing configuration/runtime tests**

Assert optional legacy defaults are disabled, the new configuration enables semantic protection and counterfactual consistency, invalid gain/switch combinations are rejected, runtime forwards all controls, and trainer normalizes the paired image exactly like the main image.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_ifdr_config.py tests/test_ifdr_runtime.py tests/test_ifdr_trainer.py -q`
Expected: FAIL because the new controls are absent.

- [ ] **Step 3: Implement strict controls**

Add `semantic_protection` and `counterfactual_consistency` component switches plus `counterfactual_gain`. Keep absent fields disabled for legacy files, reject unknown fields, require interventions and factor supervision when consistency is enabled, forward controls through runtime/model/trainer, and normalize the paired image in `preprocess_batch`.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_ifdr_config.py tests/test_ifdr_runtime.py tests/test_ifdr_trainer.py -q`
Expected: PASS.

### Task 5: Regression and smoke verification

**Files:**
- Modify only files implicated by a reproduced failure.

- [ ] **Step 1: Run the focused suite**

Run: `pytest tests/test_ifdr_data.py tests/test_reliability_gated_concat.py tests/test_ifdr_model.py tests/test_ifdr_detection_loss.py tests/test_ifdr_config.py tests/test_ifdr_runtime.py tests/test_ifdr_trainer.py -q`
Expected: PASS.

- [ ] **Step 2: Run the full suite**

Run: `pytest -q`
Expected: all tests PASS.

- [ ] **Step 3: Run one-epoch smoke training on the server**

Run: `python scripts/train_ifdr.py --config configs/experiments/mechanisms/kitti_ifdr_protected_cf_s17.yaml --dry-run`, followed by the project smoke-data command and one epoch.
Expected: finite losses, counterfactual auxiliary loss present, `results.csv`, `last.pt`, and `best.pt` created.

- [ ] **Step 4: Record verification evidence**

Record the exact test count, smoke run directory, source commit, configuration hash, parameter delta, and peak GPU memory before launching a 60-90 epoch mechanism screen.
