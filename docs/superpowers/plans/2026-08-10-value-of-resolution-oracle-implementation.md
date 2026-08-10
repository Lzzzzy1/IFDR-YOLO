# Value-of-Resolution Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute this plan task-by-task.

**Goal:** Decide, without training a new selector, whether one selective high-resolution re-observation can plausibly improve a leakage-free P2 detector by at least 3.0 Moderate KITTI AP40 points.

**Architecture:** First create a plain P2 seed-17 reference trained only on the registered 3341-image fit split and evaluate its `last.pt` on the untouched 371-image development split. Then run two frozen, one-crop-per-image oracle levels with the same checkpoint: O1 may center a fixed 2x-resolution crop on an eligible ground-truth target; O2 may choose only from a candidate pool frozen from first-pass predictions and a fixed coarse grid. Both reuse the exact KITTI AP40 evaluator, deterministic classwise fusion, incremental checkpoints, and a second persistent mirror.

**Tech Stack:** Python 3.11, PyTorch/Ultralytics YOLOv8, NumPy, Pillow/OpenCV only where already required, `unittest`, existing IFDR provenance and KITTI AP40 modules.

---

## Frozen scientific rules

- Screening split: registered seed-17 fit/development split, `3341/371`, hashes `50a1f8d72b747d1e2f460a4e0a355469484d33dbf1e9928b64ba6594c1873362` and `b1b6b6ee7e5398e93868fab407a2e8a86a53c753667002ef9b8381734ef2cda8`.
- Reference: plain `models/kitti-p2-m.yaml`, pretrained `yolov8m.pt`, seed 17, 300 epochs, image size 640, and the existing baseline optimizer/prediction settings. Protected IFDR and historical full-3712 P2 checkpoints are forbidden.
- Reference score: the new fit-only P2 `last.pt` evaluated on the same 371 images. Historical `best.pt` or the published three-seed mean is context only and is not the oracle delta denominator.
- Crop geometry: preserve the source image aspect ratio; crop width and height are exactly one half of the original dimensions and are clamped to the image, producing 2x effective target resolution at the same detector input size.
- Eligible O1 targets: Car, Pedestrian, or Cyclist ground-truth boxes with pixel height below 40; each target defines one centered crop, but the selected output still uses at most one crop per image.
- O2 candidates: six fixed 3-by-2 grid windows plus at most eighteen unique windows centered on first-pass predictions with pixel height below 40 and confidence in `[0.001, 0.25]`; proposal candidates are ordered by box area, confidence, class, and coordinates. Ground truth may select among these frozen windows but may not create, move, or resize one.
- Fusion: map crop boxes back to original pixels, then deterministic per-class greedy NMS at IoU 0.70; order by descending confidence, base-pass before crop on exact ties, then coordinates; retain at most 300 detections.
- Oracle selection utility is frozen before results: `delta_TP + 0.25 * delta_mean_matched_IoU - 0.25 * delta_FP - 0.10 * delta_duplicate`; exact KITTI class IoU thresholds and Moderate eligibility/ignore rules apply. A non-positive best utility selects no crop.
- O1 advances only at `delta Moderate macro AP40 >= 3.0`. O2 advances only at `delta >= 5.0`, both Pedestrian and Cyclist small/distant slices move positively, no class loses more than 1.0 AP overall, and near/large AP loss is at most 0.5 per reported class.
- O1 failure stops Route A immediately. O2 failure stops selector training. Results are decision evidence, not a final three-seed claim.

## Task 1: Build the leakage-free plain-P2 reference job

**Files:**

- Create: `ifdr_yolo/experiments/p2_fit_reference.py`
- Create: `scripts/run_p2_fit_reference.py`
- Create: `tests/test_p2_fit_reference.py`
- Modify only if required for a fixed output directory: `ifdr_yolo/experiments/baseline.py`
- Modify only if required for power-loss adoption: `ifdr_yolo/experiments/baseline_recovery.py`
- Modify matching CLI tests when either existing module changes.

- [ ] Write RED tests that reject the 3712-image P2 split, any overlap, wrong split hash/count, a non-plain-P2 model, `best.pt` as primary reference, and an existing job with a different identity.
- [ ] Write RED tests for an interrupted fixed-output run: a live owner is rejected; a stale run with non-empty `last.pt` and contiguous `results.csv` is resumable; resume keeps the same job directory and identity.
- [ ] Implement the smallest preparation layer that copies the registered split manifests into the job, builds exact hard-link train/val views, writes a resolved data YAML/config/manifest atomically, and records input/model/pretrained/code/config hashes.
- [ ] Use the existing baseline trainer and exact evaluator; add only the minimum fixed-output or interrupted-run hook needed for deterministic recovery.
- [ ] Persist `status.json`, PID/hostname, current epoch, `results.csv`, `weights/last.pt`, elapsed time, next action, and checkpoint identity. Ultralytics epoch checkpoints satisfy the less-than-five-minute checkpoint interval; the wrapper must flush a progress record at least once per completed epoch.
- [ ] Before evaluation, read the actual Ultralytics `args.yaml` and training label cache, materialize the observed training image IDs, and prove they exactly equal the 3341 fit manifest and have zero intersection with the 371 development manifest. Missing/unreadable cache, a different data YAML, any missing/extra ID, or any development ID is an immediate NO-GO.
- [ ] Atomically publish `observed_train_ids.txt`, `post_training_leakage_audit.json`, and `checkpoint_provenance.json`; bind `last.pt` SHA/role, both split manifests, config/code identity, actual cache SHA, and post-audit SHA. Mirror these small artifacts before permitting evaluation or `complete`.
- [ ] Write the small manifest/status/recovery command to a required external sibling mirror before declaring a stage complete. Mirror failure must fail closed without deleting the primary artifacts.
- [ ] Run focused tests, `py_compile`, and `git diff --check`.

## Task 2: Prove interruption recovery and benchmark the reference run

**Files:**

- Modify: `tests/test_p2_fit_reference.py`
- Create runtime evidence under an ignored job directory only; do not commit data or weights.

- [ ] Run a local fake-runtime active interruption test and prove resumed output identity equals uninterrupted output.
- [ ] On the confirmed YOLO RTX 5090, run dry-run and a one-epoch smoke with the real 3341/371 manifests and a separate output directory.
- [ ] Intentionally stop only the smoke after a saved epoch, resume it in place, and verify checkpoint/config/split identity plus non-duplicated epochs.
- [ ] Measure real epoch time, GPU memory/utilization, data-loading time, and checkpoint interval; use this measurement for the 300-epoch ETA.
- [ ] Start the 300-epoch reference only if all recovery and identity gates pass. Report PID, log, run directory, mirror, checkpoint, GPU use, measured ETA, and the exact resume command summary.

## Task 3: Implement pure crop, mapping, fusion, and candidate logic

**Files:**

- Create: `ifdr_yolo/eval/resolution_oracle.py`
- Create: `tests/test_resolution_oracle.py`

- [ ] Write RED tests for fixed crop clamping, 2x geometry, crop-to-full box mapping, clipping, normalized YOLO round-trip, stable same-class NMS, cross-class preservation, base-first tie handling, and `max_det=300`.
- [ ] Write RED tests proving O2 candidates are completely frozen before GT access, contain at most six grid plus eighteen proposal windows, and never use GT to create/move/resize a window.
- [ ] Write RED tests for the one-crop budget, non-positive-utility no-crop choice, fixed utility coefficients, Moderate matching thresholds, and deterministic selection ties.
- [ ] Implement immutable dataclasses and pure functions only; reuse existing `Detection`, `BoundingBox`, image-size parsing, IoU, and KITTI label/evaluator semantics instead of adding a second evaluator.
- [ ] Run focused tests and `py_compile`.

## Task 4: Implement the resumable O1/O2 inference pipeline

**Files:**

- Create: `configs/experiments/oracles/kitti_p2_resolution_oracle_s17.yaml`
- Create: `scripts/run_resolution_oracle.py`
- Modify: `tests/test_resolution_oracle.py`

- [ ] Write RED tests for checkpoint SHA/role, plain-P2 model SHA, code/config/split identity, 371-image completeness, primary/mirror separation, and fail-closed resume on any identity mismatch.
- [ ] Write RED tests for per-image atomic journals, completed image IDs, next image, elapsed/ETA, candidate-pool SHA, prediction-directory completeness, and interrupted-versus-uninterrupted equivalence.
- [ ] Implement phases `base`, `o1_candidates`, `o1_select`, `o2_pool`, `o2_candidates`, `o2_select`, and `evaluate`; each image is the smallest recovery unit and is mirrored before the phase cursor advances.
- [ ] Materialize `candidate_pool.jsonl` and its SHA before any O2 GT selection. Save raw base/crop predictions, mapped predictions, selected crop, utility components, fused prediction text, and provenance per image.
- [ ] Call `UltralyticsAdapter.predict` and the existing exact `evaluate_prediction_directory`; do not use Ultralytics mAP as a result.
- [ ] Emit `base_metrics_ap40.json`, `o1_metrics_ap40.json`, `o2_metrics_ap40.json`, stratified no-harm tables, latency/compute tables, and one decision JSON containing the registered gates without post-hoc threshold changes.
- [ ] Run focused tests, active interruption/resume equivalence, `py_compile`, and `git diff --check`.

## Task 5: Sol review and full local gate

**Files:** all files changed by Tasks 1-4.

- [ ] Review only five highest-risk points: leakage/provenance, use of `last.pt`, O2 GT isolation, crop mapping/fusion determinism, and checkpoint/mirror transaction order.
- [ ] Run the focused suites for baseline, recovery, exact AP40, prediction I/O, and resolution oracle.
- [ ] Run the full suite in `D:\ana\envs\yolo\python.exe` with the worktree `YOLO_CONFIG_DIR` and process-scoped Git safe-directory setting.
- [ ] Require zero failures other than an explicitly documented pre-existing environment skip; run `git diff --check` and confirm no data, weights, runs, secrets, or private keys are tracked.
- [ ] Commit the verified code before any formal remote oracle job and record the commit plus file hashes in `CODEX_HANDOFF.md`.

## Task 6: Execute the staged decision experiment

**Files:** server job artifacts only; no code changes while results are running.

- [ ] Evaluate the completed fit-only P2 `last.pt` on the locked 371 images and freeze its base prediction directory and exact AP40 hash.
- [ ] Run O1 first with incremental checkpoints and a second persistent mirror. If O1 delta is below 3.0, write `route_a_decision.json` as STOP and do not materialize or run O2.
- [ ] If O1 passes, materialize the GT-free O2 candidate pool, freeze its SHA, benchmark the first 16 images, and launch the remaining resumable O2 inference only if the measured ETA/cost is acceptable.
- [ ] Apply the registered overall and no-harm gates exactly once. Save a compact archive containing config, identities, base/O1/O2 metrics, candidate pool, selection journal, stratified table, latency table, decision, and SHA256 manifest.
- [ ] Update `CODEX_HANDOFF.md` with the final decision and whether the server can be safely paused.

## Stop conditions

- Stop immediately on split overlap/hash mismatch, a checkpoint trained on the development IDs, a `best.pt`/`last.pt` role mismatch, or changed model/config/evaluator identity.
- Stop the performance route if O1 is below +3.0 or O2 is below +5.0; do not train a selector and do not attach IFDR, loss, attention, NWD, RFLA, or assignment changes to rescue the result.
- Do not inspect the public Chen validation result to tune crop geometry, candidates, utility coefficients, NMS, or thresholds.
- Do not claim the oracle itself as the final model. A positive O2 result authorizes a later learned value predictor and compute-matched three-seed experiment; it does not prove that experiment will pass.
