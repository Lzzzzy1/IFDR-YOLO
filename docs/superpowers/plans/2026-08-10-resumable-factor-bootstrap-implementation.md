# Resumable Factor Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing 10,000-replicate F0-relative mechanism bootstrap fast, observable, checkpointed, and exactly resumable without changing its statistical definition.

**Architecture:** Add a focused resumable bootstrap runner that owns candidate checkpoint identity, deterministic replicate execution, progress, and finalization. Cache parsed evidence clusters once, reuse every F0 draw across all candidates, and refactor selection so precomputed paired deltas feed the unchanged eligibility/tie-breaking rule. Choose CPU process blocks or CUDA tensor batches only from measured equivalence, throughput, and memory evidence.

**Tech Stack:** Python 3.12, NumPy, dataclasses, atomic JSON, `concurrent.futures`, `unittest`.

---

### Task 1: Prepared evidence cache and pure replicate primitive

**Files:**
- Modify: `ifdr_yolo/eval/factor_repair_evidence.py`
- Modify: `ifdr_yolo/eval/factor_repair_gate.py`
- Test: `tests/test_factor_repair_evidence.py`
- Test: `tests/test_factor_repair_gate.py`

- [ ] Add RED tests proving repeated draws parse/group raw observations once and cached/legacy endpoint values match.
- [ ] Add a RED test for a single indexed candidate-minus-F0 replicate.
- [ ] Run the focused tests and record the expected failures.
- [ ] Implement one lazy immutable per-image observation cache and one indexed replicate helper using the registered draw schedule.
- [ ] Run the focused tests to GREEN and keep existing `paired_image_cluster_delta` behavior unchanged.

### Task 2: Atomic checkpoint identity and exact resume

**Files:**
- Create: `ifdr_yolo/eval/resumable_factor_bootstrap.py`
- Test: `tests/test_resumable_factor_bootstrap.py`

- [ ] Add RED tests for periodic atomic checkpoints, explicit resume, completed replicate/value consistency, and fail-closed evidence/image/code/config/RNG mismatches.
- [ ] Add a RED interruption test: interrupt after a non-boundary replicate, resume, and compare the complete replicate vector, point, endpoints, and linear 95% CI with an uninterrupted run.
- [ ] Run the new test module and record expected failures.
- [ ] Implement immutable run identity, atomic checkpoint replacement, deterministic RNG schedule state, block execution, resume validation, and final `PairedDelta` construction.
- [ ] Enforce a maximum 300-second durable-save interval independent of replicate count, with `last_saved_at`, cumulative elapsed time, completed range, and next replicate in every checkpoint.
- [ ] Run the new module to GREEN.

### Task 3: Progress, ETA, precomputed selection, and candidate concurrency

**Files:**
- Modify: `ifdr_yolo/eval/factor_repair_gate.py`
- Modify: `scripts/select_factor_repair_evidence.py`
- Test: `tests/test_factor_repair_gate.py`
- Test: `tests/test_select_factor_repair_evidence_cli.py`

- [ ] Add RED tests for progress fields, checkpoint paths, explicit `--resume`, create-once final outputs, and identity-safe reuse of complete checkpoints.
- [ ] Add RED tests for the five-minute forced-save rule, per-unit flushed logs, and atomic milestone mirroring to a second persistent directory.
- [ ] Add RED tests proving precomputed deltas use the same eligibility/tie-break rule as `select_repair_against_f0` and sequential/parallel candidate execution yields identical results.
- [ ] Run the focused tests and record expected failures.
- [ ] Refactor the selector into a pure precomputed-delta path shared by the legacy API.
- [ ] Update the CLI to run F1–F3 through independent checkpointed candidates, emit atomic progress, and finalize only after all candidates complete.
- [ ] Keep the formal CLI fixed at 10,000 replicates; expose only checkpoint interval, worker count, checkpoint directory, required persistent mirror directory, and `--resume` as execution controls.
- [ ] Run the focused tests to GREEN.

### Task 4: Verification and performance gate

**Files:**
- Create or modify only focused benchmark/test support under `tests/` or `scripts/` if required; no production behavior expansion.

- [ ] Run `py_compile` for every changed Python file.
- [ ] Run focused factor evidence/gate/selection/resume tests.
- [ ] Run the full `unittest` regression suite.
- [ ] Run a small deterministic benchmark that reports cold-load, cached replicate rate, projected 10,000-replicate time per candidate, and three-candidate wall time at the selected worker count.
- [ ] Profile call counts and hot paths to prove raw mapping conversion, `by_image` grouping, draw-index construction, and F0 endpoint work are not repeated unnecessarily.
- [ ] Compare NumPy single-process, safe NumPy process blocks, and locally executable PyTorch CUDA batches on identical deterministic draws; report numerical equivalence, throughput, peak host memory, and peak VRAM. Record CuPy as unavailable rather than installing it when absent.
- [ ] Demonstrate interruption and resume in a temporary directory and record checkpoint path, completed count before interruption, completed count after resume, and result equality.
- [ ] Run `git diff --check`, inspect `git status`, and perform one Sol specification/quality review.
- [ ] Freeze the formal server command, checkpoint paths, resume command, runtime range, and exact server-opening gate in `CODEX_HANDOFF.md`.

## Acceptance gate

The server remains off until every focused and full test passes, interruption/resume equality is demonstrated, the benchmark produces a bounded estimate, all identity mismatches fail closed, and Sol confirms the immutable scientific contract is unchanged.
