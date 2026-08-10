# Task Plan: Resumable factor bootstrap

## 2026-08-10 current gate: leakage-free P2 and resolution oracle

The project now advances one evidence gate at a time. A later gate cannot start without fresh raw evidence from the preceding gate. Any P0 stops the route; it does not trigger an automatic rerun or a rescue module.

| Gate | Success criterion | Required proof | Status |
|---|---|---|---|
| L0 local originality | Collision matrix rejects ordinary P2/crop/zoom/geometry/uncertainty as novelty and leaves one falsifiable narrow gap | `docs/research-notes/novelty-p2-20260810/NOVELTY_COLLISION_MATRIX.md`; frozen design spec | GO for F0/O1/O2 feasibility; F4 remains conditional |
| L1 local executable | O1/O2 gates both equal +2.0 AP points; focused tests, syntax check, and persistent synthetic smoke pass with input/config/code identity | `reports/local_initial_20260810/`; RED/GREEN logs; SHA manifest | GO: main verification 28/28, py_compile exit 0, smoke 1/1 STOP at +0.5 below +2.0 |
| R0 remote preflight | fit/development intersection=0; code/config/pretrained/split hashes fixed; real 1-epoch 640 benchmark; active interruption resumes equivalently; checkpoint <=5 min; independent mirror; measured ETA | remote job `status.json`, `results.csv`, `args.yaml`, cache audit, provenance, checkpoint, mirror manifest, benchmark log | Blocked until L1 passes and server is opened |
| R1 clean F0 | 300-epoch leakage-free plain P2 uses only 3,341 fit IDs; post-training cache audit again proves development intersection=0; `last.pt` and exact KITTI AP_R40 are non-empty and hashed | F0 weights, provenance, observed IDs, leakage audit, exact AP JSON | Pending R0 |
| R2 O1 oracle | Same 371 development manifest; O1-F0 >=+2.0 Pedestrian/Cyclist Moderate macro AP_R40 | base/O1 predictions, exact AP JSON, per-image journal, latency/budget, mirror | Pending R1; `<+2.0` is route NO-GO |
| R3 O2 oracle | Frozen GT-free candidate pool retains >=+2.0 AP and passes registered no-harm | candidate-pool SHA, O2 AP, strata/no-harm, latency/budget | Pending R2 |
| R4 F1-F4 | Same split/seed/budget/evaluator; plain P2 F0 must beat original P3-P5 by >=4.0 AP, full F4 must beat F0 by >=2.0 AP (target +2 to +3), and end-to-end F4 must beat original by >=6.0 AP (target +6 to +8); also report CI, far/small strata, FP, near/medium no-harm, latency and budget | raw JSON/CSV, hashes, three-seed confirmation before a performance claim | Conditional on R3 |

Originality wording is frozen to the collision matrix. P2, cropping, zooming, geometry priors, uncertainty routing, and expected coarse-versus-fine gain are prior art and cannot be presented as original.

## Goal

Repair only the execution lifecycle of the registered F0–F3 10,000-replicate paired bootstrap. Preserve the estimand, random schedule, four endpoints, CI, and selection rule.

## Phases

- [x] Root cause and approved design fixed.
- [x] RED tests for caching, checkpoint identity, resume equivalence, progress, and selection integration.
- [x] Implement atomic checkpoints, explicit resume, identity/RNG binding, progress/ETA, and shared F0 cache.
- [x] Implement transactional three-file finalization and independent exact-byte mirror.
- [x] Run representative 371-image performance benchmark and prove cached/legacy numerical equivalence.
- [x] Sol five-risk review, focused 55/55, full 813/813, explicit recovery demo 4/4, py_compile, and diff-check.
- [ ] Start the real-input preflight and formal 10,000-replicate run after the YOLO data-disk server is online.
- [ ] Minimal GREEN implementation by Luna/max. (Luna/max in progress)
- [ ] Focused and full regression plus performance/interruption evidence.
- [ ] One Sol final review and frozen server launch gate.

## Errors encountered

- Git rejected the worktree as dubious ownership under the current Windows identity. Resolution: use per-command `-c safe.directory=...`; do not change global Git configuration.
- An initial PowerShell file query used multiple values with `-Filter`. Resolution: use `Where-Object` for the bounded filename set.

## Skills

- `systematic-debugging`: root-cause and performance data flow.
- `test-driven-development`: mandatory RED→GREEN.
- `subagent-driven-development`: Luna/max implementation with Sol review.
- `using-git-worktrees`: reuse the existing isolated worktree.
- `writing-plans`: executable implementation plan.
- `verification-before-completion`: fresh evidence before completion claims.
- `planning-with-files`: persistent recovery state.
