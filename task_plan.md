# Task Plan: Resumable factor bootstrap

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
