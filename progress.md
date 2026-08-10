# Progress: Resumable factor bootstrap

## 2026-08-10 local P2/oracle gate

- Fresh novelty evidence is on disk. The broad value-of-resolution claim is NO-GO; the narrow benefit/harm plus calibrated abstention formulation is provisionally defensible, but F4 coding is blocked until O1 and O2 each retain at least +2.0 AP_R40 and pass no-harm.
- Focused local baseline evidence: 28 tests passed; direct syntax checks passed. Raw logs and command manifests are under `E:/myyolo/kitti_project/reports/local_initial_20260810/`.
- Persistent synthetic smoke ran end to end and produced a 66-entry SHA manifest with zero missing/mismatched files. It proves only execution, identity rejection, and STOP behavior; it is not an AP result.
- The smoke exposed a threshold mismatch. An intermediate `+1/+1` interpretation produced genuine RED/GREEN logs, but the user explicitly superseded it before release. The current frozen gates are O1 `+2` and O2 `+2`; prior `+1` artifacts are process evidence only and cannot authorize a server run.
- Current performance targets are separate end-to-end measurements, not added claims: plain P2 F0 versus original P3-P5 `>=+4.0` AP; full F4 versus F0 `>=+2.0` AP with a +2 to +3 target; full F4 versus original `>=+6.0` AP with a +6 to +8 target.
- The minimal `+2/+2` correction passed RED-to-GREEN. Main-agent fresh verification: 28/28 focused tests in 15.723 seconds; four-file `py_compile` exit 0; persistent synthetic smoke 1/1 with O1 delta `+0.5`, deterministic `STOP`, reason `O1 below registered +2.0 gate`, and no O2 pool. The smoke is a runnability/safety result, not AP improvement.
- L1 is GO. The next and only authorized gate is the real RTX 5090 one-epoch clean-P2 preflight with interruption/resume, zero-overlap audit, mirror growth, and measured ETA; no 300-epoch run starts until that preflight passes.
- Each following phase must publish its success criterion and proof path before execution. Runtime health requires log/checkpoint/output growth, not PID alone. Final results must include absolute AP_R40 points, uncertainty, strata, no-harm, false positives, latency, and budget; negative and zero results remain in the record.

## 2026-08-10

- Final Sol gate passed for timing/ETA, source identity, fixed 10,000 replicates, resume equivalence, transactional final output, and the independent exact-byte mirror.
- Fresh main-agent verification: focused 55/55, full 813/813 (skipped=1), explicit recovery demo 4/4, py_compile and diff-check all passed.
- Representative 371-image/62,328-row benchmark: cached/legacy max abs diff 8.33e-17, peak RSS 322.94 MB, local linear projection 42.6 minutes for F0+three candidates at 10k. This remains a projection pending the real 93 MB evidence preflight.
- Formal execution was not started because the sole YOLO server endpoint was still off and the four raw evidence files are not local. No remote write occurred.

- Read the latest handoff and required workflow skills.
- Confirmed the existing linked worktree `feature/degradation-evidence-gate` is clean.
- Traced the interrupted path from the selection CLI through `paired_image_cluster_delta` into raw evidence recomputation.
- Wrote the approved design and implementation plan.
- Baseline focused suite before changes: 38/38 passed in 1.765 seconds.
- Dispatched the bounded RED→GREEN implementation to Luna/max with exclusive ownership of the four production modules and their focused tests; no server use.
- Luna produced the required RED evidence: the cache test observed 1,512 parses instead of 504 across three draws, the indexed helper was absent, and the resumable runner tests failed because the module did not exist (five tests: one failure, four errors).
- Local backend inventory: PyTorch 2.5.1+cu121, CUDA available on one RTX 3060 Laptop GPU; NumPy available; CuPy absent. The benchmark will therefore compare NumPy CPU with PyTorch CUDA and record CuPy as unavailable rather than installing it.
- Extended the approved specification and plan with explicit F0 reuse, one-time grouping, no Python object copies in the hot loop, and CPU/GPU equivalence/throughput/memory gates.
- Next: wait for GREEN implementation, then run fresh regression, interruption/resume demonstration, representative profiling, and one Sol specification/quality review.
- First implementation gate passed 48 focused tests and 806 full tests, but Sol review rejected server launch: persisted wall time polluted resume ETA, formal byte-level input identity and transactional finalization were incomplete, the mirror was not a mandatory independent location, `workers` was accepted but unused, and the 4-image benchmark was not representative enough for a formal ETA.
- A bounded Luna correction is in progress with 371-image/roughly 50k-70k-row local profiling and a required real-input preflight before formal server execution.
