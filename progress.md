# Progress: Resumable factor bootstrap

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
