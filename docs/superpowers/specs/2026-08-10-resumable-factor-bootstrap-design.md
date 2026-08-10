# Resumable Factor Bootstrap Design

## Status and authority

This specification is the approved repair for the interrupted F0–F3 development selection statistic. It does not authorize any new GPU experiment. The formal statistic remains a 10,000-replicate paired image-cluster bootstrap over the registered four mechanism endpoints.

## Root cause

The interrupted command was scientifically deterministic but operationally non-resilient:

1. `paired_image_cluster_delta` held every replicate only in memory and returned after all 10,000 replicates.
2. `select_factor_repair_evidence.py` created the three final artifacts only after every candidate statistic completed.
3. No intermediate checkpoint, progress record, or resume identity existed. A shutdown therefore discarded all in-memory replicate values.
4. `FactorRepairEvidenceBundle.recompute_endpoints` reparsed and regrouped the complete raw observation payload for every replicate. With about 93 MB of evidence per condition, this dominated runtime.
5. F1, F2, and F3 were evaluated sequentially. The independent candidate statistics did not use the available CPU cores.

The 300-epoch reference, F0–F3 calibration runs, exact KITTI AP40, checkpoints, observer rows, absolute gates, and raw factor evidence are complete and remain authoritative.

## Immutable scientific contract

The repair must not change any of the following:

- reference condition `F0` and candidates `F1`, `F2`, `F3`;
- development image identity and paired image-cluster sampling unit;
- 10,000 formal replicates;
- bootstrap seed `20260805` and per-replicate SHA-256-derived draw schedule;
- candidate-minus-F0 composite estimand;
- the four registered endpoints and their equal-weight composite;
- percentile interval `(0.025, 0.975)` with NumPy linear quantiles;
- absolute-gate eligibility and selection tie-breaking rules;
- final `selection_decision.json`, `mechanism_table.json`, and `mechanism_table.csv` semantics.

Small replicate counts are permitted only through an internal test/benchmark API. The production CLI must not expose an option that changes the formal 10,000-replicate contract.

## Execution architecture

### Prepared evidence cache

Each loaded evidence bundle lazily converts raw mappings to immutable observation objects and groups them by development image exactly once per process. Bootstrap draws reuse those groups. The sampled rows and draw-local image identities remain identical to the existing implementation, so endpoint mathematics do not change.

### Candidate checkpoint

Each candidate owns an independent atomic checkpoint under:

`<output-dir>/checkpoints/<condition>.json`

The checkpoint contains:

- schema and state (`running` or `complete`);
- candidate and F0 evidence hashes;
- evidence-file byte hashes and canonical shared image identity;
- image count, endpoint names, stage, reference/candidate names;
- total and completed replicate counts;
- all completed finite replicate deltas, in replicate-index order;
- bootstrap seed, percentile method, estimand identifier, and immutable statistical-config hash;
- deterministic RNG schedule state: scheme name, seed, image hash, and next replicate index;
- hashes of the CLI and statistical modules that define the calculation;
- timestamps, elapsed seconds, and last measured rate.

Canonical evidence digests and source-file byte SHA256 values are separate identity fields. The formal CLI streams each F0-F3 source file once, caches those four byte hashes, and reuses the F0 identity for all candidates. It must not serialize the complete F0 raw payload again for each candidate merely to derive an identity.

The checkpoint is replaced atomically at a fixed replicate interval, whenever 300 seconds have elapsed since the previous durable save, and at normal completion. The time limit is authoritative: a large or slow block must be split so no expensive work can remain uncommitted for more than five minutes. An interruption inside a block can lose only the current uncommitted block; resume recomputes it from its deterministic replicate index.

Every committed checkpoint also records `last_saved_at`, cumulative elapsed time, the completed block/replicate range, `next_replicate_index`, output paths, and schema/version identity. Checkpoint JSON, progress JSON, manifests, and final tables are written to a temporary sibling, flushed and validated, then atomically replaced.

Cumulative elapsed time measures actual compute segments. Resume adds the current monotonic segment to the previously persisted elapsed value; powered-off or paused wall time must not enter replicate rate or ETA.

### Resume and fail-closed identity

Resume is explicit. If a checkpoint exists without `--resume`, execution fails. With `--resume`, every persisted identity field, code hash, statistical-config hash, completed count, replicate list length, finite value, and RNG schedule field must match the current run. Any mismatch fails before new computation or output modification.

Completed checkpoints remain as provenance and are never silently deleted.

### Progress and ETA

After every committed block the runner emits one flushed progress line containing condition, completed/10,000, percent, elapsed time, replicate rate, ETA, checkpoint path, and last-save time. It also atomically updates `<output-dir>/progress.json`. ETA is reported only after at least one completed block and is explicitly an estimate. Two missed five-minute save windows are a P0 and must not be described as healthy execution.

At initialization, every candidate completion, and final selection, a small recovery bundle containing checkpoint index, manifest, progress/summary JSON or CSV, run identity, hashes, and resume instructions is atomically mirrored to a second persistent directory supplied by the formal CLI. Large evidence and weights remain in their registered primary locations. Mirror failure blocks promotion to the next milestone; no secret may enter either copy.

The mirror directory is mandatory and must not equal or be nested inside the primary output/checkpoint tree. Supporting mirror files carry one generation ID and their hashes; the manifest is the last atomic commit marker. A partial mirror generation is invalid and is safely regenerated on resume.

### Safe parallelism

F1, F2, and F3 are independent conditional statistics, but the implementation must not recompute the same F0 draw three times merely because there are three candidates. Evidence parsing, `by_image` construction, draw indices, and the F0 four-endpoint result are prepared once and reused across candidates wherever the selected backend permits. Python observation replacement/copying is excluded from the hot loop.

CPU execution may use deterministic replicate blocks and up to three processes only when the benchmark shows a wall-time benefit after accounting for evidence sharing and process memory. Worker count is recorded as execution provenance but does not enter the estimand or random schedule. The final selection is performed only after all three candidate checkpoints are complete.

### Backend selection and profiling gate

Backend choice is evidence-driven rather than assumed. A representative local benchmark must separately measure:

- cold evidence parsing and one-time `by_image`/array construction;
- deterministic bootstrap-index generation;
- four-endpoint computation with F0 reused across F1-F3;
- checkpoint serialization and atomic replacement overhead;
- total throughput and peak host memory for NumPy single-process and safe process-block execution;
- when local CUDA is available, PyTorch CUDA batched index/endpoint execution, including host-to-device setup, steady-state throughput, peak VRAM, and numerical equivalence to NumPy.

CuPy is evaluated only if already available; it is not a required dependency. A GPU backend is admissible only if it preserves the registered rank/regression/intervention endpoint logic, the exact draw schedule, ordered replicate vector, and final linear percentile interval within the declared numerical tolerance. Otherwise the CPU backend remains authoritative even if a partial tensor kernel is faster. No GPU performance number may be inferred from device specifications or fabricated when the backend cannot be executed locally.

## Output lifecycle

No final selection artifact is written until all three candidate checkpoints are complete and verified. Before publishing the three user-facing files, the runner atomically persists a finalization journal containing their complete identities and payload hashes. With explicit resume, verified existing files are reused and missing files are recreated from that journal; any mismatch fails closed. A power loss after the first final file therefore cannot require result deletion or an unregistered rerun.

## Verification requirements

1. RED tests must first demonstrate the absence of checkpoint/resume and identity rejection.
2. An interruption simulation must stop after a non-boundary replicate, resume from the last committed block, and produce the same ordered replicate vector, point estimate, and CI as an uninterrupted run.
3. Identity changes in evidence, image manifest, code, statistical config, RNG schedule, or completed-value shape must fail closed before computation.
4. Cached and uncached endpoint recomputation must be numerically identical; raw observation parsing/grouping must occur once per loaded bundle.
5. Sequential and multi-process candidate execution must produce identical final candidate statistics.
6. Progress output must contain real completed counts, rate, ETA, and checkpoint path.
7. Focused tests, the complete regression suite, `py_compile`, and a small local performance benchmark must pass before any server use.
8. The formal server command, output directory, checkpoint paths, resume command, and runtime estimate must be frozen before requesting the RTX 5090.
9. The benchmark must include equivalence, throughput, peak host-memory/VRAM evidence, and a call-count or profiling assertion proving F0 endpoint work and evidence grouping are not repeated per candidate or replicate.
10. A time-controlled test must prove a checkpoint is forced before 300 seconds even when the replicate-count interval has not been reached, and a mirror test must prove milestone manifests are atomically copied to the registered second persistent location.
11. A representative local benchmark must use 371 image clusters and roughly the real per-image observation structure (about 50,000-70,000 valid rows), not a four-image micro-test. It provides a bounded local estimate only. The first reopened-server action is a short checkpointed preflight on the real approximately 93 MB evidence; the formal 10,000 run starts only if that preflight validates equality, RSS, checkpoint growth, and a measured ETA.

## Non-goals

- No change to training, F0–F3 checkpoints, AP40, observer data, gates, or endpoint definitions.
- No reduced formal replicate count.
- No result-dependent tuning, candidate filtering, or post-hoc choice of statistic.
- No automatic server connection or launch during implementation.
