# Findings: Resumable factor bootstrap

## Confirmed root cause

## Final performance and safety finding (2026-08-10)

- The dominant historical cost was repeated evidence parsing/grouping/object copying inside every replicate. Preaggregation plus shared F0 draws preserves the estimand while removing this repeated work.
- Cached and legacy endpoints agree to 8.33e-17 on the representative 371-image benchmark; checkpoint/resume produces the same ordered replicate vector as uninterrupted execution.
- The correct backend is currently single-process CPU. No full equivalent GPU kernel has been demonstrated, so GPU acceleration and multi-worker claims are intentionally rejected.
- A real-input checkpointed preflight is still required before treating any runtime estimate as a formal server ETA.

- Final selection artifacts are create-once and appear only after all candidate bootstraps complete.
- `paired_image_cluster_delta` keeps all replicate values in memory and has no checkpoint hook.
- `FactorRepairEvidenceBundle.recompute_endpoints` reparses and regroups the full raw evidence on every draw.
- The selector computes candidate statistics sequentially.
- Per-replicate sampling is already deterministic from seed, stage, image hash, and replicate index; exact resume therefore needs an authenticated next index and completed ordered values, not a mutable shared RNG stream.

## Preserved evidence

- Training, F0–F3 calibration, AP40, checkpoints, absolute gates, observer rows, and four raw evidence files are complete.
- Only the interrupted selection bootstrap lacked a durable result.

## Scientific boundary

- No formal replicate reduction, endpoint change, candidate filtering, CI change, or result-dependent choice is allowed.

## Local performance environment

- Local PyTorch is `2.5.1+cu121`; CUDA is available on an NVIDIA GeForce RTX 3060 Laptop GPU.
- NumPy is installed; CuPy is not installed and will not be added solely for this repair.
- GPU execution is acceptable only after equality with the registered NumPy statistic is demonstrated on identical draws. Device specifications are not evidence of throughput or equivalence.
