# IFDR Evidence and Conflict-Aware Research Design

Date: 2026-08-01

## Objective

Use the remaining experimental week to turn the existing KITTI result into a reproducible conference-paper result. The work has two coordinated tracks:

- Track A establishes statistical evidence for the baseline, P2 detector, and strongest existing fusion condition.
- Track B diagnoses and resolves the negative interaction between reliability-gated fusion and DCLI.

The design must keep the RTX 5090 productive without running experiments that lack a paper-facing purpose.

## Established Evidence

All values below use the same 3,769-image KITTI validation split and the project-owned AP40 evaluator.

| Condition | Moderate mean AP40 |
|---|---:|
| YOLOv8m baseline | 56.8265 |
| YOLOv8m + P2 | 63.7814 |
| IFDR full | 63.9801 |
| Factor control | 66.4091 |
| Fusion only | 66.9096 |
| DCLI only | 66.8043 |

Fusion and DCLI are individually effective, but their joint condition underperforms both. The current shared reliability estimator receives gradients from factor supervision, detection through the fusion router, and DCLI uncertainty calibration. The research hypothesis is that incompatible updates cause degradation-semantic drift and negative transfer.

The current router already predicts spatial branch weights with shape `[B, 2, H, W]`. Adding another generic spatial-attention module is therefore out of scope.

## Track A: Statistical Evidence Queue

### Conditions

Run seed 29 and seed 41 for each of:

1. YOLOv8m baseline.
2. YOLOv8m + P2.
3. IFDR fusion-only.

Together with the completed seed-17 runs, these provide three seeds for the principal progression from baseline to P2 to the strongest existing method. The other component conditions remain single-seed ablations unless later evidence requires replication.

### Invariants

- Reuse the locked KITTI train/validation split and source hashes.
- Reuse the same pretrained checkpoint and semantic-prefix initialization policy.
- Change only the experiment seed and the intervention base seed where applicable.
- Train for 300 epochs with the existing deterministic settings.
- Evaluate every completed run with the same KITTI AP40 implementation.
- Preserve `results.csv` and `last.pt` every epoch, plus an epoch checkpoint every 10 epochs.

### Queue Behavior

The queue runs sequentially on one GPU. For every condition it must:

- reject a configuration that fails dry-run validation;
- run a one-epoch smoke test before a formal run;
- skip an already complete formal run;
- resume an incomplete formal run from `last.pt`;
- write independent PID, status, and append-only pipeline logs;
- retry a transient failure once, then mark that condition failed and continue
  with the next independent condition so the GPU is not left idle;
- mark the overall queue `PARTIAL` when any condition fails, rather than
  silently treating incomplete evidence as successful;
- leave completed artifacts on `/root/autodl-tmp`.

Expected wall time is 21–23 hours, based on the measured 3.4-hour duration of a 300-epoch run plus evaluation overhead.

## Track B: Conflict-Aware Method

### Gradient Diagnostic

Measure gradient norms and pairwise cosine similarity on the reliability estimator for:

- detection loss propagated through the fusion router;
- degradation-factor supervision;
- DCLI uncertainty calibration.

Aggregate conflict frequency, mean negative cosine, and gradient-norm ratios. Report them globally and, where target metadata permits, for small or degraded examples. The diagnostic is training-only and must add no inference cost.

### Candidate Mechanisms

Screen three isolated candidates before any new 300-epoch run:

1. **Delayed DCLI:** learn factor semantics and fusion routing first, then ramp DCLI.
2. **Calibration gradient isolation:** prevent DCLI calibration from rewriting the supervised factor semantics while retaining the forward uncertainty signal.
3. **Semantic anchor with task adapters:** use the supervised factor representation as a protected anchor and pass it through small fusion- and localization-specific adapters, combined with delayed DCLI.

Each candidate receives a 60–80 epoch screening run. Candidate 3 may add parameters, but total parameter growth must remain below 5%.

### Selection Rule

A candidate advances only if it satisfies all applicable conditions:

- complete model performance no longer falls below both single-module conditions;
- Moderate mean AP40 improves over the current full condition by at least 1.0, or reaches within 0.5 AP of the fusion-only condition while materially improving robustness or variance;
- gradient conflict frequency or negative cosine magnitude decreases;
- no class loses more than 1.0 AP40 Moderate without a documented small-object trade-off;
- training remains finite and recoverable.

The winning candidate is trained for 300 epochs with seeds 17, 29, and 41. Failed candidates remain recorded as negative evidence and are not expanded into formal multi-seed runs.

## Data Flow

1. Controlled sampling and visibility interventions produce auditable factor targets.
2. The reliability estimator predicts degradation semantics at all six bidirectional fusion nodes.
3. Track-B isolation protects the semantic representation from incompatible downstream gradients.
4. The fusion adapter routes P2–P5 features using the protected semantics.
5. The localization adapter combines protected semantics with DFL entropy for DCLI.
6. Detection, factor, calibration, and gradient-agreement metrics are logged separately.
7. The unchanged KITTI AP40 evaluator produces Easy, Moderate, and Hard results.

## Verification

Before server launch:

- unit tests cover seed materialization, component switches, staged schedules, gradient isolation, queue skip/resume behavior, and invalid configuration rejection;
- all existing tests continue to pass;
- every new configuration passes the project dry-run;
- one-epoch smoke runs produce finite losses and all expected artifacts.

After training:

- validate run status, epoch count, checkpoint presence, split hash, and AP40 output;
- report mean, standard deviation, and per-class AP40 across seeds;
- compare peak epoch and late-training degradation;
- retain complexity, throughput, and robustness measurements for the final selected model.

## One-Week Boundary

- Day 1: Track-A multi-seed queue and Track-B diagnostic implementation.
- Day 2: gradient audit and three short candidate screens.
- Day 3: formal training of the winning mechanism.
- Day 4: additional seeds and controlled-degradation robustness.
- Day 5: external transfer if core evidence has passed; otherwise resolve core failures.
- Days 6–7: statistical consolidation, missing ablations, and result freeze.

No new generic attention, backbone replacement, or unrelated dataset expansion is allowed before the conflict-aware mechanism passes its selection rule.
