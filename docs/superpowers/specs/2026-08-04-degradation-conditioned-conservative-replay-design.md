# Degradation-Conditioned Conservative Replay for IFDR-YOLO

Date: 2026-08-04
Status: approved research design; implementation requires written-spec review

## Objective

Extend the accepted protected IFDR-YOLO detector into a closed research loop
for distant, small, occluded road users. The method must use target-level
sampling and visibility degradation evidence to improve Cyclist detection
without materially reducing Car or Pedestrian AP40.

The primary formal targets are:

- three-seed mean KITTI Moderate AP40 of at least 70;
- three-seed Cyclist Moderate AP40 of at least 60;
- mean Car and Pedestrian degradation no worse than 1 AP relative to the
  accepted protected IFDR checkpoint;
- a paired image-bootstrap 95% confidence interval above zero for the Cyclist
  improvement.

These values are goals rather than guaranteed outcomes. With the current
three-seed class means of 91.1656 Car and 64.9359 Pedestrian, Cyclist AP40 of
53.8985 is sufficient for a three-class mean of 70. Cyclist AP40 of 60 would
produce a mean near 72 if the other classes are preserved.

## Research Question

Can an object-level degradation representation learned from controlled
interventions and aligned to natural road-scene metadata identify difficult
distant targets, then control training allocation so that rare degraded
Cyclists improve without negative transfer to common road-user classes?

This question replaces a module-stacking claim. P2 detection, weighted
bidirectional fusion, IoU variants, generic hard-example mining, staged
fine-tuning, class balancing, and gradient surgery are prior techniques and
must not be claimed individually as novel.

## Established Evidence

The current protected IFDR model has completed 300 epochs for seeds 17, 29,
and 41. Its Moderate AP40 class means are:

| Class | Mean | Standard deviation |
|---|---:|---:|
| Car | 91.1656 | 0.8498 |
| Pedestrian | 64.9359 | 2.4330 |
| Cyclist | 42.6205 | 5.2073 |
| Three-class mean | 66.2407 | 2.6349 |

Cyclist error is concentrated in degraded strata: approximately 3.69 AP40 for
small targets, 1.00 for targets beyond 40 metres, and 2--3 for partially or
heavily occluded targets. The protected-versus-unprotected counterfactual
queue is still in progress. Until it finishes, semantic protection is a tested
hypothesis rather than a solved mechanism.

## Critical Validity Gap

The current intervention transform derives the natural sampling target from
box height but passes `natural_occlusion=0.0`. The sampling factor therefore
has a direct natural-scale anchor, while the visibility factor is learned
mainly from synthetic interventions. The project must not use predicted
visibility as a natural curriculum variable until transfer validity is
measured.

The first implementation stage is consequently an audit, not a new training
run.

## Method Overview

The proposed extension has four gated components:

1. natural factor-transfer audit;
2. natural degradation alignment when required;
3. degradation-conditioned conservative replay;
4. optional degradation-adaptive assignment only when positive-coverage
   diagnostics justify it.

The existing asymmetric semantic protection remains part of the method. It is
evaluated by matched protected and unprotected controls rather than attributed
to gradient conflict alone.

## Component 1: Natural Factor-Transfer Audit

### Inputs

- accepted protected IFDR checkpoints for seeds 17, 29, and 41;
- immutable KITTI raw metadata and processed split manifests;
- predicted sampling and visibility maps at the six reliability nodes;
- ground-truth height, depth, occlusion, and truncation values.

### Measurements

For every eligible ground-truth object, pool the factor maps over the object
region and record:

- sampling factor versus pixel height and depth;
- visibility factor versus ordinal occlusion and truncation;
- factor response under controlled intervention severity;
- target-region response versus matched background intervention;
- residual visibility association after controlling for class and box height;
- residual sampling association with depth after controlling for box height.

Report Spearman correlation, ordinal group trends, bootstrap confidence
intervals, and pairwise monotonicity. A factor is accepted as naturally aligned
only when its expected direction is stable across seeds, its bootstrap
interval excludes zero, and controlled severity ordering is correct for at
least 80% of eligible pairs.

The audit uses no optimizer and does not alter checkpoints.

### Gate

- If both factors pass, proceed directly to conservative replay.
- If sampling passes but visibility fails, use sampling-only replay as the
  factor-guided condition and add natural visibility alignment as an ablation.
- If both fail, do not describe the replay as factor-guided. Run the
  metadata-only control and repair factor alignment before any formal claim.

## Component 2: Natural Degradation Alignment

Natural alignment is training-only auxiliary supervision built from KITTI
metadata:

- sampling degradation combines normalized target height and depth;
- visibility degradation combines ordinal occlusion and truncation;
- supervision is active only within matched target support;
- uncertain or invalid metadata receives zero weight;
- synthetic intervention targets remain active so factor selectivity is not
  replaced by metadata regression.

Geometric augmentations that break object-to-metadata identity are disabled in
the short alignment stage. Box matching must reject ambiguous or low-IoU
matches instead of assigning guessed metadata. Alignment is fit and screened
using an internal split of the existing training set; the standard KITTI
validation split is reserved for the frozen formal recipe.

Natural metadata are treated as difficulty proxies, not causal ground truth.
The paper uses `counterfactual-inspired intervention` unless explicit causal
assumptions and identifiability conditions are established.

## Component 3: Degradation-Conditioned Conservative Replay

### Replay Score

Each Cyclist instance receives a degradation priority from the validated
sampling and visibility factors. A metadata-only score constructed from
height, depth, occlusion, and truncation is retained as a mandatory control.
Image priority is the maximum eligible Cyclist priority in that image, capped
to prevent a few extreme samples from dominating training.

Class identity determines eligibility, but degradation determines priority.
This distinction separates the proposed method from class-only oversampling.

### Schedule

Initialize from the corresponding accepted protected IFDR checkpoint for each
seed and reset optimizer state. At epoch `t`, sample images from

`P_t = (1 - eta_t) * P_original + eta_t * P_degradation`.

The focus stage ramps `eta_t` from zero to a bounded maximum. The recovery
stage returns `eta_t` to zero and lowers the learning rate while training on
the original all-class distribution. Mosaic, MixUp, and aggressive geometric
cropping remain disabled during target-identity-sensitive stages.

The inference graph is unchanged. Replay adds no parameters, FLOPs, or runtime
latency.

### No-Harm Selection

Hyperparameters and checkpoint choice use a deterministic internal training
holdout. Selection maximizes Cyclist AP subject to Car and Pedestrian drops no
worse than the pre-registered bound. After the recipe is frozen, formal runs
use the full locked training split and are evaluated once on the standard
validation split.

## Component 4: Conditional Degradation-Adaptive Assignment

Before changing assignment, instrument the existing TaskAlignedAssigner and
record positive candidates per ground-truth object by class, pyramid level,
height, depth, occlusion, and truncation.

Only if distant or small Cyclists receive materially fewer valid positives is
a per-object positive budget implemented. The budget may increase P2/P3
candidates as a bounded function of validated degradation, while leaving
ordinary targets and P4/P5 unchanged.

Mandatory controls are the unmodified assigner, class-only positive expansion,
and metadata-degradation expansion. This component is omitted if the coverage
audit does not support its mechanism. Adaptive assignment and rare-class
positive expansion are prior work; originality is limited to their
factor-conditioned, no-harm use inside this evidence loop.

## Semantic Protection and Negative Transfer

The protected semantic anchor continues to receive factor and counterfactual
supervision. Fusion and localization consume detached factor semantics through
task adapters. The matched unprotected model permits downstream detection and
localization gradients to update the anchor.

Gradient cosine and norm measurements are diagnostic evidence, not proof of
causation. The claim is accepted only when the protected model improves target
validation performance under the same data, initialization, schedule, and
loss weights. PCGrad or CAGrad may be screened as generic baselines but do not
replace the protected/unprotected comparison.

## Experiment Matrix

Short seed-17 screens precede any three-seed expansion:

| ID | Condition | Purpose |
|---|---|---|
| R0 | Accepted protected IFDR | Frozen reference |
| R1 | Equal-budget ordinary continuation | Extra-training control |
| R2 | Class-only Cyclist replay | Class imbalance control |
| R3 | Metadata-only degradation replay | Metadata control |
| R4 | Factor-guided replay without recovery | Recovery ablation |
| R5 | Full conservative replay | Proposed training control |
| R6 | R5 without semantic protection | Negative-transfer control |
| R7 | R5 plus adaptive assignment | Conditional extension only |

A screen advances when Cyclist improves by at least 5 AP, the three-class mean
improves by at least 1.5 AP, neither Car nor Pedestrian falls by more than 1.5
AP, training remains finite, and the distant/small/occluded slices do not
degrade collectively.

The frozen winning recipe is run for seeds 17, 29, and 41 from their
corresponding protected checkpoints. At minimum, R1, R3, and R5 receive matched
formal budgets when the final paper attributes gain to factor guidance and
recovery rather than additional optimization time.

## Evaluation

Primary results:

- KITTI Easy, Moderate, and Hard AP40 by class;
- three-seed mean and standard deviation;
- height, depth, occlusion, and truncation slices;
- paired image-bootstrap 95% confidence intervals;
- per-object positive coverage and localization-gradient diagnostics;
- factor monotonicity and natural-transfer evidence;
- parameters, FLOPs, throughput, and peak memory;
- false-positive and false-negative case studies.

BDD100K is a cross-domain mechanism test. BDD Rider is not reported as if it
were identical to KITTI Cyclist. The first BDD experiment tests factor response
under controlled object sampling and visibility interventions. A mapped-class
detection experiment is optional and must disclose taxonomy differences.

## Failure Handling

- Non-finite factor targets, ambiguous metadata matches, or empty eligible
  replay sets fail configuration validation before training.
- Every run saves `last.pt`, `best.pt`, results, factor audit records, sampling
  manifests, and selection decisions to an independent directory.
- Recovery resumes only from artifacts whose configuration, split, source,
  and checkpoint hashes match.
- A failed factor audit blocks factor-guided claims but does not erase the
  metadata-only result.
- A method that raises Cyclist but violates the Car/Pedestrian bound is
  reported as a trade-off, not as a successful no-harm method.

## Required Tests

Before server launch, local tests must cover:

- deterministic metadata indexing and replay manifests;
- height, depth, occlusion, and truncation normalization boundaries;
- ambiguous and missing metadata rejection;
- factor ROI pooling and monotonicity statistics;
- mixture-sampler reproducibility and probability normalization;
- focus-to-recovery schedule transitions;
- no-harm selection on synthetic result tables;
- checkpoint provenance and per-seed initialization;
- assignment diagnostics without changing baseline outputs;
- conditional assignment limits if that component is activated;
- one-epoch smoke artifacts and recovery behavior.

All existing tests must continue to pass.

## Citation and Originality Boundary

The paper cites FPN, PANet, EfficientDet/BiFPN, QueryDet, SNIP, OHEM, Focal
Loss, ATSS, class-balanced loss, EQL, LOCE, LTTSS, Step-wise learning, SimLTD,
WIoU, UGS, PCGrad, CAGrad, ForkMerge, KITTI, BDD100K, counterfactual data
generation, and degradation-manifold detection where their concepts are used.

Permitted contribution claims, subject to successful controls, are:

1. intervention and natural-metadata alignment of object-level sampling and
   visibility degradation factors;
2. degradation-conditioned conservative replay that improves rare degraded
   targets while returning to the original all-class distribution;
3. conditional use of factor-guided positive allocation when assignment
   diagnostics support it;
4. asymmetric semantic protection evaluated through matched optimization and
   generalization evidence;
5. a closed evaluation loop spanning stratified diagnosis, targeted
   intervention, no-harm constraints, multiple seeds, uncertainty intervals,
   and cross-domain mechanism validation.

The paper must not claim that P2, BiFPN, WIoU, hard-example mining, curriculum
learning, class balancing, adaptive assignment, or gradient surgery is new.

## One-Week Execution Boundary

- Day 1: factor-transfer and positive-coverage audits.
- Day 2: natural alignment, replay sampler, tests, dry-run, and smoke run.
- Day 3: seed-17 short controls and recipe freeze.
- Day 4: formal seed-17 run and decisive ablations.
- Day 5: formal seeds 29 and 41.
- Day 6: BDD100K mechanism transfer and optional YOLOv8s scale check.
- Day 7: bootstrap, stratified analysis, failure cases, efficiency, and result
  freeze.

No new generic attention block, backbone replacement, unrelated IoU variant,
or uncontrolled dataset augmentation enters the formal matrix.
