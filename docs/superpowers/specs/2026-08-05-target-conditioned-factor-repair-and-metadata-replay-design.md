# Target-Conditioned Factor Repair and Metadata Replay for IFDR-YOLO

Date: 2026-08-05
Status: user-approved design; implementation requires written-spec review

## Objective

Repair the mismatch between IFDR-YOLO's trained degradation semantics and its
natural KITTI audit, while running an independent metadata-only replay control.
The resulting study must determine whether difficult Cyclists improve because
of degradation-conditioned allocation rather than extra optimization, class
oversampling, or an unvalidated factor map.

The formal outcome goals remain:

- three-seed mean KITTI Moderate AP40 of at least 70;
- three-seed Cyclist Moderate AP40 of at least 60;
- mean Car and Pedestrian regressions no worse than 1 AP relative to the
  accepted protected IFDR checkpoints;
- a paired image-bootstrap 95% confidence interval above zero for the Cyclist
  improvement.

These are outcome goals, not validity thresholds and not guaranteed results.
Mechanism claims are accepted only through the gates defined below.

## Frozen Evidence and Motivation

The natural-factor audit at implementation commit
`10dc0374f0154068ebc9f49729eafea90abe83af` completed on 742 training images,
three protected checkpoints, six fusion nodes, and 2,000 image-cluster
bootstrap replicates. It produced 402,282 observations and failed the strict
mechanism gate.

The main measurements were:

- sampling raw Spearman rho 0.2513, but residual rho -0.0682 after controlling
  class and target height;
- sampling controlled-bin monotonicity 1.0, but only 24.75% of intervention
  pairs were severity ordered;
- visibility raw rho -0.0140 and residual rho -0.0086;
- visibility controlled-bin monotonicity 0.5;
- visibility target response 0.0736 versus matched-background response 0.0958.

This is a successful negative result. It blocks claims that the present raw
factor maps are naturally aligned. The source code explains the result:

1. training sampling targets use target height but no independent depth;
2. training visibility uses `natural_occlusion=0.0` rather than KITTI
   occlusion and truncation;
3. background interventions are trained to produce high degradation factors,
   while the audit asks for target-specific response;
4. dense pixel-normalized factor loss lets large objects dominate small
   objects.

The completed evidence is immutable. The original output remains under
`/root/autodl-tmp/jobs/natural-factor-audit-10dc037`, and the recoverable
archive is
`/root/autodl-tmp/transfer/natural-factor-audit-10dc037-20260805.tar.gz`
with SHA256
`b2e2abf97e8a7833833bc8fde39df92958dc2bce17ecf6cf89c641e0c9af367b`.

## Scope and Non-Goals

This stage contains two coordinated but scientifically separate tracks:

- Track M: metadata-conditioned conservative replay;
- Track F: target-conditioned factor repair followed by task adaptation.

Track M may run before Track F passes because it does not consume learned
factors. Track F may not be called factor guided until its audit passes.

This stage does not add an attention block, replace the backbone, change the
inference graph, introduce another IoU variant, or alter the KITTI validation
labels. P2, weighted fusion, class replay, curriculum sampling, hard-example
mining, natural metadata regression, and ranking loss are prior ideas and are
not individually claimed as new.

## Factor Semantics

Both factor channels are target-conditioned degradation risks in `[0, 1]`:

- sampling risk: insufficient target sampling caused by small projected
  height and large depth;
- visibility risk: insufficient target visibility caused by occlusion and
  truncation.

They are not generic image-quality maps. A degraded background patch without
an annotated target is a specificity control, not a positive target.

Natural scores retain the registered formulas:

`sampling = 1 - (1 - height_score) * (1 - depth_score)`

`visibility = 1 - (1 - occlusion_score) * (1 - truncation)`

where height, depth, occlusion, and truncation normalization boundaries remain
those in `ifdr_yolo.data.natural_degradation`. Missing or invalid depth removes
the depth contribution and marks that object in provenance; ambiguous metadata
matching receives zero alignment weight rather than a guessed assignment.

The raw factor maps are the semantic anchor and are used for the natural audit.
The existing residual factor adapter is task specific and is evaluated
separately for localization utility. Adapted factors must not be reported as
the raw semantic evidence.

## Shared Metadata Index

Tracks M and F use one immutable metadata index keyed by source image ID and
object identity. Each entry binds:

- class ID and class name;
- original XYXY box;
- height, positive camera-Z depth when valid, occlusion, and truncation;
- sampling, visibility, and joint degradation scores;
- source metadata SHA256 and split SHA256.

The joint score is

`joint = 1 - (1 - sampling) * (1 - visibility)`.

Metadata-to-label matching occurs before any geometry-changing augmentation.
Class must match and box IoU must be at least 0.99. Zero matches, multiple
matches, duplicate identities, non-finite values, and hash mismatches fail
preflight. Natural-alignment training disables Mosaic, MixUp, copy-paste,
perspective, translation, scaling, cropping, and other transforms that break
identity. Photometric augmentation may remain only when it does not change
geometry or metadata identity.

## Leakage-Free Development Protocol

The accepted protected checkpoints were trained on all 3,712 training images,
so they must not be screened on an internal subset of those same images.

A development split is created before repair:

- image-level only, with no overlap;
- deterministic seed `20260805`;
- 10% development images selected by stable hash within strata defined by
  Cyclist presence and the maximum Cyclist joint-degradation tertile;
- every stratum with at least two images contributes to both fit and
  development sets;
- exact IDs and hashes are committed before training.

A seed-17 development checkpoint is retrained from the same registered
initialization and recipe as the accepted protected model, using only the 90%
fit split. M1/M2/M3 and F1/F2/F3 are selected on the unseen development split.
The standard KITTI validation split is not used for recipe selection.

After one recipe is frozen, formal runs start from the corresponding accepted
full-training protected checkpoints for seeds 17, 29, and 41, use the complete
3,712-image training split, and evaluate the standard validation split once.

## Track M: Metadata-Conditioned Conservative Replay

Track M changes the training sampler only. It adds no inference parameters or
latency. All conditions reset optimizer state and use the same initialization,
60-epoch budget, learning-rate schedule, augmentation policy, image size, and
checkpoint rule.

The registered conditions are:

- M0: frozen protected reference, with no continuation;
- M1: 60-epoch ordinary continuation on the original distribution;
- M2: class-only replay, uniform over images containing Cyclists;
- M3: metadata replay, weighted by the maximum eligible Cyclist joint score in
  each image.

M1 is the extra-optimization control. M2 is the class-imbalance control. M3 is
the degradation-allocation control and must not be described as factor guided.

For M2 and M3, training samples from

`P_t = (1 - eta_t) * P_original + eta_t * P_focus`.

The registered schedule is:

- epochs 1-5: linear ramp from `eta=0` to `eta=0.30`;
- epochs 6-40: `eta=0.30`;
- epochs 41-60: linear recovery to `eta=0`.

M3 clips image priorities at the fit-split 95th percentile before
normalization. A small positive floor of 0.05 is added only inside the eligible
Cyclist focus pool so moderate examples remain sampleable. Images without an
eligible Cyclist remain available through `P_original` but not through the
focus pool. Sampler probabilities, selected IDs, epoch, source hashes, and
draw counts are journaled for exact recovery.

## Track F: Target-Conditioned Factor Repair

### Natural object-balanced alignment

Natural metadata supervision is object level, not pixel-count weighted. For
each valid object and node, the raw two-channel factor map is pooled within the
mapped target ROI. Smooth-L1 is computed against the object's sampling and
visibility scores.

Loss reduction proceeds in this order:

1. average spatial values within one object ROI;
2. average objects within each present class;
3. macro-average the present classes;
4. average the registered nodes.

This prevents large Cars from dominating small Pedestrians and Cyclists. An
invalid channel receives zero channel weight without deleting the object's
other valid channel.

### Controlled target specificity

Synthetic dense supervision remains active so natural regression does not
replace intervention selectivity. Matched target and empty-background
interventions use common randomness and identical severity.

For an intervention factor, specificity compares changes from the clean image:

`target_delta >= background_delta + 0.05`.

The margin loss is applied only when the matched background has no overlap
with any annotated object and the registered intervention severity is at least
0.25. Background is not globally forced to zero. Missing, overlapping, or
malformed pairs are rejected and counted.

The alignment objective is

`L_align = L_synthetic + 1.0 * L_natural + 0.5 * L_specificity`.

The three equal-budget 30-epoch repair ablations are:

- F1: synthetic plus natural object-balanced alignment;
- F2: synthetic plus target-specificity alignment;
- F3: all three terms.

The existing protected checkpoint without repair is F0.

### Semantic calibration stage

During F1/F2/F3:

- fusion schedule is zero;
- DCLI schedule is zero;
- the backbone, C2f blocks, detection head, routers, fusion adapters, and
  localization adapter are frozen;
- only the six input projections and shared reliability estimator, including
  the factor head, are trainable;
- optimizer state is reset;
- validation performs no optimizer step.

The factor audit runs immediately after calibration. Failure blocks factor
replay but does not block Track M.

### Task adaptation stage

If F3 passes the development audit, its semantic parameters are frozen. The
detector, routers, fusion adapters, localization adapter, and detection head
are trained with the same 60-epoch focus/recovery schedule as M3. Factors may
condition replay only after the audit passes; metadata scores remain a
mandatory control.

Upstream detector updates can change the frozen factor encoder's inputs, so the
complete natural audit is repeated after task adaptation. A post-adaptation
audit failure rejects the factor-guided claim even when AP improves.

## Pre-Registered Factor Gate

Nodes 17, 20, 23, and 26 are primary because their contexts correspond to the
final P2-P5 detection paths consumed by localization. Nodes 11 and 14 are
intermediate-fusion diagnostics. This selection is architecture based and is
fixed before any repaired model is evaluated.

For each factor, a development candidate passes only when:

- pooled primary-node raw and registered residual associations have the
  expected positive direction and image-cluster bootstrap 95% lower bound
  above zero;
- at least 10 of the 12 seed-by-primary-node directions are positive in the
  three-seed formal audit; the seed-17 development screen requires at least
  three of four primary-node directions positive;
- controlled severity ordering is at least 80%;
- target response is stronger than matched-background response and paired mean
  response is positive;
- no primary node has a statistically significant reverse association;
- neither diagnostic node has a reverse association with absolute rho at
  least 0.10 and a 95% interval excluding zero;
- malformed intervention count is zero and all required observations exist.

All six nodes and all failures remain in the report. The gate threshold may not
be changed after a repaired checkpoint is observed.

## Detection Screen and Formal Advancement

M1, M2, M3, and the factor-guided candidate use matched development budgets.
A seed-17 method advances only when:

- Cyclist Moderate AP40 improves by at least 5 points over its matched
  reference;
- three-class Moderate AP40 improves by at least 1.5 points;
- neither Car nor Pedestrian drops by more than 1.5 points;
- training remains finite and produces non-empty best and last checkpoints;
- small, far, and occluded Cyclist slices do not collectively regress;
- a factor-guided candidate passes both pre- and post-adaptation factor gates.

After the recipe is frozen, formal success uses three seeds and the stricter
1-AP no-harm target. Report mean, standard deviation, paired image-bootstrap
confidence intervals, per-class Easy/Moderate/Hard AP40, target slices,
calibration, efficiency, and failure cases. AP70 and Cyclist AP60 remain
ambitious result targets rather than conditions for suppressing a negative
result.

## Experiment Order

The single GPU runs one formal job at a time:

1. build and audit the leakage-free development split;
2. train the seed-17 development protected checkpoint;
3. run M1, M2, and M3 development screens;
4. run F1, F2, and F3 semantic-calibration screens;
5. audit F0-F3 and select at most one factor repair;
6. if a factor repair passes, run task adaptation and repeat the audit;
7. freeze one metadata recipe and, only if valid, one factor-guided recipe;
8. run matched formal seeds 17, 29, and 41;
9. run paired bootstrap, stratified evaluation, reliability analysis, and
   archive creation;
10. run BDD100K as a mechanism-transfer test after KITTI evidence is frozen.

Local implementation and CPU tests may proceed while the GPU runs a validated
earlier job. No unreviewed experiment may enter the server queue.

## Recovery, Provenance, and Failure Handling

Every split, sampler, calibration run, training run, and audit binds:

- clean implementation Git commit;
- source split and metadata hashes;
- initialization checkpoint hash;
- full resolved configuration;
- random seed, schedule, and trainable parameter names;
- runtime environment and device;
- per-epoch or per-image resumable progress.

Outputs use independent directories and atomic status transitions. Existing
results are never overwritten. A partial artifact resumes only when its
scientific identity matches exactly. A missing hash, dirty checkout, ambiguous
metadata assignment, non-finite loss, duplicate process, or mismatched
checkpoint fails closed before GPU work.

When Track M improves AP but Track F fails, the result is reported as
metadata-conditioned replay, not factor-guided replay. When Track F passes but
detection does not improve, the result supports semantic calibration but not a
detection gain. When Cyclist improves beyond the no-harm bound, the result is a
trade-off, not a successful no-harm method.

## Required Tests Before Server Launch

Implementation must add tests for:

- deterministic stratified fit/development split and exact hash stability;
- proof that a development checkpoint never sees development IDs;
- immutable metadata indexing and 0.99-IoU identity matching;
- rejection of missing, duplicate, ambiguous, and invalid metadata;
- exact sampling, visibility, and joint-score boundaries;
- object-balanced ROI loss invariance to object area and class frequency;
- invalid-channel masking without removing the other channel;
- matched target/background delta ranking and margin boundaries;
- exact trainable/frozen parameter sets in semantic calibration;
- zero fusion and DCLI schedules during calibration;
- deterministic M1/M2/M3 mixture probabilities and recovery schedule;
- draw journaling, interruption, and exactly-once resume;
- primary/diagnostic node gate thresholds and direction checks;
- post-adaptation audit enforcement;
- equal-budget experiment configuration checks;
- one-batch CPU dry run, one-epoch CUDA smoke, checkpoint recovery, and
  non-empty scientific artifacts;
- all existing tests remaining green.

## Contribution and Claim Boundary

Subject to successful controls, the contribution is the closed loop rather
than any isolated component:

1. target-conditioned natural and counterfactual calibration of cross-scale
   degradation semantics;
2. object-balanced alignment that prevents large common targets from masking
   small rare targets;
3. metadata and learned-factor replay under the same conservative recovery and
   no-harm protocol;
4. separation of semantic calibration from task adaptation, with mandatory
   post-adaptation re-audit;
5. leakage-free development, three-seed formal validation, stratified
   diagnosis, uncertainty intervals, and cross-domain mechanism testing.

The paper must disclose the completed failed audit and show how it motivated
the repaired definition. It must not retroactively present the original factor
maps as valid or hide negative nodes, classes, seeds, or controls.
