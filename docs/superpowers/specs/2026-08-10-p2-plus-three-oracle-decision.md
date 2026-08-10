# P2 +3 AP Decision Draft: Value-of-Resolution vs Tiny-Safe Assignment

Date: 2026-08-10  
Status: design discussion only; no implementation or server authorization

## Frozen score question

The only score suitable for the new performance promise is the same three-seed KITTI/Chen Moderate AP40 three-class macro used by the existing baseline table:

| Condition | Moderate AP40 macro | Delta vs original | Delta vs P2 |
|---|---:|---:|---:|
| original YOLOv8m | 57.57 | 0.00 | -7.77 |
| P2 YOLOv8m | 65.34 | +7.77 | 0.00 |
| reliability fusion | 65.54 | +7.97 | +0.20 |
| protected IFDR, three-seed | 66.24 ± 2.63 | +8.67 | +0.90 |

Therefore P2 already demonstrates approximately +7.77 points over the original model under the frozen metric. The hard unsatisfied target is **at least +3.00 over P2**, i.e. a three-seed mean of at least **68.34**. Meeting that target would automatically exceed the original by about +10.77.

The following numbers are not comparable and cannot be used to satisfy the target:

- 66.91 fusion-only and 66.80 localization-only are single-run ablations, not three-seed means;
- F0-F3 exact KITTI AP40 on the internal 371-image split is about 96.08 macro and already saturated; all four prediction outputs are identical and the corresponding reference is about 96.29;
- Ultralytics mAP and KITTI AP40 are different evaluators;
- per-class Cyclist or small-object AP cannot be substituted for the three-class Moderate macro.

Current evidence makes +3 over P2 ambitious but not logically impossible. Familiar context or contrastive additions commonly produce only sub-point to roughly one-point gains, and the current IFDR is only +0.90 over P2. A route should advance only if an oracle shows at least five points of reachable headroom under a deployable compute budget; otherwise a learned selector is unlikely to retain a full three-point gain.

## Three mutually exclusive routes

### Route A — Counterfactual value-of-resolution selective re-observation

**Mechanism.** Keep the P2 detector as the first pass. For a bounded set of candidate regions, obtain a second observation at greater effective target resolution using the same frozen P2 detector. During training, supervise a selector with the *real detection-utility difference* between the low-resolution and re-observed predictions: added correct detection, improved KITTI localization, avoided false positive, and compute cost. The selector predicts whether extra pixels can change the decision, not objectness, density, uncertainty, box geometry, or generic saliency.

**Novelty collision.** SAHI, FOVEA, QueryDet, ESOD, ZoomDet, ordinary adaptive zoom, perspective cropping, and multi-pass tiling already cover selective computation and re-observation broadly. The defensible distinction is the counterfactual value label tied to actual detection utility under a frozen P2 decision, paired low/high observations of the same region, and a no-harm compute-constrained policy. If implemented as “crop small objects and run again,” novelty collapses.

**Two-day executability.** High for the decisive oracle because it needs no training and reuses the frozen P2 detector/evaluator. Medium for a learned selector; it should be attempted only after the oracle gate.

**Oracle upper bound.** A deployable-candidate oracle chooses at most one re-observation among candidate windows generated without ground truth and uses a fixed fusion rule. If this cannot exceed P2 by at least +5 on the locked internal development metric while preserving near/large targets, a learned selector is not expected to yield +3.

**Maximum rejection.** “This is another zoom/crop detector whose gains come from extra FLOPs.” Required defense: same candidate generator and fusion policy for all controls; report utility-supervision ablations versus objectness/uncertainty/geometry selectors; fix average extra compute; show small/distant gains and near/large no-harm.

**Cost.** Low for the oracle (inference only); medium for selector training and formal three-seed confirmation. Inference cost must be reported and cannot be hidden by comparing only parameters.

### Route B — P2-aware tiny-safe TaskAlignedAssigner

**Mechanism.** First audit the number and quality of positive assignments for tiny/distant Cyclists at P2/P3. Only if the audit proves under-allocation, change the TaskAlignedAssigner so tiny targets receive a bounded, degradation-aware positive set while normal objects retain the original assignment. NWD and RFLA are mandatory existing baselines, not the proposed novelty.

**Novelty collision.** NWD, RFLA, ATSS-style scale adaptation, tiny-object positive expansion, and quality-aware assignment already occupy most of this space. A defensible new question would be whether modern YOLO/P2 TaskAligned alignment becomes unsafe under sampling/visibility degradation and whether a bounded no-harm correction fixes that specific failure.

**Two-day executability.** Medium-low. The assignment audit is quick, but AP impact cannot be established without retraining. Hidden interactions with TAL classification/localization quality make debugging risky.

**Oracle upper bound.** Assignment coverage can show whether positives are missing, but it cannot provide an AP upper bound without optimization. A simulation that merely adds GT positives is not a valid performance oracle.

**Maximum rejection.** “This is NWD/RFLA or a top-k tweak transplanted into YOLO.” It also risks improving recall while increasing false positives or degrading ordinary objects.

**Cost.** Medium-high: implementation, matched short training, then formal three seeds. If the coverage audit is negative, the route must stop before code changes.

### Route C — Degradation-reliability replay / current IFDR continuation

**Mechanism.** Use validated sampling/visibility reliability to allocate training toward difficult Cyclists while protecting common classes and returning to the original distribution.

**Novelty collision.** Curriculum replay, hard-example mining, class balancing, metadata replay, counterfactual training, and gradient protection are established. Novelty depends on validated target-specific factors and a closed no-harm mechanism loop.

**Two-day executability.** Low for a credible +3 claim. The current F0-F3 mechanism audit fails all absolute gates, and the protected IFDR three-seed mean is only +0.90 over P2. Continuing before factor validity would contradict the registered evidence.

**Oracle upper bound.** No valid score oracle exists without training. Metadata strata can identify difficult samples but cannot guarantee recoverable AP.

**Maximum rejection.** “The factors are not target-specific, so replay is ordinary hard-example/class oversampling with extra machinery.” Current evidence directly supports this concern.

**Cost.** Highest because it needs factor repair, adaptation controls, three seeds, and cross-domain confirmation.

## Decision

**Primary route: Route A, but only through the no-training oracle first.** It is the only route that can cheaply answer whether +3 over P2 is reachable before committing to new training, and its main claim can be falsified with existing assets.

**Backup route: Route B, conditional on a positive P2/TAL coverage audit.** It is reserved for the case where the resolution oracle shows insufficient recoverable headroom but tiny objects demonstrably receive inadequate or poor-quality positives. The audit, not preference, decides whether it is allowed.

Route C remains the reliability/diagnostic paper line and supplies negative evidence, but it is not the performance route for the new +3 target. F0-F3 must not be attached to Route A or B to add apparent novelty or score.

## Minimal no-training oracle experiment

### Frozen inputs and identities

- one registered P2 seed-17 `last.pt` checkpoint for screening; checkpoint SHA, model YAML, code commit, image IDs, labels, evaluator, image size, NMS, confidence threshold, and class mapping are frozen in a manifest;
- the locked 371-image internal development split is used for route selection; the Chen public development split is not used to tune the selector or oracle;
- the formal claim remains the three-seed Chen Moderate AP40 macro and is evaluated only after the recipe is frozen;
- all prediction files, crop manifests, mapped boxes, metrics, and progress are written incrementally with checkpoint/resume and a second persistent mirror.

### Two oracle levels

1. **Reachability oracle O1.** For every missed or poorly localized eligible tiny target, evaluate a GT-centered high-resolution crop with fixed context and the same frozen P2 checkpoint. This is an intentionally optimistic ceiling, not a deployable method. If O1 improves the same-split Moderate macro by less than +3, resolution cannot meet the target and Route A stops immediately.
2. **Deployable-candidate oracle O2.** Candidate windows are generated without GT from frozen first-pass low-confidence P2 proposals plus a fixed coarse grid. Ground truth may choose which *existing* candidate receives one re-observation, but cannot create or resize the candidate. Fusion back into the full-image predictions is fixed before scoring.

### Fixed compute budget

- at most one second-pass crop per image;
- one fixed crop geometry/context policy and one frozen fusion/NMS policy;
- report both worst-case two-pass cost and dataset-average extra FLOPs/latency;
- a deployable selector later must use the same candidate pool and cannot exceed the oracle budget.

### Utility label for the later selector

For candidate region `r`, the registered value target is the paired difference between fixed-policy predictions with and without re-observing `r`:

`value(r) = ΔTP + λ_loc Δlocalization - λ_fp ΔFP - λ_dup Δduplicates - λ_cost normalized_cost`.

The coefficients, KITTI IoU rules, and tie handling are frozen before viewing the candidate results. The oracle AP is computed from complete prediction directories with the existing KITTI AP40 evaluator; per-box utility is supervision/provenance, not a replacement metric.

### Advance and stop gates

Advance Route A only if all hold on the locked internal split:

- O1 delta is at least +3.0 Moderate AP40 macro;
- O2 delta is at least +5.0, leaving a conservative margin for an imperfect learned selector to retain +3.0;
- Pedestrian and Cyclist small/distant strata show positive direction rather than the gain coming only from easy Cars;
- near/large-target AP drops by no more than 0.5 point per reported class and no class drops by more than 1.0 point overall;
- the result remains under the fixed one-crop budget and fixed fusion policy;
- prediction identity, crop mapping, checkpoint/resume, and evaluator provenance all pass.

Stop Route A if O1 is below +3, O2 is below +5, gains require GT-generated windows, fixed fusion harms near/large objects, or the improvement appears only after changing thresholds post hoc. Do not train a selector after a failed oracle.

The oracle is not a paper result and cannot be compared numerically with the existing three-seed table. It is only a route-decision gate. After a positive oracle, the selector must beat P2 by at least +3.0 in an end-to-end, compute-matched three-seed experiment; module deltas are never added.

## Reuse and discard boundary

### Reuse

- P2 model definition, seed-17/29/41 checkpoints and three-seed result registry;
- locked KITTI data/splits, exact AP40 evaluator, class/size/occlusion/distance-proxy slicing;
- provenance, checkpoint, atomic output, resume, mirror, and no-harm infrastructure;
- existing crop/box-coordinate utilities if their geometry tests pass;
- F0-F3 failures as motivation that generic reliability factors/background response do not justify further stacking.

### Do not combine into the performance model

- reliability fusion, DCLI, WIoU variants, F0-F3 calibration, replay, NWD, and RFLA are not additive score coupons;
- the 96.08 internal AP40 mechanism-preservation result is not a performance baseline;
- single-seed best ablations cannot be mixed with three-seed means;
- no attention module, extra neck, context block, or loss swap is added to Route A;
- Route B is run only if Route A stops and the assignment audit passes; the two routes are not stacked.
