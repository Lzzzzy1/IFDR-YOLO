# P3–P5 Semantic Anchor / P2 Candidate-Survival Audit Design

## Decision

Keep the registered P3–P5 path numerically unchanged and treat P2 as a
candidate residual, not as the new semantic owner. Before changing any model
or starting a training screen, locate the first boundary at which useful P2
candidates disappear:

1. training assignment;
2. pre-NMS score/rank;
3. cross-level filtering/NMS.

Only one cause-matched repair may be implemented. Results from different
levels or repair routes are not additive.

## Frozen evidence

- Same-protocol P3–P5 macro AP_R40: `96.76868378801165`.
- Same-protocol plain-P2 macro AP_R40: `95.17692276580817` (point delta
  `-1.59176102220349`; paired interval crosses zero).
- Seed-29 DCLI produces a low-score tail: Pedestrian TP `+1`, FP `+228`;
  Cyclist TP `0`, FP `+243`. This identifies a proximal precision/ranking
  failure, not its training cause.
- DCLI `B-C` across seeds 17/29/41 is `+0.7091/-0.5603/+0.8282`; the mean is
  `+0.3257` and its interval crosses zero. Fusion adds no demonstrated value
  because `AB-B` is negative in all three seeds.

## Scope and data use

- Use only the registered 3,341 fit images to select a mechanism or threshold.
- The 371 images remain development and may only evaluate a frozen screen.
- No official-test or independent-confirmation claim is made.
- The audit consumes a fixed checkpoint, fixed data order, fixed model/config
  hashes and the actual dataloader IDs. It must fail closed on identity or
  split mismatch.

## Audit record

For every valid GT and each P2/P3/P4/P5 level, record only detached values:

- image ID, class, GT box/height and registered small/far stratum;
- number of anchors whose centers are legal for the GT;
- number selected before and after collision resolution;
- maximum IoU, maximum task-alignment value and rank of the best candidate;
- assigned-positive count;
- pre-NMS candidate count, best class score and best IoU when available;
- post-filter/post-NMS survival and winning level when available.

Level boundaries are derived from the actual feature-map sizes and flattened
anchor order; no hard-coded anchor counts are allowed.

## Numeric non-interference

Diagnostics are default-off. Enabling the audit may observe detached tensors
or replay the registered assigner on detached inputs, but must not replace the
assigner, mutate its tensors, change RNG state, change loss values or alter
predictions. A focused test must prove audit-off and audit-on loss/assignments
are exactly equal for identical inputs.

## Recoverability and outputs

- Append one canonical JSONL unit per image and fsync it.
- Atomically replace a checkpoint containing input identity, completed image
  IDs, deterministic position and cumulative elapsed time.
- Resume skips completed IDs and rejects changed code/config/checkpoint/split.
- Publish summary JSON/CSV and a SHA manifest last; remote execution mirrors
  the small journal/checkpoint/summary to a second persistent location.
- No output may contain secrets.

## Mutually exclusive decisions

### A — Assignment coverage rescue

Advance only if, among GTs with a legal P2 candidate, small/far Cyclist has a
zero-P2-positive rate at least 10 absolute percentage points above near/large
Cyclist, with image-clustered 95% CI lower bound above zero, and the deficit is
not shared by every level. The repair may add at most the highest-alignment
currently-unassigned P2 candidate while preserving all original assignments.

### B — Score ownership isolation

Consider only if assignment coverage is adequate and good-IoU P2 candidates
exist pre-NMS but have systematically inferior class-score rank. P3–P5 retain
classification/assignment ownership; a P2 localization-quality correction
must be detached from semantic logits. This route needs a separately frozen
fit-only effect and CI gate before coding.

### C — Cross-level NMS ownership

Consider only if a good, sufficiently scored P2 candidate exists before NMS
and is removed by cross-level competition. The repair may change only the
ownership/suppression rule and must preserve the P3–P5 baseline outputs when
the P2 candidate is not demonstrably better.

If no boundary shows a preregistered P2-specific defect, all three routes are
NO-GO and the project retains P3–P5.

## Training gate after a supported diagnosis

Only a single 30-epoch matched control/repair screen is allowed. GO requires
all of the following:

- at least `+1.1` absolute Pedestrian/Cyclist Moderate macro AP_R40 over the
  matched P3–P5 anchor system;
- 10,000 image-cluster paired bootstrap interval lower bound above zero;
- mechanism diagnostic moves in the preregistered direction;
- small/far TP, FN and AP improve;
- Pedestrian overall, near and large harm is at most `0.5` AP each;
- latency/FLOPs/candidate budget and all negative results are reported.

Passing one seed authorizes multi-seed confirmation; it is not a final claim.
Any job over 30 minutes additionally requires a representative benchmark,
checkpoint interval at most five minutes, same-identity resume equivalence,
atomic writes, dual persistent copies and a measured ETA.

## Originality boundary

P2/FPN, scale assignment, one-candidate supplementation, quality scoring,
stop-gradient, semantic anchoring and NMS ownership all have direct prior art.
The maximum defensible contribution is an evidence-backed identification of a
specific P2 candidate-loss boundary in road small-object detection and a
validated bounded repair under no-harm constraints.
