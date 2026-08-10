# Hard-Budget Value-of-Resolution P2: One-Day Frozen Design

## 1. Objective and claim boundary

The project will test whether a P2 detector can convert degradation evidence into an actual KITTI detection gain under a fixed per-frame compute budget. P2, cropping, zooming, NWD, RFLA, coarse-to-fine inference, and prediction of coarse-versus-fine detection gain are prior art and are not claimed as original. The narrowed candidate contribution is a risk-constrained decision: estimate separate paired benefit and harm outcomes of one re-observation, condition their reliability on current-frame road geometry and first-pass residual uncertainty, select at most one crop, and emit the exact F0 output when a calibrated lower-bound net benefit is non-positive.

This one-day study may deliver a leakage-free single-seed GO/NO-GO and raw ablation artifacts. It is not multi-seed final paper evidence, and no positive outcome is assumed.

## 2. Frozen data, model, and metric

- Dataset: KITTI 2D detection, Car/Pedestrian/Cyclist labels.
- Development protocol: seed 17, 3,341 fit images and 371 development images.
- Registered fit ID SHA256: `50a1f8d72b747d1e2f460a4e0a355469484d33dbf1e9928b64ba6594c1873362`.
- Registered development ID SHA256: `b1b6b6ee7e5398e93868fab407a2e8a86a53c753667002ef9b8381734ef2cda8`.
- Training may reference only the fit manifest. Oracle and AP evaluation may reference only the development manifest.
- Reference detector: leakage-free plain YOLOv8m-P2, seed 17, 300 epochs, image size 640, primary checkpoint `last.pt`.
- Primary metric: the absolute-point change in the arithmetic mean of KITTI Pedestrian Moderate AP_R40 and Cyclist Moderate AP_R40 on the identical frozen image list.
- Secondary metrics: Car Moderate AP_R40, per-class Easy/Moderate/Hard AP_R40, target-height and truncation/occlusion strata, TP/FP, latency, re-observation rate, and crop budget.
- Values from the internal 371-image protocol must never be numerically mixed with the historical 3,769-image Chen validation scores.

## 3. Hypotheses and success criteria

### Primary effectiveness hypothesis

- `H0-effect`: full F4 does not improve Pedestrian/Cyclist Moderate macro AP_R40 over F0 (`delta <= 0` absolute AP points).
- `H1-effect`: full F4 improves the same frozen metric over F0 (`delta > 0`).
- `H0-novel-mechanism`: under the same candidate pool and zero-or-one crop budget, F4 does not outperform the prior-art-style expected-gain F3 or does not satisfy the registered harm constraints.
- `H1-novel-mechanism`: F4 outperforms F3 while satisfying all registered harm constraints.

### Research target

Three end-to-end targets are registered and must not be satisfied by arithmetically adding results from different protocols: leakage-free plain P2 F0 must improve over original P3-P5 YOLOv8 by at least `+4.0` absolute AP_R40 points; full F4 must improve over F0 by at least `+2.0` points, with a research target of `+2.0 to +3.0`; and full F4 must improve over original P3-P5 by at least `+6.0` points, with a target range of `+6.0 to +8.0`. Historical numbers are context only and cannot satisfy these gates.

A single-seed F4-versus-F0 point estimate of at least +2.0 is only a screening signal. A positive performance claim additionally requires three seeds with directionally consistent gains, a paired confidence interval supporting a positive effect (or an equivalently preregistered statistical gate), improvement in the registered far/small strata, no harm to near/medium objects, and complete FLOPs/latency/budget reporting. A gain below +2.0, or a gain confined to one seed or one favorable slice, is diagnostic evidence rather than performance success.

### Statistical and safety requirements

- Report paired image-cluster uncertainty for the AP delta; do not report a relative percentage as an AP gain.
- F4 cannot advance if either Pedestrian or Cyclist Moderate AP_R40 decreases.
- No class may lose more than 1.0 overall AP_R40 point.
- Near/large-object AP loss may not exceed 0.5 point per reported class.
- The hard budget is at most one fixed half-image crop per frame; selecting no crop is always permitted.

## 4. Frozen ablation matrix

All variants use the same F0 checkpoint, 371 development IDs, candidate pool, image size, prediction thresholds, crop geometry, NMS, maximum detections, random seed, and one-crop budget. Only the selection evidence changes.

| Variant | Frozen definition | Purpose |
|---|---|---|
| F0 | Leakage-free plain P2, no second observation | Strong base detector and delta denominator |
| F1 | Geometry-only selector using road/perspective position, candidate height, and fit-derived geometry confidence; no detection uncertainty | Tests whether road geometry alone explains useful resolution allocation |
| F2 | Detection-residual/uncertainty-only selector using first-pass confidence deficit, class ambiguity, and deterministic first-pass instability; no road geometry | Tests whether detector uncertainty alone identifies useful re-observations |
| F3 | Prior-art-style expected coarse-versus-fine gain selector with the frozen candidate pool and at most one crop | Tests whether ordinary expected-gain zoom is already sufficient |
| F4 | Separate paired benefit/harm prediction, current-frame geometry-reliability conditioning, residual uncertainty, and calibrated lower-bound abstention to exact F0 under the same hard budget | Candidate original risk-constrained method |

Paired outcomes are computed from actual base-versus-crop detections on fit data only. Benefit records recovered true positives and improved matched localization; harm separately records lost true positives, new false positives, duplicates, and class flips. F3 predicts their expected scalar net gain. F4 predicts benefit and harm separately and may act only when its fit-calibrated lower-bound net benefit is positive and its harm gate passes. Development results never train or tune either policy.

## 5. Feasibility gates before F1-F4

### Clean F0 training gate

Before any development evaluation, the run must publish and mirror:

- actual training IDs recovered from the Ultralytics cache;
- observed training count and SHA256;
- `intersection_count = 0` against development IDs;
- actual `args.yaml.data` identity;
- code, model, pretrained weight, config, split, cache, `last.pt`, and post-training audit hashes;
- checkpoint/resume and mirror provenance.

Any mismatch is an immediate NO-GO.

### O1 resolution-reachability oracle

O1 may center one fixed half-image crop on an eligible development ground-truth target only to measure an optimistic upper bound. It is not a deployable model. If `O1 - F0 < +2.0` Pedestrian/Cyclist Moderate macro AP_R40 points, stop the re-observation route and preserve the negative result. Do not run O2 or add rescue modules. Reaching +2.0 only authorizes O2; it is not a final positive result.

### O2 frozen-pool oracle

Only if O1 passes, freeze a GT-free candidate pool before GT access: six fixed 3-by-2 grid windows plus at most eighteen first-pass small/low-confidence proposal windows. GT may select among frozen candidates but may not create, move, or resize a window. O2 must retain at least a +2.0 absolute-point reachable gain under the frozen pool and pass the registered no-harm checks before F1-F4 implementation is justified. O2 remains a screening upper bound, not a final model result.

## 6. Novelty boundary and prior-art controls

The study must compare or discuss official P2, generic sliced inference, Dynamic Zoom-In, Uzkent WACV 2020, AdaZoom, FOVEA/Two-Plane, ESOD/ZoomDet/coarse-to-fine selection, uncertainty-guided dynamic inference, and NWD/RFLA-style tiny-object assignment. The claim is not that crops, P2, geometry priors, uncertainty, or expected high-resolution gain are individually new. The narrowed claim candidate is a risk-constrained one-action decision with:

1. road-geometry confidence;
2. first-pass detection residual/uncertainty;
3. separate candidate-specific paired benefit and harm outcomes from extra pixels;
4. a per-frame one-crop hard budget;
5. calibrated lower-bound abstention and exact base-result fallback;
6. leakage-free, compute-matched end-to-end AP evidence.

The full collision decision and source registry are frozen in `docs/research-notes/novelty-p2-20260810/NOVELTY_COLLISION_MATRIX.md`. The broad value-of-resolution claim is NO-GO; only this narrowed formulation is provisionally allowed, and F4 implementation remains blocked until O1 and O2 each retain at least +2.0 AP_R40 under the frozen feasibility gates.

## 7. One-day execution schedule

The clock is measured from restoration of a confirmed RTX 5090 instance. Long-run ETA is not treated as fact until a representative 640-pixel epoch and resume cycle are measured.

| Time box | Deliverable | Gate |
|---|---|---|
| 0-1 h | Server identity, data, code, split, prior smoke, and GPU-state audit | Stop on identity or data mismatch |
| 1-2 h | Real 640-pixel smoke, saved epoch, intentional interruption, in-place resume, leakage and mirror evidence | All recovery/identity gates must pass |
| 2 h onward | Start or resume 300-epoch clean F0; ETA = measured representative epoch rate, reported with PID/log/checkpoint/mirror | No unbenchmarked long run |
| F0 completion + 0-1 h | Post-training leakage audit and exact F0 AP_R40 on 371 development images | Intersection must be zero |
| Next 1-2 h | O1 oracle with per-image checkpoints and measured ETA | `< +2.0`: immediate route STOP |
| Remaining time, GO only | O2; then single-seed F1-F4 in frozen order | No implementation or run after a failed preceding gate |
| End of day | Raw JSON/CSV, prediction hashes, AP table, latency/budget table, GO/NO-GO, limitations | Never relabel single-seed evidence as final multi-seed proof |

### Absolute delivery deadlines (Asia/Shanghai)

- `2026-08-10 17:00`: local initial gate: novelty collision matrix, falsifiable contribution statement, frozen F0-F4 specification, focused syntax/tests, persistent synthetic smoke, novelty GO/NO-GO, updated handoff, and explicit server-start permission. The smoke proves runnability only, not AP improvement.
- If novelty is GO and the server is available, `2026-08-10 19:00`: real one-epoch smoke, intentional interruption/resume evidence, zero-leakage audit, and measured ETA.
- `2026-08-10 23:59`: leakage-free F0 and O1 output, exact F0 AP, O1 delta, paths/hashes, and route GO/NO-GO. O1 below +2.0 stops the route that night.
- If O1 passes, `2026-08-11 12:00`: same-protocol single-seed F0-F4 preliminary AP, far/small slices, false positives, budget/latency, F4-versus-F0 delta, and full-system-versus-P3-P5 delta.
- `2026-08-12 10:00`: three-seed confirmation, paired CI, no-harm and cost table for any viable route, or an evidence-backed completion percentage and explicit remaining gaps. A single seed cannot be presented as paper-level completion.
- GPU deadlines move by the actual server-offline duration if the server is unavailable after a novelty GO. Local novelty and code gates do not move.

## 8. Long-job safety and stop conditions

- Every task expected to exceed 30 minutes requires a representative benchmark, checkpoint interval no greater than five minutes, input/code/config identity, deterministic unit or RNG state, atomic writes, flushed progress and ETA, an external persistent mirror, and an active interruption RED-to-GREEN demonstration.
- Do not launch duplicate jobs. A live owner, foreign host, changed identity, incomplete mirror transaction, or uncertain checkpoint role fails closed.
- Stop on leakage, missing provenance, changed evaluator/config, O1 below +2.0, O2 below +2.0 or its frozen no-harm gate, or two consecutive supervision intervals without progress.
- Do not add attention, IoU losses, fusion blocks, NWD, RFLA, or assignment changes to rescue a failed oracle.

## 9. Today’s acceptable outcomes

1. `GO`: clean F0 plus O1/O2 evidence supports implementing and running frozen F1-F4.
2. `NO-GO`: O1 or O2 fails; archive the result and end the re-observation route.
3. `EXTERNAL BLOCK`: server, data, or environment prevents a valid run; report the exact evidence and do not fabricate progress.

Only raw, traceable outputs support any conclusion.
