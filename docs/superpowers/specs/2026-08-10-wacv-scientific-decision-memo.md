# WACV Scientific Decision Memo: From Module Stack to Degradation Reliability

Date: 2026-08-10  
Status: decision frozen before reading the pending F0-relative paired result

## Fixed evidence boundary

The paper cannot use P2, weighted fusion, an IoU variant, or a higher AP number as its main novelty. The defensible question is whether a detector can learn target-specific sampling and visibility reliability for distant road users, preserve the primary detector, and demonstrate that the learned semantics respond to target degradation rather than scene background.

Existing evidence already establishes a useful but incomplete chain:

- the original natural-factor audit failed and identified natural alignment, target/background specificity, and severity ordering as concrete failure modes;
- F0-F3 use a leakage-free 3341/371 fit/development split and matched 30-epoch budgets;
- F0-F3 preserve the exact KITTI development detector output, so semantic calibration did not trade away detector AP;
- all four absolute mechanism gates fail; F1 is the closest partial repair but still fails target/background specificity, paired target response, and severity ordering;
- the missing F0-relative 10,000-replicate CI may quantify partial improvement, but cannot override an absolute-gate failure.

## Distinct research paths

### Path A — Target-conditioned nuisance-invariant reliability repair

**Question.** Can reliability semantics be made target-specific by explicitly removing matched-background response while preserving ordered response to target degradation?

**Method principle.** One coherent correction, not a module stack: learn the target residual relative to the matched clean/background response, with the registered severity order inside the same three-view intervention protocol. The detector remains frozen during semantic calibration. The correction is admitted only if target response exceeds background response and severity ordering improves under the pre-registered audit.

**Novelty.** High relative to the current project because it turns a failed three-view audit into a falsifiable representation constraint. It is not a generic attention or loss swap.

**Falsifiability.** Strong. The method fails if the target-background gap, paired target response, severity order, or natural association does not improve against compute-matched F0 with a paired CI, even if training loss falls.

**WACV fit.** High: visual recognition under small-target degradation, intervention-based diagnosis, reliability, and failure-aware evaluation.

**Evidence reuse.** Very high: same split, observer, F0 control, four endpoints, audit, exact AP40, and paired bootstrap.

**New experiments.** At most one seed-17 targeted calibration candidate after the current paired result; if it passes, matched F0/candidate task adaptation, three seeds, and one cross-domain mechanism test.

**Failure value.** High. A failed targeted correction still supports a rigorous boundary result: detection-preserving factor calibration is insufficient for target-specific reliability.

**Cost.** Medium initially (one 30-epoch seed-17 calibration plus existing observer/AP40/statistics); high only after passing gates because three-seed and external confirmation then become justified.

**Strongest rejection.** “The new constraint is an ad-hoc combination of contrastive and ordinal losses.” Mitigation: present it as one conditional-invariance objective, pre-register its two necessary consequences, and include compute-matched single-principle controls.

### Path B — Negative-transfer and semantic-protection study

**Question.** When degradation semantics and detection optimization conflict, does asymmetric protection preserve meaningful reliability without sacrificing detection?

**Method principle.** Compare protected and unprotected semantic anchors under identical data, initialization, budget, and losses; use gradient conflict only as diagnosis, not causation.

**Novelty.** Medium. The application to cross-scale small-target reliability is specific, but gradient isolation/detachment is known.

**Falsifiability.** Strong if matched protected/unprotected controls and post-training semantic audits are complete.

**WACV fit.** Medium-high, especially as a representation-learning and multi-task interference study.

**Evidence reuse.** High: existing gradient-conflict diagnostics, protected detector results, and mechanism observer.

**New experiments.** Matched protected/unprotected multi-seed runs plus post-run mechanism audits; potentially expensive.

**Failure value.** Medium. A null result still bounds the role of gradient conflict but is less compelling without a positive reliability mechanism.

**Cost.** High because causal attribution needs matched multi-seed runs rather than one screen.

**Strongest rejection.** “Stop-gradient protection is standard and the contribution is mainly engineering.” Mitigation requires unusually strong mechanism and generalization evidence, which is not yet available.

### Path C — Leakage-free degradation audit and negative-result protocol

**Question.** Do learned sampling/visibility maps actually encode target degradation rather than class, size, or background shortcuts?

**Contribution.** A three-view target/background/natural audit, object-balanced endpoints, leakage-free selection, paired image-cluster uncertainty, exact checkpoint provenance, and explicit failure boundaries.

**Novelty.** Medium-high as an evaluation protocol; lower as a detector method.

**Falsifiability.** Very high because every claim is tied to a registered gate and negative results are retained.

**WACV fit.** Medium for the main track, stronger for a reliability/diagnostics framing or a workshop/secondary venue.

**Evidence reuse.** Maximal. The current F0-F3 evidence is almost the complete paper core.

**New experiments.** One multi-seed confirmation of the audit and one BDD100K mechanism-transfer check; no new detector module required.

**Failure value.** Very high. Failure is itself the result if it is reproducible and exposes shortcut semantics.

**Cost.** Low-to-medium relative to method development; observer/evaluation dominates, not retraining.

**Strongest rejection.** “This is an evaluation report, not a new recognition method.” Mitigation: keep it as the fallback paper, not the primary WACV method claim.

### Path D — Cross-domain degradation replay for rare Cyclists/Riders

**Question.** Does degradation-conditioned replay improve rare degraded road users across KITTI and BDD100K without harming common classes?

**Novelty.** Medium. Replay and curriculum learning are prior art; novelty depends entirely on validated reliability factors and a no-harm closed loop.

**Falsifiability.** Strong with class-only, metadata-only, extra-training, and recovery controls.

**WACV fit.** High if cross-domain gains are real and statistically supported.

**Evidence reuse.** Medium-high, but factor validity is a hard dependency.

**New experiments.** Multiple replay controls, three KITTI seeds, and BDD taxonomy-aware validation.

**Failure value.** Medium because metadata-only replay remains informative, but the factor-guided claim disappears.

**Cost.** Highest of the four paths.

**Strongest rejection.** “Gains come from rare-class oversampling or extra optimization.” The full compute-matched control matrix is mandatory and expensive.

## Adversarial review and decision

### Mechanism reviewer

The strongest rejection is that the factors respond to background texture and object size rather than target degradation. Path A directly attacks this failure; Path C measures it honestly. Paths B and D do not solve it by themselves.

### Experimental-fairness reviewer

The strongest rejection is development leakage, checkpoint cherry-picking, or unequal compute. The 3341/371 split, exact `last.pt`, fixed budgets, F0 control, common image clusters, and paired CI already address most of this. Any new candidate must reuse them unchanged.

### WACV-novelty reviewer

The strongest rejection is a YOLO module bundle with many familiar ingredients. The paper must therefore describe P2/fusion/localization only as the detector substrate. The contribution must be the failure-driven target-specific reliability loop and its falsifiable evidence.

## Frozen choice

**Primary line: Path A**, conditional on the pending F0-relative statistic showing at least one meaningful directional improvement worth repairing. The only admissible new method experiment is a minimal target-conditioned nuisance-invariance repair aimed specifically at the failed background-specificity and severity-order endpoints.

**Low-cost fallback: Path C.** If no F1-F3 candidate has a positive paired improvement, or a targeted repair still fails its absolute gate, stop method expansion and assemble the leakage-free audit/negative-result study. Do not hide the failure with P2/fusion/IoU gains.

Path B remains an explanatory ablation if already available; Path D is deferred until factor validity exists. Neither is the next experiment.

## Time-boxed execution and stop rules

### Gate 0 — Statistical infrastructure (now, local only)

Deliverables: checkpoint/resume/identity/progress implementation; interruption equivalence; focused/full regression; representative CPU/CUDA benchmark; frozen command and measured ETA.

Stop condition: any identity mismatch not rejected, any resumed vector not equivalent, or no bounded ETA. The server stays off.

### Gate 1 — Existing F0-F3 paired result

Run exactly the registered 10,000 replicates after Gate 0. Produce ordered checkpointed replicates, four endpoint deltas, composite delta, linear 95% CI, identity/hash provenance, and selection table.

Decision:

- if no candidate has a positive paired composite CI, select Path C and stop new repair training;
- if a candidate has a positive paired composite CI but fails specific absolute endpoints, permit exactly one seed-17 Path A candidate targeting those pre-existing failures;
- an absolute-gate failure can support only “partial improvement,” never “successful repair.”

### Gate 2 — One targeted seed-17 repair

Pre-register the candidate before reading its result. Use the same initialization, 30 epochs, three views, frozen detector, split, checkpoint rule, observer, AP40, and 10,000-replicate statistic.

Advance only if:

- detector preservation remains within the existing no-harm bound;
- target-background specificity and severity ordering both improve against F0;
- the absolute mechanism gate passes;
- the paired composite 95% CI is above zero;
- no new P0/P1 reproducibility issue exists.

If any condition fails, stop Path A and publish/submit under Path C or a lower-risk venue. Do not create F5/F6.

### Gate 3 — Confirmation only after Gate 2

Freeze the recipe, then run the minimum matched three-seed confirmation and one BDD100K mechanism-transfer test. The BDD result tests mechanism transfer and must not equate Rider with KITTI Cyclist without a taxonomy statement.

Stop if seed instability reverses the direction, no-harm fails, or cross-domain factor behavior is inconsistent. In that case downgrade the claim rather than adding modules.

## Required paper artifacts

- main detector-preservation table using exact KITTI AP40;
- four-endpoint mechanism table with absolute gates and F0-relative paired CIs;
- target-versus-background and severity-response plots;
- split/checkpoint/code/config provenance table;
- failure-case figure grouped by target size, occlusion, truncation, and distance proxy;
- three adversarial-review responses and an explicit limitations section;
- if Path C is selected, a clear negative-result statement and a reusable audit protocol rather than a claimed repair.
