---
task_id: f
role: Doctoral research-methodology reviewer
status: complete
sources_found: 12
---

## Sources

[1] Uncertainty-Aware Gradient Stabilization for Small Object Detection | https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Uncertainty-Aware_Gradient_Stabilization_for_Small_Object_Detection_ICCV_2025_paper.html | Source-Type: academic | As Of: 2025-10 | Authority: 10/10
[2] The Importance of Anti-Aliasing in Tiny Object Detection | https://arxiv.org/abs/2310.14221 | Source-Type: academic | As Of: 2023-10 | Authority: 8/10
[3] SET: Spectral Enhancement for Tiny Object Detection | https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html | Source-Type: academic | As Of: 2025-06 | Authority: 10/10
[4] TOOD: Task-aligned One-stage Object Detection | https://openaccess.thecvf.com/content/ICCV2021/papers/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.pdf | Source-Type: academic | As Of: 2021-10 | Authority: 10/10
[5] QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection | https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.pdf | Source-Type: academic | As Of: 2022-06 | Authority: 10/10
[6] LER-YOLO: Reliability-Aware Expert Routing for Misaligned RGB-Infrared UAV Detection | https://arxiv.org/abs/2605.20667 | Source-Type: academic | As Of: 2026-05 | Authority: 7/10
[7] KITTI Object Detection and Orientation Estimation Evaluation (2D) | https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d | Source-Type: official | As Of: 2026-07-28 | Authority: 10/10
[8] Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection | https://proceedings.neurips.cc/paper_files/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html | Source-Type: academic | As Of: 2020-12 | Authority: 10/10
[9] Labels Are Not Perfect: Inferring Spatial Uncertainty in Object Detection | https://arxiv.org/abs/2012.12195 | Source-Type: academic | As Of: 2020-12 | Authority: 7/10
[10] Learn Discriminative Features for Small Object Detection through Multi-Scale Image Degradation with Contrastive Learning | https://www.jstage.jst.go.jp/article/transinf/E108.D/4/E108.D_2024EDP7204/_article/-char/en | Source-Type: academic | As Of: 2025-04 | Authority: 8/10
[11] Scale-Aware Relay and Scale-Adaptive Loss for Tiny Object Detection in Aerial Images | https://arxiv.org/abs/2511.09891 | Source-Type: academic | As Of: 2025-11 | Authority: 7/10
[12] Pedestrian Emergence Estimation and Occlusion-Aware Risk Assessment for Urban Autonomous Driving | https://arxiv.org/abs/2107.02326 | Source-Type: academic | As Of: 2021-07 | Authority: 7/10

## Findings

- The strongest surviving program is not a new neck or loss but a counterfactually supervised latent degradation state that is required to be calibrated and identifiable before it is reused for feature routing, assignment, and probabilistic regression. [1][2][4][6][9][10]
- This program has bounded novelty because reliability-gated fusion, shared assignment-loss quality, sparse high-resolution routing, predictive uncertainty, and degradation contrastive learning each exist separately, but the reviewed sources do not place an identifiable road-scene degradation posterior across all three control points. [4][5][6][8][9][10]
- A viable latent should distinguish at least effective sampling ratio, alias/noise energy, visibility or occlusion, and boundary uncertainty, because one undifferentiated attention score cannot tell missing evidence from corrupted evidence or annotation ambiguity. [1][2][3][9]
- Counterfactual intervention is needed for identifiability: known blur, resampling, sub-pixel translation, occlusion, and boundary-jitter interventions provide supervision for what changed, whereas ordinary attention can correlate with class or background without representing degradation. [2][9][10]
- The second candidate program is a theory-first gradient-budget framework that extends scale-dependent curvature analysis from regression into assignment selection and pyramid-level compute, but its novelty risk is high because UGS, TOOD, GFL, and scale-adaptive loss occupy much of the component space. [1][4][8][11]
- The third candidate program is risk-calibrated conditional perception for potentially occluded road users, where contextual emergence risk allocates high-resolution compute without being treated as direct positive-box evidence; it has high safety relevance but the largest evaluation and hallucination risk. [5][7][12]
- A two-to-three-week period can produce only a falsifiable pilot: repair the KITTI protocol, build a controlled degradation diagnostic set, estimate one calibrated degradation variable, route one P2/P3 branch, and couple at most one optimization component. [5][7][10]
- A dissertation-scale claim requires at least three linked contributions, multi-dataset validation, parameter/FLOP/training-budget matched controls, calibration and failure analysis, and an explicit result that would disprove the central mechanism. [1][5][7]
- KITTI should be the first road-scene test bed rather than the sole evidence source, because its official metric excludes sub-25-pixel targets and couples scale, occlusion, and truncation in difficulty labels. [7]
- No reviewed evidence supports the absolute statement that nobody has studied the surviving program; the defensible wording is that no functionally equivalent public method was identified within the documented source and query boundary as of 2026-07-28.

## Deep Read Notes

### Source [1]: Candidate Program A — Counterfactual Degradation-Calibrated Detection
Key data: Central question: can a locally identifiable posterior over resolution, aliasing, visibility, and boundary uncertainty explain small-object failure and non-redundantly control representation, assignment, and regression?
Key insight: Chapter 1 builds a degradation-factor benchmark and causal interventions; Chapter 2 learns a calibrated posterior \(q_\theta(z\mid F)\); Chapter 3 uses \(z\) for expected-compute-constrained P2/P3 routing and anti-alias/spectral control; Chapter 4 uses the same posterior for uncertainty-aware assignment and probabilistic box likelihood.
Useful for: Bounded novelty versus UGS/TOOD/QueryDet/LER-YOLO is the counterfactual identifiability and probabilistic coupling, not the existence of routing or uncertainty; a two-to-three-week pilot should repair KITTI evaluation, generate paired interventions, predict one factor such as effective sampling ratio, and test one routed branch plus one coupled loss term.

### Source [4]: Candidate Program B — Scale-Conditioned Gradient-Budget Theory
Key data: Central question: can a detector equalize useful localization information rather than raw loss magnitude across object scale, stride, assignment probability, and visibility?
Key insight: Chapter 1 derives curvature and positive-selection bias; Chapter 2 formulates an expected gradient or Fisher-information budget per instance; Chapter 3 designs joint assignment and distributional regression; Chapter 4 validates calibration and cross-scale generalization.
Useful for: Required controls include UGS, TOOD/TAL, GFL/DFL, NWD or scale-adaptive loss, equal-compute heads, and synthetic pixel-jitter experiments; falsification occurs if gradient equalization does not predict optimization stability or if gains reduce to simple area weighting, with overall research risk assessed as high (8/10).

### Source [12]: Candidate Program C — Risk-Calibrated Conditional Perception
Key data: Central question: can contextual risk of a small or occluded road user allocate sensing and high-resolution computation while keeping visual detection probabilities calibrated?
Key insight: Chapter 1 models contextual emergence risk; Chapter 2 separates context prior from visual likelihood; Chapter 3 uses risk for constrained regional P2 computation; Chapter 4 evaluates safety-weighted miss cost, false alarms, and cross-city calibration.
Useful for: Required datasets extend beyond KITTI to CityPersons/BDD100K or equivalent occlusion-rich road scenes; falsification occurs if context only increases hallucinations or if ordinary objectness querying matches it, and research risk is very high (9/10) because ground-truth emergence and safety metrics are difficult.

## Gaps

- Counterclaim: Program A can be described as LER-YOLO reliability routing plus TOOD alignment plus QueryDet sparse P2 plus probabilistic regression; it survives only if counterfactual supervision makes the latent identifiable and interaction ablations show that joint probabilistic coupling beats independent modules. [4][5][6][8][9]
- Counterclaim: controlled blur, resampling, occlusion, and label jitter may not reproduce real camera distance, motion, compression, weather, or human annotation processes, so real-condition calibration and cross-dataset tests are mandatory. [2][10]
- Counterclaim: a shared latent can create negative transfer because visual degradation, label ambiguity, assignment confidence, and regression uncertainty are not the same variable; a factorized posterior must be compared with one scalar and fully independent controls. [1][4][9]
- Counterclaim: Program B overlaps strongly with UGS and may become a mathematical reformulation without practical benefit unless it predicts new failure regimes before experiments. [1]
- Counterclaim: Program C may optimize a safety prior rather than object detection and can hallucinate pedestrians near contextual cues, so it may require a separate task definition and ethics/safety evaluation. [12]
- Evidence gap: the source set contains 2025-2026 preprints whose claims may change after review, and it does not cover non-English databases, subscription-only patents, unpublished work, or all code implementations.
- Implementation gap: the current project has no official KITTI AP40 evaluator, uses fixed-size label normalization, discards DontCare, and collapses classes, so no new-method conclusion is valid until the data and evaluation pipeline are rebuilt.
