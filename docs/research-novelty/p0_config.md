# P0 Research Configuration

- AS_OF: 2026-07-28
- Mode: General Research / Standard
- Topic: Dissertation-level novelty for small and difficult road-user detection
- Target codebase: YOLOv8m on KITTI 2D object detection
- Source policy: public sources only; prioritize peer-reviewed papers, official benchmarks, author repositories, and public preprints
- Accessibility: public
- Time horizon: foundational work plus 2020-2026 state of the art
- Geographic scope: global public research

## Primary Research Question

What research program could grow from the current YOLOv8m-P2-feature-fusion-IoU project into a defensible graduate or doctoral contribution, with a precise novelty claim, falsifiable mechanism, reproducible evaluation, and a documented search boundary?

## Counter-Review Plan

The review must attempt to disprove novelty by checking:

1. Whether the same computation graph already exists under another module name.
2. Whether the proposed loss is only target-area reweighting already covered by scale-aware losses.
3. Whether improvements could be explained by increased resolution, parameters, or training budget.
4. Whether KITTI-only evidence is too narrow to support general conclusions.
5. Whether an idea solves a real measured failure mode or merely adds architectural complexity.

## Scope Boundaries

- In scope: RGB 2D object detection, small/far/occluded road users, feature pyramids, dynamic routing/fusion, assignment, box regression, label uncertainty, evaluation.
- Adjacent evidence allowed: general tiny-object detection, scale-aware learning, uncertainty-aware detection, frequency-domain feature enhancement.
- Out of scope for the first implementation cycle: LiDAR fusion, monocular 3D detection, tracking, foundation-model pretraining, deployment hardware co-design.

