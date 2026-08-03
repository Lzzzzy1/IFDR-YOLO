# Research Task Board

## Group A — Independent Core Literature Chains

### Task A: Cross-Scale Representation and Fusion

- Role: Small-object architecture researcher
- Objective: Map P2, FPN/PAN/BiFPN, spatially dynamic fusion, scale routing, high-resolution branches, and context modeling; identify saturated and open claims.
- Queries:
  1. small object detection dynamic feature pyramid spatial adaptive fusion CVPR ICCV ECCV
  2. tiny object detection high resolution P2 scale routing feature fusion
  3. conditional computation feature pyramid object detection scale aware
- Depth: DEEP
- Output: `research-notes/task-a.md`

### Task B: Localization Loss, Assignment, and Uncertainty

- Role: Detection optimization researcher
- Objective: Map IoU variants, WIoU, scale-aware losses, distributional regression, sample assignment, label uncertainty, and small-object gradient instability.
- Queries:
  1. small object detection localization loss scale adaptive IoU gradient uncertainty
  2. bounding box regression label uncertainty tiny object detection
  3. task assignment scale aware small objects object detection
- Depth: DEEP
- Output: `research-notes/task-b.md`

### Task C: KITTI and Road-User Evaluation

- Role: Autonomous-driving benchmark researcher
- Objective: Map 2D KITTI road-user work, official difficulty evaluation, common data splits, occlusion/truncation handling, and gaps in YOLO-based studies.
- Queries:
  1. KITTI 2D pedestrian cyclist small object detection occlusion truncation
  2. KITTI object detection AP40 Easy Moderate Hard evaluation official
  3. YOLOv8 KITTI pedestrian cyclist improved small object detection
- Depth: DEEP
- Output: `research-notes/task-c.md`

## Group B — Adjacent Mechanisms and Collision Search

### Task D: Frequency, Uncertainty, and Context Mechanisms

- Role: Adjacent-methods researcher
- Objective: Investigate frequency-domain enhancement, uncertainty-aware gradients, contextual reasoning, and scale-conditioned learning as possible deeper mechanisms.
- Queries:
  1. tiny object detection frequency enhancement CVPR 2025
  2. uncertainty aware gradient stabilization small object detection ICCV 2025
  3. context reasoning far pedestrian detection object detection
- Depth: DEEP
- Output: `research-notes/task-d.md`
- Dependency: Group A notes

### Task E: Novelty Collision Scan

- Role: Prior-art and reproducibility auditor
- Objective: Search titles, abstracts, public code, and patents/preprints for candidate combinations and synonymous implementations; record collisions and unverifiable gaps.
- Queries:
  1. dynamic spatial weighted BiFPN small object detection code
  2. scale balanced Wise IoU small object detection
  3. P2 dynamic fusion uncertainty loss road object detection
- Depth: DEEP
- Output: `research-notes/task-e.md`
- Dependency: Group A notes

## Group C — Dissertation Program Synthesis

### Task F: Falsifiable Research Program

- Role: Doctoral research-methodology reviewer
- Objective: Synthesize evidence into 2-3 dissertation-scale candidate programs, each with hypothesis, novelty boundary, required baselines, failure criteria, multi-dataset validation, and staged implementation.
- Queries:
  1. doctoral contribution criteria computer vision object detection ablation generalization
  2. small object detection open problems survey 2025
- Depth: DEEP
- Output: `research-notes/task-f.md`
- Dependency: Tasks A-E

