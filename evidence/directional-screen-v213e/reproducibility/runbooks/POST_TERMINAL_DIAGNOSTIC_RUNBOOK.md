# v213e post-terminal diagnostic runbook

Execute only after `frozen_gate.json` has been written. These diagnostics are explanatory and cannot change the frozen decision.

## Frozen paths

```powershell
$source = 'E:\myyolo\stage11-v213-anchored-rerun-20260827e'
$run = "$source\runs\b2-anchored-rerun-s0-b2-e15"
$evaluation = 'E:\myyolo\stage11-v214-v213e-terminal-evaluation-20260827a'
$plain = 'E:\myyolo\stage11-v205-local-runtime-plain-20260826b\runs\plain-s0-b2-e15'
$b0 = 'E:\myyolo\stage11-v206-local-runtime-b0-20260826d\runs\b0-s0-b2-e15'
$b1 = 'E:\myyolo\stage11-v209-local-runtime-b1-20260826b\runs\b1-s0-b2-e15'
$v212 = 'E:\myyolo\stage11-v212-local-runtime-b2-20260826a\runs\b2-anchored-s0-b2-e15'
$tools = 'C:\Users\张玮倩\Documents\Codex\2026-08-27\recovered-myyolo\work\ifdr-8h-20260827\tools'
$python = 'D:\ana\envs\yolo\python.exe'
```

## D1 — disjoint reliability diagnostic

```powershell
& $python "$source\scripts\evaluate_detection_reliability.py" --run "PLAIN=$plain\predictions\labels" --run "B0=$b0\predictions\labels" --run "B1=$b1\predictions\labels" --run "ANCHORED=$run\predictions\labels" --label-dir "$source\kitti_raw\training\label_2\training\label_2" --image-dir "$source\kitti_raw\training\image_2\training\image_2" --split "$source\inputs\development_ids.txt" --output-dir "$evaluation\diagnostics\reliability" --split-seed 20260803 --bins 25
if ($LASTEXITCODE -ne 0) { throw 'reliability diagnostic failed' }
```

Interpretation boundary: the existing evaluator uses HARD-valid objects but `matching_iou_threshold=0.0`; therefore LaECE0/LRP describe score-to-overlap/error behavior only. They are not KITTI AP and cannot override the AP gate.

## D2 — exactly eight paired image-bootstrap jobs

```powershell
& $python "$source\scripts\run_paired_bootstrap_matrix.py" --run "PLAIN:0=$plain\predictions\labels" --run "ANCHORED:0=$run\predictions\labels" --comparison PLAIN=ANCHORED --class-name Pedestrian --class-name Cyclist --slice small_25_40 --slice far_gt_40m --slice near_0_20m --slice large_gt_80 --iterations 1000 --bootstrap-seed 17 --workers 4 --label-dir "$source\kitti_raw\training\label_2\training\label_2" --image-dir "$source\kitti_raw\training\image_2\training\image_2" --split "$source\inputs\development_ids.txt" --output-dir "$evaluation\diagnostics\paired-bootstrap"
if ($LASTEXITCODE -ne 0) { throw 'paired bootstrap matrix failed' }
```

The matrix dimensions are one comparison x one matched seed x two classes x four slices = exactly eight jobs. Intervals are descriptive local image-cluster uncertainty only. The core attempts 1000 draws but retains only draws containing a valid target; therefore `comparison.iterations` is the effective count. Cyclist/far retained 868/1000 because 132 draws contained no valid far-Cyclist GT. The original wrapper requires effective=requested and consequently preserves a false-negative matrix state of `failed`; `validated_summary.json/csv` independently validates all eight child artifacts and records both counts without rerunning or rewriting them.

## D3 — object-identity benefit overlap

```powershell
& $python "$source\scripts\analyze_benefit_overlap.py" --split "$source\inputs\development_ids.txt" --image-dir "$run\view\images\val" --label-dir "E:\myyolo\stage11-v211-mechanism-diagnosis\raw-label-view" --p2-dir "$plain\predictions\labels" --a-dir "$b1\predictions\labels" --b-dir "$run\predictions\labels" --class-name Pedestrian --class-name Cyclist --bootstrap-iterations 1000 --bootstrap-seed 17 --journal "$evaluation\diagnostics\benefit-overlap\per_image.journal.jsonl" --output-json "$evaluation\diagnostics\benefit-overlap\benefit_overlap.json" --output-csv "$evaluation\diagnostics\benefit-overlap\benefit_overlap.csv"
if ($LASTEXITCODE -ne 0) { throw 'benefit overlap failed' }
```

Historical field binding is frozen as `P2=PLAIN`, `A=corrected B1`, `B=ANCHORED`; A/B are aliases, not mechanism labels.

The first invocation with the raw 7481-file image/label trees failed before creating an output because the tool requires exact split-ID parity. The successful command above uses the immutable 371-file development views; the split, predictions, matching logic, bootstrap count and seed are unchanged.

## D4 — exact gradient trajectory

```powershell
& $python "$tools\summarize_gradient_trajectory.py" --run "B1=$b1\gradient_diagnostics.jsonl" --run "v212_partial=$v212\gradient_diagnostics.jsonl" --run "ANCHORED=$run\gradient_diagnostics.jsonl" --epoch-start 6 --epoch-end 15 --output "$evaluation\diagnostics\gradient-trajectory\gradient_trajectory.json"
if ($LASTEXITCODE -ne 0) { throw 'gradient trajectory diagnostic failed' }
```

The statistic is the epoch mean cosine, stored conflict count over valid records, and ratio of epoch-mean counterfactual/factor norms. Partial v212 must remain marked as missing epoch 15.

## Final evidence manifest

Run this only after every evaluation and diagnostic artifact is closed. Rerunning it is deterministic; the manifest and its JSON metadata exclude themselves to avoid a circular hash.

```powershell
& $python "$tools\create_evidence_manifest.py" --root $evaluation --manifest "$evaluation\MANIFEST.sha256" --metadata "$evaluation\manifest.json"
if ($LASTEXITCODE -ne 0) { throw 'final evidence manifest failed' }
```

## Tool identities

- `evaluate_detection_reliability.py`: `74df61ff6c1d6f4db8e1fb8103adb3e0ca18b00be1163ecc7b7e280a71b57c83`
- `run_paired_bootstrap_matrix.py`: `d07b0e4d43c221ac25b7abee16fbf7398800ba5005ef2c076e13612f63e6b8b2`
- `analyze_benefit_overlap.py`: `f1ce78a845e1cf89f3e60272cf4e7be91560e1c70221d4406a7c204adab1210c`
- `summarize_gradient_trajectory.py`: `998a889fc046a5188bc385e6a4a9d00925523bdc91accbca957f32c394a0235c`
- `create_evidence_manifest.py`: `e264a2f54a4f59b94de7843d7693580ed6215cdf49ee6c3de043e0ca921a49df`
