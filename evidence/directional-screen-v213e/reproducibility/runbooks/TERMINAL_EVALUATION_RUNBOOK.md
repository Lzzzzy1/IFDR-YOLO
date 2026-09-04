# v213e terminal evaluation runbook

All commands run in ordinary PowerShell; administrator rights are not required. Do not execute them until the v213e runner itself publishes `state=complete`, `epoch=15`, terminal predictions and `metrics_ap40.json`. The runner already performs the frozen terminal prediction from epoch-15 `last.pt`; this runbook must not select another checkpoint.

## Frozen paths

```powershell
$source = 'E:\myyolo\stage11-v213-anchored-rerun-20260827e'
$run = "$source\runs\b2-anchored-rerun-s0-b2-e15"
$mirror = "$source\mirror\b2-anchored-rerun-s0-b2-e15"
$evaluation = 'E:\myyolo\stage11-v214-v213e-terminal-evaluation-20260827a'
$plain = 'E:\myyolo\stage11-v205-local-runtime-plain-20260826b\runs\plain-s0-b2-e15'
$tools = 'C:\Users\张玮倩\Documents\Codex\2026-08-27\recovered-myyolo\work\ifdr-8h-20260827\tools'
$python = 'D:\ana\envs\yolo\python.exe'
```

## 1. Fail-closed readiness and new evaluation generation

```powershell
$status = Get-Content "$run\status.json" -Raw | ConvertFrom-Json
if ($status.state -ne 'complete' -or $status.epoch -ne 15 -or $status.identity_sha256 -ne 'b5047a2d097b1bb4679f58136aaf557484059bc131cf878cc47c4334d3ef78c3') { throw 'v213e terminal state is not frozen-complete' }
if (Test-Path -LiteralPath $evaluation) { throw "evaluation generation already exists: $evaluation" }
New-Item -ItemType Directory -Path $evaluation | Out-Null
```

## 2. Engineering evidence

```powershell
& $python "$tools\collect_terminal_evidence.py" --run-root $run --mirror-root $mirror --launcher-stdout "$source\launcher.stdout.log" --launcher-stderr "$source\launcher.stderr.log" --fit-ids "$source\inputs\fit_ids.txt" --development-ids "$source\inputs\development_ids.txt" --expected-identity b5047a2d097b1bb4679f58136aaf557484059bc131cf878cc47c4334d3ef78c3 --output "$evaluation\engineering\terminal_evidence.v2.json"
if ($LASTEXITCODE -ne 0) { throw 'terminal engineering evidence failed' }
```

Historical note: the first collector version wrote `terminal_evidence.json` and falsely failed because it hard-required an optional assignment stream and a hardcoded 380 records. That artifact is retained. The test-backed current collector checks primary/mirror existence parity and derives the expected manifest from the 7 present controls + 371 predictions + checkpoint, producing the authoritative 379-record `terminal_evidence.v2.json`.

## 3. Frozen HARD-conditional slice gate

```powershell
& $python "$source\scripts\evaluate_stratified.py" --run "PLAIN=$plain\predictions\labels" --run "ANCHORED=$run\predictions\labels" --label-dir "$source\kitti_raw\training\label_2\training\label_2" --image-dir "$source\kitti_raw\training\image_2\training\image_2" --split "$source\inputs\development_ids.txt" --output-dir "$evaluation\hard-gate"
if ($LASTEXITCODE -ne 0) { throw 'frozen HARD-conditional evaluation failed' }
```

## 4. Separate Moderate-valid descriptive slices

```powershell
& $python "$tools\evaluate_moderate_stratified.py" --source-root $source --run "PLAIN=$plain\predictions\labels" --run "ANCHORED=$run\predictions\labels" --label-dir "$source\kitti_raw\training\label_2\training\label_2" --image-dir "$source\kitti_raw\training\image_2\training\image_2" --split "$source\inputs\development_ids.txt" --output-dir "$evaluation\moderate-descriptive"
if ($LASTEXITCODE -ne 0) { throw 'Moderate-valid descriptive evaluation failed' }
```

The Moderate-valid report is descriptive only. It must never be passed to the frozen gate evaluator.

## 5. Single frozen decision

```powershell
& $python "$tools\evaluate_frozen_gate.py" --baseline-metrics "$plain\metrics_ap40.json" --candidate-metrics "$run\metrics_ap40.json" --stratified-report "$evaluation\hard-gate\stratified_ap40.json" --baseline-run PLAIN --candidate-run ANCHORED --engineering-evidence "$evaluation\engineering\terminal_evidence.v2.json" --output "$evaluation\gate\frozen_gate.json"
if ($LASTEXITCODE -ne 0) { throw 'frozen gate evaluator execution failed' }
```

The scientific decision inside `frozen_gate.json` is authoritative for this local screen: overall `>= +1.1`, small/far `>0`, near/large `>=0`, and every engineering check PASS. A `NO_GO` is a valid negative result and must not trigger gain/threshold/seed/evaluator retuning.

## 6. Independent same-checkpoint prediction replay

Run only after the frozen decision has already been written, so replay cannot become a result-selection source.

```powershell
& $python "$tools\replay_terminal_prediction.py" --source-root $source --config "$source\configs\experiments\development\kitti_clean_anchor_b1_s0_local_low_memory.yaml" --run-root $run --development-ids "$source\inputs\development_ids.txt" --output-dir "$evaluation\prediction-replay"
if ($LASTEXITCODE -ne 0) { throw 'terminal prediction replay differs from runner prediction' }
```

The replay manifest is written to `$evaluation\prediction_replay_manifest.json`; it binds the same `last.pt`, frozen args and all 371 per-file hashes.
