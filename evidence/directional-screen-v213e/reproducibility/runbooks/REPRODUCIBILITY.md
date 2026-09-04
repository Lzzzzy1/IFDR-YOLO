# v213e reproducibility

## Environment and frozen identity

- Windows PowerShell; administrator rights not required.
- Python `3.11.15`, Ultralytics `8.4.98`, torch `2.5.1+cu121`.
- GPU observed: NVIDIA GeForce RTX 3060 Laptop, 6144 MiB.
- Source root: `E:\myyolo\stage11-v213-anchored-rerun-20260827e`.
- Config SHA-256: `c0c713dcebcaecc0408ae6f93cd9618b51804297116ae22cd6b70ca21267c94d`.
- Code aggregate SHA-256: `6a408a005dcf04fcc03a4cbecc60a5309267ca3efd4476ebcc8d6769dccd2575`.
- Model/pretrained SHA-256: `0d2cbd9215dc62c4c9920127964ce0ab61b5d10581861d8e83c33fc168aa302a` / `5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5`.
- Fit/development SHA-256: `50a1f8d72b747d1e2f460a4e0a355469484d33dbf1e9928b64ba6594c1873362` / `b1b6b6ee7e5398e93868fab407a2e8a86a53c753667002ef9b8381734ef2cda8`.
- Frozen identity: `b5047a2d097b1bb4679f58136aaf557484059bc131cf878cc47c4334d3ef78c3`.

The configuration resolves to seed0, 15 epochs, batch2, 640 px, SGD `lr0=0.01`, workers2, deterministic mode, frozen first five epochs and ten ramp epochs. This is a local low-memory direction screen, not the formal 30ep/b16 run.

## Safe fresh rerun

The command below deliberately refuses an existing destination. It reads the frozen v213e source/config/data but writes to a new generation. Choose a genuinely new `$reproRoot`; never point it at v213e or v214.

```powershell
$python = 'D:\ana\envs\yolo\python.exe'
$source = 'E:\myyolo\stage11-v213-anchored-rerun-20260827e'
$reproRoot = 'E:\myyolo\stage11-v215-v213e-independent-repro-YYYYMMDDa'
$config = "$source\configs\experiments\development\kitti_clean_anchor_b1_s0_local_low_memory.yaml"

if (Test-Path -LiteralPath $reproRoot) { throw "destination already exists: $reproRoot" }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $config).Hash.ToLower() -ne 'c0c713dcebcaecc0408ae6f93cd9618b51804297116ae22cd6b70ca21267c94d') { throw 'config hash mismatch' }
New-Item -ItemType Directory -Path $reproRoot | Out-Null

$env:PYTHONPATH = $source
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:YOLO_CONFIG_DIR = "$reproRoot\runtime\yolo-config"
$env:MPLCONFIGDIR = "$reproRoot\runtime\matplotlib"

& $python "$source\scripts\run_p2_interaction_s0.py" `
  --config $config `
  --fit-ids "$source\inputs\fit_ids.txt" `
  --development-ids "$source\inputs\development_ids.txt" `
  --output-dir "$reproRoot\runs\b2-anchored-rerun-s0-b2-e15" `
  --mirror-dir "$reproRoot\mirror\b2-anchored-rerun-s0-b2-e15" `
  --mode full `
  --device 0 `
  --execution-purpose local_low_memory_seed0_diagnostic
if ($LASTEXITCODE -ne 0) { throw 'fresh anchored rerun failed' }
```

Observed v213e wall time was `7797.95` seconds. The runner itself evaluates only terminal `last.pt` and writes `metrics_ap40.json`; do not substitute `best.pt` or an earlier epoch.

## Terminal evaluation

Use `TERMINAL_EVALUATION_RUNBOOK.md` after the fresh runner publishes `state=complete`, `epoch=15`. It performs, in order:

1. fail-closed engineering collection;
2. byte-frozen HARD-conditional slice evaluation;
3. separately labeled Moderate-valid descriptive evaluation;
4. one frozen GO/NO_GO decision;
5. independent same-checkpoint replay of all 371 predictions.

Then use `POST_TERMINAL_DIAGNOSTIC_RUNBOOK.md` for reliability, exactly eight PLAIN→candidate paired bootstraps, object-identity overlap and gradient trajectories. These diagnostics run only after the decision and cannot replace it.

## Result identities to expect for the recorded v213e run

- `results.csv`: `9b7cbbeb26bab7259b19fc431c1109b915c92c9223b391406c729f1aa13d28f6`
- terminal `metrics_ap40.json`: `47b3d98eab79d0fd61e88fa5198bcaeed8575cc5e71fb47742873440f68dfbb4`
- terminal `last.pt`: `c6157a321dd539af1694a8a3cb8e36688e3e8004354d7edcb0446ad8cb85333d`
- frozen gate report: `524435ba9e52a594cd45fc673a553a2fa9b86259b598e0ca22f7b01159061bdb`
- exact replay manifest: `5b9e0c9755b4e1600141cad5883169a4989e4984ed7470de8c583ec4c00cd630`

A new independent run may not be byte-identical across hardware/runtime changes; it must still preserve the declared contract and report any numerical difference rather than overwriting this run.
