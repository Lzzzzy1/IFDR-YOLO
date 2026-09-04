# v214 evaluation notes

This directory evaluates the sole frozen v213e epoch-15 local direction screen. It cannot be used as formal 30ep/b16 evidence.

Authoritative decision: `gate/frozen_gate.json` → `NO_GO` with failed gates `overall`, `small`, `far`, `near`; `large` passes. Authoritative engineering report: `engineering/terminal_evidence.v2.json` → all checks PASS. The original `engineering/terminal_evidence.json` is retained as a collector-v1 false failure.

Interpretation boundaries:

- `hard-gate/` is the preregistered decision metric.
- `moderate-descriptive/` is a secondary validity-definition cross-check.
- `diagnostics/` was executed only after the frozen decision and cannot alter it.
- `prediction-replay/` and `prediction_replay_manifest.json` prove same-checkpoint deterministic prediction parity.
- Paired-bootstrap `status.json` files preserve a wrapper false negative for far Cyclist. Their `validated_summary.json/csv` files distinguish 1000 requested from 868 effective samples and validate all eight child outputs without rerunning them.
- `diagnostics/prediction-scores/prediction_score_summary.json` and its partial v1 CSV are preserved from a CSV-writer failure. The complete corrected outputs are `.v2.json/.v2.csv`.
- `diagnostics/posthoc-b1-vs-anchored-bootstrap/` is explicitly post-hoc failure attribution, not a selection result.

The final `MANIFEST.sha256` and `manifest.json` are generated only after this note, tool/test snapshots and all evidence artifacts are closed.
