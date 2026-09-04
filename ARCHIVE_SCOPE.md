# Archive scope and integrity boundary

This repository is a complete **safe GitHub research archive**: it preserves
everything needed to audit the reported tables, trace the formal protocol to
source and configuration, inspect teacher/manuscript deliverables, and rebuild
the experiment after the user supplies licensed external assets.

## Included

- Project source, tests, scripts, model YAML, experiment YAML, fixed split IDs,
  and provenance/audit utilities.
- The frozen formal-runtime source closure and seed configurations.
- Ten-run five-seed raw metric JSON files, training-result CSV files, terminal
  and epoch publication receipts, binding receipts, statistical summaries,
  teacher-report evidence, and repository-level SHA-256 manifests.
- The formal teacher report, English Chapter 3, venue-neutral IEEE manuscript,
  and AIAC 2026 anonymous review manuscript in editable DOCX and PDF form.
- The later local directional screen as an explicitly separate `NO_GO`
  development artifact, including aggregate diagnostics and reproducibility
  tools but excluding its full per-image prediction dump.
- Project-authored literature synthesis notes without third-party paper PDFs.

## Intentionally excluded

- KITTI or BDD100K images, labels, archives, derived image trees, and dataset
  caches. Dataset licences and redistribution terms still apply in a private
  repository.
- `yolov8m.pt`, trained checkpoints, exported engines, and optimizer states.
- Full prediction-label dumps, repeated offload mirrors, transport bundles,
  large training archives, and duplicate historical ZIP/TAR packages.
- Conda/virtual environments, downloaded dependencies, `site-packages`, IDE
  metadata, bytecode, test caches, render caches, and temporary worktrees.
- Passwords, API tokens, cookies, private keys, credential files, remote-login
  material, host addresses, and machine-specific authorization records.
- Third-party source-paper PDFs. Citations and project-authored notes remain.

## Scientific boundary

The formal five-seed package is authoritative for paper claims. The separate
15-epoch/batch-2 directional screen is only a local feasibility result and is
labelled `NO_GO`. No artifact in this repository turns that short run into a
formal result, selects a best epoch, or changes the frozen split, seed,
evaluation definition, or terminal endpoint.

The repository does not claim an official KITTI test-server score, guaranteed
external generalisation, a valid plagiarism percentage, an AI-authorship
percentage, conference acceptance, IEEE Xplore publication, or EI Compendex
indexing.

## Integrity

`ARCHIVE_MANIFEST.sha256` and `ARCHIVE_MANIFEST.json` cover every committed
artifact except the manifest files themselves. Hashes are computed from the
Git index blob bytes, so they describe the exact content stored by Git rather
than a platform-specific working-tree line-ending view. Regenerate them after
an intentional archive update and review the Git diff before committing.
