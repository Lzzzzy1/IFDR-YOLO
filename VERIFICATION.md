# GitHub archive verification

Verification date: 2026-09-04 (Asia/Shanghai).

## Executed checks

- Repository unit suite: `310/310 PASS` under Python 3.11.15. The one test
  requiring the frozen upstream `yolov8m.pt` was executed only after verifying
  the local artifact SHA-256 against
  `5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5`.
  That temporary test copy was removed before the archive scan and is ignored
  by Git.
- Selected formal-runtime source, formal raw evidence, and copied artifact
  trees were compared against their source files by SHA-256: pass.
- Text secret scan for GitHub tokens, cloud access keys, private-key headers,
  passwords, API keys, authorization values, and secret assignments: no real
  credential found. A deliberately fake `example.invalid` credential-shaped
  URL remains in a unit-test fixture and cannot authenticate anywhere.
- Filename scan for environment files, credential stores, cookies, private
  keys, model/checkpoint extensions, raw-dataset directories, downloads,
  caches, and training-run directories: no included match.
- DOCX internal XML scan for token/private-key signatures and unrelated-project
  manuscript names: zero matches.
- Cross-project filename/text scan: zero unrelated manuscript/project matches.
- Size scan of the Git commit candidate: 1,011 files, approximately 27.28 MiB;
  zero files above 50 MiB and zero files above 100 MiB. The largest included
  artifact was approximately 1.59 MiB.
- Root manuscript/evidence copy audit: pass; full prediction-label dumps,
  ignored logs, raw image data, weights, checkpoints, repeated archives, and
  local caches are absent from the commit candidate.

## Test command

```powershell
python -m unittest discover -s tests -q
```

The repository does not download or redistribute the pretrained weight. A
tester must provide a lawfully obtained file with the recorded hash before
running the single semantic-prefix transfer test.

## Claim boundary

These checks verify archive integrity, source-level behaviour, and faithful
copying. They do not rerun the 10 formal GPU trainings, generate a new AP value,
or convert the local 15-epoch/batch-2 direction screen into formal evidence.
