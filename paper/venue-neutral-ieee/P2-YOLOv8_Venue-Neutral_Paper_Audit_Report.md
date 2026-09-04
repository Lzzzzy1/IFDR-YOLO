# P2-YOLOv8 venue-neutral paper audit report

**Audit date:** 2026-09-02 (Asia/Shanghai)  
**Research state:** venue not locked; scientific evidence frozen; no new training or experiment was launched.

## 1. Delivered manuscript state

The paper has been rebuilt as a venue-neutral IEEE conference-style manuscript. It does not name or imply a submission target on the title page, in the author block, header/footer, filename, metadata, body claim, or template declaration. The primary PDF is generated with the official `IEEEtran` conference class; the editable DOCX mirrors the conference geometry with a one-column title/abstract front matter, a two-column body, full-width method/contract blocks, and column-flow result subtables.

The title, abstract, index terms, introduction framing, contribution paragraph, and negative-result interpretation were adapted for a computer-vision/intelligent-transportation audience. No data partition, seed, hyperparameter, endpoint, evaluation rule, metric, experiment value, confidence interval, engineering-cost measurement, or scientific conclusion was changed.

## 2. Hard acceptance results

| Gate | Result | Evidence |
|---|---:|---|
| Primary PDF total pages, references included | **PASS — 8 pages** | `verification-report.json`; all 8 pages rendered |
| Editable DOCX Word-rendered total pages | **PASS — 7 pages** | Word/WPS PDF QA render; all 7 pages rendered |
| Main PDF page stock | **PASS — US Letter** | 612 × 792 pt on every page |
| Main PDF two-column body | **PASS** | `IEEEtran` conference source and 8-page visual inspection |
| DOCX two-column body | **PASS** | 10 continuous sections with sequence `1,2,1,2,1,2,1,2,1,2` columns |
| Tables and figures preserved | **PASS** | DOCX contains 6 editable tables and 2 embedded registered figures |
| Reference list preserved | **PASS** | Both PDF and DOCX QA reach reference `[18]` |
| Visible colored elements | **PASS — 0** | All 15 rendered pages have 0 pixels with RGB channel gap > 4 |
| Decorative blue elements | **PASS — 0** | 0 blue-dominant pixels across PDF and DOCX QA renders |
| DOCX hidden/latent color | **PASS — 0** | 0 non-gray OOXML colors; 0 theme-color refs; 0 colored highlights; 0 colored raster pixels |
| Crop/edge collision | **PASS** | 15/15 rendered pages nonblank and clear of page edges |
| Embedded PDF fonts | **PASS** | 12/12 detected font programs embedded; 0 unembedded |
| Former submission identities | **PASS — 0** | 0 occurrences of the audited old-identity strings in PDF and DOCX QA text/metadata |
| Frozen numeric evidence | **PASS** | 15 registered tokens present in both PDF and DOCX QA |

### Artifact checksums

- Primary PDF SHA-256: `E2A234FE7C6B17A366128CE591DCDF2161CC9E39C556CC6C79F53EF45701A5C0`
- Editable DOCX SHA-256: `4ADB95713D51C8E8EF76AA3682256BE0F76C852214B7174163008ECDF2C9D987`
- DOCX QA PDF SHA-256: `0E14A3B554F7B8599024D00AF37610820770115995A385A9D25426AC524BA840`

The machine-readable audit is at `work/venue_neutral_ei/verification-report.json`. It records page-level margins, ink fraction, grayscale counts, DOCX package colors, column sections, preserved values, reference reach, and PDF font embedding.

## 3. Scientific freeze audit

The following evidence remains unchanged in both deliverables:

- 3,341 fit images and a disjoint 371-image internal development subset derived from KITTI;
- five paired seeds, epoch 30 `last.pt` only, 640-pixel input and batch 16;
- PLAIN_P2 macro Moderate AP_R40 `95.3886 ± 0.5779`;
- DCLI macro Moderate AP_R40 `94.3702 ± 0.8473`;
- paired PLAIN_P2-minus-DCLI mean `+1.0184`, 95% t interval `[-0.4055, 2.4423]`, four of five positive pairs;
- small diagnostic means `67.5344` versus `56.3170` and far means `57.7167` versus `47.6637`, with wide intervals and sparse support preserved;
- seed-0 local latency change `+18.305%` and training-time change `+26.980%`;
- the 371-image subset is explicitly not the official KITTI test set;
- mechanism activity and finite gradients are explicitly not treated as terminal AP evidence;
- PLAIN_P2 is retained as the tested local development choice;
- DCLI is reported as a failed exploratory extension, not an improvement;
- no causal P2-versus-three-scale effect is claimed because the matched three-scale control is absent.

The audit does not claim population-level superiority, external validity, official KITTI leaderboard performance, cross-dataset transfer, or a universally optimal detector.

## 4. Content and citation audit

The paper contains 18 references. The citation structure was checked against the rendered reference list; both artifact forms reach `[18]`, and the main evidence chain remains tied to the method and evaluation statements. Publication venues appearing inside bibliographic records remain factual reference metadata, not a manuscript submission identity.

The existing seven-paper layout study was retained at `work/conference_paper/PAPER_LAYOUT_LESSONS.md`. It learns only transferable practices from official published papers: compact problem-to-method framing, self-contained captions, bounded contribution language, high information density, compact three-line tables, and the separation of optimizer behavior from endpoint evidence. No sentence, figure, table, or code was copied from those papers.

Text-overlap or AI-style tools can only flag review risk; they cannot produce a valid plagiarism percentage or AI-authorship verdict. This delivery therefore makes no claim about a “guaranteed” similarity score, plagiarism rate, or AI-detection rate.

## 5. Venue and EI audit

The separate candidate ledger evaluates four real conference series using only organizer, IEEE, official submission-system, or Elsevier Engineering Village sources. Its ranked result is:

1. IEEE IV 2027 — best direct intelligent-vehicle/VRU fit, but the current deadline sources conflict and 2027 author rules are incomplete.
2. IEEE ITSC 2027 — equally direct transportation/perception fit, but the edition-specific 2027 CFP and author rules are not yet public.
3. IEEE ICME 2027 — broader vision/multimedia fit; an official 2023 page gives past EI Compendex evidence, while 2027 remains unverified.
4. IEEE ICASSP 2027 — current rules are public, but the 4+1-page ceiling is incompatible with this eight-page intermediate without a separate rewrite.

IEEE states that indexing partners make independent editorial decisions and that IEEE cannot guarantee inclusion in a specific database. Elsevier separately applies Compendex source-selection criteria. The audit therefore records IEEE Xplore publication and EI Compendex evidence as different fields and labels every unverified edition honestly.

## 6. Verified and not yet verified

### Verified now

- primary PDF is 8 total pages, references included;
- editable DOCX Word-rendered output is 7 total pages;
- both artifacts use a real two-column body structure;
- all rendered pages are black/white/grayscale with visible colored elements = 0;
- no audited former submission identity remains in the new artifact names, front matter, headers/footers, metadata, or manuscript text;
- registered figures, tables, references, frozen numbers, and conclusion boundaries are present;
- PDF fonts are embedded and no page is blank or cropped;
- no new training or experimental rewrite occurred.

### Not yet verified / intentionally deferred

- final target conference;
- edition-specific deadline, timezone, paper limit, anonymous-review rule, template revision, and generative-AI policy for IV/ITSC/ICME 2027;
- 2027 EI Compendex inclusion for any candidate;
- the exact Engineering Village source record, which normally requires direct/institutional database access;
- A4 migration, author affiliation, ORCID, acknowledgments, funding statement, ethics/disclosure text, and final camera-ready metadata;
- an official KITTI test-server result, external road dataset result, matched three-scale control, or new experiment.

## 7. Submission boundary and next action

The present PDF and DOCX are verified **venue-neutral intermediate artifacts**, not a claimed camera-ready submission. The next safe action is to lock one specific conference only after its edition-specific official author page and current indexing evidence are available. Then apply that venue’s exact page stock, anonymity, author block, template, policy, and disclosure requirements without changing the frozen scientific evidence.

## 8. Official policy sources

- IEEE conference author templates: https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/
- IEEE publication and indexing boundary: https://events.ieee.org/planning-basics/ieee-conference-publications/publishing-information-for-ieee-conference-authors/
- Elsevier Engineering Village databases: https://www.elsevier.com/products/engineering-village/databases
- Elsevier Compendex selection criteria: https://www.elsevier.com/products/engineering-village/databases/selection-criteria

