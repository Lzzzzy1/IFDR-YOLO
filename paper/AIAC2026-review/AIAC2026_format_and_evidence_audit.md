# AIAC 2026 format and evidence audit

Audit date: 2026-09-03 (Asia/Shanghai)

## Deliverables

- `P2-YOLOv8_AIAC2026_Review_Manuscript.docx`
- `P2-YOLOv8_AIAC2026_Review_Manuscript.pdf`

The manuscript is a double-blind review version. The visible author line and document metadata are anonymized; no affiliation, funding, e-mail address, or old target-conference identity is present in the title block, headers, footers, or file names.

## Official AIAC authority used

- Conference home: <https://www.icaiac.org/>
- Submission instructions: <https://www.icaiac.org/submission>
- Official downloads: <https://www.icaiac.org/download>
- Editorial policy: <https://www.icaiac.org/EditorialPolicy>
- AI-tool policy: <https://www.icaiac.org/GuidelinesforAITools>
- Publication statement: <https://www.icaiac.org/Publication>

The official AIAC Word template downloaded from the conference website is the layout authority. Its SHA-256 is `8a2eb3467b19a909a5a431be2f2e9f6ba3a512a9b3be8ad04aa8ee7a12a5416c`. The downloaded author-instructions PDF has SHA-256 `0c13c1a68d78c5ddadde037f274075f63f7443120002ee78e2c6b64fad8a39e8`.

AIAC states that review is double-blind and that generative-AI assistance must be disclosed. The manuscript therefore uses `Anonymous Author(s)` and includes a specific disclosure in the experimental-protocol section. The disclosure names OpenAI Codex (GPT-5), the access month, the permitted writing/formatting functions, the authors' verification responsibility, and the fact that the tool did not generate or alter data, runs, checkpoints, evaluation outputs, or statistics.

The conference publication page describes possible IEEE Xplore inclusion subject to IEEE requirements; neither IEEE Xplore inclusion nor EI Compendex indexing is represented here as guaranteed.

## Layout acceptance

- PDF pages: **7**, including references and AI disclosure; hard limit requested by the user: **8**.
- Paper size: **US Letter, 612 × 792 pt**.
- Margins in every section: left/right **0.625 in**, top **0.75 in**, bottom **1.00 in**.
- Column structure: single-column title/full-width islands alternating with the two-column technical body; the ten section column counts are `1, 2, 1, 2, 1, 2, 1, 2, 1, 2`.
- Two-column gap: **0.25 in**.
- Page numbers: **0**; headers and footers are empty.
- Typography: Times New Roman for prose, 24-pt title, 11-pt anonymous author line, 9-pt abstract/keywords, 10-pt body, 8-pt captions/tables/references; mathematical objects retain equation typography.
- Section hierarchy: six Roman-numbered primary sections and fifteen lettered secondary sections.
- Figures: **2**, both high-resolution embedded grayscale PNGs (`3013 × 1203` and `3013 × 1033`).
- Tables: **6**; all header rows carry Word table-header semantics. The six-column diagnostic table was losslessly reorganized into five columns by combining the mean difference and interval into one statistics column.
- Captions: **2 figure captions** and **4 table captions**, using the full words `Figure` and `Table` and ending with periods.
- References: **18/18** present, **18/18** cited in the body, and **18/18** carry a DOI or an official clickable source link. Hyperlinks render in black without decorative blue styling.
- Accessibility audit: **0 high**, **0 medium**, **0 low** findings after the final pass.

## Color and visual acceptance

The final PDF was rendered at 180 dpi and all seven pages were inspected. The two figures, all tables, equations, text, links, rules, and headings are readable without overlap or clipping.

Pixel scan across **21,205,800** rendered pixels:

- Non-gray pixels at RGB channel-spread tolerance greater than 2: **0**.
- Blue-dominant pixels at channel difference greater than 8: **0**.
- Visible colored elements: **0**.
- Blue decorative elements: **0**.

## Scientific-content freeze

The formatting pass did not change the dataset split, seeds, evaluator, endpoint, training settings, reported evidence, or conclusion boundary. The following frozen values were rechecked in the DOCX and PDF:

- PLAIN_P2: `95.3886 ± 0.5779` Moderate Pedestrian/Cyclist macro AP_R40.
- DCLI: `94.3702 ± 0.8473`.
- Paired PLAIN_P2 minus DCLI mean: `+1.0184` AP.
- Paired 95% t interval: `[-0.4055, 2.4423]`.
- Pair direction: PLAIN_P2 wins `4/5`; DCLI wins seed 0 only.
- Small: `67.5344` versus `56.3170`, mean difference `+11.2175`.
- Far: `57.7167` versus `47.6637`, mean difference `+10.0530`.
- Near: `96.7938` versus `95.9563`, mean difference `+0.8375`.
- Large: `97.1207` versus `96.6724`, mean difference `+0.4483`.

The paper retains PLAIN_P2 as the tested development choice, reports DCLI as a falsified exploratory extension rather than an improvement, and explicitly states that no causal P2 gain is established without a matched three-scale control.

## Integrity hashes

- Final DOCX SHA-256: `db757b23d4cd88b54ac077678a6f4e31064b38f7b39308e14a386fc28912a64e`.
- Final PDF SHA-256: `ffe0d590f0e045852cf82227670be85f02eb2994e796d60fc8b1e4a92f0e9c7b`.

## Verified and not claimed

Verified: Word opens and exports the DOCX without a repair prompt; the PDF has seven Letter pages; every rendered page was inspected; grayscale, structure, references, anonymity, and frozen key evidence pass automated checks.

Not claimed: no Turnitin similarity percentage, third-party AI-detection percentage, official KITTI test-set score, population-level superiority, or guaranteed IEEE/EI indexing. Before a camera-ready submission, the anonymous author block must be replaced with the real author names, affiliations, corresponding-author mark, e-mail addresses, and any required funding/copyright text under the final AIAC instructions.
