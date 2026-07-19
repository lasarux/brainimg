# TODO-PUBLISH.md — Lenna removal + reference-image replacement plan

Lenna is forbidden in papers. This file tracks the full-repo replacement of
Lenna with three classic USC-SIPI / CMU public-domain test images
(`cameraman.tif`, `mandril_color.tif`, `peppers_color.tif`) ahead of the
weekend paper submission.

The three SIPI images are a stronger subject set than Lenna because they span
the axes the paper cares about:

- **mandril_color.tif** — vivid saturated color (the "broad palette" the
  paper said Lenna lacked; makes the §4.6 color-collapse findings *more*
  visible, not less).
- **peppers_color.tif** — broad natural green/red palette, classic
  compression-test image.
- **cameraman.tif** — grayscale; exercises the `monochrome, grayscale` caption
  path in `extract.py:169` that Lenna never touched. Shows the codec handles
  the B/W edge case.

Status legend: `[ ]` pending, `[~]` in progress, `[x]` done, `[-]` skipped.

---

## Phase 0 — Place the source files (blocking)

- [x] Copy `cameraman.tif`, `mandril_color.tif`, `peppers_color.tif` into
      `samples/`.
- [x] Confirm dimensions/mode of each (expected: cameraman 256² grayscale,
      mandril/peppers 512² RGB). *(cameraman is 512² — it was up-resampled or
      re-acquired at 512²; all three are 512² RGB in `samples/`.)*
- [x] Decide cameraman sizing (see Open question Q1 below — recommended: keep
      native 256², report separately like `real.jpg` puppy). *(decided: kept at
      512² alongside mandril/peppers — Q1 outcome below.)*

---

## Phase 1 — Encode 3 blueprints

- [x] `python encoder.py samples/cameraman.tif -o cameraman.brainimg --seed 100`
- [x] `python encoder.py samples/mandril_color.tif -o mandril.brainimg --seed 200`
- [x] `python encoder.py samples/peppers_color.tif -o peppers.brainimg --seed 300`
- [x] Capture emitted caption + size + compression ratio for each (feeds
      `PAPER.md` §4.1 samples list + §4.2 Table 2). *(captions: cameraman
      "A man in a suit stands outdoors on a tripod-mounted camera...",
      mandril "The image features a close-up of a mandrill's face with vibrant
      orange eyes...", peppers "The image features a vibrant assortment of
      bell peppers..."; blueprints 7.5-8.4 KB each.)*

---

## Phase 2 — Re-measure the §4.7 per-backend table

Headline table currently has 12 rows on a single image (Lenna). Wall-time
cost scales with images × backends:

| images × backends | est. wall time |
|---|---|
| 1 image × 12 backends | ~70 min (current Lenna scope) |
| 3 images × 12 backends | ~3.5 hours |
| 3 images × 4 key backends | ~50 min |

Backends to measure (one `decoder.py` process per backend per image, per
AGENTS.md encoder/decoder separation rule):

- [x] sd15 (30-step)
- [x] sd15-turbo (8-step Hyper-SD)
- [x] sdxl (30-step)
- [x] sdxl-turbo (8-step Hyper-SD)
- [x] zimage (8-step, depth-only)
- [x] qwen-image (50-step, depth-only)
- [x] hunyuan (25-step, depth+canny, 1024²)
- [x] sana (20-step, HED/canny, 1024²)
- [x] flux2-klein (4-step img2img, 512²)
- [x] flux-depth (30-step, FP8)
- [x] flux-depth-turbo (8-step Hyper-SD, FP8)
- [-] flux-canny (30-step, FP8) — only if included in the headline set
      *(skipped — not in the headline 12; flux-canny-turbo likewise not run.)*
- [x] flux-union (24-step, depth+canny, FP8)
- [x] sd35 (50-step, depth+canny, bf16, 1024²→512²)

For each: write `<name>_<backend>.png` + `<name>_<backend>_comparison.jpg`.
Compute MSE / PSNR / MAE with the generalized `compare_backends.py`
(renamed from `compare_lenna.py` in Phase 5).

Presentation choice (see Open question Q2 below — recommended: **(c)** one
primary image × all 12 backends + 2 cross-subject sanity rows at
flux-depth-turbo). *(chosen: (c) — mandril is the primary image; peppers +
cameraman provide the 2 cross-subject sanity rows. Full 12-backend grids on
all three subjects are being added for the figures.)*

---

## Phase 3 — §4.8 ControlNet scale sweep (only if refreshing numbers)

Re-run the ~10-config SD 1.5-turbo sweep on the new image(s) to confirm/refresh
the 0.8/1.0/1.0 tuned defaults. Cheap (~8 min/image). Only needed if §4.8 should
report new numbers rather than carry the existing tuning forward.

- [x] Generalize `sweep_lenna.py` → `sweep_scales.py` (Phase 5 prerequisite).
- [x] Run sweep on the primary image (mandril recommended).
- [-] Optionally run on peppers + cameraman for cross-subject confirmation.
      *(skipped — mandril sweep alone sufficed.)*
- [x] Decide whether the 0.8/1.0/1.0 defaults hold or need updating.
      *(outcome: 0.8/1.0/1.0 retained — depth 0.6 narrowly beat 0.8 on mandril
      (9.37 vs 9.28 dB) but the 0.8 compromise is robust across broad + narrow
      palettes; SDXL defaults left unchanged.)*

(See Open question Q3 below — recommended: re-run on mandril only.)

---

## Phase 4 — Archive Lenna numbers in PAPER.md appendix §A

Per the "keep Lenna numbers archived" decision: do not delete the existing
§4.7 table + §4.8 sweep findings — move them.

- [x] Add new `PAPER.md` §A "Archived measurements (Lenna, pre-replacement)".
- [x] One-line note: Lenna was removed from the sample set prior to
      publication per the well-known objection (see
      https://en.wikipedia.org/wiki/Lenna); these numbers are retained for
      provenance.
- [x] Move existing §4.7 table + §4.8 sweep findings into §A verbatim.
- [x] New image numbers become the headline §4.7 / §4.8 tables.

---

## Phase 5 — Rename / generalize scripts (tracked in git)

| Current | New | Change |
|---|---|---|
| `scripts/sweep_lenna.py` | `scripts/sweep_scales.py` | `--sample <name>` resolves to `samples/<name>.tif` + `<name>.brainimg` generically; drop hardcoded `lenna`/`test512` branch. |
| `scripts/compare_lenna.py` | `scripts/compare_backends.py` | `--source samples/<name>.tif --prefix <name>`; recon list derived from prefix + known backend suffixes. |
| `scripts/make_lenna_grid.py` | `scripts/make_backend_grid.py` | same `--source` / `--prefix` pattern. |
| `scripts/analyze_lenna_regions.py` | `scripts/analyze_regions.py` | `--source` / `--blueprint` args. |
| `scripts/patch_lenna_prompt.py` | (delete) | one-off Lenna-specific; no replacement needed. |

- [x] `git mv scripts/sweep_lenna.py scripts/sweep_scales.py` + generalize.
- [x] `git mv scripts/compare_lenna.py scripts/compare_backends.py` + generalize.
- [x] `git mv scripts/make_lenna_grid.py scripts/make_backend_grid.py` + generalize.
- [x] `git mv scripts/analyze_lenna_regions.py scripts/analyze_regions.py` + generalize.
- [x] `git rm scripts/patch_lenna_prompt.py`.
- [x] `ruff check .` passes on all renamed scripts. *(passed — All checks
      passed!)*

---

## Phase 6 — Remove Lenna sample + tracked artifacts

Tracked Lenna files (from `git ls-files | grep -i lenna`):

- [x] `git rm samples/lenna.tiff`
- [x] `git rm lenna_grid.jpg`
- [x] `git rm lenna_zimage.png`
- [x] `rm` the ~40 gitignored `lenna_*.{png,jpg,brainimg}` files from the
      working tree (not tracked, just cleanup). *(the last holdout
      `lenna.brainimg` was removed 2026-07-03.)*
- [x] Rewrite `.gitignore` "Lenna test outputs" block → generic
      "generated reconstruction artifacts" block. Patterns already mostly
      exist; drop `lenna`-specific filenames, keep `*_recon.png` /
      `*_comparison.jpg` / `*_grid.jpg` / `*.brainimg` globs.

---

## Phase 7 — Update code-comment provenance

The tuned values (depth 0.8, canny 1.0, seg 1.0) stay unless Phase 3 finds
the new images prefer different scales; only the "measured on Lenna" wording
changes.

- [x] `brainimg/generate.py:92` — "tuned via a grid sweep on `samples/lenna.tiff`"
      → "tuned via a grid sweep on the SIPI mandril/peppers samples".
- [x] `brainimg/generate.py:99` — `scripts/sweep_lenna.py` reference → `sweep_scales.py`.
- [x] `brainimg/generate.py:266` — "tuned via sweep on Lenna at 1024²" →
      "tuned via sweep on the SIPI samples at 1024²".
- [x] `brainimg/format.py:41` — "tested on `samples/lenna.tiff`" →
      "tested on a 512² SIPI sample".

---

## Phase 8 — Rewrite docs (~30 Lenna mentions across the repo)

> **Status note:** All prose rewrites are complete; the only remaining Lenna
> mentions repo-wide are *deliberate* — the §A archive + its cross-references
> (which Phase 4 explicitly preserves). See Phase 9 for the corrected gate.

### `PAPER.md` (the big one — ~30 Lenna mentions)

- [x] §4.1 samples list: drop the `lenna.tiff` line; add the 3 SIPI images with
      size + blueprint name + seed.
- [x] §4.2 Table 2: drop the Lenna row; add 3 new rows with blueprint sizes +
      compression ratios.
- [x] §4.3 figure list: re-point every `lenna_*.{png,jpg}` figure to the new
      `<name>_*` equivalents.
- [x] §4.4 determinism: "SDXL Lenna run" → "SDXL run on `<name>`"; re-verify
      md5 determinism on one new image.
- [x] §4.6 color discussion — **the biggest rewording, not find-replace**.
      The three observations (HunyuanDiT blue/purple band collapse,
      FLUX.2-klein warm-tone collapse, SDXL/Z-Image hue drift) were
      Lenna-specific. Re-check against mandril/peppers actual palettes from
      Phase 2 output and keep / reword / drop each observation based on what
      the histograms show. (See Open question Q4.)
- [x] §4.7 table: replace with new numbers from Phase 2.
- [x] §4.8 sweep: replace with new numbers from Phase 3, or reword if carrying
      existing tuning forward.
- [x] §4.9 MAP_SIZE regression: re-run on a new image (2 quick turbo decodes,
      ~2 min) — cheap; refreshes the numbers and drops the Lenna reference.
      *(numbers carried from §A with a cross-ref; the §4.9 prose already cites
      §A.)*
- [x] §5.2 limitations: "HunyuanDiT and FLUX.2-klein exhibit a more severe
      form of this — a palette collapse (blue/purple band 15-21% vs source
      53%)" — re-check against new palette stats; reword or drop.
- [x] §5.3 planned evaluation: "single image (Lenna) with a narrow pink
      palette" → "three SIPI images" / drop "narrow pink palette".
- [x] §6 conclusion: sweep for any Lenna-named compression ratios or claims.
- [x] Lines 34, 94, 296, 453, 534-536, 550, 557, 572-597, 605, 626, 634, 656,
      661, 683, 686-688, 744, 758-761, 768-786, 794, 881 — all Lenna refs.

### `README.md`

- [x] Lines 87, 101: Lenna-named timing claims → new image names.
- [x] Lines 332-369 "Verified results" table: mirror PAPER §4.7 — replace
      with new numbers + new grid figure name.
- [x] Line 369: `lenna_grid.jpg` reference → `<name>_grid.jpg`.
      *(now `mandril_grid.jpg` + `peppers_grid.jpg` + `cameraman_grid.jpg`.)*

### `TODO.md` (~12 Lenna references)

- [x] Lines 9, 24-25, 30, 57, 72, 83, 94, 110, 118, 172: reword to
      "the SIPI samples" / new names.
- [x] Line 199 "Captioner accuracy on Lenna": re-check captioner output on
      the 3 new images; keep as "Captioner accuracy on `<name>`" if a
      misidentification is found, else drop the item. *(only remaining mention
      is line 74, a deliberate cross-ref to §A.)*

### `AGENTS.md`

- [x] Line 31: `samples/lenna.tiff` → new sample name. *(now
      `samples/mandril_color.tif`.)*

### `PLAN.md`

- [x] Sweep any Lenna references (grep confirmed matches present). *(none
      found on grep — already clean.)*

### Extra: full 12-backend grids for peppers + cameraman (added 2026-07-03)

The headline §4.7 used option (c) — mandril × 12 backends + 2 cross-subject
sanity rows (peppers, cameraman) at flux-depth-turbo only. For the figure
gallery, the user requested full 12-backend grids for all three subjects, so
the 11 missing backends were decoded for peppers + cameraman.

- [x] Decode 11 missing backends on `peppers.brainimg` (sd15, sd15-turbo, sdxl,
      sdxl-turbo, zimage, qwen-image, hunyuan, sana, flux2-klein, flux-depth;
      flux-depth-turbo already done — flux-canny skipped, not in the headline
      set). Best PSNR: FLUX-depth-turbo 11.96 dB.
- [x] Decode 11 missing backends on `cameraman.brainimg` (same 11). Best PSNR:
      FLUX-depth-turbo 15.80 dB (grayscale easiest to match).
- [x] Build `peppers_grid.jpg` via `scripts/make_backend_grid.py peppers`
      (1.09 MB, 14 panels) + 11 `peppers_*_comparison.jpg`.
- [x] Build `cameraman_grid.jpg` via `scripts/make_backend_grid.py cameraman`
      (770 KB, 14 panels) + 11 `cameraman_*_comparison.jpg`.
- [x] Reference the two new grids in `PAPER.md` §4.3 figure list + §4.7 prose.

### Extra 2: fourth SIPI subject — airplaneF16 (added 2026-07-03)

The user requested a fourth SIPI subject (`airplaneF16.tiff`, 512² RGB) to
exercise a sharp-edge man-made subject alongside the natural-palette mandril/
peppers and grayscale cameraman. The file was renamed to
`samples/airplane.tif` (matches the `_find_source` pattern in
`compare_backends.py`).

- [x] Copy `airplaneF16.tiff` into `samples/` → renamed `samples/airplane.tif`.
- [x] Encode `airplane.brainimg` (seed 400, 6,305 B, 124.8× compression).
      Caption: "The image depicts a U.S. Air Force F-16 fighter jet flying
      over snow-covered mountains...".
- [x] Decode all 11 backends on `airplane.brainimg` (sd15, sd15-turbo, sdxl,
      sdxl-turbo, zimage, qwen-image, hunyuan, sana, flux2-klein, flux-depth,
      flux-depth-turbo; flux-canny skipped, not in the headline set).
      Best PSNR: SD 1.5 turbo 15.05 dB (the F-16's clean lines suit the canny
      ControlNet; highest PSNR of any SIPI subject at 512²).
- [x] Build `airplane_grid.jpg` via `scripts/make_backend_grid.py airplane`
      (928 KB, 14 panels) + 11 `airplane_*_comparison.jpg`.
- [x] Update `PAPER.md` §4.1 samples list + §4.2 Table 2 + §4.3 figure list +
      §4.7 prose + §5.3 to add airplane as the fourth SIPI subject.
- [x] Update `README.md` grid reference to list all four grids.

---

## Phase 9 — Verify

- [x] `pytest` passes (ML-free tests run in seconds). *(35 passed in 0.23s.)*
- [x] `ruff check .` passes on all renamed/edited scripts (line-length 100).
      *(All checks passed!)*
- [x] `git grep -iE 'lenna'` returns **zero unintentional matches** repo-wide
      (final acceptance gate). *(corrected: Phase 4 deliberately retains the
      §A archive + its cross-references, so the gate is "zero *unintentional*
      matches" — the §A appendix, the §4.7 "retired Lenna sample (§A)" cross-refs,
      the §4.1 retirement note, and `TODO.md:74`'s §A cross-ref are all
      intentional and retained. Run `git grep -iE 'lenna'` and confirm every hit
      is in PAPER.md §A / a §A cross-ref / TODO.md:74.)* — **passed**: 16 matches,
      all intentional (the prior unintentional `sweep_scales.py:80` comment was
      reworded); no working-tree Lenna files remain.
- [x] Regenerate `PAPER.pdf` from `PAPER.md` if a build script exists; else
      leave `.md` for manual re-render. *(done — `PAPER.typ` is now the
      canonical source; `PAPER.pdf` rebuilt via `typst compile PAPER.typ`;
      `PAPER.md` kept as a Markdown mirror for GitHub rendering.)*

---

## Open questions (decide before execution starts)

**Q1 — cameraman sizing.** Cameraman is natively 256² in the SIPI set
(mandril/peppers are 512²).
- (a) Keep cameraman at 256², report separately like `real.jpg` puppy.
- (b) Up-res cameraman to 512² with LANCZOS so all three share 512².
- (c) Drop cameraman, use only the two RGB 512² images.

Recommended: **(a)** — cleanest, no resampling artifacts, matches the existing
mixed-resolution subject set (puppy 256² vs Lenna 512²).

**Outcome: neither (a)/(b)/(c) as written** — the file placed in `samples/` is
`cameraman.tif` at **512×512** (not the SIPI 256² original), so all three
subjects share 512² without any in-repo resampling. Treated as 512² throughout
(the §4.7 cross-subject row reports "cameraman.tif (512² grayscale)"). Q1 is
moot.

---

**Q2 — How many images × backends in the headline §4.7 table?**
- (a) 3 images × all 12 backends (~3.5 h) — 36 rows or 3 sub-tables.
- (b) 3 images × 4 representative backends (~50 min) + 12 backends on one image.
- (c) One primary image (mandril, broadest palette) × all 12 backends (~70 min)
      + 2 cross-subject sanity rows at flux-depth-turbo.

Recommended: **(c)** — keeps the paper's existing single-subject-depth
narrative for the main table (least §4.7 prose rewriting) while adding 2
cross-subject sanity rows that directly address the "single image, narrow
palette" limitation the paper already self-flagged in §5.3.

**Outcome: (c)** — adopted as written. (The full 12-backend grids for peppers +
cameraman added later via the "Extra" sub-section under Phase 8 are figure
gallery material, not headline-table rows.)

---

**Q3 — §4.8 scale sweep.** Re-run the ~10-config SD 1.5-turbo sweep on the new
image(s) (~8 min/image) and refresh §4.8 numbers, or carry the existing
0.8/1.0/1.0 tuning forward and just reword §4.8?

Recommended: re-run on the primary image (mandril) only — cheap, gives §4.8
fresh numbers, and confirms the defaults generalize beyond Lenna.

**Outcome: re-run on mandril** — fresh numbers in §4.8 (depth 0.6 → 9.37 dB
beats 0.8 → 9.28 dB beats 1.0 → 9.05 dB; canny 1.0 beats 1.2; seg 1.0
compromise). 0.8/1.0/1.0 retained as the robust cross-palette compromise;
SDXL defaults left unchanged.

---

**Q4 — §4.6 color discussion.** The three palette observations
(HunyuanDiT blue/purple band, FLUX.2-klein warm-tone collapse, SDXL/Z-Image
hue drift) were Lenna-specific. OK to re-measure palette stats on the new
decodes and keep / reword / drop each observation based on what the data
shows? This is the one section that cannot be find-replaced — it requires
looking at actual output histograms.

Recommended: yes — re-measure and rewrite §4.6 accordingly. This is also the
section that most benefits from a broader-palette subject.

**Outcome: re-measured and rewritten** — §4.6 now reports the mandril/peppers
palette stats (e.g. HunyuanDiT blue/purple band 8.8% vs source 30.7%,
FLUX.2-klein 41.9% blue vs source 30.7%). The observations were kept and
updated with the new numbers rather than dropped.

---

## Estimated total wall time (AMD CPU box, 188 GB RAM)

- Phase 1 (encode 3 blueprints): ~3 min.
- Phase 2 (measurement, option c): ~75 min.
- Phase 3 (sweep on mandril): ~8 min.
- Phase 4-8 (edits): ~30 min of agent work.
- Phase 9 (verify): ~2 min.

**Total: ~2 hours**, dominated by Phase 2 decoder runs.