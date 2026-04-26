# Extension Plan — 2D Learned Index + Three-Way Representation Comparison

**Status:** Planned — awaiting greenlight to start Phase 1
**Date drafted:** 2026-04-25
**Owner:** Jiayi (jh9141@nyu.edu)
**Builds on:** Notebook 04 (supervised contrastive 64D, Approach B), Notebook 06 (MJJAS)

---

## Motivation

Three open questions from the advisor's feedback:

1. **Is BSISO genuinely 2D?** Conventional indices (PC1, PC2) are *constructed* to be 2D. We can test this by training the encoder to output 2D directly and comparing against the existing 64D + t-SNE result.
2. **What is each representation actually capturing?** Three candidates exist:
   - **Conventional**: APEC (PC1, PC2) — built from labels.
   - **Supervised contrastive**: pairs defined via BSISO phase + ENSO category.
   - **Self-supervised temporal**: pairs defined by temporal proximity only (advisor's MJO method).
3. **Which is most informative for downstream tasks?** Forecast skill on East Asian precipitation is the eventual gold standard; linear probe accuracy on atmospheric structure is the immediate proxy.

---

## Design decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| 2D model architecture | Encoder output dim = 2 (option (a)) | Forces the network to put everything in 2 numbers — direct test of 2D-sufficiency |
| SSL positive pair window | days in [d−3, d+3] \ {d} | User-specified; tight enough to capture intraseasonal continuity |
| SSL bandpass filter | 25–90 day Lanczos, per grid point per channel | Faithful to advisor's MJO method; isolates BSISO band |
| SSL negative sampling | Unrestricted (any \|Δd\| > 30, same year) | Lee preprocessing + bandpass already remove seasonal cycle confound; no need for same-month restriction |
| Data scope | MJJAS (May–Sep) | Better Phase 5 coverage than July-only |
| Preprocessing | **Lee et al. (2013)** — `X_MJJAS_lee.npy` (3-harmonic Fourier annual cycle removal + 120-day running mean + area-averaged std normalization) | Approach B is July-only; Lee is the existing MJJAS pipeline and more rigorous (also strips slow ENSO mean state via 120-day running mean) |
| Train/val split | Year-based, every 5th year held out (no leakage) | Recent fix; regenerated deterministically per notebook to be robust against cache overwrites |

---

## On the linear probe (clarification for the writeup)

A frozen representation feeds a single linear classifier; train on (embedding → target), report held-out accuracy. The point is **strictness**: a linear classifier can only read information that's linearly arranged in the representation. High accuracy = "this structure is laid out cleanly," not "a powerful classifier eventually decoded it."

**Fairness wrinkle for the three-way comparison:**
- Supervised model is trained *using* BSISO phase labels → phase-at-lag-0 probe rewards what it was directly told.
- SSL model never sees phase labels → phase-at-lag-0 asks "does temporal smoothness alone recover phase?"
- Conventional index *is* phase → phase-at-lag-0 is trivially perfect.

Mitigation: probe at **lag +5 and +10 days** as well — none of the three were trained on these, so it's a level playing field, and it's a step toward forecast skill.

---

## Phase 0 — Prep (no new training)

- [ ] Verify MJJAS Approach B array exists (`X_MJJAS_B.npy` from notebook 06); confirm year-based split indices are reproducible from a saved seed.
- [ ] Create `notebooks/extension_2d/` with three planned notebooks: `07_supervised_2d.ipynb`, `08_ssl_temporal_2d.ipynb`, `09_threeway_comparison.ipynb`.
- [ ] Save the conventional (PC1, PC2) BSISO vector from `labels.csv` aligned to the MJJAS frame indices — this is "representation #1" for Phase 3.

---

## Phase 1 — Supervised 2D model (sanity check)

**Goal:** Does squeezing the encoder to 2D preserve the structure currently visible in 64D + t-SNE?

1. Copy notebook 04 training pipeline. Change encoder's final FC from `→ 64` to `→ 2`. Everything else identical: InfoNCE temperature, ENSO-stratified hard negatives, 50 epochs, MJJAS Approach B inputs, year-based split.
2. Train. Save embeddings for all train + val frames.
3. **Diagnostics:**
   - 2D scatter of val embeddings, colored by BSISO phase (1–8) and by ENSO category. Compare visually to the existing 64D + t-SNE figure (`results/lee/tsne_overview.png`).
   - Linear probe **BSISO phase** on val → compare to **67.7%** (Lee MJJAS 64D, year-based split baseline).
   - 5-fold GroupKFold-by-year CV → compare to **68.6% ± 1.0%**.
   - ENSO **balanced accuracy** linear probe on val (new — addresses class-imbalance issue flagged in `analysis_results_lee_mjjas.md`).
   - ENSO displacement z-score in 2D space (recompute notebook 05's statistic) → compare to **3.83**.
4. **Decision criterion:**
   - If 2D probe accuracy is within ~5pp of 64D (i.e. ≥ ~62%) and z-score is similarly large (≥ ~3) → BSISO-as-encoded-by-supervised-CNN is genuinely ≤2D. Proceed to Phase 2.
   - If 2D probe accuracy collapses (e.g. drops to ~30%) → BSISO is not 2D in this representation. Run Phase 4 dimension sweep to find the elbow before continuing.

---

## Phase 2 — Self-supervised temporal 2D model

**Goal:** Does temporal continuity alone — no labels — discover the same phase structure?

1. **Bandpass preprocessing:** Apply 25–90 day Lanczos bandpass to **Lee** anomalies along the time axis, per grid point, per channel. Compute on the full record before splitting (pad-and-trim for filter edge effects). Save as `X_MJJAS_lee_bp.npy`.
2. **Positive-pair sampler:** For each anchor day d, positive = uniform random day in [d−3, d+3] \ {d}, restricted to the same year (no cross-year leakage).
3. **Negative sampler:** Random day with |Δd| > 30, same year as anchor. (Unrestricted by month — see design decision rationale.)
4. **Architecture:** Same 2D encoder as Phase 1.
5. **Loss:** InfoNCE (consistent with current setup).
6. **Train, save embeddings.**
7. **Diagnostics:**
   - 2D scatter colored by BSISO phase, by ENSO category, **and by calendar month** (May/Jun/Jul/Aug/Sep — the new diagnostic). If month-clustering dominates, the seasonal-cycle confound was not fully removed and we revisit negative sampling.
   - Linear probe BSISO phase, ENSO balanced accuracy on val.

---

## Phase 3 — Three-way comparison (the headline result)

**Three representations** (all 2D, all aligned to same MJJAS val-year frames):

| # | Name | Definition |
|---|---|---|
| 1 | Conventional | APEC (PC1, PC2) from `labels.csv` |
| 2 | Supervised contrastive 2D | From Phase 1 |
| 3 | SSL temporal 2D | From Phase 2 |

**Probe targets** (each = independent linear classifier per representation, trained on train years, evaluated on val years):

| Target | Metric | What it tests |
|---|---|---|
| BSISO phase @ lag 0 | Accuracy | Standard convention; biased toward (1) and (2) |
| BSISO phase @ lag +5 days | Accuracy | Forward-looking, no model trained on it |
| BSISO phase @ lag +10 days | Accuracy | Forward-looking, closer to forecast skill goal |
| ENSO category @ lag 0 | Balanced accuracy | Slow-mode information |
| BSISO amplitude @ lag 0 | R² | Continuous regression target |

**Output:**
- A single results table (rows = representations, cols = targets).
- Side-by-side 2D scatter plots (3 panels) colored by phase.
- Optional: same plot colored by ENSO.

**This is the figure for the writeup.**

---

## Phase 4 — Recommended follow-ups (only if 2D held up in Phase 1)

- **Dimension sweep:** train the supervised model with embedding dim ∈ {1, 2, 4, 8, 16, 32, 64}; plot probe accuracy vs dim. The elbow is the intrinsic dimensionality of BSISO-as-modulated-by-ENSO. **Direct test of "is BSISO 2D?"**
- **Phase trajectory geometry:** compute angular velocity in 2D latent space as a function of phase angle; identify slow-down regions (candidates for low-predictability phases).

---

## Phase 5 — Deferred (forecast skill on East Asian precipitation)

**Trigger:** Only after Phase 3 results are in and at least one learned representation looks competitive with the conventional index.

1. **Download precipitation:** Recommend ERA5 `total_precipitation` first (same source/grid/period as circulation fields, lowest friction — extends notebook 01). Alternatives if higher-quality observations are needed:
   - **APHRODITE V1101** — observation-based, 0.25° over Asia, but ends 2015 (truncates record).
   - **GPCP 1DD** — 1°, observation-based, 1996–present.
2. **East Asia box:** e.g. 100–145°E, 20–45°N (revisit based on BSISO precipitation footprint).
3. **Linear probe** from each representation @ lag 0 → area-averaged precipitation @ lag +10 / +20 days.
4. **Comparison:** does any learned representation beat the conventional index? If yes — concrete argument the representation captures something real.

---

## Compute budget (Colab T4)

| Phase | Cost | Notes |
|---|---|---|
| 1 | ~30 min | One training run, same as notebook 04 |
| 2 | ~30 min training + ~10 min one-time bandpass preprocessing | |
| 3 | seconds | Linear probes only |
| 4 (dim sweep) | ~3.5 hours | 7× Phase 1 cost; one Colab session |
| 5 | ~30 min download + ~30 min training | After greenlight |

---

## Open methodological notes

- **MJJAS lag probes near MJJAS boundaries:** lag +5/+10 day probes will lose samples at the end of September (target falls in October). Either accept the sample loss or extend the input record by one month for probe purposes only.
- **Bandpass edge effects:** Lanczos filter has finite support; first/last ~45 days of each year (or of the full record, depending on implementation) are unreliable. Either pad with reflection or drop edge frames from training.
- **SSL model and ENSO:** the SSL model does not see ENSO labels, so the existing ENSO displacement z-score statistic is still a valid downstream test of whether ENSO modulation emerges *without* being told to look for it. This would be a strong result if it does.

---

## Status checklist

- [ ] Phase 0 complete
- [ ] Phase 1 complete — 2D supervised trained, diagnostics vs 64D recorded
- [ ] Phase 1 decision logged (BSISO is / is not 2D in this representation)
- [ ] Phase 2 complete — SSL temporal trained, month-clustering diagnostic logged
- [ ] Phase 3 complete — three-way probe table + scatter figure produced
- [ ] Phase 4 dim sweep (optional)
- [ ] Phase 5 forecast skill (deferred)
