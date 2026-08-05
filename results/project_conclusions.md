# Project Conclusions — ENSO Modulation of Tropical Intraseasonal Oscillations via Self-Supervised Learning

*Data-Driven Climate Science (NYU) · Jiayi (jh9141@nyu.edu) · synthesis of Sessions 1–59, as of 2026-07-08*

This document concludes the results of the whole project to date. It is organized as: (0) the thesis, (1) data & methods, (2) results by research thread, (3) cross-cutting scientific conclusions, (4) methodological lessons, (5) caveats/retractions/open items, (6) data-pipeline status, and (7) references (code + papers + prior summaries). Numbers and claims are cross-referenced to the specific notebook and log session that produced them.

> **Read-with note on data vintage.** Two data generations exist. The *early* representation-learning results (BSISO supervised/SSL 2-D, and the first NSV `d̂` estimates) were computed on **12:00-UTC snapshot** ERA5. In June 2026 all inputs were migrated to **daily means** (Session 34) and the newer threads (RMM reproduction nb24, collapse-fix nb07c–e, moisture diagnostics nb28–34) were run on **daily-mean full-record** data. Where a headline number is snapshot-era, it is marked *(snapshot)*. Session 35 argues the qualitative conclusions are robust to the migration; exact numbers may shift ±1 in `d̂` and ±2 in z-scores after a full daily-mean NSV rerun (still pending).

---

## 0. Thesis

Can **self-supervised learning** (label-free representation learning on ERA5 fields) discover how **ENSO modulates** tropical intraseasonal oscillations — the **BSISO** (boreal summer monsoon) and the **MJO** — *beyond* what conventional composite analysis and hand-crafted indices (APEC BSISO, Wheeler–Hendon RMM) reveal? The project answers **yes for ENSO *content*** (SSL finds strong, label-free ENSO-sensitive geometry), **with a sharp and reproducible limitation** (SSL does not recover the oscillation's *phase angle* the way a linear EOF index does), and it explains *why* via a unifying **objective→representation principle**.

---

## 1. Data & Methods (foundation)

- **Fields (ERA5, 1979–2023, 2° grid):** BSISO uses u850, v850, OLR over 60–160°E, 0–60°N (domain map 31×51); MJO uses u850, u200, OLR over the global 15°S–15°N strip (meridionally averaged → 1×180). Migrated from 12:00 snapshots to **daily means** (Session 34, notebooks [01](../notebooks/01_era5_download.ipynb)/[01b](../notebooks/01b_era5_download_mjjas.ipynb)/[01c](../notebooks/01c_era5_precip_download.ipynb)/[12](../notebooks/mjo/12_mjo_era5_download.ipynb)).
- **Labels:** APEC BSISO index (PC1/PC2, phase 1–8, amplitude); Wheeler–Hendon RMM for the MJO ([nb11](../notebooks/mjo/11_mjo_rmm_download.ipynb)); NOAA Niño-3.4 ENSO category (El Niño / Neutral / La Niña, ±0.5 K).
- **Preprocessing:** Lee et al. (2013) / Wheeler–Hendon (2004) style — 3-harmonic annual-cycle removal (base 1979–2001), 120-day preceding running-mean background removal, global std-normalization ([nb03](../notebooks/03_preprocessing.ipynb), [nb13](../notebooks/mjo/13_mjo_preprocessing.ipynb)). Lanczos filtering: 25-day lowpass (BSISO SSL/NSV) and 20–90-day bandpass (MJO) to isolate the intraseasonal band.
- **Split:** **year-based** train/val (every 5th year held out) to prevent temporal leakage — adopted Session 11 after diagnosing leakage in random splits.
- **Models:** Siamese/contrastive CNNs (InfoNCE), supervised CNNs, Neural State Variables (lag-prediction autoencoder + intrinsic-dimension estimation + SIREN refine + dynamics MLP), and a temporally-graded Barlow Twins. All training on Google Colab (T4); local machine used only for editing/pushing.

---

## 2. Results by research thread

### A. Reproducing the operational MJO index from ERA5 — "own-RMM" ([nb24](../notebooks/mjo/24_mjo_rmm_pca_from_daily.ipynb), Session 40)

- The full-record (N = 16,436) combined EOF of [u850′, u200′, OLR′] reproduces Wheeler–Hendon: **PC1 = 13.10%, PC2 = 12.69%** of variance (WH04: 12.8/12.2), a clean leading pair above PC3 (5.9%).
- Because PC1 ≈ PC2 (near-degenerate), the (RMM1, RMM2) basis is only defined **up to a rotation**; per-component correlation is therefore rotation-sensitive and initially looked poor (0.785/0.793). After **orthogonal-Procrustes rotation alignment** to the official BoM index: **RMM1 r = 0.947, RMM2 r = 0.971**, amplitude r = 0.909, **phase exact 83% / within-±1 = 100%**, and **canonical correlations = 0.97 / 0.95** (rotation-free subspace match).
- **No ENSO-dependent bias:** El Niño 0.936/0.955, Neutral 0.955/0.975, La Niña 0.949/0.978 — index quality does not vary with ENSO, so downstream ENSO analyses are not confounded by it.
- **Adopted `mjo_rmm_own_pcs.npy` (rotation-aligned) as the canonical, validated, *linear* MJO phase clock** used by all later diagnostics. This also validated the daily-mean migration and the `X_MJO` pipeline.

### B. Supervised vs self-supervised 2-D representations — BSISO ([nb07c](../notebooks/extension_2d/07c_supervised_2d_no_l2norm.ipynb), [nb08](../notebooks/08_ssl_temporal_2d.ipynb), [nb09](../notebooks/extension_2d/09_lag_correlation.ipynb); Sessions 13–18)

The central "SSL advantage" finding (all *(snapshot)* era, Lee MJJAS, N ≈ 6,579):

- **Supervised 2-D** (trained on BSISO phase labels) reproduces the BSISO index geometry almost exactly — **circular correlation ρ_c = 0.844** at lag 0 — but carries **weak ENSO modulation (displacement z = 2.53)**.
- **SSL temporal 2-D** (label-free, InfoNCE on ±3-day pairs) shows *lower* index alignment (**ρ_c = 0.305**, 42/61 lags significant) but **far stronger ENSO sensitivity — displacement z = 14.55, with no ENSO labels**. El Niño and La Niña years systematically occupy different angular arcs of its embedding ring.
- **Interpretation:** the SSL ring spends angular variance on an ENSO axis *orthogonal* to phase — the unsupervised model discovers ENSO-sensitive geometry the supervised model does not. The 64-D supervised phase probe reaches **67.7%** (vs 12.5% chance) but is also a weak ENSO separator (z = 3.83).
- **Lag cross-correlation (nb09):** supervised ↔ index nearly synchronous; SSL significantly *anti-correlated* with both the index and the supervised embedding — consistent with SSL encoding a rotated/ENSO-loaded frame.

### C. Contrastive training collapse — diagnosis & fix, then a dimension sweep ([nb07d](../notebooks/extension_2d/07d_sup_2d_collapse_sweep.ipynb), [nb07e](../notebooks/extension_2d/07e_sup_dim_sweep.ipynb); Sessions 37–42)

- On daily-mean data the 2-D, non-L2-normalized, raw-dot-product InfoNCE encoder **collapsed onto a line**. A 22-config sweep isolated the cause: **temperature**, not the daily-mean migration. Sharp phase transition — every τ ≥ 0.1 collapses (effective rank ≈ 1.0); every τ = 0.07 is healthy (eff_rank ≈ 2.0). Unit-variance renorm was a **no-op** (Lee preprocessing already standardizes).
- **Mechanism:** without normalization the net controls both direction and magnitude, so a soft (large-τ) softmax tolerates a lazy 1-D minimizer (encode class by signed magnitude on one axis); a sharp (small-τ) softmax penalizes every too-close negative, forcing a spread into 2-D.
- **Fixed recipe:** VICReg (variance hinge + covariance decorrelation) + batch 64 + **τ = 0.07** + cosine LR + **no early stopping** → ~52% phase, z ≈ 6, no collapse. **Early stopping on val loss is catastrophic here** (the embedding spreads *after* the InfoNCE val loss plateaus). Fix ported into [nb07c](../notebooks/extension_2d/07c_supervised_2d_no_l2norm.ipynb)/[nb14](../notebooks/mjo/14_mjo_supervised_2d.ipynb) (Session 47).
- **Dimension sweep {1,2,4,8,16,32,64}:** BSISO **phase is essentially 2-D** (d=2 → 48.7% ≈ 86% of achievable phase signal; phase is an angle) while **ENSO modulation is high-dimensional** (z climbs monotonically 5.8 → 17.2 from d=2 → d=64, no saturation). See §5 for the retracted "sharp elbow at d=4" over-claim.

### D. MJO three-way comparison — RMM vs supervised vs SSL ([nb14](../notebooks/mjo/14_mjo_supervised_2d.ipynb)/[nb15](../notebooks/mjo/15_mjo_ssl_temporal_2d.ipynb)/[nb16](../notebooks/mjo/16_mjo_comparison.ipynb); Sessions 22–24)

- The "SSL advantage" **generalizes from BSISO to the MJO**: MJO-average SSL reaches **ENSO z = 18.74**, exceeding the supervised MJO encoder — the ENSO-sensitivity advantage is a general property of temporal SSL, not specific to boreal-summer monsoon dynamics.
- Lag/autocorrelation analysis (nb16, and the lat16 variant nb16b) showed that **meridional averaging is a strong inductive bias**: with averaging, both supervised and SSL converge toward the RMM mode; without it (lat16), SSL discovers a *different* slow mode (e-folding τ_e ≈ 31 d vs 8 d for RMM) decorrelated from RMM at all lags.
- The lat-aware CNN experiments (nb14b/15b, Sessions 25–29) **failed** (embedding collapse from an ill-posed architecture) and were the trigger to pivot to the NSV thread; the clean scientific conclusion recorded is that meridional-average input is what keeps the 2-D methods well-posed.

### E. Neural State Variables — intrinsic dimensionality of the state manifold ([nb17b](../notebooks/nsv/17b_nsv_bsiso_data_lp25.ipynb)→[nb18c](../notebooks/nsv/18c_nsv_bsiso_stage1_lag10.ipynb)→[nb19](../notebooks/nsv/19_nsv_bsiso_id_estimation.ipynb)→[nb20](../notebooks/nsv/20_nsv_bsiso_refine_analysis.ipynb) for BSISO; nb21→22→23 for MJO; Sessions 26–32) *(snapshot)*

Method (after Chen et al. 2022): a lag-10 prediction encoder–decoder (MSE on X_{t+10}) over-parameterizes a 64-D latent; Levina–Bickel / TwoNN / local-PCA estimate the intrinsic dimension `d̂`; a SIREN refine autoencoder compresses 64-D → `d̂`; a dynamics MLP validates predictability; Pearson correlations map each NSV axis to conventional indices.

- **Lag choice was decisive:** lag-1 on Lee data absorbed synoptic noise (`d̂` ≈ 17, low confidence); lag-1 on lp25 data trivialized to persistence; **lp25 + lag-10** was the fix (PC1 = 85.8%, +18.9% over persistence). Lag-10 forces extraction of the slow *predictable* state.
- **BSISO: d̂ = 4** (Levina–Bickel 3.84; SIREN retains **99.32%** of latent variance through a 4-D bottleneck; dynamics MLP **+20.1%** over persistence). Physical axes: v0 ↔ cos(phase) (r = +0.36), v3 ↔ sin(phase) (+0.33), v2 ↔ amplitude (−0.41), **v1 a "mystery axis"** (max |r| = 0.17). ENSO displacement in 4-D v-space **z = 11.01**, indistinguishable from the 64-D supervised baseline (z ≈ 11.0) — compression to 4-D discards no ENSO information.
- **MJO: d̂ = 7** (Levina–Bickel 6.52, medium confidence; SIREN 99.47%; dynamics **+54.7%** over persistence). All 7 axes correlate with RMM phase/amplitude; **ENSO displacement z = 20.88**, exceeding the supervised MJO baseline (12.21).
- **Both indices undercount:** BSISO needs ~4 state dimensions, MJO ~7, vs the conventional 2-D (PC1, PC2) — the 2-D index is a projection of a higher-dimensional state. ENSO is a **distributed, multi-dimensional centroid shift**, not a single linear axis (no NSV dim has |r| > 0.10 with continuous ENSO, yet the permutation z-scores are 11–21).

### F. Temporally-graded Barlow Twins — the phase-vs-envelope dichotomy ([nb26](../notebooks/mjo/26_mjo_barlow_twins.ipynb)/[nb27](../notebooks/mjo/27_mjo_barlow_d7_trajectories.ipynb); Sessions 36, 43–47)

A novel loss: both Barlow invariance and redundancy terms weighted by a τ-graded schedule λ(τ) over day-lags τ ∈ {1…5}, on a D = 7 (= MJO `d̂`) cross-correlation, warm-started from the NSV encoder.

- **Definitive, thrice-confirmed result (D = 7, D = 3, steep-τ):** invariance-SSL on temporal-lag views is a **slow-feature extractor**. It recovers the slow **ENSO/amplitude envelope** (ENSO modulation effectively **~2-D** — participation ratio ≈ 2, one dominant axis ~68% + secondary ~16–24%) but **structurally cannot represent the cyclic MJO phase** (phase probe = chance at every D; the invariance term penalizes exactly what propagates). τ-decay does not rescue it, and BT fine-tuning even *erodes* the warm-start encoder's phase.
- The D = 3 projector PCA is a clean **three-armed "ENSO star"** (arms = El Niño / Neutral / La Niña, radius = amplitude, no phase loop).
- This establishes a clean **complementarity:** invariance-SSL → slow ENSO/amplitude envelope; contrastive/prediction-SSL → fast phase cycle.

### G. Moisture–convection theory diagnostics — which MJO theory, and how ENSO shifts it ([nb28](../notebooks/mjo/28_mjo_moisture_download.ipynb)/[nb29](../notebooks/mjo/29_mjo_moisture_preprocess.ipynb)/[nb30](../notebooks/mjo/30_mjo_latent_moisture_diagnostics.ipynb); Sessions 48–53) — full daily-mean record, 10,177 active MJO days

Motivated by Zhang et al. (2020), *Four Theories of the MJO*. Using own-RMM as the phase clock and the first-harmonic phase offset Δθ(field, −OLR) (negative = field *leads*/east of convection):

- **The observed MJO is moisture-mode-leaning with a skeleton-flavoured low-level recharge:** column moisture Δθ(q_col) = −19/−29/−25° (IO/MC/WP) — **nearly in phase** → moisture-mode; lower-trop Δθ(q_low) = −24/−45/−49° — **leads more, growing eastward** → skeleton recharge. The q_low−q_col gap (−6 → −15 → −24°) **quantifies the skeleton↔moisture-mode tension widening eastward**.
- **Trio-interaction signatures:** Rossby–Kelvin ratio = **1.34** (realistic coupled K–R, not the Gill ≈ 2.2); BL-convergence lead = −41/−60/−15° (convergence leads convection over the warm pool). Internal check: ∂q/∂t leads q_col by ~90° (validates the estimator).
- **ENSO modulation (the distinctive contribution):** the **longitude of the largest moisture lead follows the warm pool** — El Niño shifts active moisture–convection coupling **east** into the West Pacific (q_low −54° vs La Niña −45°); La Niña concentrates it over the **Maritime Continent** (q_low −51° vs El Niño −46°). Robust across q_col and q_low. R–K ratio EN 1.26 < Neutral 1.34 < LN 1.37. Zhang et al. barely treat ENSO, so *ENSO modulation of which-theory-the-MJO-resembles* is a novel result.
- **Do learned latents recover this?** Only the **linear own-RMM** is a clean phase clock (harmonic |A| ≈ 5–6). SSL-nb15 is partial (|A| 2–3.6, circ-corr 0.17); Barlow is **phase-blind** (|A| 0.1–0.7, circ-corr ≈ 0 — its Δθ is noise, a validated negative control).

### H. The objective→representation map — the unifying principle ([nb31](../notebooks/mjo/31_mjo_moisture_aux_latent.ipynb)–[nb34](../notebooks/mjo/34_mjo_barlow_rmm_reconstruction.ipynb); Sessions 54–59)

Attempts to *force* a neural latent's polar angle to equal MJO phase all failed — a robust, instructive negative result — which then produced the project's cleanest general statement.

- **Angle = phase is a linear-EOF / quadrature property, not achievable by contrastive SSL.** own-RMM works because PC1 ⟂ PC2 are the cos/sin of the propagating wave; contrastive "pull temporal neighbours together" never enforces quadrature, so the cycle winds/folds and the angle is not RMM phase. Demonstrated **three ways**: 2-D no-L2 (circ-corr 0.17), 2-D rebalanced with healthy training (0.17), 3-D explicit L2 circle (0.03, phase leaked onto z3).
- **What each objective makes the net encode** (frozen-latent probes + direct u850/OLR field reconstruction, [nb33](../notebooks/mjo/33_mjo_latent_content.ipynb)):

  | objective | phase acc | amp R² | R-K (wind) R² | u850 recon | OLR recon | MJO-band frac | encodes |
  |---|---|---|---|---|---|---|---|
  | Contrastive (SSL nb15) | 0.24 | 0.07 | 0.39 | 0.16 | 0.06 | 0.42 | low-level **wind-phase** (one quadrature component) |
  | + moisture aux (3-D, nb32) | 0.33 | 0.31 | 0.42 | 0.28 | 0.12 | 0.57 | wind-phase **+ amplitude** |
  | Invariance (Barlow) | 0.12 (chance) | 0.08–0.22 | ~0 | ~0 | ~0 | 0.03–0.10 | **slow envelope only** (reconstructs *neither* field) |
  | Lag-prediction (NSV-7D) | **0.46** | 0.34 | **0.58** | **0.53** | **0.21** | 0.60 | **full predictable state** (wind+convection+phase+amplitude) |

  *(baselines: phase 0.125, ENSO 0.333; every latent's ENSO probe ≈ 0.37 — ENSO is a structured signal, not a directly-decodable axis, because the 20–90 d band-pass removed slow ENSO from the input.)*
- **Two findings:** (1) **wind vs convection** — every contrastive latent reconstructs u850 ≈ 2.3–2.5× better than OLR; it latches onto the cleaner low-level *wind* signal and ignores convection longitude entirely, even when trained to predict OLR. NSV (prediction) reconstructs convection ~3× better, because forecasting *needs* it. (2) **Representation richness tracks the information the objective demands** — prediction (richest) > contrastive+aux > contrastive (minimal) > invariance (slow envelope). **The training objective, not the architecture, decides what physics the latent encodes.**
- **Direct confirmation against the operational index ([nb34](../notebooks/mjo/34_mjo_barlow_rmm_reconstruction.ipynb), Session 59):** reconstructing RMM from the Barlow latents, decomposed into amplitude vs phase with linear *and* MLP probes — **phase not recovered at all** (circ-corr ≈ 0, 8-sector at chance ~0.12, angular error ~90°; MLP fails too while recovering the own-RMM-self ceiling perfectly at circ-corr 1.0 → phase is genuinely absent, not nonlinearly hidden), **amplitude partially recovered and scaling with capacity** (D7 ~17–21% of envelope variance, D3 ~6%). Identical under official BoM RMM → robust.

---

## 3. Cross-cutting scientific conclusions

1. **Self-supervised models reveal ENSO-sensitive geometry that supervised models and hand-crafted indices miss** — label-free SSL recovers the oscillation cycle *and* a strong ENSO signature (BSISO SSL ENSO z = 14.55 vs supervised 2.53; MJO-average SSL z = 18.74; NSV z 11–21), across both BSISO and the MJO. This is the project's headline positive result.
2. **The conventional 2-D indices undercount the state.** Intrinsic dimensionality is **BSISO ≈ 4-D, MJO ≈ 7-D** (NSV); the (PC1, PC2) index is a 2-D projection. Phase is an angle (~2-D); ENSO modulation is genuinely higher-dimensional and distributed.
3. **A clean phase *angle* is a linear-EOF property; SSL captures phase/ENSO *content* but not as a polar angle.** This is reproduced across architectures, dimensions, and normalizations — it is structural, not a tuning failure.
4. **Phase-vs-slow-envelope dichotomy by objective:** contrastive/prediction SSL → fast propagating phase; invariance (Barlow) → slow ENSO/amplitude envelope. Unified by *representation richness ∝ information the objective demands* — the project's most general statement.
5. **Contrastive nets track low-level wind, not convection**, because the wind signal is the cleanest temporal discriminator; prediction objectives (NSV) keep convection because forecasting requires it.
6. **The observed MJO is a moisture-mode–skeleton hybrid**, and **ENSO shifts its active moisture–convection coupling zonally with the warm pool** (El Niño → West Pacific, La Niña → Maritime Continent) — a novel, ENSO-focused reading of Zhang et al. (2020).
7. **ERA5 daily-mean fields reproduce the operational MJO index** (own-RMM r ≈ 0.95, canonical 0.95–0.97, ENSO-unbiased), validating the entire data pipeline.

---

## 4. Methodological lessons (reusable)

- **Near-degenerate EOF pairs must be compared by Procrustes + canonical correlation, never per-component r** (per-component r is rotation-sensitive; own-RMM went 0.79 → 0.95 after rotation alignment). Same caution flagged for any 2-D EOF comparison (Session 40).
- **Non-normalized raw-dot InfoNCE needs a sharp temperature (τ ≈ 0.07) + an explicit anti-collapse term (VICReg/Barlow); never early-stop on contrastive val loss** — the embedding spreads *after* the loss plateaus (Sessions 39, 41).
- **Lag choice sets what a prediction encoder learns:** short lag absorbs synoptic noise or trivializes to persistence; lag-10 on lowpassed input extracts the slow predictable state (Sessions 30–31).
- **Judge SSL runs by geometry + downstream probes (eff_rank, phase probe, ENSO z), not raw loss** — batch size shifts the InfoNCE floor and makes raw-loss ranking meaningless.
- **Harmonic amplitude |A| cleanly rank-orders phase-awareness of a latent** (own-RMM ≫ SSL ≫ Barlow); use it as a negative-control validator (Session 53).
- **Always run a 3-seed robustness check before reading an "elbow"** — a single lucky seed produced a false "2-D is enough" and a false "sharp elbow at d=4" (Session 42).

---

## 5. Caveats, retractions, and open items

- **RETRACTED: "sharp elbow at d=4 confirms NSV d̂=4."** The full seed-42 dimension-sweep curve (Session 42 correction) shows a slow concave climb peaking at d=32, *no* sharp saturation at 4. Corrected reading: BSISO **phase** is ~2-D; **ENSO modulation is high-dimensional**; d=4 is a *pragmatic* operating point consistent with the NSV 4-D *state* decomposition (phase 2 + amplitude/ENSO ~2), **not** an independent supervised measurement of `d̂ = 4`.
- **Data vintage:** the NSV `d̂` and ENSO-z headline numbers (BSISO 4/11.0, MJO 7/20.9) are **snapshot-era**; a full daily-mean NSV rerun (nb17b→20, nb21→23) is **pending** (Session 35). RMM reproduction, moisture diagnostics, collapse-fix, and the objective→map are all daily-mean.
- **NSV `d̂` confidence:** estimators disagree (BSISO LB 3.84 / TwoNN 5.36 / lPCA 1.0 → "low"; MJO LB 6.52 / TwoNN 5.49 / lPCA 3.0 → "medium"); `d̂` is a judgement call, and sensitive to preprocessing (lag, filter).
- **The BSISO NSV "mystery axis" v1** (max |r| = 0.17) is real variance with no identified physical correlate.
- **MJO NSV axes are rotationally entangled** (multiple dims load on the same index) — a valid low-D manifold but not an orthogonally-interpretable coordinate system.
- **Precipitation forecast skill is near-zero** (nb10/10b, Session 15–16): direct ACC ≈ 0 for all representations; the usable result there is the EN−LN *composite* difference, not a forecast.
- **Small-N stratification:** ENSO-displacement per-phase uses as few as 3 EN/LN days in some phases; magnitudes have wide CIs even where permutation p-values are valid.

---

## 6. Data-pipeline status (daily-mean migration)

- All four download notebooks switched from a single 12:00 snapshot to a proper **daily average** (instantaneous winds → 4×/day mean; accumulated OLR/precip → 24×/day sum), aggregated in-notebook to preserve one-value-per-day filenames so downstream preprocessing is unchanged (Session 34). Per-request **CDS cost-limit 403s** were fixed by per-year (and half-year for MJO OLR) chunking.
- A one-time migration notebook ([nb00](../notebooks/00_migrate_snapshot_to_daily.ipynb)) moves old snapshot files to `_snapshot12z_backup/` so the skip-if-exists guards let the new code re-download.
- **Status (Session 35 diagnostic):** BSISO branch fully regenerated on daily means (`X_MJJAS_lee*.npy`); MJO winds complete; MJO OLR/moisture completed to full record for nb24/nb28–34. The NSV pipelines still need a daily-mean rerun to refresh their `d̂`/z headline numbers.
- Only **[nb01b](../notebooks/01b_era5_download_mjjas.ipynb) + [nb12](../notebooks/mjo/12_mjo_era5_download.ipynb)** are needed to regenerate all BSISO+MJO analysis (plus [nb01c](../notebooks/01c_era5_precip_download.ipynb) for precip); [nb01](../notebooks/01_era5_download.ipynb) (July-only) is legacy and feeds nothing in the current pipeline.

---

## 7. Headline numbers (quick reference)

| Quantity | Value | Source |
|---|---|---|
| own-RMM ↔ BoM (rotation-aligned r) | 0.947 / 0.971 (canonical 0.97/0.95) | nb24, S40 |
| Supervised 2-D ↔ BSISO index (circular ρ_c) | 0.844 | nb07c/09, S18 |
| SSL 2-D ↔ BSISO index (circular ρ_c) | 0.305 | nb08/09, S18 |
| **SSL 2-D BSISO ENSO z (no labels)** | **14.55** | nb08, S14c |
| Supervised 2-D BSISO ENSO z | 2.53 | nb07c, S18 |
| MJO-average SSL ENSO z | 18.74 | nb15/16, S24 |
| NSV intrinsic dimension (BSISO / MJO) | 4 / 7 | nb19/22, S32 *(snapshot)* |
| NSV ENSO z (BSISO / MJO) | 11.0 / 20.9 | nb20/23, S32 *(snapshot)* |
| NSV dynamics gain over persistence (BSISO/MJO) | +20% / +55% | nb20/23, S32 |
| BSISO 64-D supervised phase probe | 67.7% (vs 12.5% chance) | nb04/05 *(snapshot)* |
| Collapse-fixed 2-D BSISO phase probe | ~50–56% | nb07d/e, S39–42 |
| Barlow-Twins ENSO modulation dimensionality | ~2-D (PR ≈ 2) | nb26, S44–46 |
| MJO Δθ(q_col) IO/MC/WP | −19/−29/−25° (moisture-mode-leaning) | nb30, S51 |
| Rossby–Kelvin ratio (obs) | 1.34 | nb30, S51 |
| Objective→encoding: phase acc (SSL/Barlow/NSV) | 0.24 / 0.12 / 0.46 | nb33, S56–58 |
| Barlow → RMM phase (circ-corr, lin/MLP) | ≈ 0 (phase absent) | nb34, S59 |

---

## References

### Project notebooks (code)
- **Data:** [nb00 migration](../notebooks/00_migrate_snapshot_to_daily.ipynb), [nb01](../notebooks/01_era5_download.ipynb)/[01b](../notebooks/01b_era5_download_mjjas.ipynb)/[01c](../notebooks/01c_era5_precip_download.ipynb) (BSISO ERA5), [nb02](../notebooks/02_labels_download.ipynb) (labels), [nb03](../notebooks/03_preprocessing.ipynb) (BSISO preprocess).
- **BSISO 2-D / collapse:** [nb07c](../notebooks/extension_2d/07c_supervised_2d_no_l2norm.ipynb), [nb07d](../notebooks/extension_2d/07d_sup_2d_collapse_sweep.ipynb), [nb07e](../notebooks/extension_2d/07e_sup_dim_sweep.ipynb), [nb08](../notebooks/08_ssl_temporal_2d.ipynb), [nb09](../notebooks/extension_2d/09_lag_correlation.ipynb), [nb10](../notebooks/10_precip_forecast.ipynb)/[10b](../notebooks/10b_precip_composite.ipynb).
- **BSISO NSV:** [nb17b](../notebooks/nsv/17b_nsv_bsiso_data_lp25.ipynb), [nb18c](../notebooks/nsv/18c_nsv_bsiso_stage1_lag10.ipynb), [nb19](../notebooks/nsv/19_nsv_bsiso_id_estimation.ipynb), [nb20](../notebooks/nsv/20_nsv_bsiso_refine_analysis.ipynb).
- **MJO:** [nb11](../notebooks/mjo/11_mjo_rmm_download.ipynb), [nb12](../notebooks/mjo/12_mjo_era5_download.ipynb), [nb13](../notebooks/mjo/13_mjo_preprocessing.ipynb)/[13b](../notebooks/mjo/13b_mjo_preprocessing_lat16.ipynb), [nb14](../notebooks/mjo/14_mjo_supervised_2d.ipynb)/[15](../notebooks/mjo/15_mjo_ssl_temporal_2d.ipynb)/[16](../notebooks/mjo/16_mjo_comparison.ipynb), [nb24 RMM reproduction](../notebooks/mjo/24_mjo_rmm_pca_from_daily.ipynb), [nb26 Barlow Twins](../notebooks/mjo/26_mjo_barlow_twins.ipynb)/[27](../notebooks/mjo/27_mjo_barlow_d7_trajectories.ipynb).
- **MJO moisture constraints:** [nb28](../notebooks/mjo/28_mjo_moisture_download.ipynb)/[29](../notebooks/mjo/29_mjo_moisture_preprocess.ipynb)/[30](../notebooks/mjo/30_mjo_latent_moisture_diagnostics.ipynb) (diagnostics), [nb31](../notebooks/mjo/31_mjo_moisture_aux_latent.ipynb)/[32](../notebooks/mjo/32_mjo_moisture_aux3d_latent.ipynb) (physics-informed latents), [nb33](../notebooks/mjo/33_mjo_latent_content.ipynb) (latent content / objective→map), [nb34](../notebooks/mjo/34_mjo_barlow_rmm_reconstruction.ipynb) (RMM reconstruction). Folder README: [notebooks/mjo_moisture_constraints/README.md](../notebooks/mjo_moisture_constraints/README.md).

### Prior project documents (results/)
- [conversation_log.md](conversation_log.md) — full session-by-session record (Sessions 1–59).
- [summary_upto_260623.md](summary_upto_260623.md) — CV blurb + detailed results through ~Session 44.
- [mjo_moisture_theory_summary.md](mjo_moisture_theory_summary.md) — moisture-convection & objective→representation synthesis.
- [analysis_results.md](analysis_results.md) / [analysis_results_B.md](analysis_results_B.md) / [analysis_results_lee_mjjas.md](analysis_results_lee_mjjas.md) — early BSISO Approach A/B/Lee results.
- [extension_2d_analysis_report.md](extension_2d_analysis_report.md), [extension_2d_plan.md](extension_2d_plan.md) — 2-D supervised/SSL extension.
- [zhang2020_mjo_theories_extracted.txt](zhang2020_mjo_theories_extracted.txt) — extracted text of the Zhang et al. (2020) review.

### External literature
- Wheeler & Hendon (2004), *An All-Season Real-Time Multivariate MJO Index*, Mon. Wea. Rev. — RMM index.
- Lee et al. (2013), *Real-time BSISO index* (APEC Climate Center) — BSISO preprocessing.
- Chen, Huang, Raghupathi, Chandratreya, Du, Lipson (2022), *Automated discovery of fundamental variables hidden in experimental data*, Nat. Comput. Sci. — Neural State Variables ([arXiv:2112.10755](https://arxiv.org/abs/2112.10755)).
- Zbontar, Jing, Misra, LeCun, Deny (2021), *Barlow Twins: Self-Supervised Learning via Redundancy Reduction*, ICML ([arXiv:2103.03230](https://arxiv.org/abs/2103.03230)).
- Sitzmann et al. (2020), *Implicit Neural Representations with Periodic Activation Functions (SIREN)*.
- Zhang, Adames, Khouider, Wang, Yang (2020), *Four Theories of the Madden-Julian Oscillation*, Rev. Geophys.
- Bardes, Ponce, LeCun (2022), *VICReg* — variance/covariance regularization (anti-collapse term).

---
*Compiled by Claude Code from `results/conversation_log.md` (Sessions 1–59) and the results/ documents above. Marked snapshot-era numbers await the pending daily-mean NSV rerun (Session 35).*
