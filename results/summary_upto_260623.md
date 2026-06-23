# Project Summary — ENSO Modulation of Tropical Intraseasonal Oscillations via Self-Supervised Learning
*Data-Driven Climate Science · status as of 2026-06-23*

## CV blurb (short)

**Self-Supervised Representation Learning for ENSO Modulation of the MJO and BSISO** — built an end-to-end deep-learning pipeline (PyTorch, ERA5 reanalysis 1979–2023) to learn how ENSO modulates tropical intraseasonal oscillations beyond classical composite analysis. Key results:

- **Reproduced the operational Wheeler–Hendon RMM MJO index** from daily-mean ERA5 (combined EOF), validated against the official Bureau of Meteorology index with rotation-aware (Procrustes + canonical-correlation) comparison: **r ≈ 0.95** (rotation-aligned 0.947/0.971), **canonical correlation 0.97/0.95**, uniform across ENSO states.
- **Showed self-supervised models reveal ENSO-sensitive geometry that supervised models miss**: a label-free SSL 2-D encoder recovered the BSISO cycle *and* a strong ENSO signature (**ENSO displacement z = 14.55, no labels**) where the supervised encoder, despite reproducing the index almost exactly (circular corr ρ = 0.844), showed weak ENSO (z = 2.53).
- **Estimated the intrinsic dimensionality** of the oscillation manifolds with a Neural State Variables approach: **BSISO ≈ 4-D, MJO ≈ 7-D** (ENSO-displacement z up to 20.9).
- **Diagnosed and fixed a contrastive training collapse** (root cause: InfoNCE temperature) and, via a dimension sweep, independently confirmed BSISO phase is ~2-D and the full state ~4-D.
- **Designed a novel temporally-graded Barlow Twins objective**, establishing a complementarity: invariance-SSL recovers the slow ENSO/amplitude envelope (~2-D), while contrastive/prediction methods capture the fast phase cycle.

*Tools: PyTorch, scikit-learn, xarray, Copernicus CDS/ERA5, Google Colab.*

---

## Detailed results

### 1. Data & methods
- **Fields:** ERA5 u850, v850, OLR (BSISO domain 60–160°E, 0–60°N) and u850, u200, OLR (MJO global 15°S–15°N), 1979–2023, daily-mean (migrated from 12 UTC snapshots).
- **Labels:** APEC BSISO index (PC1/PC2, phase 1–8, amplitude); NOAA Niño-3.4 ENSO category (El Niño / Neutral / La Niña); Wheeler–Hendon RMM for MJO.
- **Preprocessing:** Lee et al. style; Lanczos lowpass (lp25, BSISO) / bandpass (bp20–90, MJO); year-based train/val split (every 5th year held out) to avoid leakage.
- **Models:** Siamese/contrastive CNNs (InfoNCE), supervised CNNs, Neural State Variables (lag-prediction autoencoder + intrinsic-dimension estimation), Barlow Twins.

### 2. MJO RMM index reproduction (nb24)
Full-record (1979–2023) combined EOF reproduces WH04: PC1/PC2 variance **13.1%/12.7%** (WH04 12.8/12.2). Because PC1≈PC2 (near-degenerate), per-component correlation is rotation-sensitive; a rotation-aware comparison gives **r = 0.947/0.971**, **canonical correlation 0.97/0.95**, phase exact 83% / within-±1 100%, and **no ENSO-dependent bias** (El Niño 0.94/0.96, Neutral 0.96/0.98, La Niña 0.95/0.98). Lag-correlation peak +9 d and a 30–80 d spectral peak confirm eastward propagation.

### 3. Supervised vs self-supervised 2-D representations — BSISO (nb07–09)
- **Supervised 2-D** (trained on BSISO phase labels): reproduces the BSISO index geometry almost exactly (**circular correlation ρ_c = 0.844** at lag 0) but carries weak ENSO modulation (**z = 2.53**).
- **SSL temporal 2-D** (label-free, contrastive on time-shifted pairs): captures the BSISO cycle (**ρ_c = 0.305**, 42/61 lags significant) *and* a strong ENSO signature (**ENSO displacement z = 14.55**) — i.e. the unsupervised model discovers ENSO-sensitive geometry the supervised model does not. The lower ρ_c is *because* the SSL ring devotes angular variance to ENSO, orthogonal to phase.
- **MJO-average SSL**: ENSO z = 18.74.
- 64-D supervised BSISO phase probe reaches **67.7%** (vs 12.5% random).

### 4. Neural State Variables — intrinsic dimension
Lag-10 prediction encoder forces extraction of the slow predictable state (avoids the lag-1 persistence-trivialization failure). Intrinsic-dimension estimation (Levina–Bickel / TwoNN / lPCA, N-aware): **BSISO d̂ = 4, MJO d̂ = 7**. ENSO recovered in the refined latent space with **z = 12.5 (BSISO), 20.9 (MJO)** — exceeding the supervised baselines.

### 5. Supervised-2-D collapse fix + dimension sweep (nb07c–e)
- **Collapse diagnosed:** the 2-D, non-normalized, raw-dot InfoNCE encoder collapsed onto a line. Root cause = **temperature** (τ=0.5 too soft); a sharp transition near τ≈0.1 unlocks the full 2-D embedding. (Not the daily-mean migration — input renorm was a no-op.)
- **Fixed recipe:** vicreg-regularized / batch 64 / τ=0.07 / cosine / no early stopping → ~52–57% phase, no collapse.
- **Dimension sweep {1…64}:** BSISO *phase* is essentially 2-D (an angle); the full state and ENSO modulation are higher-dimensional, consistent with the NSV d̂ = 4. Method note: judge near-degenerate EOF agreement by Procrustes + canonical correlation, never per-component r.

### 6. Temporally-graded Barlow Twins — MJO (nb26)
Novel loss: both invariance and redundancy terms weighted by a τ-graded schedule λ(τ) over day-lags τ∈{1…5}, on a D=7 (= MJO d̂) cross-correlation, warm-started from the NSV encoder. Finding: **invariance-SSL on temporal-lag views is a slow-feature extractor** — it recovers the slow **ENSO/amplitude envelope** (ENSO modulation effectively **~2-D**, robust across D=3/7 and three runs; at D=3 the PCA cleanly separates the three ENSO categories) but structurally **cannot represent the cyclic MJO phase** (invariance penalizes exactly what propagates; τ-decay does not fix it). This establishes a clean **complementarity**: invariance-SSL → slow envelope; contrastive/prediction-SSL → fast phase.

### Headline numbers
| Quantity | Value |
|---|---|
| RMM reproduction (rotation-aligned r) | 0.947 / 0.971 (canonical 0.97/0.95) |
| Supervised 2-D ↔ BSISO index (circular ρ) | 0.844 |
| SSL 2-D ↔ BSISO index (circular ρ) | 0.305 |
| Supervised 2-D ENSO z | 2.53 |
| **SSL 2-D ENSO z (no labels)** | **14.55** |
| MJO-average SSL ENSO z | 18.74 |
| NSV ENSO z (BSISO / MJO) | 12.5 / 20.9 |
| Intrinsic dimension (BSISO / MJO) | 4 / 7 |
| Barlow-Twins ENSO modulation | effectively ~2-D |
| BSISO phase probe (64-D supervised) | 67.7% |
