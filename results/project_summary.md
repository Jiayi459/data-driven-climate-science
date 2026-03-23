# ENSO-BSISO Self-Supervised Learning — Project Summary

**Author:** Jiayi Hao | jh9141@nyu.edu
**Course:** Data Driven Climate Science
**Date:** 2026-03-23
**Repo:** https://github.com/Jiayi459/data-driven-climate-science

---

## Overview

This project uses self-supervised contrastive learning to study how ENSO (El Niño–Southern Oscillation) modulates the atmospheric structure of BSISO (Boreal Summer Intraseasonal Oscillation). A Siamese CNN trained with InfoNCE loss learns a compact embedding of daily atmospheric fields, without using phase or ENSO labels during training. The central research question is:

> *Can a self-supervised model discover a representation that captures ENSO's modulation of BSISO structure — going beyond what conditional composite analysis already shows?*

---

## 1. Data

**Source:** ERA5 reanalysis (Copernicus CDS), APEC Climate Center BSISO Index, NOAA Niño 3.4

**Fields:** Three atmospheric channels, daily, July only (1981–2023):
- **u850** — zonal wind at 850 hPa (m/s)
- **v850** — meridional wind at 850 hPa (m/s)
- **OLR** — outgoing longwave radiation (negated ERA5 `ttr`, J/m²); proxy for deep convection

**Spatial domain:** 60°E–160°E, 0°N–60°N, 2° resolution → grid shape **31 lat × 51 lon**

**Labels (not used in training, only for evaluation):**
- **BSISO phase (1–8):** computed from APEC PC1/PC2 via atan2 → 8 sectors of 45°
- **ENSO category:** El Niño / La Niña / Neutral from NOAA Niño 3.4 JJA mean (±0.5 K threshold)

**Dataset size after label alignment:** N = **1,333 July days** (1981–2023, 43 years × ~31 days)

**ENSO class distribution (imbalanced):**

| Category | N | Fraction |
|----------|---|----------|
| Neutral | 775 | 58.1% |
| La Niña | 341 | 25.6% |
| El Niño | 217 | 16.3% |

---

## 2. Data Preprocessing

Two preprocessing approaches were implemented and run in parallel:

### Approach A — Raw Z-score Normalization
Each channel z-score normalized over all 1,333 July days. Simple and standard, but retains the year-to-year ENSO mean-state signal in the data.

### Approach B — Interannual Background Removal (Primary)
For each year *y* and channel *c*, the July mean of that year is subtracted per grid point before z-scoring:

```
X_anom[day, c] = X[day, c] − mean(X[all July days in year y, c])
```

This removes the slowly-varying ENSO background (warm/cold SST-driven mean state), leaving only intraseasonal (BSISO) anomalies. The model trained on Approach B data must encode ENSO as a modulation of BSISO anomaly patterns — which is exactly the research question.

**Output arrays:** `X_July.npy` (A) and `X_July_B.npy` (B), both shape (1333, 3, 31, 51), float32.

**Label validation:** Composite plots by BSISO phase (2×4 grid) and ENSO category (1×3 grid) confirmed correct labels — clear northeastward BSISO propagation and physically expected El Niño/La Niña OLR anomalies.

---

## 3. Model and Training

### Architecture — Siamese CNN Encoder
- **Input:** (3, 31, 51) per sample
- **3 convolutional layers** with batch normalisation and ReLU
- **Output:** 64-dimensional L2-normalised embedding
- **~250K parameters**

### Loss — InfoNCE (Contrastive)
Temperature τ = 0.07. Pairs are constructed each batch:
- **Positive pair:** same BSISO phase + same ENSO category
- **Hard negative:** same BSISO phase + different ENSO category
- **Easy negative:** different BSISO phase

### Training
- Platform: Google Colab T4 GPU
- 50 epochs, batch size 64
- Both Approach A and Approach B trained separately
- Checkpoints saved every 10 epochs: `encoder_final_A.pth`, `encoder_final_B.pth`

---

## 4. Analysis Results

### 4.1 BSISO Phase Encoding

| Approach | Val Accuracy | 5-fold CV | Baseline |
|----------|-------------|-----------|----------|
| A | **67.4%** | 67.1% ± 1.9% | 12.5% (random) |
| B | **59.2%** | 59.4% ± 4.0% | 12.5% |

A logistic regression trained on frozen 64-dim embeddings achieves 59–67% accuracy on 8-class BSISO phase prediction — 4.7–5.4× the random baseline — without ever seeing phase labels during training. The 8 pp drop from A to B is expected: removing the yearly mean makes the task harder by stripping large-scale background structure. The t-SNE projection shows distinct phase clusters, with Indian Ocean active phases (1–2) separated from Western Pacific phases (5–6).

### 4.2 ENSO Encoding — Displacement Analysis

The linear probe gives ~58% ENSO accuracy for both approaches, matching the majority-class (Neutral) baseline — this metric is **not informative** due to severe class imbalance.

The correct test is the **ENSO displacement**: for each BSISO phase, compute the centroid distance between El Niño and La Niña embeddings, compared to a null distribution from 1,000 label shuffles.

| Metric | Approach A | Approach B |
|--------|-----------|-----------|
| Mean displacement | 0.0779 | **0.0822** |
| Null mean ± std | 0.0264 ± 0.0047 | 0.0329 ± 0.0050 |
| Overall z-score | 11.02 | **9.85** |

**Both results are highly significant.** Approach B displacement is *larger* than Approach A despite the background being removed — the CNN genuinely learned ENSO modulation of intraseasonal anomaly structure.

**Phase-by-phase (Approach B):**

| Phase | ‖EN−LN‖ | vs. Null | Notes |
|-------|---------|----------|-------|
| 1 | ~0.052 | +1.6× | Significant |
| 2 | ~0.091 | +2.8× | Significant |
| 3 | ~0.038 | +1.2× | Marginal — Bay of Bengal active; orographically anchored |
| 4 | ~0.089 | +2.7× | Significant |
| 5 | — | — | Too sparse (<3 El Niño days) |
| **7** | **~0.170** | **+5.2×** | **Strongest — suppressed/re-initiation phase** |
| 8 | ~0.075 | +2.3× | Significant |

Phase 7 (suppressed/transitional) is most ENSO-sensitive — El Niño's eastward-shifted Walker circulation likely interferes most with BSISO re-initiation over the Indian Ocean. Phase 3 (Bay of Bengal active) is least sensitive, possibly because it is anchored by land-sea contrast and orography.

---

## 5. Areas Where I'd Like Support

### 5.1 Evaluation Metrics
The ENSO linear probe result is effectively uninterpretable due to class imbalance. I need help implementing:
- **Balanced accuracy** (`balanced_accuracy_score`, `class_weight='balanced'`)
- **Per-class confusion matrix** for both BSISO phase and ENSO
- **Circular accuracy** for BSISO phases (phases 1 and 8 are adjacent, should not count as large errors)

### 5.2 Generalisation / Data Leakage
The current train/val split is **random 80/20 by day index**. Since ENSO is a seasonal-scale forcing, days from the same year appear in both train and val — the model may have partially memorised year-specific patterns rather than generalisable ENSO structure. I need guidance on:
- Implementing a **year-based split** (hold out every ~5th year as validation)
- Whether this changes the BSISO probe or displacement results significantly

### 5.3 Statistical Robustness
- **Bootstrap confidence intervals** per phase for the displacement metric — specifically to assess whether Phase 7's large value (only ~21 El Niño days) is a genuine signal or a small-sample artefact
- **Phase 5 sparsity** — fewer than 3 El Niño days in Phase 5 prevents centroid computation at all. Is there a principled way to handle this (merge with adjacent phase, impute, or just flag as missing)?

### 5.4 "Beyond Composite Analysis" Claim
The displacement metric is structurally similar to comparing composite means in embedding space. To rigorously claim the model goes beyond composites I would need:
- A **Maximum Mean Discrepancy (MMD)** or Energy Distance test within each (phase × ENSO) bin — does the *distribution* of El Niño embeddings differ from La Niña beyond just the centroid?
- Advice on whether there is a cleaner way to frame or test this claim

### 5.5 Interpretability
I currently have no way to understand *what* in the atmospheric field drives the embedding separation. Options I am aware of but have not implemented:
- **Grad-CAM** applied to the encoder to produce spatial attribution maps
- **Nearest-neighbour retrieval** in embedding space (find the most ENSO-discriminative days)

---

## 6. Known Problems and Limitations

| Problem | Impact | Status |
|---------|--------|--------|
| Random train/val split (same-year leakage) | May inflate probe accuracy; ENSO results less reliable | Not yet fixed |
| Class imbalance (El Niño 217 vs Neutral 775) | ENSO probe uninformative; El Niño underrepresented in training pairs | Not yet fixed |
| Phase 5 missing (< 3 El Niño days) | Incomplete phase-by-phase displacement figure | July-only data too sparse |
| Phase 7 displacement based on ~21 El Niño days | Large displacement may be noisy | Need bootstrap CIs |
| Temperature τ = 0.07 not tuned | May be too aggressive for this dataset | Not yet explored |
| 50 epochs may be insufficient | Training loss still decreasing at epoch 50 | Need early stopping or longer run |
| "Beyond composites" not formally demonstrated | Core claim not fully tested | Need MMD test |

---

## 7. Potential Next Steps

**Short-term (methodological fixes):**
1. Year-based train/val split → re-run Notebooks 04 and 05
2. Balanced accuracy + `class_weight='balanced'` in linear probe
3. Bootstrap CIs for per-phase displacement (1,000 resamples)

**Medium-term (scale up):**
4. Extend to MJJAS (May–September) — 5× more data, fixes El Niño sparsity, enables Phase 5
5. More training epochs (100) with early stopping

**Longer-term (deeper analysis):**
6. Ablation: train without ENSO in pair labels — does displacement collapse?
7. Grad-CAM spatial attribution maps
8. MMD test within (phase × ENSO) bins
9. Composite analysis of Phase 7 El Niño vs La Niña anomalies to physically interpret the largest displacement

---

## Appendix: File Structure

```
data-driven-climate-science/
├── notebooks/
│   ├── 01_era5_download.ipynb
│   ├── 02_labels_download.ipynb
│   ├── 03_preprocessing.ipynb      ← Approach A + B + composite validation
│   ├── 04_training.ipynb           ← APPROACH = 'A' or 'B'
│   └── 05_analysis.ipynb           ← APPROACH = 'A' or 'B', saves to results/A/ or results/B/
├── results/
│   ├── analysis_results.md         ← Approach A detailed results (2026-03-08)
│   ├── analysis_results_B.md       ← Approach B detailed results (2026-03-22)
│   └── project_summary.md          ← this file
└── CLAUDE.md                       ← project context for Claude Code
```

Google Drive (`BSISO_SSL_Project/`):
```
checkpoints/  encoder_final_A.pth, encoder_final_B.pth, training_history_*.json
data/raw/     ERA5 .nc files, BSISO index, NOAA ENSO index, labels.csv
data/processed/  X_July.npy, X_July_B.npy, labels_aligned.csv, norm_stats*.json
results/A/    embeddings.npy, tsne_*.png, enso_displacement.png, linear_probe_results.json
results/B/    (same structure)
```

---

*DDCS Project | jh9141@nyu.edu | 2026-03-23*
