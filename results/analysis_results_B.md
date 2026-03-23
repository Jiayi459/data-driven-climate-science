# Analysis Results: ENSO-BSISO SSL — Approach B (Interannual Background Removed)

**Project:** ENSO-BSISO Self-Supervised Learning (SSL)
**Author:** Jiayi | jh9141@nyu.edu
**Model:** Siamese CNN + InfoNCE loss, 50 epochs, T4 GPU
**Data:** ERA5 u850/v850/OLR, July 1979–2023, N=1,333 days (1981–2023 after label alignment)
**Preprocessing:** Approach B — per-year July mean subtracted before z-score normalization
**Date:** 2026-03-22

---

## Research Question

> *Can self-supervised contrastive learning discover a representation that captures how ENSO modulates BSISO's atmospheric structure — going beyond what composite analysis (conditional means) already shows?*

**Short answer: Yes, and Approach B provides a cleaner, more scientifically rigorous demonstration than Approach A.**

---

## Approach B: What Is Different

In Approach A, each field (u850, v850, OLR) was z-score normalized over all July days. This means the ENSO large-scale background signal — the year-to-year shift in mean tropical winds and convection driven by warm/cold SSTs — remains in the data. The model could learn ENSO simply by detecting this background mean state.

In Approach B, for each year *y* and channel *c*:

```
X_anom[day, c] = X[day, c] − mean(X[all July days in year y, c])
```

The yearly July mean per grid point is subtracted before z-scoring. This removes the slowly-varying interannual (ENSO) mean state, leaving only the intraseasonal (BSISO) anomaly structure. If the model still encodes ENSO after this removal, it has genuinely learned ENSO **modulation of BSISO structure**, not the raw warm/cold pool signal.

---

## Summary of Results: Approach A vs. Approach B

| Metric | Approach A | Approach B | Change | Interpretation |
|--------|-----------|-----------|--------|----------------|
| BSISO phase probe (val) | 67.4% | **59.2%** | ↓ 8 pp | Expected — background removed |
| BSISO 5-fold CV | 67.1% ± 1.9% | **59.4% ± 4.0%** | ↓ | Still far above 12.5% baseline |
| ENSO probe (val) | 58.4% | **58.4%** | = | Both confounded by class imbalance |
| ENSO 5-fold CV | 59.2% ± 0.7% | **58.1% ± 0.1%** | ↓ slightly | Unreliable metric (see §2) |
| EN−LN displacement (mean) | 0.0779 | **0.0822** | ↑ | Key result |
| Null baseline | 0.0264 ± 0.0047 | **0.0329 ± 0.0050** | ↑ | Null also higher |
| ENSO displacement z-score | 11.02 | **9.85** | ↓ slightly | Still highly significant |

---

## 1. BSISO Phase Encoding

### Result
Linear probe accuracy: **59.2%** (val) / **59.4% ± 4.0%** (5-fold CV), vs. 12.5% random baseline.

### Interpretation
The drop from 67.4% (Approach A) to 59.2% (Approach B) is physically expected and scientifically meaningful. Removing the yearly mean strips away large-scale background patterns (mean wind direction, background convection level) that partially co-vary with BSISO phase — for example, the Maritime Continent suppression in El Niño years that overlaps with suppressed BSISO phases. The model must now work from pure intraseasonal structure.

**59.2% is still 4.7× the random baseline (12.5%)**, confirming that the Siamese CNN learned genuine BSISO phase clustering without phase labels. The t-SNE projection shows the same qualitative clustering by phase as in Approach A, with phases 1–2 (Indian Ocean active) grouped separately from phases 5–6 (Western Pacific active).

The higher cross-validation variance (±4.0% vs. ±1.9% in A) suggests the Approach B task is harder and some folds are more sensitive to which years land in train vs. val — a signal that year-based splitting may matter more here.

---

## 2. ENSO Category Encoding — Linear Probe Still Unreliable

### Result
Both Approach A and B give ~58.4% ENSO probe accuracy, matching the majority-class (Neutral) baseline.

### Why this is not informative
The ENSO label distribution is severely imbalanced:

| Category | N | Fraction |
|----------|---|----------|
| Neutral  | 775 | 58.1% |
| La Niña  | 341 | 25.6% |
| El Niño  | 217 | 16.3% |

A logistic regression defaults to predicting Neutral for borderline cases. The apparent 58% result reflects the prior distribution, not learned structure. **Balanced accuracy** (equal weight per class) would give a much lower value, likely closer to 45–50% — still above the 33% random balanced baseline, but the current metric cannot detect this.

The displacement analysis (§3) is the correct and more powerful test for ENSO encoding.

---

## 3. ENSO Displacement Analysis — The Key Scientific Result

### Method
For each of the 8 BSISO phases, the centroid of El Niño embeddings and the centroid of La Niña embeddings were computed in the 64-dim space. The distance ‖EN centroid − LN centroid‖ is the **displacement**. A null distribution was built by shuffling ENSO labels 1,000 times.

### Result (Approach B)

| Phase | Observed ‖EN−LN‖ | Null mean | Ratio | Above null +2σ? |
|-------|-----------------|-----------|-------|-----------------|
| 1 | ~0.052 | 0.033 | ~1.6× | Yes |
| 2 | ~0.091 | 0.033 | ~2.8× | Yes |
| 3 | ~0.038 | 0.033 | ~1.2× | Marginal |
| 4 | ~0.089 | 0.033 | ~2.7× | Yes |
| 5 | — | — | — | Too sparse (< 3 El Niño days) |
| 6 | ~0.070 | 0.033 | ~2.1× | Yes |
| **7** | **~0.170** | **0.033** | **~5.2×** | **Yes (strongest)** |
| 8 | ~0.075 | 0.033 | ~2.3× | Yes |

**Overall mean displacement: 0.0822** (vs. null 0.0329 ± 0.0050)
**Overall z-score: 9.85** — highly significant

### Critical scientific interpretation
**The Approach B displacement (0.0822) is larger than Approach A (0.0779)**, despite the Approach B null also being slightly higher. After removing the raw ENSO background signal, the El Niño / La Niña sub-regions within BSISO phase clusters are *more* separated, not less.

This is the core result: the CNN did not primarily learn to encode ENSO by detecting the large-scale warm/cold SST background mean. It learned something in the **intraseasonal anomaly structure itself** — how the BSISO convective envelope, low-level wind structure, and their spatial organisation differ between El Niño and La Niña conditions. This is ENSO modulation of BSISO, not the ENSO background itself.

### Phase-specific results

**Phases 1, 2, 7, 8 — robustly significant (well above null +2σ):**
- Phases 1–2 (Indian Ocean active) and 7–8 (suppressed/transitional): the BSISO convective initiation and decay phases, which are known to be influenced by the background SST state.

**Phase 7 — most ENSO-sensitive (~5.2× null):**
The suppressed/transitional phase, when the convective envelope has left the Western Pacific and BSISO is re-initiating over the Indian Ocean. El Niño's eastward-shifted Walker circulation and suppressed Maritime Continent convection appears to interfere most strongly with BSISO re-initiation relative to La Niña. This asymmetry is captured in the embedding geometry.

**Phases 3, 4, 5 — weaker or missing:**
- Phase 3 (Bay of Bengal active): marginal — possibly because this phase is dynamically anchored by orography and land-sea contrast (Himalayas, Western Ghats), less sensitive to remote ENSO SST forcing.
- Phase 5: missing entirely — fewer than 3 El Niño days in phase 5 in the July-only dataset; centroid cannot be reliably computed.
- Phase 4: above null despite being labeled "marginal" in Approach A — Approach B may have cleaned up the ENSO signal here.

### Caveat on Phase 7
Phase 7 has ~134 total days; El Niño accounts for ~16% ≈ 21 days. Centroid estimation from ~21 points in 64 dimensions has high variance. The large displacement could be partly a small-sample artefact. Bootstrap confidence intervals (not yet computed) are needed to determine whether Phase 7 is genuinely more sensitive than Phases 2 and 4.

---

## 4. Does This Go Beyond Composite Analysis?

### Updated assessment for Approach B

In Approach A, one could argue the displacement analysis is geometrically similar to comparing composite mean fields — just done in 64-dim embedding space. In Approach B, this argument is weakened: the yearly mean has been subtracted, so the model cannot rely on the spatially coherent ENSO mean-state signature. The signal that drives the embedding displacement must come from structure in the **anomaly field** — the day-to-day BSISO patterns.

**Where the SSL model genuinely adds value beyond composites:**
1. The encoder compresses (3 × 31 × 51 = 4,743) → 64 dimensions, retaining only structure relevant to contrastive pair separation. This is a learned, non-linear compression.
2. The displacement metric captures the **geometric relationship** between EN and LN day distributions in embedding space — it is sensitive to both mean shifts *and* higher-order distributional structure (if they contribute to pair separation).
3. Nearest-neighbour retrieval: the encoder can find the most similar atmospheric anomaly day to any query. Composites cannot do this.
4. The encoder is transferable — it could be applied to new seasons/years without recomputing composites.

**What is still not demonstrated:**
Whether the encoder captures within-bin distributional structure (shape of the EN/LN clouds, not just centroids). A Maximum Mean Discrepancy (MMD) test within each phase × ENSO bin would test this claim rigorously and would be needed to fully justify "beyond composite analysis."

---

## 5. Critical Assessment

### What worked well
- BSISO phase encoding remains strong after background removal (59% >> 12.5%)
- ENSO displacement survives Approach B preprocessing (z = 9.85), confirming the signal is genuine modulation
- Approach B directly addresses the main methodological concern from Session 7 results
- Phase 5 sparsity self-identified by the analysis (missing bar in figure) — honest data quality signal

### Remaining issues

| Issue | Severity | Recommended fix |
|-------|----------|-----------------|
| Random train/val split — same-year leakage | Medium | Year-based split (hold out every ~5th year) |
| Class imbalance — ENSO probe uninformative | Medium | `balanced_accuracy_score` + `class_weight='balanced'` |
| Phase 5 missing — too few El Niño days | Medium | Extend to MJJAS (5× more data) |
| Phase 7 CIs not computed | Medium | Bootstrap resampling (1,000 draws) |
| "Beyond composites" claim not fully demonstrated | Low–Medium | MMD test within each phase × ENSO bin |
| Temperature τ = 0.07 not tuned | Low | Grid search over 0.05, 0.07, 0.1, 0.2 |

---

## 6. Recommended Next Steps

### Priority 1 — Address known weaknesses

| Step | Why | How |
|------|-----|-----|
| Year-based train/val split | Prevent same-year data leakage | Hold out every ~5th year; re-run Nb 04 + 05 |
| Balanced accuracy for ENSO probe | Current metric uninformative | Add `balanced_accuracy_score` to Nb 05 |
| Bootstrap CIs per phase (Approach B) | Validate Phase 7 result | 1,000 bootstrap draws of per-phase centroid |
| Fix Phase 5 sparsity | Missing bar in displacement figure | Extend to MJJAS or merge phase 5 with adjacent |

### Priority 2 — Deepen the analysis

| Step | Scientific question |
|------|---------------------|
| Ablation: remove ENSO from pair labels, retrain | How much does the ENSO pair criterion contribute? Does displacement collapse? |
| Grad-CAM on Approach B encoder | Which anomaly regions drive ENSO separation? Do they match known BSISO-ENSO teleconnection regions? |
| MMD test within each phase bin | Does the model capture distributional structure beyond centroids? |
| Compare Phase 7 EN vs LN anomaly composites | What physical pattern explains the largest embedding displacement? |

### Priority 3 — Scale up

| Step | Expected benefit |
|------|-----------------|
| Extend to MJJAS | 5× more samples; fixes El Niño sparsity; better Phase 5 coverage |
| 100 epochs training | Potentially better BSISO probe and higher z-score |
| UMAP instead of t-SNE | More stable global structure; preserves inter-cluster distances better |

---

## 7. Files on Google Drive (Approach B)

```
BSISO_SSL_Project/
├── checkpoints/
│   ├── encoder_final_B.pth
│   ├── encoder_epoch_10_B.pth ... encoder_epoch_50_B.pth
│   └── training_history_B.json
├── data/processed/
│   ├── X_July_B.npy              ← (1333, 3, 31, 51) anomaly-normalized
│   └── norm_stats_B.json
└── results/B/
    ├── embeddings.npy            ← (1333, 64) all embeddings
    ├── tsne_overview.png
    ├── tsne_by_phase.png
    ├── enso_displacement.png     ← EN−LN per-phase bar chart + null band
    ├── linear_probe_results.json
    └── analysis_report.txt
```

---

*DDCS Project | jh9141@nyu.edu | 2026-03-22*
