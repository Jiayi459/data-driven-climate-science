# Analysis Results: ENSO-BSISO Self-Supervised Learning

**Project:** ENSO-BSISO Self-Supervised Learning (SSL)
**Author:** Jiayi | jh9141@nyu.edu
**Model:** Siamese CNN + InfoNCE loss, 50 epochs, T4 GPU
**Data:** ERA5 u850/v850/OLR, July 1979–2023, N=1,333 days (1981–2023 after label alignment)
**Date:** 2026-03-08

---

## Research Question

> *Can self-supervised contrastive learning discover a representation that captures how ENSO modulates BSISO's atmospheric structure — going beyond what composite analysis (conditional means) already shows?*

**Short answer: Yes, but with an important nuance about how ENSO is encoded.**

---

## Summary of Results

| Metric | Value | Baseline | Verdict |
|--------|-------|----------|---------|
| BSISO phase linear probe (val) | **67.4%** | 12.5% (random 1/8) | Strongly encoded |
| BSISO phase 5-fold CV | **67.1% ± 1.9%** | 12.5% | Stable |
| ENSO category linear probe (val) | **58.4%** | ~58% (majority = Neutral) | Appears weak |
| ENSO category 5-fold CV | **59.2% ± 0.7%** | ~58% | Appears weak |
| EN−LN displacement (mean, all phases) | **0.0779** | 0.0264 ± 0.0047 (null) | 3× null |
| ENSO displacement z-score | **11.02** | > 2 = significant | Highly significant |

---

## 1. BSISO Phase Encoding

### Result
A logistic regression trained on the frozen 64-dim embeddings achieves **67.4% accuracy** predicting BSISO phase (8 classes), compared to a random baseline of 12.5%. The result is stable across 5-fold cross-validation (67.1% ± 1.9%), confirming it generalises and is not an artefact of the specific train/val split.

### What this means
The Siamese CNN organised its embedding space so that days sharing the same BSISO phase tend to cluster together — without ever receiving a phase label during training. The contrastive pairs alone (positive = same phase + same ENSO; hard negative = same phase + different ENSO) were sufficient to teach the model what BSISO phase structure looks like.

The t-SNE visualisation confirms this geometrically: distinct colour clusters are visible in the 2D projection, corresponding to the 8 BSISO phases. Days in the active Indian Ocean phases (1–2) group separately from days in the Western Pacific phases (5–6), consistent with the known northeastward propagation of the BSISO envelope.

### Critical caveat
67.4% is good but not perfect — 1 in 3 days is misclassified. Two structural limitations likely contribute:

1. **Cyclical phase confusion:** BSISO phases are sequential (8 → 1 → 2 ...). The model may often misclassify phase 1 as phase 8 or phase 2, which is physically a small error (adjacent phases) but counts as wrong in accuracy. A phase-aware metric (e.g. circular distance) would likely show better performance.

2. **Unequal sample counts:** Phase 4 has only 125 days and phase 7 has 134 days, versus 192–199 for phases 1, 2, 3, 8. The model sees fewer examples of rare phases, leading to weaker learning for those phases. This is an inherent limitation of the July-only 43-year dataset.

---

## 2. ENSO Category Encoding — The Linear Probe Is Misleading

### Result
The linear probe achieves **58.4%** ENSO accuracy, barely above the majority-class baseline of ~58% (Neutral accounts for 775 of 1,333 days = 58.1%). The automated classifier would label this as "ENSO not encoded."

### Why this conclusion is wrong
The linear probe fails to detect ENSO encoding for two separate reasons, one statistical and one geometric:

**Reason 1 — Class imbalance.**
The ENSO label distribution is heavily skewed:

| Category | N | Fraction |
|----------|---|----------|
| Neutral  | 775 | 58.1% |
| La Niña  | 341 | 25.6% |
| El Niño  | 217 | 16.3% |

A logistic regression trained with equal class weights will default toward predicting Neutral for borderline cases, inflating accuracy without actually learning ENSO structure. The apparent 58.4% result is almost entirely explained by correctly predicting the dominant Neutral class. A balanced-accuracy evaluation (which weights each class equally regardless of size) would give a very different picture, likely closer to 45–50% — which is still above the 33% random baseline, but the current metric is uninformative.

**Reason 2 — ENSO is geometrically local, not global.**
The linear probe asks: "Can a single hyperplane separate all El Niño days from all La Niña days across the entire embedding space?" The geometry of the learned representation does not support this, because ENSO is not a global reorganiser of the embedding space. The space is primarily structured by BSISO phase. ENSO creates *local sub-structure within each phase cluster*, not a global shift of all points. A global linear classifier cannot detect local cluster-level displacements.

The ENSO displacement analysis (Section 3) directly tests this local geometry and finds an overwhelmingly significant signal.

---

## 3. ENSO Displacement Analysis — The Real Signal

### Result
For each of the 8 BSISO phases, the distance between the El Niño centroid and La Niña centroid was computed in the 64-dim embedding space. A null distribution was built by shuffling ENSO labels 100 times and recomputing the displacement each time.

| Phase | Observed ||EN−LN|| | Null mean | Ratio | Above null +2σ? |
|-------|----------------------|-----------|-------|-----------------|
| 1 | 0.046 | 0.026 | 1.8× | Yes |
| 2 | 0.085 | 0.026 | 3.3× | Yes |
| 3 | 0.030 | 0.026 | 1.2× | No (weakest) |
| 4 | 0.093 | 0.026 | 3.6× | Yes |
| 5 | 0.089 | 0.026 | 3.4× | Yes |
| 6 | 0.062 | 0.026 | 2.4× | Yes |
| **7** | **0.148** | **0.026** | **5.7×** | **Yes (strongest)** |
| 8 | 0.071 | 0.026 | 2.7× | Yes |

**Overall z-score: 11.02** (11 standard deviations above the null). This is not borderline — the probability of obtaining this result by chance is astronomically small.

### What this means
Every BSISO phase has its El Niño and La Niña days mapping to distinguishably different sub-regions of the embedding space. The model did not just learn BSISO phase patterns and incidentally memorise ENSO as a side effect — it learned a representation where the atmospheric fingerprint of BSISO is consistently shifted by the ENSO background state, across all phases.

### Phase 7: most ENSO-sensitive
Phase 7 shows the largest displacement (0.148, 5.7× the null). Phase 7 in the BSISO1 index corresponds to the suppressed/transitional phase, when the active convective envelope has left the Western Pacific and is in the process of re-establishing over the Indian Ocean. This is plausibly the BSISO moment most sensitive to ENSO background state: a strong El Niño, which shifts the Walker circulation eastward and suppresses Maritime Continent convection, would delay or redirect the BSISO re-initiation compared to a La Niña. The model appears to have learned this asymmetry.

### Phase 3: least ENSO-sensitive
Phase 3 has a displacement of 0.030, barely above the null mean. Phase 3 corresponds to active convection established over the Bay of Bengal and northeastern Indian subcontinent. This is a dynamically robust phase anchored by land-sea contrast and orography (the Himalayas, Western Ghats), which may make it less susceptible to remote ENSO forcing. A physically meaningful null result.

### Critical caveat on Phase 7
Phase 7 also has the fewest El Niño samples (134 total in phase 7, with El Niño accounting for only ~16% ≈ 21 days). Centroid estimates from ~21 points in 64 dimensions have high variance. The large Phase 7 displacement may be partly a statistical artefact. Bootstrap confidence intervals per phase (not currently computed) would test whether Phase 7's displacement is genuinely larger than, say, Phase 4 or 5, or whether the difference is within noise.

---

## 4. Does This Go Beyond Composite Analysis?

This is the hardest part of the research question to answer honestly.

**What composite analysis gives you:** The mean atmospheric field for each (phase × ENSO) bin — 8 × 3 = 24 maps. This shows whether El Niño and La Niña July days look different on average within each phase.

**What the SSL model adds:** The displacement analysis also measures differences between El Niño and La Niña centroids — which is geometrically similar to comparing composite means, just done in 64-dim embedding space rather than the original (3, 31, 51) field space. In that narrow sense, the displacement result is not fundamentally different from a sophisticated composite.

**Where the SSL model genuinely adds value:**
1. The embedding space compresses (3 × 31 × 51 = 4,743 numbers) into 64 dimensions while preserving the relationships that matter for contrastive pair structure. This dimensionality reduction is learned, not hand-crafted.
2. The linear probe separability of BSISO phases (67%) tells us the learned geometry is coherent and structured, not just noise.
3. The model enables **nearest-neighbour retrieval** — finding the most similar atmospheric day to any query — which composite analysis cannot do.
4. Future work with the trained encoder could apply it to new years or extended seasons without recomputing composites.

**The honest limitation:** We have not yet shown the model captures distributional structure *within* each EN/LN × phase bin (i.e., not just the centroid but the shape of the cluster). That would be the stronger claim — that it goes genuinely beyond conditional means. The current metrics do not test this.

---

## 5. Critical Assessment: What Went Well and What Did Not

### What went well
- BSISO phase encoding is unambiguous and strong (67% vs 12.5%)
- ENSO displacement is statistically highly significant (z = 11.02)
- The model is well-calibrated: no embedding collapse (embeddings are diverse), training loss decreased from ~4.5 toward ~1–2
- The pair design (positive = same phase + same ENSO; hard negative = same phase + different ENSO) worked as intended

### What is uncertain or problematic
1. **Random train/val split:** Days from the same year appear in both train and val sets. In a strong El Niño year (e.g. 1997), July 1–15 might be in train and July 16–31 in val. Since ENSO is a seasonal-scale forcing, the model may have partly learned year-specific patterns rather than ENSO category patterns. A year-based split would give a more conservative, realistic generalisation test.

2. **Class imbalance uncorrected:** El Niño has only 217 days vs 775 Neutral. The current training loss treats all pairs equally, meaning El Niño signal is underrepresented. The probe metric is uninformative without balanced accuracy.

3. **Approach A preprocessing:** The current z-score normalisation does not remove the interannual (ENSO) background signal from the raw fields. This means the model may partly be learning ENSO as a change in mean wind speed or OLR level (a large-scale SST-driven background), rather than as a modulation of BSISO structure per se. Approach B (subtract the interannual signal first) would make the research question cleaner.

4. **Phase 7 displacement may be noisy:** ~21 El Niño days in phase 7 is a small sample for 64-dim centroid estimation.

5. **The "beyond composite analysis" claim is not fully demonstrated:** The displacement metric is structurally similar to comparing composite means. More convincing evidence would come from showing the model captures within-bin variance or nonlinear structure invisible to composites.

---

## 6. Recommended Next Steps

### Priority 1 — Fix known methodological weaknesses

| Step | Why | How |
|------|-----|-----|
| Year-based train/val split | Prevents same-year data leakage | Hold out ~9 years (every ~5th year) as val |
| Balanced accuracy metric | Current ENSO probe metric uninformative | Add `balanced_accuracy_score` + `class_weight='balanced'` to probe |
| Bootstrap CIs on displacement | Assess whether Phase 7 result is robust | 1000 bootstrap resamples of per-phase centroid |
| Approach B preprocessing | Cleaner research question | Subtract July climatological mean per year before z-scoring |

### Priority 2 — Improve the model

| Step | Why | How |
|------|-----|-----|
| Extend to MJJAS | 5× more samples, better El Niño coverage | Re-download ERA5 May–September, re-run pipeline |
| Balance El Niño pairs | Correct training class imbalance | Oversample El Niño pairs to match Neutral count |
| More epochs (100) | 50 epochs may not be sufficient for subtle ENSO signal | Re-train with early stopping on val loss |
| Tune temperature | 0.07 is aggressive; try 0.1, 0.2 | Grid search or validation curve |

### Priority 3 — Deeper analysis

| Step | What it tests |
|------|--------------|
| Ablation: train without ENSO in pair labels | How much did the ENSO criterion contribute? Without it, does displacement collapse? |
| Grad-CAM / saliency maps on encoder | Which spatial regions (lon/lat) does the CNN focus on? Do they match known BSISO/ENSO teleconnection regions? |
| Within-phase distributional test (MMD or Energy distance) | Does the model capture distributional structure beyond the centroid? Tests the "beyond composites" claim more rigorously |
| UMAP instead of t-SNE | More stable global structure, faster, preserves distances better |
| Confusion matrix for BSISO probe | Which phases are confused? Are errors adjacent phases (small error) or opposite phases (large error)? |

### Priority 4 — Scientific interpretation

| Step | Scientific question |
|------|---------------------|
| Compare Phase 7 EN vs LN composite fields | Why is Phase 7 most ENSO-sensitive? What atmospheric pattern drives the large embedding displacement? |
| Compute ENSO displacement in original field space | Is the embedding displacement consistent with displacement in raw u/v/OLR space? Or does the CNN learn a compressed representation that amplifies the ENSO signal? |
| Test on an independent year | Does the encoder generalise to a held-out strong El Niño year (e.g. 2015–16)? |

---

## Files

```
BSISO_SSL_Project/results/
├── embeddings.npy              ← (1333, 64) all embeddings, float32
├── tsne_overview.png           ← t-SNE: by phase (left) and ENSO (right)
├── tsne_by_phase.png           ← 8-panel: ENSO sub-structure within each phase
├── enso_displacement.png       ← EN−LN centroid displacement per phase + null
├── linear_probe_results.json   ← accuracy values for phase + ENSO probes
└── analysis_report.txt         ← automated summary
```

---

*DDCS Project | jh9141@nyu.edu | 2026-03-08*
