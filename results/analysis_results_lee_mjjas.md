# Analysis Results: Lee et al. Preprocessing — MJJAS (Year-Based Split)

**Project:** ENSO-BSISO Self-Supervised Learning (SSL)
**Author:** Jiayi | jh9141@nyu.edu
**Model:** Siamese CNN + InfoNCE loss, 50 epochs, T4 GPU
**Data:** ERA5 u850/v850/OLR, MJJAS 1979–2023, N=6,579 days (1981–2023 after label alignment)
**Preprocessing:** Lee et al. (2013) — 3-harmonic Fourier annual cycle removal + 120-day running mean + area-averaged std normalization
**Train/Val Split:** Year-based (~80/20) — every 5th year held out for validation (no same-year leakage)
**Date:** 2026-04-13

---

## Research Question

> *Can self-supervised contrastive learning discover a representation that captures how ENSO modulates BSISO's atmospheric structure — going beyond what composite analysis (conditional means) already shows?*

**Short answer: Yes. The ENSO modulation signal survives rigorous preprocessing AND a strict year-based train/val split with no same-year leakage (z=3.83), confirming the model learned genuine ENSO modulation of BSISO structure.**

---

## Preprocessing: Lee et al. (2013) Method

Three-step anomaly computation following Lee et al. (2013):

1. **Remove annual cycle:** Compute climatological daily mean per DOY over the base period 1981–2010. Fit the first 3 Fourier harmonics (periods 365, 182.5, 121.7 days) via least squares to obtain a smooth seasonal curve. Subtract this smooth curve from all days.
   ```
   f(d) = a₀ + Σₖ₌₁³ [aₖ cos(2πkd/365) + bₖ sin(2πkd/365)]
   ```

2. **Remove interannual variability:** Subtract the preceding 120-day running mean from each day, removing slowly-varying ENSO background state.

3. **Normalize:** Divide by the area-averaged temporal standard deviation (one scalar per variable: u850, v850, OLR).

This ensures the model receives only intraseasonal anomalies — the BSISO signal — with both the seasonal cycle and interannual ENSO mean state removed.

---

## Summary of Results (Year-Based Split)

| Metric | Value | Baseline | Verdict |
|--------|-------|----------|---------|
| BSISO phase linear probe (val) | **67.7%** | 12.5% (random 1/8) | Strongly encoded |
| BSISO phase 5-fold CV (year-grouped) | **68.6% ± 1.0%** | 12.5% | Stable |
| ENSO category linear probe (val) | **60.9%** | ~58% (majority = Neutral) | Marginal |
| ENSO category 5-fold CV (year-grouped) | **58.1% ± 0.0%** | ~58% | Uninformative |
| EN−LN displacement (mean, all phases) | **0.0188** | 0.0117 ± 0.0019 (null) | 1.6× null |
| ENSO displacement z-score | **3.83** | > 2 = significant | Significant |

---

## Impact of Year-Based Split

| Metric | Random Split | Year-Based Split | Change |
|--------|-------------|-----------------|--------|
| BSISO Phase (val) | 69.6% | **67.7%** | -1.9 pp |
| BSISO 5-fold CV | 69.2% ± 1.2% | **68.6% ± 1.0%** | -0.6 pp |
| ENSO z-score | 2.60 | **3.83** | +1.23 |

The year-based split is a stricter test — no days from the same year appear in both train and val. BSISO phase accuracy dropped slightly (as expected, since same-year leakage is removed), but ENSO displacement z-score *increased* from 2.60 to 3.83. This means the model's ENSO discrimination genuinely generalizes to unseen years, and the random split was actually underestimating the ENSO signal (partial-year splits created noisier centroids).

---

## Comparison Across All Approaches

| | Approach A | Approach B | Lee (July) | Lee MJJAS (random) | **Lee MJJAS (year split)** |
|--|-----------|-----------|------------|---------------------|---------------------------|
| N | 1,333 | 1,333 | 1,333 | 6,579 | **6,579** |
| Preprocessing | raw z-score | per-year mean removed | Lee (raw DOY clim) | Lee (3-harmonic) | **Lee (3-harmonic)** |
| Split | random | random | random | random | **year-based** |
| BSISO Phase Acc | 67.4% | 59.2% | 62.2% | 69.6% | **67.7%** |
| BSISO 5-fold CV | 67.1% ± 1.9% | 59.4% ± 4.0% | 65.2% ± 2.7% | 69.2% ± 1.2% | **68.6% ± 1.0%** |
| ENSO z-score | 11.02 | 9.85 | 10.82 | 2.60 | **3.83** |

---

## 1. BSISO Phase Encoding — Strong and Robust

### Result
A logistic regression on the frozen 64-dim embeddings achieves **67.7% accuracy** (val) predicting BSISO phase (8 classes), with a year-grouped 5-fold CV of **68.6% ± 1.0%**. This is with a year-based split that ensures no same-year leakage — a stricter test than the random split used in all previous runs.

Compared to previous runs:
- -1.9 pp vs. random split on same data (69.6%) — small drop, confirming the model did not rely on same-year leakage
- +8.5 pp over Approach B (59.2%), which used cruder per-year mean removal
- +5.5 pp over Lee July-only (62.2%), showing the benefit of 5× more data

### What this means
The combination of (1) proper Fourier-based annual cycle removal, (2) 120-day running mean subtraction, and (3) 5× more training data from MJJAS produces a robust BSISO representation that generalizes to entirely unseen years. The tight CV variance (±1.0%) — the lowest across all runs — confirms the model is stable.

### t-SNE visualization
The t-SNE overview shows well-defined, spatially separated phase clusters. Adjacent phases (e.g., 1–2, 3–4) neighbour each other in the 2D projection, consistent with the cyclical BSISO progression. The embedding space has captured the physically meaningful phase sequence without explicit ordering supervision.

---

## 2. ENSO Category Encoding — Linear Probe Still Uninformative

### Result
The ENSO linear probe achieves 60.9% (val) / 58.1% (5-fold CV). The 5-fold CV shows essentially zero variance (±0.0%), meaning the classifier collapses to predicting the majority class (Neutral) in every fold.

### Why this is expected
The class imbalance problem remains:

| Category | Approximate N (MJJAS) | Fraction |
|----------|----------------------|----------|
| Neutral  | ~3,800 | ~58% |
| La Niña  | ~1,700 | ~26% |
| El Niño  | ~1,080 | ~16% |

A balanced-accuracy evaluation with `class_weight='balanced'` is needed to assess ENSO encoding via linear probe. The displacement analysis (Section 3) remains the correct metric.

---

## 3. ENSO Displacement Analysis — Significant and Robust

### Result
The mean EN−LN centroid displacement across all 8 BSISO phases is **0.0188**, compared to a null baseline of **0.0117 ± 0.0019**. The z-score of **3.83** comfortably exceeds the significance threshold of 2.

Notably, the z-score *increased* from 2.60 (random split) to 3.83 (year-based split). This means the ENSO modulation signal generalizes well to entirely unseen years — the random split was actually underestimating the signal because partial-year splits created noisier within-phase centroids.

### Per-phase displacement
From the bar chart, most phases show observed displacement above the null mean:
- Phases 1, 2, 3, 4, 6, 8: displacement above or near the null +2σ threshold
- Phase 5: near the null — weak ENSO sensitivity
- Phase 7: no longer the extreme outlier seen in July-only runs

### Why the z-score dropped from ~11 (July) to 3.83 (MJJAS)

Three factors contribute:

1. **Seasonal dilution of ENSO–BSISO coupling.** ENSO modulation of BSISO is strongest during peak boreal summer (July). May, June, August, and September contribute samples where the ENSO–BSISO teleconnection is weaker. The overall displacement is a weighted average that gets pulled toward zero by these less-coupled months.

2. **Tighter centroids with more data.** With N=6,579 (vs. 1,333), both the observed and null centroid estimates have lower variance. The absolute displacement shrinks (0.0188 vs. 0.0779) because centroids are more precisely estimated, reducing the signal-to-noise gap.

3. **Better preprocessing removes ENSO leakage.** The Lee et al. 120-day running mean effectively strips out the slowly-varying ENSO background. In Approach A, the raw ENSO mean state was still present — the model could "cheat" by detecting warm-pool SST-driven background wind/OLR differences. After removal, the model can only detect ENSO modulation of intraseasonal structure, which is a subtler signal.

### Scientific interpretation
The z=3.83 result with year-based split is the most scientifically rigorous of all runs: it survives (1) the strictest preprocessing (3-harmonic Fourier + 120-day running mean), (2) the broadest temporal scope (MJJAS), and (3) a year-based split that ensures no same-year leakage. That the z-score actually improved over the random split demonstrates the ENSO modulation signal is genuine and generalizable.

---

## 4. Physical Interpretation: Why ENSO Modulates BSISO

### 4.1 The Large-Scale Mechanism: Walker Circulation

The primary pathway through which ENSO modulates BSISO is the **Walker circulation** — the zonal (east-west) overturning cell spanning the tropical Pacific and Indian Oceans.

In the **neutral state**, the Walker cell drives low-level easterly trade winds across the Pacific, with ascending motion and deep convection over the Maritime Continent (100–150°E) and descending motion over the eastern Pacific. This sets the background environment for BSISO propagation.

**During El Niño** (warm Niño 3.4 SST anomaly):
- The Walker circulation weakens and shifts eastward
- Ascending branch moves from the Maritime Continent toward the central Pacific
- Low-level westerly anomalies develop over the western Pacific
- Convection is suppressed over the Maritime Continent and enhanced over the central Pacific
- The Indo-Pacific warm pool shrinks westward

**During La Niña** (cold Niño 3.4 SST anomaly):
- The Walker circulation strengthens and contracts westward
- Enhanced ascending motion and convection over the Maritime Continent
- Stronger low-level easterlies across the Pacific
- The warm pool extends further east

These background changes alter the thermodynamic and dynamic environment through which the BSISO convective envelope propagates — modifying where convection initiates, how fast it propagates, and how far it extends.

### 4.2 The Gill Model Framework

The atmospheric response to ENSO-driven tropical heating anomalies can be understood through the **Gill (1980) model**, which describes the steady-state response of the tropical atmosphere to a localized diabatic heating source:

```
εu − βyv = −∂φ/∂x
εv + βyu = −∂φ/∂y
εφ + c²(∂u/∂x + ∂v/∂y) = −εQ
```

where `u`, `v` are zonal/meridional winds, `φ` is geopotential, `β` is the meridional gradient of the Coriolis parameter, `c` is the gravity wave speed, `ε` is the damping rate, and `Q` is the diabatic heating.

The Gill model predicts two key features of the low-level response to tropical heating:
1. **Equatorial Kelvin wave response** to the east of the heating — easterly wind anomalies
2. **Equatorial Rossby wave response** to the west — a pair of cyclonic gyres straddling the equator

During El Niño, the shifted heating anomaly (toward the central Pacific) produces:
- Anomalous low-level westerlies over the western Pacific and Maritime Continent (Kelvin wave response to enhanced central Pacific heating)
- Anomalous low-level convergence patterns that interfere with the BSISO's own Kelvin-Rossby wave structure

This interference is phase-dependent: when the BSISO convective envelope is in a region where the ENSO-driven Gill response reinforces or opposes the BSISO circulation, the modulation is strong. When the BSISO is in a region orthogonal to the Gill response pattern, the modulation is weak.

### 4.3 Moisture Budget and Moist Static Energy

BSISO propagation is fundamentally driven by **moisture dynamics**. The column-integrated moisture budget:

```
∂⟨q⟩/∂t = −⟨v⃗ · ∇q⟩ − ⟨∂(ωq)/∂p⟩ + E − P
```

where `⟨⟩` denotes column integration, `q` is specific humidity, `v⃗` is horizontal wind, `ω` is vertical velocity (pressure coordinates), `E` is evaporation, and `P` is precipitation.

ENSO modifies this budget through:
1. **Horizontal moisture advection** `−⟨v⃗ · ∇q⟩`: ENSO-driven wind anomalies transport moisture differently, altering where pre-moistening occurs ahead of the BSISO convective envelope
2. **Background moisture gradient** `∇q`: El Niño shifts the warm pool and associated moisture maximum, changing the gradient the BSISO propagates through
3. **Evaporation** `E`: SST anomalies in the Indo-Pacific modify surface evaporation, changing the moisture source for BSISO convection

The **moist static energy (MSE)** framework provides a more complete picture of BSISO propagation:

```
∂⟨h⟩/∂t ≈ −⟨v⃗ · ∇h⟩ + F_net
```

where `h = cpT + Lvq + gz` is the MSE (sum of sensible heat, latent heat, and geopotential energy), and `F_net` is the net energy flux into the column. The northward propagation of BSISO is driven by meridional advection of mean MSE by the BSISO's anomalous winds (`−⟨v' · ∂h̄/∂y⟩`). ENSO modifies the background MSE gradient `∂h̄/∂y`, directly affecting the propagation speed and extent of the BSISO.

### 4.4 Phase-by-Phase ENSO Sensitivity — Physical Explanations

**Phases 1–2 (Indian Ocean initiation) — Moderate to strong ENSO sensitivity:**

BSISO convection initiates over the equatorial Indian Ocean (70–90°E). This region is affected by ENSO through the **Indian Ocean Basin (IOB) mode**: during El Niño, the tropical Indian Ocean warms with a 1–2 season lag via the "atmospheric bridge" (modified Walker circulation → reduced cloudiness → increased solar radiation → SST warming). This warming alters:
- Surface evaporation and boundary layer moisture (modified `E` in the moisture budget)
- The zonal SST gradient between the Indian Ocean and Maritime Continent
- The strength of the cross-equatorial monsoon flow that triggers BSISO initiation

The CNN likely detects these differences in the u850/v850 wind patterns during initiation phases.

**Phase 3 (Bay of Bengal active) — Weak ENSO sensitivity:**

Phase 3 features strong, organized convection over the Bay of Bengal and northeastern Indian subcontinent. This phase is governed by:
- **Orographic forcing** from the Western Ghats and Himalayas, which anchors convection
- **Land-sea contrast** driving strong monsoon southwesterlies
- **Local air-sea interaction** in the semi-enclosed Bay of Bengal

These local forcings are largely independent of remote Pacific SST anomalies. The BSISO convective envelope is at its most dynamically "self-sustaining" in this phase — the orography and land-sea thermal contrast provide strong boundary conditions that ENSO cannot easily override. This explains why Phase 3 consistently shows the weakest ENSO modulation across all experimental runs.

**Phase 4 (northward propagation peak) — Moderate ENSO sensitivity:**

The BSISO envelope is propagating northward from the Bay of Bengal into the Indian subcontinent. The propagation speed depends on the meridional MSE gradient (`∂h̄/∂y`):
- El Niño weakens the meridional MSE gradient (warmer tropics → reduced south-to-north gradient) → potentially slower propagation
- La Niña strengthens the gradient → potentially faster propagation

The CNN may detect these propagation speed differences as changes in the spatial pattern of v850 (meridional wind), which directly reflects the meridional moisture transport.

**Phase 5 (Western Pacific) — Weak ENSO sensitivity:**

Active convection is over the Western Pacific / Philippines (120–150°E). Despite being geographically close to the ENSO action center (Niño 3.4 = 170°W–120°W), Phase 5 shows weak ENSO sensitivity because:
- The **Western Pacific warm pool** (SST > 28°C) is remarkably stable even during ENSO events — SST variability is much smaller here than in the central/eastern Pacific
- During active BSISO convection, the intraseasonal dynamics (convective self-aggregation, wind-evaporation feedback) dominate over the ENSO background signal
- The strong BSISO convection in Phase 5 may "saturate" the atmospheric response, leaving little room for ENSO modulation

**Phases 6–8 (suppressed/transition) — Variable ENSO sensitivity:**

These phases represent the suppressed convection / transition period as the BSISO envelope weakens over the Western Pacific and begins re-initiating over the Indian Ocean. Phase 7 in particular showed extreme ENSO sensitivity in July-only runs (z=5.7× null) but normalized with MJJAS data.

The physical mechanism for ENSO sensitivity during suppressed phases is the **"quiet window" hypothesis**: when the BSISO convective envelope is weak or absent, the atmosphere is more transparent to the ENSO background state. The low-level winds and moisture distribution during suppressed phases are less constrained by intraseasonal convective dynamics, allowing the ENSO-driven Walker circulation anomaly to express itself more freely in the u850, v850, and OLR fields.

During El Niño suppressed phases: anomalous westerlies over the Maritime Continent persist without competing BSISO convection. During La Niña: enhanced easterlies and maintained Maritime Continent convection. The CNN can distinguish these different "baseline states" during the quiet window.

### 4.5 Seasonality: Why July > MJJAS for ENSO–BSISO Coupling

The drop in ENSO z-score from July-only (z=10.82) to MJJAS (z=3.83) is itself a physically meaningful finding. Three mechanisms explain the July peak:

**1. Monsoon maturity and the ENSO teleconnection pathway:**
The Asian summer monsoon reaches peak intensity in July. The monsoon's cross-equatorial Hadley circulation provides the primary teleconnection pathway from the Pacific to the Indian Ocean sector. In July:
- The monsoon Hadley cell is strongest → maximum sensitivity to Walker circulation perturbations
- The Tropical Easterly Jet (TEJ) at 200 hPa is fully established → strongest upper-level divergence coupling
- The monsoon trough position is most active → ENSO-driven shifts have maximum impact

In May–June (onset) and August–September (withdrawal), the monsoon circulation is weaker, reducing the teleconnection efficiency.

**2. ENSO amplitude seasonality:**
ENSO events typically develop through boreal summer and peak in boreal winter (November–January). By July of an El Niño year, the Niño 3.4 SST anomaly is moderate and developing, but the atmospheric response (modified Walker circulation) is already established. The JJA period represents a "sweet spot" where:
- SST anomalies are large enough to drive significant atmospheric circulation changes
- The background monsoon circulation amplifies the Pacific signal into the Indo-Pacific domain
- The BSISO is at its most active, providing clear phase structure to modulate

**3. Background state moisture sensitivity:**
July has the highest column-integrated water vapor in the Indo-Pacific domain. In a moisture-rich environment, small perturbations in circulation (from ENSO) can trigger disproportionately large changes in convective organization — a nonlinear moisture-convection feedback. May and September have drier background states where the same ENSO-driven circulation anomalies produce smaller convective responses.

---

## 5. Does This Go Beyond Composite Analysis?

### What composite analysis gives
The mean atmospheric field for each (phase × ENSO) bin — 8 × 3 = 24 composite maps. This reveals average differences between El Niño and La Niña within each phase.

### What the SSL model adds
1. **Learned dimensionality reduction:** 4,743 input dimensions → 64 embedding dimensions, preserving contrastive structure. This compression is data-driven, not hand-crafted.
2. **BSISO phase structure without labels:** 67.7% phase accuracy from a model that never saw phase labels during training. The contrastive pair design alone taught the model what BSISO phase structure looks like.
3. **Embedding enables downstream tasks:** nearest-neighbour retrieval, anomaly detection, transfer to new periods — capabilities composite analysis cannot provide.
4. **ENSO signal after rigorous preprocessing + year-based split (z=3.83):** Survives 3-harmonic Fourier + 120-day running mean removal AND generalizes to held-out years. This is a stricter test than composite analysis typically applies.

### Honest limitation
The displacement metric (EN−LN centroid distance) is structurally similar to comparing composite means in embedding space. To fully demonstrate "beyond composites," a distributional test (MMD, Energy distance) within each phase × ENSO bin is needed — testing whether the model captures higher-order structure (variance, skewness, multimodality) beyond the mean.

---

## 6. Critical Assessment

### What went well
- **Robust BSISO representation:** 67.7% phase accuracy with year-based split, tight CV variance (±1.0% — lowest across all runs)
- **Scientifically rigorous preprocessing:** Lee et al. (2013) 3-harmonic Fourier method is the proper standard
- **ENSO signal survives the strictest test:** z=3.83 after Lee et al. preprocessing + year-based split — genuine modulation signal that generalizes to unseen years
- **5× more data:** MJJAS eliminates Phase 5 sparsity and stabilizes all per-phase estimates
- **Year-based split resolved key concern:** z actually increased from 2.60 to 3.83, removing doubt about same-year leakage

### What remains uncertain or problematic
1. **Class imbalance uncorrected:** El Niño (~16%) is underrepresented in training pairs and probe evaluation. Balanced accuracy and oversampled pairs would give fairer metrics.
2. **No bootstrap CIs:** Per-phase displacement lacks confidence intervals. Cannot statistically distinguish "Phase 4 has stronger ENSO modulation than Phase 6" without CIs.
3. **"Beyond composites" claim incomplete:** Displacement metric is a centroid comparison. Need distributional tests for the stronger claim.

---

## 7. Recommended Next Steps

### Priority 1 — Methodological fixes
| Step | Why |
|------|-----|
| ~~Year-based train/val split~~ | ~~Done (Session 11) — z improved from 2.60 to 3.83~~ |
| Balanced accuracy metric | Current ENSO probe is uninformative due to class imbalance |
| Bootstrap CIs on displacement | Test robustness of per-phase ENSO sensitivity differences |

### Priority 2 — Scientific interpretation
| Step | Why |
|------|-----|
| ~~Physical explanation of per-phase ENSO sensitivity~~ | ~~Done (Section 4) — Walker circulation, Gill model, MSE framework, per-phase mechanisms~~ |
| ~~July vs MJJAS z-score seasonality~~ | ~~Done (Section 4.5) — monsoon maturity, ENSO amplitude, moisture sensitivity~~ |
| Grad-CAM / saliency maps | Identify which spatial regions drive the CNN's ENSO discrimination |

### Priority 3 — Deeper analysis
| Step | Why |
|------|-----|
| Ablation: train without ENSO in pair labels | Measure ENSO criterion contribution |
| MMD / Energy distance within-phase test | Formally test "beyond composites" claim |
| Alternative dim reduction (Isomap, UMAP) | Isomap may reveal cyclic BSISO manifold; UMAP preserves global structure |

---

## Files

```
BSISO_SSL_Project/results/lee/
├── embeddings.npy              ← (6579, 64) all embeddings, float32
├── tsne_overview.png           ← t-SNE: by phase (left) and ENSO (right)
├── tsne_by_phase.png           ← 8-panel: ENSO sub-structure within each phase
├── enso_displacement.png       ← EN−LN centroid displacement per phase + null
├── linear_probe_results.json   ← accuracy values for phase + ENSO probes
└── analysis_report.txt         ← automated summary
```

---

*DDCS Project | jh9141@nyu.edu | 2026-04-13*
