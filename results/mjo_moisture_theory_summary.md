# MJO Moisture–Convection & Theory Diagnostics — Results Summary
*ENSO-BSISO SSL · MJO moisture-constraint experiment (motivated by Zhang et al. 2020, "Four Theories of the MJO") · 2026-06-29*

---

## 0. What "own-RMM" means (definition)

**RMM** = the **Real-time Multivariate MJO index** (Wheeler & Hendon 2004), the standard operational MJO index. It is the leading **two principal components (RMM1, RMM2)** of a *combined EOF* of meridionally-averaged (15°S–15°N) **u850, u200, and OLR** anomalies. The MJO **phase** is the octant of the (RMM1, RMM2) plane (1–8); the **amplitude** is √(RMM1²+RMM2²). The *official* RMM is published by the Australian Bureau of Meteorology (BoM).

**"own-RMM"** = our **own reproduction of that index, computed from ERA5** (notebook nb24): we ran the same combined-EOF pipeline on our ERA5 fields, then **rotation-aligned** the result (Procrustes) to the official BoM RMM and validated it — **r ≈ 0.95** (canonical correlation 0.97/0.95), phase exact 83% / within-±1 100%, and **no ENSO-dependent bias**. We use own-RMM (not the published values) because it lives on our exact daily date axis and is a **validated, ENSO-unbiased, *linear* MJO phase clock**. File: `mjo_rmm_own_pcs.npy`; phase angle `θ_own = atan2(PC2, PC1)`.

> In one line: **own-RMM = our ERA5-derived Wheeler–Hendon RMM**, validated against the official index, used as the trustworthy phase coordinate.

---

## 1. Question

Using the phase relationship between **moisture and convection**, which of the four MJO theories does the **observed** MJO most resemble, and **how does ENSO modulate it**? And independently: do the project's **self-supervised latents** recover that relationship?

Theory expectations (Zhang et al. 2020):
- **Moisture-mode** (§5): precipitation ∝ *column* moisture → q and convection roughly **in phase**.
- **Skeleton** (§4.6): *lower-tropospheric* moisture and convective activity in **quadrature** (recharge ~quarter-cycle *ahead* of convection).
- **Trio-interaction** (§7): **boundary-layer convergence leads** convection; coupled Kelvin–Rossby structure (Rossby–Kelvin ratio ≈ 1, not the Gill value ≈ 2.2).

---

## 2. Method

- **Phase clock:** own-RMM angle θ (oriented eastward; circular correlation with official phase = **0.87**).
- **Phase-offset estimator:** for each field `f(day, longitude)` the first complex harmonic `A_f(x) = ⟨f·e^{iθ}⟩` over active MJO days; `Δθ = arg(A_field) − arg(A_conv)`, with `conv = −OLR′`. **Negative Δθ = field *leads* convection (sits *east* of it).** Computed per longitude and summed (coherently) per region — Indian Ocean (60–90°E), Maritime Continent (100–130°E), West Pacific (140–170°E) — with bootstrap 95% CIs, all **ENSO-stratified**.
- **Fields:** `q_col` (total column water vapour — moisture-mode variable), `q_low` (1000–700 hPa integrated humidity — skeleton variable), `∂q_col/∂t` (propagation tendency). Plus the **Rossby–Kelvin ratio** (u850 asymmetry) and the **BL-convergence lead** (1000/925 hPa divergence — trio-interaction).
- **Data:** ERA5 1979–2023, processed identically to `X_MJO` (nb13: 3-harmonic annual cycle, 120-day running-mean background removal, std-normalize). **10,177 active MJO days.**
- Notebooks: **nb28** (download), **nb29** (preprocess), **nb30** (diagnostics).

---

## 3. Result — which theory the MJO resembles

| Field | Indian O. | Maritime C. | West Pac. | reads as |
|---|---|---|---|---|
| `Δθ(q_col)` | −19° | −29° | −25° | column q ~**in phase** → moisture-mode |
| `Δθ(q_low)` | −24° | −45° | −49° | lower-trop q **leads more**, grows east → skeleton recharge |
| **gap (q_low−q_col)** | **−6°** | **−15°** | **−24°** | skeleton↔moisture-mode tension, widening east |
| `Δθ(∂q_col/∂t)` | −105° | −115° | −128° | moistening **east** of convection = propagation driver |

- **Rossby–Kelvin ratio = 1.34** (observed MJO ≈ 1.0, Gill ≈ 2.2) → a **realistic coupled Kelvin–Rossby** structure, *not* a Gill pattern (supports trio-interaction, §7.5).
- **BL-convergence lead** = IO −41°, MC −60°, WP −15° → low-level convergence **leads convection over the warm pool** (trio-interaction BL feedback, §7).
- Internal-consistency check: `∂q/∂t` leads `q_col` by ~90° (a tendency must) — validates the sign convention and the whole estimator.

**Verdict:** the real MJO is **moisture-mode-leaning** (column moisture broadly in phase with convection, *not* the 90° skeleton quadrature), **with a skeleton-flavoured lower-tropospheric recharge** that strengthens eastward and a **trio-interaction BL-convergence lead** over the warm pool. A **hybrid** — consistent with Zhang et al.'s conclusion that the theories are complementary rather than mutually exclusive.

---

## 4. Result — ENSO modulation (the headline / novel contribution)

The **longitude of the largest moisture lead follows the warm pool**:

| | Maritime Continent | West Pacific |
|---|---|---|
| **El Niño** | smaller lead (q_low −46°) | **larger** lead (q_low −54°) |
| **La Niña** | **larger** lead (q_low −51°) | smaller lead (q_low −45°) |

Robust across **both** `q_col` and `q_low`. Interpretation: **El Niño shifts the active moisture–convection coupling east** (into the West Pacific); **La Niña concentrates it over the Maritime Continent** — the MJO's moisture-convection phase structure tracks the displaced warm pool. The Rossby–Kelvin ratio also shifts monotonically: **EN 1.26 < Neutral 1.34 < La Niña 1.37** (El Niño slightly more Kelvin-dominated). Zhang et al. (2020) barely treat ENSO, so *ENSO modulation of which-theory-the-MJO-resembles* is the project's distinctive result.

---

## 5. Do the self-supervised latents recover this? — and the *angle = phase* insight

Same `Δθ(q_col)`, but using each **learned** latent's angle `θ_z = atan2(z2,z1)` (date-aligned), reporting circ-corr(θ, phase) and harmonic amplitude `|A|`:

| latent | circ-corr | Δθ(q_col) IO/MC/WP | `|A|` IO/MC/WP | verdict |
|---|---|---|---|---|
| **own-RMM** (linear EOF) | **0.65–0.87** | −19/−29/−25 | **5.9/5.8/4.7** | clean phase clock |
| **SSL-temporal-2D** (nb15) | 0.17 | −2/−22/+4 | 3.6/2.0/0.7 | partial; angle is a poor clock |
| **Barlow-D3 / base** (nb26) | ~0.00 | incoherent | **0.1–0.7** | **phase-blind** (Δθ is noise) |
| **aux2d** (nb31, +moisture aux) | 0.17 | +25/+17 | — | clean disk (eff_rank 2.0), ENSO z 18.5, **still angle≠phase** |
| **aux3d** (nb32, L2 circle + z₃) | **0.03** | incoherent | — | phase leaked to z₃; ENSO *not* on z₃ |

**Why even an explicit circle fails:** `atan2 = phase` requires the two axes to be a **quadrature pair** (the cos and sin of the propagating wave) — a property that **linear EOF guarantees** (own-RMM) but **contrastive SSL never enforces**. "Pull temporal neighbours together" has many solutions where the cycle winds/folds around the circle; the optimizer picks one whose angle is *not* RMM phase. Demonstrated **three independent ways**: 2-D no-L2 (0.17), 2-D rebalanced with healthy training (0.17), 3-D explicit L2 circle (0.03).

**Conclusion:** a clean **angular** phase coordinate is a **linear-EOF / quadrature property**; self-supervised networks capture the **phase + ENSO *content*** strongly (ENSO displacement z ≈ 13–19, label-free) but **not as a polar angle**, regardless of dimension or normalization.

---

## 6. Adopted framing — division of labor

- **own-RMM (linear EOF)** = the clean, ENSO-unbiased **MJO phase clock** → used for all the moisture-convection science above.
- **Self-supervised nets** (SSL-temporal, NSV d̂=7, Barlow Twins) = strong **label-free ENSO / slow-envelope representations** (ENSO z 13–21) → the ENSO-modulation engine.
- They are **complementary**, consistent with the project's recurring **phase-vs-slow-envelope** dichotomy (invariance/contrastive SSL → slow envelope; linear EOF → phase angle).

---

## 7. Open question (next investigation): *what does the SSL latent encode, if not phase?*

The SSL latent has high ENSO z but a near-useless phase angle — so what is it organizing by?

**Hypothesis:** the **slow ENSO/amplitude envelope** (as Barlow Twins does), with MJO phase present only weakly/linearly. Rationale: the temporal-contrastive objective rewards structure that is *slowly varying* across the ±3-day positive pairs; the MJO phase advances ~24° over 3 days (so it is partly "changed" between a pair, hence partly suppressed), whereas the ENSO/amplitude envelope is near-constant over 3 days.

**Concrete test (proposed nb33 — latent-content analysis):**
1. **Linear probes** from the frozen SSL latent for: MJO phase (8-class accuracy), MJO **amplitude** (regression R²), **ENSO** category (3-class balanced accuracy), **longitude of minimum-OLR** (convection centre), and **calendar month/season** (confound check).
2. **Regress each physical variable onto the latent** to recover its direction; quantify how much variance each explains.
3. **Prediction:** amplitude- and ENSO-probes high; phase-probe modest; season ≈ chance. If borne out, it confirms the SSL latent is a *slow ENSO/amplitude-envelope* representation, not a phase representation — closing the loop on why its angle isn't phase.

---

## Artifacts

- **Notebooks:** `28_mjo_moisture_download`, `29_mjo_moisture_preprocess`, `30_mjo_latent_moisture_diagnostics`, `31_mjo_moisture_aux_latent` (2-D physics-informed), `32_mjo_moisture_aux3d_latent` (3-D disentangled).
- **Diagnostics outputs:** `MJO/moisture_constraints/results/diagnostics/` — `diagnostics_summary.json`, `delta_theta_regions.csv`, `enso_stratified_delta_theta.csv`, `bl_convergence_lead.csv`, `secondary_latent_delta_theta.csv`, + 7 PNGs.
- **Conversation log:** Sessions 48–54 (`results/conversation_log.md`).
