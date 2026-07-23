# MJO Moisture-Convection Constraint Experiment

Status: planning, created 2026-06-26.

This experiment is motivated by Zhang et al. (2020), *Four Theories of the Madden-Julian Oscillation*.

Core question:

Can we train a 2D MJO latent space whose geometry is physically interpretable in terms of the phase relationship between column moisture and convection, and whose ENSO modulation can be diagnosed in that same space?

## Motivation

Zhang et al. (2020) contrast several MJO theories using different assumptions about the relationship among moisture, convection, dynamics, and radiation.

The most useful distinction for this project is:

- Skeleton theory: lower-tropospheric moisture and convective activity oscillate against each other on the intraseasonal timescale. Moisture leads or is in quadrature with convective activity.
- Moisture-mode theory: precipitation/convection is tightly coupled to column moisture; precipitation is approximately proportional to column/free-tropospheric moisture.
- Trio-interaction theory: boundary-layer moisture convergence and low pressure lead major convection, coupling Kelvin-Rossby dynamics, moisture, and heating.

The project already has learned MJO latent spaces. The next step is to ask whether a new latent space can be made interpretable by aligning it with physically meaningful moisture-convection phase relationships.

## Recommendation

Try a 2D Barlow Twins-style representation, but do **not** use the previous temporal-lag invariance objective alone.

Reason:

The previous temporal-lag Barlow Twins objective made embeddings invariant between `X_t` and `X_{t+tau}`. That is structurally hostile to MJO phase, because MJO phase is exactly what changes over 1-5 days. It recovered a slow ENSO/amplitude envelope but phase was random.

Recommended new 2D setup:

- Use 2D latent space for visualization and direct phase interpretation.
- Use same-day views, not temporal-lag views, for the Barlow Twins invariance term.
- Add a phase/dynamics term so the representation still organizes the cyclic MJO.
- Add moisture/convection diagnostics first; then add physics regularization only after the target phase relation is measured.

## Data

Use the existing MJO fields:

- u850
- u200
- OLR
- domain: 15S-15N, all longitudes
- preprocessing: Wheeler-Hendon-style MJO preprocessing already used in the project

Add moisture:

- Primary: total column water vapor or column-integrated specific humidity.
- Secondary: lower-tropospheric moisture, ideally 850-700 hPa or 1000-700 hPa.

Column-integrated moisture comes first because it aligns best with moisture-mode theory, which uses column/free-tropospheric moisture. Lower-tropospheric moisture should be added as a second diagnostic because skeleton theory emphasizes lower-tropospheric moisture.

Preprocess moisture with the same temporal treatment as MJO:

- remove annual cycle / first harmonics
- remove slowly varying background
- normalize by an area-averaged temporal standard deviation
- apply the same 20-90 day bandpass for phase diagnostics

## Diagnostic Phase Before Training

Before adding a loss constraint, run diagnostics on existing latent spaces.

Candidate latent spaces:

- existing MJO SSL 2D InfoNCE embedding
- existing MJO NSV 7D latent, projected to first two physically meaningful coordinates
- existing Barlow D=3/D=7 projector for slow ENSO/amplitude envelope

Diagnostics:

1. Compute latent phase:

   ```text
   theta_z(t) = atan2(z2(t), z1(t))
   ```

2. Bin days into 8 or 16 equal bins of `theta_z`.

3. Composite:

   - column moisture
   - lower-tropospheric moisture
   - convection proxy `-OLR`
   - optionally u850/u200 for Kelvin-Rossby structure

4. Compute first-harmonic phase per longitude or region:

   ```text
   A_q(x)    = sum_k composite_q(k, x)    * exp(i theta_k)
   A_conv(x) = sum_k composite_-OLR(k, x) * exp(i theta_k)

   delta_theta_q_conv(x) = arg(A_q(x)) - arg(A_conv(x))
   ```

5. Summarize over the Indian Ocean, Maritime Continent, and western Pacific:

   - mean phase offset
   - circular standard deviation
   - amplitude of first harmonic
   - bootstrap confidence interval

6. Interpret:

   - near 0 degrees: moisture and convection in phase; moisture-mode-like
   - positive moisture lead / quarter cycle: skeleton-like recharge before convection
   - strong longitude dependence: propagation-dependent relation
   - ENSO-dependent offsets: possible ENSO modulation of MJO physics

## Training Plan

Create a new notebook sequence under this experiment folder:

1. `01_mjo_moisture_download.ipynb`
   - Download ERA5 moisture data.
   - Prefer total column water vapor if available.
   - Otherwise download specific humidity on pressure levels and vertically integrate.

2. `02_mjo_moisture_preprocess.ipynb`
   - Match the existing MJO preprocessing.
   - Save aligned moisture arrays and labels.

3. `03_mjo_latent_moisture_diagnostics.ipynb`
   - Diagnose existing embeddings before training any new model.
   - Produce moisture/OLR phase offset figures.

4. `04_mjo_bt2d_physics_train.ipynb`
   - Train new 2D representation with same-day Barlow Twins views plus phase/dynamics and physics constraints.

5. `05_mjo_bt2d_physics_analysis.ipynb`
   - Compare to RMM, previous InfoNCE SSL, NSV, and previous Barlow Twins.
   - Evaluate ENSO displacement and moisture/convection interpretability.

## Step-by-Step Implementation Plan

### Phase 0: Lock Inputs and Naming

Goal: avoid ambiguity before downloading or training.

Decisions:

- Experiment root: `notebooks/mjo_moisture_constraints/`
- Google Drive output root: `BSISO_SSL_Project/MJO/moisture_constraints/`
- Primary moisture variable: total column water vapor if available from ERA5 single levels.
- Fallback moisture variable: pressure-level specific humidity integrated vertically.
- Primary latent target: new 2D physics-informed Barlow Twins representation.
- Diagnostic baselines: previous MJO SSL 2D, MJO NSV 7D, and Barlow D=3/D=7.
- Convection proxy: `-OLR`, not raw OLR, so larger values mean stronger convection.

Deliverable:

- A short config cell reused by all notebooks with paths, years, domain, variables, and run tag.

Gate:

- Confirm all dates align with existing MJO daily data and RMM labels before any model training.

### Phase 1: Download Moisture Data

Notebook: `01_mjo_moisture_download.ipynb`

Goal: obtain moisture data over the same MJO domain and period as the existing MJO pipeline.

Preferred download:

- ERA5 total column water vapor / vertically integrated water vapor.
- Domain: 15S-15N, 0-360 longitude, 2 degree resolution if possible.
- Period: 1979-2023 daily.
- Season: all days if storage allows; otherwise match current MJO processed record.

Fallback download:

- ERA5 pressure-level specific humidity `q`.
- Suggested levels: 1000, 925, 850, 700, 600, 500, 400, 300 hPa for column estimate.
- Add a lower-tropospheric product from 1000-700 or 850-700 hPa.

Outputs:

```text
MJO/moisture_constraints/data/raw/tcw_1979_2023.nc
MJO/moisture_constraints/data/raw/q_plev_1979_2023.nc   # fallback only
```

Gate:

- No missing days after date parsing.
- Moisture grid can be regridded or selected onto the same longitude convention as MJO fields.

### Phase 2: Preprocess Moisture

Notebook: `02_mjo_moisture_preprocess.ipynb`

Goal: make moisture directly comparable to existing MJO inputs.

Steps:

1. Load moisture and existing MJO labels/date index.
2. Convert longitude convention if needed.
3. Meridionally average over 15S-15N for direct Wheeler-Hendon-style comparison.
4. Remove annual cycle using the same first-three-harmonic method.
5. Remove slow interannual background with the same running-mean method.
6. Normalize by area-averaged temporal standard deviation.
7. Apply the same 20-90 day bandpass used for MJO phase diagnostics.
8. Save column moisture and lower-tropospheric moisture if available.

Outputs:

```text
MJO/moisture_constraints/data/processed/qcol_mjo_processed.npy
MJO/moisture_constraints/data/processed/qlow_mjo_processed.npy
MJO/moisture_constraints/data/processed/moisture_dates.csv
MJO/moisture_constraints/data/processed/moisture_preprocess_meta.json
```

Gate:

- Dates exactly intersect with existing `X_MJO` / RMM dates.
- Processed moisture has reasonable variance and no annual-cycle leakage visible in monthly means.

### Phase 3: Diagnose Existing Latent Spaces

Notebook: `03_mjo_latent_moisture_diagnostics.ipynb`

Goal: measure the moisture-convection phase relationship before imposing any loss constraint.

Inputs:

- Existing MJO SSL 2D embedding.
- Existing MJO NSV 7D latent.
- Existing Barlow D=3/D=7 embeddings.
- Processed column moisture.
- Processed lower-tropospheric moisture if available.
- Processed OLR / `-OLR`.
- RMM phase and ENSO category.

Core computations:

1. Compute latent phase `theta_z = atan2(z2, z1)`.
2. Bin days into 8 and 16 phase bins.
3. Composite `qcol`, `qlow`, `-OLR`, u850, and u200 by latent phase.
4. Compute first-harmonic complex phase:

   ```text
   A_q    = sum_k composite_q(k)    * exp(i theta_k)
   A_conv = sum_k composite_-OLR(k) * exp(i theta_k)
   delta_theta = arg(A_q) - arg(A_conv)
   ```

5. Compute regional summaries over Indian Ocean, Maritime Continent, and western Pacific.
6. Bootstrap days within phase bins for confidence intervals.
7. Repeat diagnostics by ENSO category.

Figures:

```text
phase_composites_q_olr.png
delta_theta_by_longitude.png
delta_theta_regions_with_ci.png
enso_stratified_delta_theta.png
latent_phase_vs_rmm_phase.png
```

Decision gate:

- If moisture and convection are near in phase: use a moisture-mode-compatible regularizer.
- If moisture leads convection by a stable positive phase: use a skeleton/recharge-compatible regularizer.
- If the lag is longitude-dependent or ENSO-dependent: use an auxiliary prediction loss first, not a hard phase-lag target.
- If existing 2D latent has no coherent moisture/OLR composites: train a new model before making theory claims.

### Phase 4: Train Baseline 2D Same-Day Barlow Twins

Notebook: `04_mjo_bt2d_physics_train.ipynb`

Goal: train a non-collapsed 2D latent without destroying MJO phase.

Baseline views:

- Same-day augmented view A and B from the same atmospheric state.
- Possible augmentations: small Gaussian noise, channel dropout, weak longitude jitter, masking, or OLR-only/wind-only cross-view.

Initial loss:

```text
L = L_BT + lambda_var * L_var
```

where:

- `L_BT`: Barlow Twins redundancy reduction.
- `L_var`: VICReg-style variance floor to prevent 2D collapse.

Do **not** use `X_t` and `X_{t+tau}` as invariant views in this baseline.

Outputs:

```text
MJO/moisture_constraints/checkpoints/bt2d_baseline_encoder.pth
MJO/moisture_constraints/results/bt2d_baseline/embeddings.npy
MJO/moisture_constraints/results/bt2d_baseline/training_history.json
```

Gate:

- Effective rank close to 2.
- Nonzero embedding variance in both dimensions.
- RMM phase accuracy / circular correlation better than random.
- Moisture and OLR composites are coherent enough to interpret.

### Phase 5: Add Phase/Dynamics Constraint

Notebook: continue in `04_mjo_bt2d_physics_train.ipynb` with run tag `bt2d_dyn`.

Goal: preserve cyclic temporal organization without enforcing invariance across time.

Candidate terms:

```text
L_speed = mean_t (||z_{t+1} - z_t|| - s0)^2
L_turn = penalty for extreme angular jumps
L_smooth = mean_t circular_distance(theta_{t+1}, theta_t - omega_t)^2
```

Use weak weights. The MJO is irregular; the loss should discourage noise, not force a perfect oscillator.

Gate:

- Lag circular correlation with RMM improves or at least does not collapse.
- ENSO displacement remains measurable.
- Moisture/OLR phase composites remain coherent.

### Phase 6: Add Moisture-Convection Constraint

Notebook: continue in `04_mjo_bt2d_physics_train.ipynb` with run tags based on diagnostic result.

Use only after Phase 3 diagnostic establishes the target.

Option A: auxiliary prediction, safest first physics constraint.

```text
L_aux = MSE(q_hat(z), q_target) + MSE(olr_hat(z), olr_target)
```

This encourages the 2D latent to retain moisture/convection information without forcing a theory.

Option B: moisture-mode-compatible phase regularizer.

```text
L_q_phase = 1 - cos(theta_q - theta_conv)
```

Use only if diagnostics show moisture and convection are robustly in phase.

Option C: skeleton/recharge-compatible phase-lag regularizer.

```text
L_q_lead = 1 - cos((theta_q - theta_conv) - phi0)
```

Use only if diagnostics show a stable moisture lead `phi0`.

Option D: ENSO/radius slow-envelope regularizer.

```text
r = ||z||
L_radius_slow = temporal smoothness on r
```

Use if ENSO or amplitude appears mainly in radius while phase appears in angle.

Gate:

- Physics-constrained model must beat or match baseline on interpretability without destroying RMM alignment.
- Report tradeoff explicitly if moisture interpretability improves but phase skill drops.

### Phase 7: Final Analysis

Notebook: `05_mjo_bt2d_physics_analysis.ipynb`

Compare:

- RMM / own ERA5 RMM
- previous MJO SSL 2D InfoNCE
- MJO NSV 7D
- previous Barlow D=3/D=7
- new BT2D baseline
- new BT2D dynamics
- new BT2D physics-constrained model

Metrics:

- RMM phase probe
- circular correlation with RMM
- lag circular correlation
- ENSO balanced accuracy
- ENSO displacement z-score
- effective rank and collapse metrics
- moisture-convection phase offset with CI
- ENSO-stratified phase offset
- Kelvin-Rossby wind composite coherence

Final figures:

```text
embedding_phase_enso_amplitude.png
q_olr_phase_composite_grid.png
delta_theta_summary.png
enso_delta_theta_summary.png
lag_correlation_comparison.png
model_comparison_table.csv
```

Final decision:

- If new 2D BT has phase + interpretable moisture/convection lag: adopt as the explainable MJO latent.
- If new 2D BT has only ENSO/amplitude envelope: keep it as slow-envelope representation and pair with NSV/InfoNCE for phase.
- If 2D cannot hold both phase and moisture/ENSO: escalate to 3D or 4D latent, with 2D phase plane plus extra slow coordinate.

## Candidate Loss Constraints

Use these in stages. Do not turn all on at once.

### Stage A: Representation Health

Use Barlow Twins redundancy reduction on same-day augmented views:

```text
L_BT = sum_i (1 - C_ii)^2 + lambda_off * sum_{i != j} C_ij^2
```

For 2D, increase `lambda_off` compared with the previous D=7 runs because there is only one off-diagonal pair and decorrelation must be strong enough to prevent collapse.

Add VICReg-style variance floor if needed:

```text
L_var = mean_i max(0, gamma - std(z_i))^2
```

### Stage B: Phase/Dynamics Constraint

Avoid temporal invariance. Instead, preserve temporal order:

```text
L_speed = mean_t (||z_{t+1} - z_t|| - s0)^2
L_turn  = penalty for erratic angular jumps
```

Possible angular smoothness:

```text
theta_t = atan2(z2_t, z1_t)
L_smooth = mean_t circular_distance(theta_{t+1}, theta_t - omega_t)^2
```

Use weak weights so the model can learn irregular MJO speed.

### Stage C: Moisture-Convection Interpretability Constraint

After the diagnostic target is known, add one of:

Moisture-mode prior:

```text
L_q_phase = 1 - cos(theta_q - theta_conv)
```

Skeleton-like prior:

```text
L_q_lead = 1 - cos((theta_q - theta_conv) - phi0)
```

where `phi0` is measured from diagnostics, not imposed blindly.

A safer alternative is not to enforce skeleton or moisture-mode a priori, but to make the latent phase predictive of both:

```text
L_aux = MSE(q_hat(z), q) + MSE(olr_hat(z), olr)
```

Then measure the learned phase offset afterwards.

### Stage D: ENSO Disentanglement Constraint

If the goal is to expose ENSO modulation:

- keep one circular pair `(z1, z2)` for MJO phase
- add one optional scalar or radial dimension for slow ENSO/amplitude

For strict 2D, use radius as the slow coordinate:

```text
theta = atan2(z2, z1)
r = ||z||
```

Then evaluate whether ENSO separates by radius, angular sector, or both.

Possible weak regularizer:

```text
L_radius_slow = temporal smoothness on r
```

Do not directly supervise ENSO unless the experiment is explicitly semi-supervised.

## Main Risks

- OLR is not exactly skeleton-theory wave activity. It is only a convection proxy.
- Column moisture best matches moisture-mode theory; skeleton theory emphasizes lower-tropospheric moisture, so lower-level q should be added before strong claims.
- A 2D latent may be too small for full MJO physics. This is acceptable if the goal is interpretable phase-plane visualization, but compare against NSV 7D.
- If Barlow Twins uses temporal-lag views again, it will likely erase phase again.
- A loss that forces a desired phase relation can create an attractive figure but weaken scientific value. Diagnostics should determine the target before regularization.

## Success Criteria

Minimum:

- Non-collapsed 2D latent.
- Significant circular correlation with RMM or own ERA5 RMM.
- Clear moisture and OLR composites by latent phase.
- Quantified `delta_theta_q_conv` with bootstrap confidence intervals.

Strong:

- The 2D latent separates phase and slow ENSO/amplitude structure better than previous InfoNCE or Barlow embeddings.
- Moisture-convection phase offset differs by ENSO category in a physically interpretable way.
- Kelvin-Rossby wind composites remain coherent in latent phase bins.

Best-case result:

The learned 2D phase plane becomes a physically interpretable MJO diagram: angular position tracks the convective/moisture cycle, radius or sector structure tracks slow ENSO/amplitude modulation, and the phase offset between moisture and convection provides a concrete link to MJO theory.
