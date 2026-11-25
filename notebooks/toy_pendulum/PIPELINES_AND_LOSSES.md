# Active five-frame pipelines and losses

Restored 2026-11-25. Notebooks40/41 use the original `l1_learning.py`.
L1a: g=L=1, damping0.02, observation noise0, trajectory split60/20/20,
training stride2, evaluation stride4. The original independent training-view
augmentation std0.01 remains; it is disabled for exported features.
All learned latent plots use original coordinates, without PCA or t-SNE.
The formulas below are in English.

Let X_t=[I_(t-4),...,I_t], and Y_t=X_(t+5). The shared CNN outputs h in R^64.
Each SSL projector is Linear64→128→LayerNorm→SiLU→Linear128→2.

```text
InfoNCE:
X_t     → shared encoder → h64 → shared MLP projector → z2 ┐
X_(t+5) → shared encoder → h64 → shared MLP projector → z2 ┴→ symmetric InfoNCE
After training: clean X_t → encoder → projector → z2 → scatter(z1,z2)

Barlow Twins:
X_t → shared encoder → h64 → shared MLP projector → z2 ┐
Y   → shared encoder → h64 → shared MLP projector → z2 ┴→ Barlow loss
Y=X_(t+5) for temporal BT; Y=X_t for independently augmented same-time BT.
After training: clean X_t → encoder → projector → z2 → scatter(z1,z2)

NSV:
Stage1: X_t → encoder → h64 → image decoder → predicted X_(t+5) → image MSE
Then: frozen encoder h64 → train-only ID diagnostic
Stage2: h64 → training-set normalization → refiner encoder → z_d
        → refiner decoder → reconstructed normalized h64 → latent MSE
After training: clean X_t → encoder → normalization → refiner encoder → z_d2
               → scatter(z1,z2)
```

**InfoNCE loss.** Define q_i^A=z_i^A/||z_i^A|| and q_i^B=z_i^B/||z_i^B||.
Let s_ij=(q_i^A)^T q_j^B/T, with T=0.1. Retain the matching row i=j;
mask other rows from the same trajectory whose anchor times differ by fewer
than window+lag=10 frames. Denote the allowed column/row sets by A_i and B_i.

\[
\mathcal L_{\mathrm{NCE}}=-\frac{1}{2B}\sum_i
\left[\log\frac{e^{s_{ii}}}{\sum_{j\in A_i}e^{s_{ij}}}
+\log\frac{e^{s_{ii}}}{\sum_{j\in B_i}e^{s_{ji}}}\right].
\]

The encoder and projector are optimized jointly. Plotting uses raw z, not q.

**Barlow Twins loss.** Standardize each feature across the batch using population
variance and epsilon=1e-5. Define C=(tilde Z_A)^T tilde Z_B/B.

\[
\mathcal L_{\mathrm{BT}}=\sum_k(1-C_{kk})^2
+0.005\sum_{k\ne l}C_{kl}^2.
\]

The encoder and projector are optimized jointly. Temporal BT uses one lag5 pair,
not the climate replica's weighted sum over lags1..5.

**NSV prediction loss.** With P=5*64*64 image values per target window,

\[
\mathcal L_{\mathrm{prediction}}=\frac{1}{BP}\sum_{i,p}
\left[D(E(X_i))_p-Y_{i,p}\right]^2.
\]

Only the input receives training-view augmentation. The target remains the future
observation. The prediction encoder and image decoder are optimized jointly.

**NSV refinement loss.** Normalize frozen h using training-set mean and coordinate
standard deviations (with the numerical scale floor in `Refiner`). Let this vector
be tilde h, R_e the refiner encoder, and R_d its decoder.

\[
\mathcal L_{\mathrm{refinement}}=\frac{1}{64B}\sum_i
\left\|R_d(R_e(\widetilde h_i))-\widetilde h_i\right\|_2^2.
\]

The two refiner components are optimized jointly; the original encoder stays
fixed. Fit d in {1,2,3,4,8}. Plotting z_d2 does not mean an ID estimator selected2.
This original pipeline has no separately trained latent dynamics MLP.

---

The remaining sections document the separate climate-replica implementation.
They are not the active notebook40/41 recipe after the five-frame restoration.

# Retained single-frame climate-replica pipelines

Protocol: `climate_bsiso_v1`. Date: 2026-11-04.
These diagrams describe the retained alternative in `l1_climate.py`, not the active
historical common-backbone experiments. Each encoder input is one grayscale image,
shape `[1,64,64]`. Batch size is denoted by B; the number of image pixels is P=4096.
PCA and t-SNE are post-training display transforms and are absent from all losses.

## 1. InfoNCE

```text
I_t       → shared CNN → GAP → pre_fc32 → Linear(32→2) → z_t
I_(t+δ)   → shared CNN → GAP → pre_fc32 → Linear(32→2) → z_(t+δ)
                                                        ↓
                                    raw dot-product InfoNCE loss
                                                        ↓
                              update CNN and final Linear jointly

δ ∈ {−3, −2, −1, +1, +2, +3}, within the same trajectory and split
```

The final Linear layer is part of the encoder. There is no separate nonlinear MLP
projector. The full image-to-z mapping remains nonlinear because the CNN is nonlinear.
Both branches share parameters; branch B is not a frozen target encoder.

**Loss: raw, one-direction InfoNCE**

For paired batches A and B, let z_i^A and z_i^B be the direct 2D encoder outputs.
The matching pair has index i; other opposite-branch batch rows are negatives.

\[
s_{ij}=\frac{(z_i^A)^\top z_j^B}{T},\qquad T=0.5,
\]
\[
\boxed{\mathcal L_{\mathrm{InfoNCE}}
=-\frac{1}{B}\sum_{i=1}^{B}
\log\frac{\exp(s_{ii})}{\sum_{j=1}^{B}\exp(s_{ij})}.}
\]

No L2 normalization, reverse-direction term, or nearby-negative mask is applied.
The loss updates the CNN and its final Linear layer together. No physical-state
labels enter this objective.

## 2. Temporal Barlow Twins

```text
Selected NSV predictive encoder checkpoint → initialize BT encoder

I_t       → shared encoder → h64 → shared MLP projector → z_t       (D=3 or 7)
I_(t+τ)   → shared encoder → h64 → shared MLP projector → z_(t+τ)   (D=3 or 7)
                                                           ↓
                                            cross-correlation C^(τ)
                                                           ↓
                                     weighted BT losses at τ=1,2,3,4,5
                                                           ↓
                                              sum → one backward pass
                                                           ↓
                                     update encoder + projector jointly

MLP projector:
64 → Linear128 → BatchNorm → ReLU → Linear64 → BatchNorm → ReLU → LinearD
```

NSV warm start initializes the encoder; the encoder is subsequently fine-tuned,
not frozen. Each lag uses its own paired minibatch. The NSV refiner is not used in BT.

**Loss: weighted temporal Barlow Twins**

For each lag τ and each branch V in {A,B}, define the batch mean and sample standard
deviation independently for each coordinate k:

\[
\mu_k^V=\frac1B\sum_{i=1}^{B}z_{ik}^V,\qquad
s_k^V=\sqrt{\frac1{B-1}\sum_{i=1}^{B}(z_{ik}^V-\mu_k^V)^2},
\]
\[
\widetilde z_{ik}^V=\frac{z_{ik}^V-\mu_k^V}{s_k^V+\epsilon},\qquad
C_{kl}^{(\tau)}=\frac1B\sum_{i=1}^{B}\widetilde z_{ik}^A\widetilde z_{il}^B,
\quad \epsilon=10^{-5}.
\]
\[
\boxed{\mathcal L_{\mathrm{BT}}
=\sum_{\tau=1}^{5}w_\tau
\left[\sum_{k=1}^{D}(1-C_{kk}^{(\tau)})^2
+\lambda\sum_{k\ne l}(C_{kl}^{(\tau)})^2\right],
\quad w_\tau=0.05^{(\tau-1)/4},\quad\lambda=0.005.}
\]

The diagonal term promotes agreement between corresponding coordinates; the
cross-coordinate term penalizes redundancy. The loss updates encoder and projector
together. The sample-standard-deviation convention and epsilon placement match the
reference notebook. Optional BT-same uses the same bracketed objective with one
same-time pair of augmented views and weight1; it is not the main climate replica.

## 3. Neural State Variables (NSV)

```text
Stage 1 — Predictive representation
I_t → CNN encoder E → h_t64 → image decoder D_img → predicted I_(t+10)
                                                       ↓
                                    MSE against observed I_(t+10)
                                                       ↓
                                   update encoder + image decoder

Stage 2 — State refinement
Freeze E → extract h64 → estimate intrinsic dimension → choose d

h64 → refiner encoder r → z_d → refiner decoder q → reconstructed h64
                                                        ↓
                                            MSE against original h64
                                                        ↓
                                         update r + q; keep E frozen

r: 64 → 128 → 64 → 32 → d
q: d → 32 → 64 → 128 → 64
Hidden layers use sine activations; bottleneck and final output are linear.

Stage 3 — Latent dynamics
Freeze E and r → z_t → dynamics MLP F → predicted z_(t+10)
                                              ↓
                                MSE against fixed target z_(t+10)
                                              ↓
                                          update F only

F: d → 32 → 64 → 64 → 32 → d, with ReLU hidden layers
```

Main d is ID-selected. A separately trained d=2 refiner is an additional ablation
when different. Raw h is used without standardization. The three stages are trained
sequentially; their objectives are not summed into one joint training loss.

**Stage 1 loss: future-image prediction**

\[
\boxed{\mathcal L_{\mathrm{prediction}}
=\frac1{BP}\sum_{i=1}^{B}
\left\|D_{\mathrm{img}}(E(I_{t_i}))-I_{t_i+10}\right\|_2^2.}
\]

Trainable: E and D_img. Target: observed future image, including condition noise.

**Stage 2 loss: latent reconstruction**

\[
h_i=\operatorname{stopgrad}(E(I_{t_i})),\qquad
\boxed{\mathcal L_{\mathrm{refinement}}
=\frac1{64B}\sum_{i=1}^{B}\left\|q(r(h_i))-h_i\right\|_2^2.}
\]

Trainable: r and q. Frozen: E. Target: raw 64D encoder features.

**Stage 3 loss: latent dynamics**

\[
z_t=\operatorname{stopgrad}(r(E(I_t))),\qquad
\boxed{\mathcal L_{\mathrm{dynamics}}
=\frac1{Bd}\sum_{i=1}^{B}\left\|F(z_{t_i})-z_{t_i+10}\right\|_2^2.}
\]

Trainable: F only. Targets are frozen refined representations from the same
trajectory. This is a one-step transition loss, not autonomous-rollout training.

All equations above are data objectives. Adam weight decay is an optimizer setting
and is documented separately in README; it is not an additional encoder-only or
projector-only learning objective.

## Does the MLP projector matter for pendulum?

It can materially change representation geometry, accessibility to linear readouts,
and gradients reaching the CNN. For z=g(h), the feature gradient is
`dL/dh = J_g(h)^T (dL/dz)`. Thus the projector affects encoder training as well as the
final plotted coordinates; it is not post-training PCA. Removing a nonlinear MLP
still leaves a learned linear output layer and a nonlinear CNN.

A nonlinear head can support useful upstream features even when its output is less
useful for downstream readouts. SimCLR demonstrated this on ImageNet, including
better pre-head than post-head readout performance; it does not establish the sign
or size of the effect for this pendulum or its temporal objective.
[SimCLR, Section4.2](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf).

Historical single-frame L1a, seed42, supports examining both layers:

| Legacy model | Angle ridge R² from h64 | Angle ridge R² from z2 | Angle MLP R² from z2 |
|---|---:|---:|---:|
| InfoNCE | 0.999978 | 0.983096 | 0.996215 |
| BT-same | 0.999069 | 0.478910 | 0.847874 |
| BT-temporal | 0.999888 | 0.999335 | 0.998291 |

Source: [archived analysis](../../results/toy_pendulum/173b0c7eaac813d1/single_frame/45eeb9463f6b0f10/analysis.md).
BT-same's MLP readout reached its iteration limit. These are same-run layer
comparisons from one seed, not projector-on/off ablations, and are not new
climate-replica results. Nonlinear recovery suggests a coordinate effect in some
cases; failed finite probes do not prove all information is absent.

The broad BT and NSV block diagrams remain familiar, but training changed materially:
BT gains NSV warm start, a deeper BN/ReLU projector and weighted multi-lag loss;
NSV changes encoder/decoder, lag, ID selection, SIREN depth, raw-h target scaling,
training budget, and adds dynamics. InfoNCE also changes its CNN, loss normalization,
temperature, directionality, negative masking and pairing, not just its projector.
Old-versus-new performance cannot isolate the projector's causal effect.

A focused follow-up would keep the same BSISO CNN up to GAP32, the same pairs/splits,
raw InfoNCE loss, D=2 and training recipe, comparing Linear32→2 against a specified
nonlinear MLP32→128→2. Compare upstream and final features, linear/nonlinear angle
readouts, noise sensitivity, variance/collapse and several training seeds. This is
a proposed ablation, not implemented or run here. A projector cannot recover unique
angular velocity absent from a position-only single frame.
