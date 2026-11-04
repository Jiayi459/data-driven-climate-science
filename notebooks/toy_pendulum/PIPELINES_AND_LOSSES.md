# Pendulum learning pipelines and loss functions

Protocol: `climate_bsiso_v1`. Date: 2026-11-04.
These diagrams describe the current implementation in `l1_climate.py`, not the
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
