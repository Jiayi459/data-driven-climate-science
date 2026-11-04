# Climate-pipeline parity audit

2026-11-04, protocol `climate_bsiso_v1`. The user selected BSISO 2-D CNNs as the
spatial reference. This ports representation-learning recipes; images do not
reproduce climate preprocessing, information content, sample size or results.

## Source mapping

Cell indices are zero-based JSON indices, hashed in `climate_reference_manifest.json`.
The notebook builder rebuilds/checks the manifest. Original climate notebooks are unchanged.

| Component | Reference | Port |
|---|---|---|
| InfoNCE | `extension_2d/08_ssl_temporal_2d.ipynb`, cells2,10,12,14; compare `mjo/15_mjo_ssl_temporal_2d.ipynb`, cells2,12 | BSISO kernels; direct2D output; raw one-way loss; no MLP projector |
| Predictive CNN/decoder | `nsv/18c_nsv_bsiso_stage1_lag10.ipynb`, cells2,6,8,10 | Single-channel64×64 images; reference layers/initialization/lag10 recipe |
| ID | `nsv/19_nsv_bsiso_id_estimation.ipynb`, cells2,4,6,12 | Raw train+val h64; original MLE calls; TwoNN/lPCA; noise/shuffle controls; recorded sample cap |
| SIREN/dynamics | `nsv/20_nsv_bsiso_refine_analysis.ipynb`, cells4,6,16 | Reference layers/initialization, raw-latent MSE, separate dynamics MLP |
| BT projector/loss/training | `mjo/26_mjo_barlow_twins.ipynb`, cells2,8,10 | Same projector/weighted lag objective; BSISO predictive encoder replaces MJO longitude encoder |
| BT dimensions | nb26 D=3, nb27 D=7 | Separate runs; actual output shape/config overrides legacy `z7` names |

## Architecture and objectives

**InfoNCE:** `[B,1,64,64]` → Conv1→16(k3,p1)+BN+ReLU+MaxPool2 →
Conv16→32(k3,p1)+BN+ReLU+MaxPool2 → Conv32→32(k3,p1)+BN+ReLU → GAP →
`pre_fc`32 → Linear32→2 → `z`2. Conv initialization: Kaiming fan-out; BN1/0;
final linear Normal(0,0.01), zero bias. No L2 normalization or separate MLP projector.

\[
s_{ij}=z_i^{A\top}z_j^B/0.5,\qquad
\mathcal L_{\rm NCE}=-\frac1B\sum_i\log\frac{\exp s_{ii}}{\sum_j\exp s_{ij}}.
\]

Positives are available ±1,±2,±3 frames in the same trajectory/split, excluding zero.
Other opposite-branch batch rows are negatives, without nearby-negative masking;
physically similar states can therefore be negatives. Norms affect logits, and
mean/std/max norms are saved per epoch. Cosine loss would be a separate ablation.

**NSV predictive encoder:** Conv1→32(k4,s2,p1) → Conv32→32(k3,s1,p1) →
Conv32→64(k4,s2,p1) → Conv64→64(k3,s1,p1) → Conv64→128(k4,s2,p1), each with
BN+ReLU; GAP → Linear128→64. Decoder: Linear64→128, reshape128×1×1; bilinear
resize8×8 + Conv128→64+BN+ReLU; resize16×16 + Conv64→64+BN+ReLU;
resize32×32 + Conv64→32+BN+ReLU; resize64×64 + Conv32→1, linear output.
Decoder convolutions usek3,p1. The original resize ladder3×6→7×12→15×25→31×51
is adapted to the image size.

\[
\mathcal L_{\rm pred}=\operatorname{mean}_{i,p}
[D(E(I_{t_i}))_p-I_{t_i+10,p}]^2.
\]

CNN and image decoder train jointly. Selected h64 features are then frozen. SIREN:
**64→128→64→32→d→32→64→128→64**, sine(30·linear) at the three intermediate
layers on each side, linear bottleneck/final output. First sine-layer weights:
Uniform(±1/n_in); later: Uniform(±sqrt(6/n_in)/30); sine biases: Uniform(±0.001).
Bottleneck/output use Kaiming uniform with linear gain and zero bias.

\[
\mathcal L_{\rm refine}=\operatorname{mean}_{i,j}[q(r(h_i))_j-h_{ij}]^2.
\]

There is **no h standardization**. Refiner encoder r and decoder q train jointly;
the CNN stays fixed. Main `z=r(h)` uses estimated d; separately trained d=2 is an
ablation. The third stage trains **d→32→64→64→32→d**, ReLU hidden layers,
on fixed z, minimizing `mean((F(z_t)-z_(t+10))**2)`. Pairs stay in their trajectory
and split. Validation/test MSE and persistence MSE are saved. Recursive autonomous
and pixel rollouts are not implemented.

**BT:** NSV-warm-started predictive encoder → h64 → **Linear64→128+BN+ReLU →
Linear128→64+BN+ReLU → Linear64→D**, D=3 or7. Encoder/projector both train;
projector uses original PyTorch default initialization. No NSV refiner is used.

As in nb26, standardize each coordinate with batch **sample std** (B−1 denominator),
epsilon1e-5 **outside** the square root:

\[
\widetilde z_{ik}=\frac{z_{ik}-\overline z_k}{s_k+10^{-5}},\quad
C^{(\tau)}=\widetilde Z_A^\top\widetilde Z_B/B,
\]
\[
\mathcal L_{\rm BT}=\sum_{\tau=1}^{5}0.05^{(\tau-1)/4}
\left[\sum_k(1-C^{(\tau)}_{kk})^2+0.005\sum_{k\ne l}(C^{(\tau)}_{kl})^2\right].
\]

One batch per lag contributes to one backward/update. Epoch length is the minimum
loader length. Fixed validation-batch C matrices are saved initially/after each
epoch, plus training diagonal/off-diagonal losses. Sample std makes identical
nonconstant views' diagonal approximately(B−1)/B, not exactly1; this is preserved.

[README](README.md#method-defaults) records all training defaults. Checkpoint choices
follow the original: InfoNCE final epoch; BT minimum training loss; NSV stages minimum
validation MSE. A unified policy is a separate experiment.

## ID audit

nb19 (also nb22) varies nominal neighborhood sizes with
`skdim.id.MLE(K=k).fit(h)`. With default `neighborhood_based=True`, **K does not
set the local neighborhood size**; `fit(..., n_neighbors=k)` does, according to the
[official MLE API](https://scikit-dimension.readthedocs.io/en/stable/skdim.id.MLE.html).
A flat original “k sweep” is therefore not evidence of neighborhood stability.
This identifies a sweep issue, not a demonstrated numerical correction to earlier
BSISO/MJO ID estimates or proof of any physical dimension.

The port saves:

- `LB_by_k_legacy_call`: original calls. Main refiner still uses
  `max(1,round(mean(...)))`, preserving the historical recipe explicitly.
- `LB_by_explicit_n_neighbors`: actual local-neighborhood sweep, diagnostic only.
- TwoNN/lPCA; Gaussian and coordinate-shuffled controls with TwoNN and original MLE.
  Original automatic “physical hypothesis confirmed” interpretations are not copied.

ID uses train+validation h (as in climate code), deduplicated, with recorded random
sample cap5000 for Colab memory. Original code uses all unique points; increase
`id_max_samples` for that ablation. Capping changes neighborhood fractions and can
change the estimate. Test is excluded. Separate `id_train.json` estimates train-only
ID excluding same-group neighbors. `id_backend='group_mle'`/`id_scope='train'` are
explicit ablations used in tiny local fixtures, not the default climate recipe.

## Deliberate differences and limits

| Aspect | Adaptation |
|---|---|
| Space | Three31×51 BSISO fields become one64×64 image; MJO1D longitude encoder is not ported. BSISO internal layers retained. |
| Time | Days become simulation frames atT0/50. Matching integer offsets does not match physical timescales automatically. |
| Preprocessing | Climate anomalies, bandpass/lowpass, spatial standardization and active-day filtering have no identical image counterpart and are not replicated. |
| Splits/sampling | Trajectories/length groups replace years. Independent test groups shared by all methods. Positive-pair RNG is deterministic per epoch/anchor, not identical climate RNG draws. |
| Warm start | Instead of silently cold-starting on a missing file, require one compatible checkpoint with lineage/checksum. |
| Invalid anchors | Original Info sampler can fall back to self; replica raises without a neighbor. Dense trajectories have valid neighbors. |
| Loaders | num_workers0, deterministic seeds, smaller batches in small fixtures. BT drops incomplete train batches and skips singleton validation batches. |
| Validation | BT additionally records full validation loss without using it for selection; original mainly records fixed-batch C. |
| ID | Sample cap and explicit-neighbor diagnostic added. ENSO-conditioned ID and automatic physical claims omitted. |
| Refiner | Add fixed d=2 comparison alongside estimated d; main d is never forced to2. |
| Displays | Native, train-fitted PCA and test-subset t-SNE separated. BT original uses up to5000 active climate days; toy up to1000 test samples, no activity filter. Scores use original features. |
| Capacity | Architectures, epochs and warm starts differ by method, matching climate recipes. Loss-only conclusions require a separate matched-capacity experiment. |

Position-only single frames omit velocity. Poor v readout alone does not diagnose
training failure; attractive rings or ID≈2 do not prove state recovery. Noise,
finite sampling and nonlinear coordinates affect ID/probes. The toy can expose code
behavior and controlled observation failures, but cannot establish all climate
generalizability, preprocessing correctness, ENSO causation or parameter extrapolation.

## Verification boundary

`test_climate_parity.py` compares actual original notebook definitions against new
modules: forward outputs and input/parameter gradients under matching weights;
SIREN/projector initialization; loss gradients; sampler boundaries; warm-start
weights; tiny pipelines. Display tests check original coordinates are unchanged,
1D padding, separate BT runs and finite t-SNE. `test_l1.py` retains simulator and
legacy/evaluation tests.

These are software checks, not equivalent trained outcomes. Formal T4 training and
the pinned scikit-dimension branch require Colab. Local scikit-dimension is absent;
end-to-end fixtures explicitly use grouped-MLE. GPU/library differences and different
datasets prevent a claim of bitwise reproduction of climate training runs.
