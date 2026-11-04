# L1 — Noisy damped pendulum

**Current protocol: `climate_bsiso_v1` (2026-11-04).** Reproduce the project's
climate representation pipelines on single-frame pendulum images, using **BSISO
2-D CNNs**, as requested. Each method retains its reference architecture, loss,
pairing and training recipe. This is a pipeline comparison; differences cannot be
attributed to the loss alone. For arrow diagrams and English formulas, see
[PIPELINES_AND_LOSSES.md](PIPELINES_AND_LOSSES.md). See [PIPELINE_PARITY.md](PIPELINE_PARITY.md) for source
mapping, exact losses, adaptations and the ID-estimation audit issue.

The old common-backbone implementation remains in `l1_learning.py`. Both five-frame
and single-frame legacy L1a results exist; neither comes from the new replica.
**Retrain models. Reuse data:** `l1_data.py` is unchanged, and dataset
`173b0c7eaac813d1` remains compatible.

## Run in Colab

Each notebook embeds its helpers and can be uploaded directly to Colab. Formal
training belongs on a T4; no local PyTorch installation is needed.

| Notebook | Runtime | Work |
|---|---|---|
| [39_pendulum_data.ipynb](39_pendulum_data.ipynb) | CPU | Simulate; check integration/energy; save states and whole-group splits |
| [40_pendulum_nsv.ipynb](40_pendulum_nsv.ipynb) | T4 | Future-frame prediction → ID → SIREN refinement → latent dynamics MLP |
| [41_pendulum_ssl_barlow.ipynb](41_pendulum_ssl_barlow.ipynb) | T4 | Direct-output InfoNCE; NSV-warm-started temporal BT at D=3 and D=7 |
| [42_pendulum_comparison.ipynb](42_pendulum_comparison.ipynb) | CPU | Held-out probes/forecasts; native/PCA/t-SNE displays; report ZIP |

1. Reuse the existing dataset ID, or run 39 (`PROFILE='pilot'`; `smoke` checks code only).
2. In 40/41 set `DATASET_ID`, `CONDITIONS_TO_RUN=['L1a']`, `SEEDS=[42]`.
   Blank ID works only with exactly one dataset. Run **40 before BT in 41**.
   Notebook 40 installs `scikit-dimension==0.3.4` into Colab only.
3. Retain method-specific defaults below for the replica; changes are ablations.
   BT requires one completed compatible NSV run for the same condition and seed.
   If several exist, set the exact directory name in `NSV_RUN_IDS` in 41.
   Missing/ambiguous warm starts raise an error; parent run/checksum are recorded.
4. Run 42 with the same dataset, window1, stride4 and protocol. It excludes legacy
   runs, reports missing methods, and retains each configuration/seed separately.
   `SELECTED_RUN_IDS` selects a prespecified subset, not a best-test run.
5. Download reports in 42 Cell6. Repeat for L1b/L1c and more seeds after L1a.
   No new climate-replica scientific training results have been produced locally.

## Method defaults

| | InfoNCE | Barlow temporal | NSV |
|---|---|---|---|
| Input per branch | One `[1,64,64]` image | One `[1,64,64]` image | One `[1,64,64]` image |
| Representation | BSISO temporal CNN → **z2** directly; no MLP projector | Predictive CNN → h64 → **64→128→64→D** projector; D=3,7 | Predictive CNN → h64; fixed h64 → **SIREN → z_d** |
| Pair/target | Random ±1/2/3 frames in same trajectory | Lags1–5; exponential weights1→0.05 | Image t+10; refined-state dynamics also t→t+10 |
| Objective | Raw dot product, one-way InfoNCE, temperature0.5 | Original sample-std cross-correlation; off-diagonal weight0.005; sum lag losses before one update | Future-image MSE; raw h64 reconstruction MSE; latent transition MSE |
| Initialization | Random | NSV encoder warm start; encoder + projector both trained | Random stage1; separate SIREN/dynamics fits |
| Stage1 epochs / batch | 50 / 64 | 100 / 256 | 100 / 64 |
| Optimizer | Adam lr0.001, weight decay0.0001 | Same | Same |
| Schedule / clipping | Cosine to1e-5 / norm1 | Cosine to1e-5 / none | Cosine to1e-5 / none |
| Selected checkpoint | Final epoch | Minimum **training** objective | Minimum validation MSE |
| Added view augmentation | None | None | None |

NSV refinement: 500 epochs, batch64, Adam lr1e-4, weight decay0, cosine minimum1e-6,
minimum validation MSE. Main bottleneck d is estimated, **not fixed at2**; a fixed
d=2 refiner is an additional comparison when different. Dynamics MLP: 300 epochs,
batch64, Adam lr1e-3, weight decay1e-4, cosine minimum1e-5, minimum validation MSE.

`bt_same` is an optional toy control (`RUN_SAME_TIME_CONTROL=False`), not a climate
replica: same-time views with independent pixel augmentation std0.01 and independent
observation-noise draws. NSV and BT are themselves self-supervised; “SSL” in the
filename means contrastive InfoNCE.

## Data and interpretation

L1a/b/c are data conditions, each usable with all methods:

| Condition | Observation | Independent split unit |
|---|---|---|
| L1a | Fixed L, no noise | Trajectory |
| L1b_s002 / L1b_s005 | Same L1a states, pixel noise std0.02/0.05 | Same trajectory split |
| L1c_visible | Continuous L, visible rod length, noise std0.02 | Length group |
| L1c_hidden | L changes dynamics but not displayed rod length, noise std0.02 | Length group |

- Dynamics: `theta_dot=v`, `v_dot=-0.02*v-(g/L)*sin(theta)`, g=L0=1, T0=2π.
  Eight periods, 50 observation intervals/period, 8 RK4 substeps/interval,
  401 frames/trajectory. Integration convergence and energy are checked.
- Pilot: 100 fixed-L trajectories, 60/20/20 split. Variable-L: 60 continuous draws
  in[0.8,1.2], five trajectories each, whole length groups split60/20/20. L is fixed
  within each trajectory; synthetic “years” are independent environment groups.
- Grayscale64×64 images. No stacking, numerical velocity channel, cropping,
  amplitude rescaling, per-frame standardization or clipping noisy training arrays.
- One interval=T0/50: NSV10frames=0.2T0, InfoNCE1–3frames=0.02–0.06T0,
  BT1–5frames=0.02–0.10T0. These are **simulation frames, not calendar days**.
- Condition noise is fixed per trajectory/frame, shared across temporal branches
  and evaluation. Training augmentation is separate and zero in the three main
  methods. NSV predicts noisy future observations, not privileged clean truth.
- **Observability:** fixed-L images depend on theta, not v. A deterministic encoder
  cannot uniquely distinguish opposite velocities at the same position. Clean
  snapshot geometry is locally1D, or2D in(theta,L) when continuous L is visible.
  Physical state dimension2/3 is an oracle reference, not a required learned ID.
  Single-day multivariable climate fields have a different observation mapping.

## Evaluation and saved outputs

Physical labels enter simulator checks and post-training probes, not representation
losses. Whole trajectories/length groups stay in one split. Scaling, PCA and probes
fit train; probe hyperparameters use validation. Climate ID uses train+validation
as in nb19, with independent test excluded; train-only grouped ID is also saved.
See [the parity audit](PIPELINE_PARITY.md#id-audit) for the original MLE parameter issue.

Readouts use **original features**: InfoNCE `pre_fc`32 and `z`2; BT `h`64 and `z`3/7;
NSV `h`64, ID-selected `z`, and additional `z_d2` when applicable. Display transforms
never feed probes:

- `latent_native_test.png`: first two original coordinates; 1D is zero-padded;
  higher coordinates of D>2 are omitted.
- `latent_test.png`: PCA fitted on train, applied to test. For2D z this centers and
  rotates/reflects coordinates without whitening or dimensional reduction.
- `latent_tsne_test.png`: descriptive t-SNE on the displayed test subset, up to1000
  points. It is not an out-of-sample mapping or physical phase portrait.
- Each run has a separate panel labelled with dimension/run suffix. Native/t-SNE
  coordinates are saved. Colors: theta(rad), v(rad/simulation-time unit); latent
  axes have no physical unit.

PCA baseline fits train on block-averaged observed images. Its legacy `delay_pca`
name does not imply history when window1. Oracle-state persistence receives true
current theta/v. Grouped bootstrap CIs condition on a trained model; seed variability
stays separate. Direct ridge forecasts at ~T0/4,T0,2T0 and NSV's one-step latent MLP
are **not autonomous rollouts**.

```text
MyDrive/BSISO_SSL_Project/toy/pendulum/
  datasets/<dataset_id>/manifest.json, fixed.npz, variable.npz, oracle_*
  runs/<dataset_id>/<condition>/<method>_s<seed>_<hash>/
    run.json, training_history.json, checkpoint_selection.json
    encoder_initial.pt, encoder_selected.pt, model_final.pt, model_selected.pt
    features.npz, id_train.json, complete.json
    intrinsic_dim.json, dense_features.npz, refinement_summary.json  # NSV
    refine_d*.pt, refine_d*_history.json, dynamics*                   # NSV
    bt_corr_history.npz                                             # BT
  analysis/<dataset_id>/<condition>/<evaluation_hash>/
    evaluation.json, scorecard.csv, probe_selection.json, id_train.json
    evaluation_truth.npz, display_protocol.json, latent_*.png
    latent_*_coordinates.npz, state_readout_test.png, forecast_test.png
```

Uint8 frame caches stay in Colab `/content/ddcs_l1_frames` (~164MB fixed-L,
~493MB per variable-L rendering). ZIP includes JSON/CSV/PNG, not arrays/checkpoints.
Source/config/protocol/parent-checkpoint/reference-cell hashes identify runs.
Completed runs reuse exact identities/checksums. Partial runs are not overwritten;
inspect them or change `RUN_TAG`. Automatic epoch-level resume is not implemented.

## Development checks

Edit Python sources and rebuild embedded copies:

```bash
python notebooks/toy_pendulum/build_notebooks.py
python notebooks/toy_pendulum/build_notebooks.py --check
python -m unittest discover -s notebooks/toy_pendulum -p 'test_*.py' -v
```

`test_climate_parity.py` extracts original notebook definitions and checks
outputs/gradients after documented image adaptations, losses, initialization,
pair sampling, warm-start lineage, refinement/dynamics, protocol isolation and plots.
`test_l1.py` retains simulator/legacy/evaluation regression tests. Tiny CPU fixtures
check code, not scientific performance. The pinned scikit-dimension branch requires
Colab; local fixtures explicitly use grouped-MLE because that dependency is absent.

Historical analysis: [five-frame L1a](../../results/toy_pendulum/173b0c7eaac813d1/analysis.md)
and [single-frame legacy L1a](../../results/toy_pendulum/173b0c7eaac813d1/single_frame/45eeb9463f6b0f10/analysis.md).
L1b/L1c learning, process noiseL1d, autonomous rollouts and L2/L3 remain future work.
