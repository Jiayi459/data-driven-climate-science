# L1 — Noisy damped pendulum

This folder continues the project's global notebook numbering after notebook 38.
All four notebooks are self-contained: upload/open the `.ipynb` in Colab and run its
cells. They embed their Python helpers, so an unpublished GitHub commit or separate
module upload is not required. No packages are installed into the local conda environment.

## Experimental matrix

L1a/b/c are **data conditions**, not different learning methods. Every condition can
run the same method comparison:

| Condition | NSV (40) | InfoNCE (41) | BT temporal (41) | BT same-time (41) |
|---|---|---|---|---|
| L1a: clean, fixed L | yes | yes | yes | yes |
| L1b: same states + observation noise | yes | yes | yes | yes |
| L1c: continuous L across synthetic years | yes | yes | yes | yes |

NSV and Barlow Twins are themselves self-supervised. “SSL” in the notebook filename
means the project's contrastive InfoNCE method. Start by checking L1a with NSV, then
run all objectives; use the same protocol for L1b and L1c.

## Run order

| Notebook | Runtime | Function |
|---|---|---|
| [39_pendulum_data.ipynb](39_pendulum_data.ipynb) | CPU | RK4 simulation, immutable dataset, whole-group splits, energy/convergence checks, observation preview, oracle ID |
| [40_pendulum_nsv.ipynb](40_pendulum_nsv.ipynb) | T4 GPU | Future-window encoder/decoder, train-only ID, SIREN refiners at d=1/2/3/4/8 |
| [41_pendulum_ssl_barlow.ipynb](41_pendulum_ssl_barlow.ipynb) | T4 GPU | Matched CNNs with InfoNCE, BT-temporal and BT-same-time |
| [42_pendulum_comparison.ipynb](42_pendulum_comparison.ipynb) | CPU | Train/validation-fitted probes, held-out readouts and direct forecasts, grouped CIs and plots |

1. Run 39. Keep its printed `DATASET_ID`. `PROFILE='smoke'` reduces trajectory count;
   `PROFILE='pilot'` creates the initial experiment-sized data. A smoke run is only a
   pipeline check, not scientific evidence.
2. In 40 and 41, set that ID and `CONDITIONS_TO_RUN=['L1a']`. Blank ID resolves only
   when exactly one dataset exists; it never silently chooses the latest dataset.
3. Keep `BASE.window`, `BASE.lag`, `BASE.eval_stride`, `SEEDS`, and other shared
   hyperparameters matched. Default is five-frame input and five-frame lag (no overlap),
   30 stage-1 epochs, 60 NSV refinement epochs, and seed 42.
4. Run 42 with the same ID/history/stride. It lists missing methods and saves all
   compatible completed runs separately. Use `SELECTED_RUN_IDS` for a prespecified subset.
5. Repeat for `L1b_s002` / `L1b_s005`; then `L1c_visible` / `L1c_hidden`. Add 3 to
   `PROJECTION_DIMS` for the L1c three-coordinate comparison. Expand seeds only after the pilot works.

## Data and views

- Unit-mass pendulum: `theta_dot=v`, `v_dot=-gamma*v-g/L*sin(theta)`.
  Defaults: g=L0=1, gamma=0.02, 8 reference periods, 50 observations/period,
  8 RK4 substeps/observation. Step-halving and energy decay are checked before saving.
- Fixed-L data: 100 trajectories with independent angle/velocity initial conditions.
  L1a and every L1b noise condition use exactly the same states and splits.
- Variable-L data: 60 continuous length draws in [0.8,1.2], five independent
  trajectories per length. Each synthetic year/length group remains in one split.
  Length is constant within each trajectory; trajectories are not joined across years.
- Conditions: L1a sigma_obs=0; L1b sigma_obs=0.02/0.05; both L1c conditions sigma_obs=0.02.
  Visible L changes the rendered rod length; hidden L uses a fixed display length and
  only changes the dynamics. No year IDs/text labels or state values enter training.
- 64×64 grayscale images; no per-frame standardization, cropping, amplitude scaling,
  clipping of noisy arrays, or time shuffling. Quantized clean images form a reusable
  local cache; observed noise is added as float32.
- Observation noise is keyed by trajectory/frame/view. Overlapping windows have
  consistent pixels within a view; all methods see the same seeded observations.
- A separate **training augmentation** `view_noise_std=0.01` is applied to input
  branches even in L1a. It is independent of the condition's observation noise and
  saved in each run's config. Evaluation uses condition noise only. Set to zero for
  an explicit no-augmentation experiment; BT-same then has identical L1a views.
- NSV predicts noisy future observations, not privileged clean truth. BT-same uses
  independent observation views of the same window; this is a synthetic two-view
  protocol, documented separately from temporal views.

## Evaluation and interpretation

- Optimize representation losses without theta, v or L labels. Physical labels are
  used for simulator checks and post-training probes only.
- One common evaluation index includes all required future horizons. Checksum and
  exact row matching prevent mixing cached features or mismatched observation conditions.
- Shared backbone output h is 64-D. Compare projector/NSV outputs at matched dimensions;
  InfoNCE additionally saves its unit-normalized projector features, which discard norm.
- PCA uses 16×16 block-averaged observed delay windows and fits on training data only.
  Ridge scalers fit train; ridge alpha uses validation. The optional fixed-width MLP
  readout fits train with no random temporal validation split.
- Report theta/v readouts (plus L in L1c), direct ridge forecasts at approximately
  T0/4, T0 and 2T0, and train-only kNN ID estimates excluding same-group neighbors.
  Current-state nonlinear readouts diagnose nonlinear coordinate changes; the main
  forecasts remain linear readouts and **are not autonomous rollouts**.
- The persistence baseline is explicitly **oracle-state persistence**: it gets the
  true current theta/v. Do not describe it as an unlabeled image baseline.
- Bootstrap over independent trajectories (L1a/b) or length groups (L1c). These CIs
  condition on one trained model; model-seed variation is not included. Run seeds and
  configurations remain separate in the CSV; no “best test seed” is selected.
- Figures display the first available run and low-dimensional feature per method in
  deterministic order. They do not display every dimension/seed; consult the complete CSV.
- Fixed L has a two-variable physical state. A continuously sampled, constant-per-run
  L supports a three-variable augmented-state reference if observable. Noisy image-cloud
  ID need not equal 2/3, and a few discrete lengths do not establish a 3-D local manifold.

## Output layout and reruns

```text
MyDrive/BSISO_SSL_Project/toy/pendulum/
  datasets/<dataset_id>/
    manifest.json, fixed.npz, variable.npz, oracle_*.png/json
  runs/<dataset_id>/<condition>/<method>_s<seed>_<config_hash>/
    run.json, model_best.pt, training_history.json
    id_train.json, features.npz, complete.json
    refiner_d*.pt, refine_history_d*.json                  # NSV only
  analysis/<dataset_id>/<condition>/<evaluation_hash>/
    evaluation.json, scorecard.csv, probe_selection.json, id_train.json
    evaluation_truth.npz, latent_test.png, forecast_test.png, state_readout_test.png
```

Full image arrays live only in `/content/ddcs_l1_frames`, not Drive. One default
fixed-L cache is about 164 MB; one variable-L rendering cache about 493 MB (uint8).
Local Colab caches may be deleted/regenerated without deleting saved states/results.

Dataset identity includes simulation settings and data-code hash. Runs additionally
include condition, seed, learning settings and training-code hash. Completed artifacts
are reused only for the same identity; incomplete data/runs are not overwritten.
If a runtime stops, inspect the partial run and use a new `run_tag` to restart.
Automatic epoch-level resume is not implemented. Updating data helper code requires
regenerating notebooks together and a new dataset ID; stale code fails explicitly.

This implementation covers L1a/b/c. Random force noise (L1d), a continuously changing
physical pendulum length, autonomous latent rollouts, and L2/L3 remain future work.
No real scientific training results are included in these notebooks.

## Maintaining and validating the notebooks

The `.py` files are the editable source of truth. `build_notebooks.py` embeds them
verbatim into the notebooks; do not edit an embedded helper independently.

```bash
python notebooks/toy_pendulum/build_notebooks.py
python notebooks/toy_pendulum/build_notebooks.py --check
python -m unittest discover -s notebooks/toy_pendulum -p 'test_l1.py' -v
```

Local checks use only an already available scientific environment; do not install
PyTorch into climate-sci. Torch tests skip if it is absent/unusable. Tiny one-epoch
fixtures, when supported, validate code paths and do not constitute project training.
Full GPU experiments belong on Colab.

Validation on 2026-10-27: all eight tests passed, including the analytic small-angle
reference, integration/energy checks, group/noise/cache integrity, held-out-label
isolation for linear and nonlinear probes, notebook/schema/source validation,
loss/gradient checks, and tiny one-epoch runs of all four objectives through evaluation.
The noisy hidden-length branch and generated plot layouts were also checked. These
are correctness checks; full Colab GPU runs and scientific conclusions remain pending.
