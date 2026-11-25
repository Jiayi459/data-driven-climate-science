# L1 — Five-frame damped pendulum

**Active notebooks: original five-frame common-backbone protocol, restored 2026-11-25.**
L1a uses fixed g=L=1, damping0.02, zero observation noise, and100 trajectories split
60/20/20 by whole trajectory. Input is five consecutive grayscale images in time
order, `[I(t-4),...,I(t)]`, shape `[5,64,64]`. Training stride=2; evaluation stride=4.
Temporal pairs are separated by5 frames: the other window is `[I(t+1),...,I(t+5)]`.

BT-same, BT-temporal, InfoNCE and NSV use the original `l1_learning.py` again.
InfoNCE therefore has its original MLP projector. BT trains independently and does
not require an NSV warm start. `l1_climate.py` and its reference audit remain available
as a separate single-frame implementation, but notebooks40/41 no longer invoke it.
See [pipelines and losses](PIPELINES_AND_LOSSES.md) and the retained
[climate parity audit](PIPELINE_PARITY.md).

## Start with notebook42: check saved runs before training

1. Open the rebuilt [42_pendulum_comparison.ipynb](42_pendulum_comparison.ipynb) in
   Colab CPU and run Cells1–3. Default dataset is `173b0c7eaac813d1`.
2. Cell3 lists every saved run, its input window, seed, completion marker, presence
   of `features.npz`, and compatibility with the original complete training recipe.
   New single-frame/climate runs are excluded. The inventory does not train anything.
3. If the four old five-frame runs are ready, continue42. It uses saved features,
   fits evaluation probes/baselines on CPU and creates new plots. **No representation
   retraining is needed just to remove display PCA.** Output folders are hashed;
   prior analyses remain intact.
4. If models are genuinely absent, use40 for NSV and41 for InfoNCE/BT on a Colab T4.
   Their original default configuration and source hash are restored exactly, so
   completed original runs are reused by identity/checksum instead of retrained.
   If only a method is missing, run only that method. Partial runs raise an error;
   they are not silently overwritten or resumed.
5. Run39 only if simulator data are absent. Its implementation is unchanged, so the
   original dataset and60/20/20 split remain reusable after newer training runs.

| Saved material | Reuse |
|---|---|
| `datasets/<id>/fixed.npz` and manifest | Physical states/splits; reusable for either input protocol |
| Original five-frame `features.npz`, `run.json`, `complete.json` | Directly usable by42; no GPU needed |
| Only original `model_best.pt` and NSV `refiner_d*.pt` | Can recover features by inference; the default42 expects already exported features |
| Single-frame or climate-replica models | Different architecture/input protocol; do not substitute for five-frame models |
| Downloaded report ZIP alone | JSON/CSV/PNG only; cannot regenerate exact latent coordinates without features/models |

Original run directory names:

```text
bt_same_s42_965cdae7e094e5fb
bt_temporal_s42_11fbe85137e2fe3f
infonce_s42_81eb6af4797e3c95
nsv_s42_4acd97a965814085
```

They live under `MyDrive/BSISO_SSL_Project/toy/pendulum/runs/173b0c7eaac813d1/L1a/`.
New runs have different hashes and do not overwrite these directories. Whether old
artifacts still exist on Drive must be established by the inventory, not by a local
report ZIP. The rebuilt notebook embeds all required helper modules.

## Noise and training settings

L1a **data observation noise is0**. To reproduce the original trained models,
`view_noise_std=0.01` remains an independent training-view augmentation. It is optional
regularization, not process noise and not a source of velocity information. Without
it, clean BT-same receives two identical views. Removing it defines a new training
configuration; update both `BASE` in40/41 and `EXPECTED_TRAINING` in42 for that experiment.
Exported features are always computed without training-view augmentation.

Original settings: seed42; h64; projector64→128→2 with LayerNorm/SiLU; batch128;
30 prediction/SSL epochs; AdamW lr0.001, weight decay1e-5, gradient norm clip5;
minimum validation objective checkpoint. InfoNCE uses L2-normalized vectors,
symmetric cross-view loss, temperature0.1 and same-trajectory nearby-negative masking.
BT uses lag5 (temporal) or lag0 (same-time), off-diagonal weight0.005.
NSV uses a future-window image MSE, train-only grouped ID diagnostics, and separate
SIREN refiner dimensions1/2/3/4/8 (60 epochs, Adam lr0.001). ID is not forced to2.

## Raw latent plots

`latent_test.png` now displays learned **z1 versus z2 directly**, with no PCA,
centering, scaling or t-SNE. `h64` remains available for numerical readouts but is
not the selected latent scatter. NSV displays its explicit `z_d2` refiner output.
The delay-PCA baseline alone fits PCA64 to block-averaged observed windows, then
shows its already available PC1/PC2 without fitting another PCA.

- Each point is one five-frame window. Rows share coordinates and differ only in
  the physical label used for color: theta or velocity at the final input frame.
- All20 held-out trajectories are pooled. Evaluation has1500 windows (75 per
  trajectory); the scatter displays every third one, giving500 points per panel.
- `latent_coordinates.npz` saves exact plotted coordinates and trajectory/frame
  indices. `display_protocol.json` records the plotting rules.
- Numerical probes use full original features, including all64 PCA baseline
  coordinates. Removing display PCA does not change these inputs or the ID logic.
- Compatibility output `latent_native_test.png` also uses raw coordinates. Optional
  t-SNE support remains in the evaluator but is disabled in notebook42.
- No closed ring or visually spread2D cloud alone proves full physical state recovery.
  Five-frame inputs contain motion information; evaluate both theta and velocity.

Notebook42 exports JSON/CSV/PNG reports in Cell6. Keep feature arrays and checkpoints
on Drive; the report ZIP intentionally does not contain them.

`state_space_by_latent_test.png` reverses the view: true final-frame theta/velocity
are the axes, and each learned method's raw z1/z2 supplies colors in two rows.
It uses the exact500 saved samples from `latent_test.png`, without re-fitting or
resampling, and excludes the PCA baseline. All point/color values are exported in
`state_space_by_latent_test.csv` (included in the report ZIP). Notebook42 Cell5
creates this figure from `evaluation_truth.npz` and `latent_coordinates.npz`.
**Cell5a is standalone:** run only that cell to mount Drive and draw the existing
`c45df90165111f0f` results. It embeds all plot code and does not require Cells1–5,
the module-upload cells, training, or readout evaluation to run first.
For an already evaluated folder, run only the plotting helper on Colab CPU:

```bash
python l1_state_plot.py /path/to/analysis/173b0c7eaac813d1/L1a/c45df90165111f0f
```

## Development checks

```bash
python notebooks/toy_pendulum/build_notebooks.py
python notebooks/toy_pendulum/build_notebooks.py --check
python -m unittest discover -s notebooks/toy_pendulum -p 'test_*.py' -v
```

Tests use tiny CPU fixtures, not scientific training runs. They check original run
identity reuse, saved-run filtering, exact equality of plotted versus model feature
coordinates, data split integrity, losses and notebook embedding. Climate-reference
checks remain separate regression coverage. Do not install PyTorch locally for this.
