"""Build self-contained Colab notebooks from the locally testable source modules.

Run: python notebooks/toy_pendulum/build_notebooks.py [--check]
Only the standard library is needed. No data generation or model training occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent

REFERENCE_CELLS = {
    'extension_2d/08_ssl_temporal_2d.ipynb': [2, 10, 12, 14],
    'nsv/18c_nsv_bsiso_stage1_lag10.ipynb': [2, 6, 8, 10],
    'nsv/19_nsv_bsiso_id_estimation.ipynb': [2, 4, 6, 12],
    'nsv/20_nsv_bsiso_refine_analysis.ipynb': [4, 6, 16],
    'mjo/15_mjo_ssl_temporal_2d.ipynb': [2, 12],
    'mjo/26_mjo_barlow_twins.ipynb': [2, 6, 8, 10, 22],
}


def reference_manifest():
    refs = {}
    for name, indices in REFERENCE_CELLS.items():
        notebook = json.loads((HERE.parent/name).read_text())
        refs[name] = {str(i): hashlib.sha256(''.join(notebook['cells'][i]['source']).encode()).hexdigest()
                      for i in indices}
    return {'protocol': 'climate_bsiso_v1', 'cell_indexing': 'zero-based JSON cell indices',
            'source_cell_sha256': refs}


def cell(source, markdown=False):
    source = textwrap.dedent(source).strip()+"\n"
    result = {"cell_type": "markdown" if markdown else "code",
              "id": hashlib.sha256(source.encode()).hexdigest()[:12],
              "metadata": {}, "source": source.splitlines(keepends=True)}
    if not markdown:
        result.update(execution_count=None, outputs=[])
    return result


def md(source):
    return cell(source, True)


def module(name):
    result = cell(f"%%writefile {name}\n"+(HERE/name).read_text())
    result["metadata"] = {"collapsed": True}
    return result


BOOTSTRAP = '''
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import os, sys, tempfile, json

PROJECT_DIR = Path('/content/drive/MyDrive/BSISO_SSL_Project')
L1_ROOT = PROJECT_DIR / 'toy' / 'pendulum'
L1_ROOT.mkdir(parents=True, exist_ok=True)
# Embedded code needs no GitHub clone/download or external .py upload.
RUNTIME_DIR = Path(tempfile.mkdtemp(prefix='ddcs_l1_code_'))
os.chdir(RUNTIME_DIR)
sys.path.insert(0, str(RUNTIME_DIR))
for name in ('l1_data', 'l1_learning', 'l1_climate', 'l1_evaluation'):
    sys.modules.pop(name, None)
CACHE_DIR = Path('/content/ddcs_l1_frames')  # disposable; never store frame arrays on Drive

def resolve_dataset_id(value):
    if value:
        if not (L1_ROOT / 'datasets' / value / 'manifest.json').exists():
            raise FileNotFoundError(f'Dataset {value} not found; run notebook 39 first.')
        return value
    available = sorted(p.parent.name for p in (L1_ROOT / 'datasets').glob('*/manifest.json'))
    if len(available) != 1:
        raise ValueError(f'Set DATASET_ID explicitly from notebook 39. Available: {available}')
    return available[0]

print('Drive outputs:', L1_ROOT)
print('Local code:', RUNTIME_DIR)
'''


def notebooks():
    data = [
        md('''
        # Notebook 39 — L1 Pendulum: Data + Oracle Checks
        **Project:** ENSO–BSISO SSL — toy dynamical systems / L1<br>
        **Author:** Jiayi (jh9141@nyu.edu)<br>
        **Runtime:** Colab CPU. Run all cells, then copy the printed `DATASET_ID` to 40–42.

        L1a = clean weakly damped pendulum; L1b = the **same trajectories/splits** with
        observation noise; L1c = continuous length values across independent synthetic years.
        Every condition is compared with NSV, InfoNCE, BT-temporal and BT-same-time.
        These are data conditions, not successive learning algorithms.

        The default learning input is one frame per sample (updated 2026-11-04).
        The simulator's two-state oracle is unchanged; a position-only snapshot does
        not identify angular velocity. Earlier five-frame results are a separate protocol.

        $\\dot\\theta=v$, $\\dot v=-\\gamma v-(g/L)\\sin\\theta$.
        All L1a–c trajectories have constant L internally. No force noise or continuous
        physical length change in this implementation. Store small state arrays on Drive;
        training notebooks render reusable uint8 image caches in `/content`.

        This notebook checks integration, energy decay, split integrity, and oracle ID.
        ID estimates are scale-dependent diagnostics, not a hard 2/3 acceptance test.
        ''') ,
        md('## Cell 1 — Drive and isolated paths'), cell(BOOTSTRAP),
        md('## Cell 2 — Embedded data implementation (expand to inspect)'), module('l1_data.py'),
        md('## Cell 3 — Simulation configuration'),
        cell('''
        from dataclasses import replace
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from l1_data import (SimulationConfig, CONDITIONS, create_dataset, load_dataset,
                             render, energy, estimate_id, json_write)

        PROFILE = 'pilot'  # 'smoke' creates fewer trajectories; it is NOT a scientific result
        CONFIG = SimulationConfig(
            seed=20261027, n_fixed=100, n_length_groups=60, trajectories_per_length=5,
            periods=8.0, frames_per_period=50, substeps=8,
            g=1.0, reference_length=1.0, damping=0.02,
            length_min=0.8, length_max=1.2, image_size=64)
        if PROFILE == 'smoke':
            CONFIG = replace(CONFIG, n_fixed=10, n_length_groups=10,
                             trajectories_per_length=2, periods=3.0)
        elif PROFILE != 'pilot':
            raise ValueError(PROFILE)
        print('Observation conditions:', CONDITIONS)
        print('Reference period:', CONFIG.period, 'sampling dt:', CONFIG.dt)
        ''') ,
        md('## Cell 4 — Generate immutable data and validate numerical integration'),
        cell('''
        DATASET_ID = create_dataset(L1_ROOT, CONFIG)
        DATA_DIR = L1_ROOT / 'datasets' / DATASET_ID
        fixed, _, manifest = load_dataset(L1_ROOT, DATASET_ID, 'fixed')
        variable, _, _ = load_dataset(L1_ROOT, DATASET_ID, 'variable')
        print('DATASET_ID =', repr(DATASET_ID))
        display(pd.DataFrame(manifest['checks']).T)
        for name, bundle in [('fixed (L1a/b)', fixed), ('variable (L1c)', variable)]:
            print(name, 'trajectory counts train/val/test:', np.bincount(bundle['splits'], minlength=3))
        print('Saved:', DATA_DIR)
        ''') ,
        md('## Cell 5 — Inspect state coverage, energy decay and continuous length groups'),
        cell('''
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for tr in range(min(5, len(fixed['states']))):
            axes[0].plot(*fixed['states'][tr].T, alpha=0.7)
            axes[1].plot(fixed['times']/CONFIG.period, energy(fixed['states'][tr:tr+1],
                         fixed['lengths'][tr:tr+1], CONFIG.g)[0])
        axes[0].set(xlabel='theta', ylabel='angular velocity', title='Weakly damped trajectories')
        axes[1].set(xlabel='time / reference period', ylabel='energy (unit mass)', title='Energy must decay')
        for split, label in enumerate(['train', 'val', 'test']):
            mask = variable['splits'] == split
            axes[2].scatter(variable['groups'][mask], variable['lengths'][mask], s=12, label=label)
        axes[2].set(xlabel='synthetic year / length group', ylabel='L', title='Whole groups stay in one split')
        axes[2].legend()
        fig.tight_layout(); fig.savefig(DATA_DIR / 'oracle_dynamics.png', dpi=140); plt.show()
        ''') ,
        md('## Cell 6 — The actual observation model'),
        cell('''
        times = np.linspace(0, min(60, len(fixed['times'])-1), 5, dtype=int)
        clean = render(fixed['states'][0, times, 0], fixed['lengths'][0])
        rng = np.random.default_rng(7)
        fig, axes = plt.subplots(3, len(times), figsize=(12, 7))
        for row, sigma in enumerate([0.0, 0.02, 0.05]):
            noisy = clean + sigma*rng.standard_normal(clean.shape)
            for col, t in enumerate(times):
                axes[row, col].imshow(noisy[col], cmap='gray', vmin=0, vmax=1)
                axes[row, col].set_title(f'frame={t}, sigma={sigma}')
                axes[row, col].axis('off')
        fig.tight_layout(); fig.savefig(DATA_DIR / 'observation_preview.png', dpi=140); plt.show()
        print('Display colors clip to [0,1]; training arrays are NOT clipped or normalized per frame.')
        ''') ,
        md('## Cell 7 — Train-only oracle dimension reference'),
        cell('''
        oracle = {}
        for name, bundle in [('fixed', fixed), ('variable', variable)]:
            trajectories = np.flatnonzero(bundle['splits'] == 0)
            samples = bundle['states'][trajectories, ::10]
            group = np.repeat(bundle['groups'][trajectories], samples.shape[1])
            state = samples.reshape(-1, 2)
            if name == 'variable':
                state = np.column_stack((state, np.repeat(bundle['lengths'][trajectories], samples.shape[1])))
            oracle[name] = estimate_id(state, group)
            print(name, ': physical reference =', 2 if name == 'fixed' else 3)
            display(pd.DataFrame(oracle[name]['scales']))
        json_write(DATA_DIR / 'oracle_id_train.json', oracle)
        print('Next: notebook 40 (NSV), then 41 (InfoNCE/BT), then 42 (held-out evaluation).')
        print('Use DATASET_ID =', repr(DATASET_ID))
        ''') ,
    ]
    train_intro = md('''
        ## Protocol shared by notebooks 40 and 41

        **climate_bsiso_v1:** replicate each climate method's architecture/training recipe,
        using the BSISO 2-D CNN for single-channel 64x64 pendulum images. Methods share
        the data split, one-frame observation and evaluation rows, but have different
        backbones/heads/optimizers just as the original pipelines do.

        InfoNCE: direct 2-D CNN output, raw single-direction loss, temperature=0.5,
        positive offsets +/-1..3 frames. BT: NSV-warm-started encoder, nb26 projector,
        sum of weighted losses at lags 1..5 (weight 1 -> 0.05). NSV: predict t+10,
        estimate ID, fit the nb20/23 raw-latent SIREN, then the latent dynamics MLP.
        One frame is T0/50 by default, not one calendar day. NSV lag10 is 0.2T0.
        The three climate methods add **no training-view noise**. Optional BT-same is
        a toy-only control with independent std0.01 pixel augmentation.

        Model selection follows the reference: InfoNCE final epoch; BT minimum training
        loss; NSV prediction/refinement minimum validation MSE. Original ID uses train+val
        features; independent test trajectories never enter ID/model selection.
        No simulator state labels enter representation training. Climate preprocessing
        (bandpass/anomalies) and the MJO 1-D spatial encoder are not replicated by images.
        A clean fixed-L snapshot depends on theta, not velocity. Velocity readout and
        future prediction measure information missing from this observation; do not
        require recovery of the full two-state oracle from one position-only frame.
        For L1c, visible-length snapshots depend on (theta,L), hidden-length ones on theta.
        Earlier common-backbone runs remain historical controls; notebook42 defaults
        to this new protocol only. Existing simulator data can be reused; retrain 40/41.
        A completed run is reused by exact config hash. Incomplete runs are not overwritten;
        inspect them or change `run_tag`. Full rerun/resume of partial epochs is not implemented.
        ''')
    train_config = '''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import torch
        from dataclasses import replace
        from l1_data import CONDITIONS, load_dataset, frame_cache, observation
        from l1_climate import PROTOCOL, ClimateConfig, ClimateModel, climate_config, run_training

        DATASET_ID = ''  # paste from notebook 39; blank works only if exactly one dataset exists
        CONDITIONS_TO_RUN = ['L1a']  # later: L1b_s002, L1b_s005, L1c_visible, L1c_hidden
        SEEDS = [42]  # pilot first; then [42, 43, 44] with the same list in notebook 41
        # Each method uses its own reference defaults; do not share one BASE optimizer.
        RUN_TAG = 'climate_replica'
        DATASET_ID = resolve_dataset_id(DATASET_ID)
        if not torch.cuda.is_available():
            raise RuntimeError('Select Colab Runtime > Change runtime type > T4 GPU for training.')
        print('GPU:', torch.cuda.get_device_name(0), '| torch:', torch.__version__)
        print('Dataset:', DATASET_ID, '| conditions:', CONDITIONS_TO_RUN)
        for condition in CONDITIONS_TO_RUN:
            if condition not in CONDITIONS:
                raise ValueError(condition)
    '''
    nsv = [
        md('''
        # Notebook 40 — L1 Pendulum: Neural State Variables
        **Runtime:** Colab T4 GPU. **Prerequisite:** notebook 39.

        Replicate nb18c -> nb19 -> nb20: predict t+10 through the BSISO encoder/decoder,
        compute the original scikit-dimension ID calls, refine through the estimated d,
        and fit a latent dynamics MLP. Also fit d=2 as an explicit ablation.
        No physical labels enter these losses. Do not force an ID estimate to equal
        2 or 3. A single position-only frame does not identify velocity, so physical
        state dimension is not the expected clean snapshot dimension. Compare with
        the project NSV pipeline and with same-dimensional SSL outputs.
        Full autonomous latent rollouts are not implemented here; notebook 42 evaluates
        common direct forecasts and state readouts.
        ''') , train_intro,
        md('## Cell 1 — Drive and isolated paths'), cell(BOOTSTRAP),
        md('## Cell 2 — Embedded implementations and reference ID dependency'),
        cell("%pip install -q scikit-dimension==0.3.4"),
        module('l1_data.py'), module('l1_climate.py'), module('climate_reference_manifest.json'),
        md('## Cell 3 — Condition and training configuration'), cell(train_config),
        md('## Cell 4 — Predictive encoder, train+validation ID, and NSV refinement'),
        cell('''
        RUNS = []
        for condition in CONDITIONS_TO_RUN:
            for seed in SEEDS:
                config = climate_config('nsv', seed=seed, run_tag=RUN_TAG)
                # Defaults: 100 prediction epochs, 500 refinement, 300 dynamics.
                # cfg.id_backend='skdim' reproduces the pinned climate estimator calls.
                path = run_training(L1_ROOT, DATASET_ID, condition, CACHE_DIR, config)
                RUNS.append(path)
                print('Completed:', path)
        ''') ,
        md('## Cell 5 — Validation curves and ID scale sensitivity'),
        cell('''
        for path in RUNS:
            history = pd.read_json(path / 'training_history.json')
            history.plot(x='epoch', y=['train', 'val'], title=path.name)
            plt.ylabel('Future-observation MSE'); plt.show()
            diagnostic = json.loads((path / 'id_train.json').read_text())
            display(pd.DataFrame(diagnostic['scales']))
            rows = []
            print('Climate ID:', json.loads((path / 'intrinsic_dim.json').read_text()))
            for p in sorted(path.glob('refine_d*_history.json')):
                history = pd.read_json(p)
                rows.append({'dimension': int(p.stem.split('_d')[-1].split('_')[0]), 'best_val_mse': history.val.min()})
            display(pd.DataFrame(rows).sort_values('dimension'))
            if (path / 'dynamics_scores.json').exists():
                print('Latent dynamics:', json.loads((path / 'dynamics_scores.json').read_text()))
        ''') ,
        md('## Cell 6 — Inspect future-frame prediction on validation only'),
        cell('''
        path = RUNS[0]
        meta = json.loads((path / 'run.json').read_text())
        config = ClimateConfig(**meta['training'])
        condition = meta['condition']
        bundle, sim, _ = load_dataset(L1_ROOT, DATASET_ID, CONDITIONS[condition]['kind'])
        frames = frame_cache(bundle, sim, condition, CACHE_DIR, DATASET_ID)
        trajectory = int(np.flatnonzero(bundle['splits'] == 1)[0])
        end = config.window + 10
        x = observation(frames, trajectory, end, config.window, CONDITIONS[condition]['noise_std'], sim.seed)
        y = observation(frames, trajectory, end+config.lag, config.window,
                        CONDITIONS[condition]['noise_std'], sim.seed, draw=0)
        model = ClimateModel(config).eval()
        model.load_state_dict(torch.load(path / 'model_selected.pt', map_location='cpu', weights_only=True))
        with torch.no_grad():
            predicted = model(torch.from_numpy(x[None]))[1][0].numpy()
        fig, axes = plt.subplots(3, config.window, figsize=(3*config.window, 8), squeeze=False)
        for row, (clip, label) in enumerate([(x, 'input'), (y, 'future observation'), (predicted, 'prediction')]):
            for col in range(config.window):
                axes[row, col].imshow(clip[col], cmap='gray', vmin=0, vmax=1)
                axes[row, col].set_title(label); axes[row, col].axis('off')
        fig.tight_layout(); fig.savefig(path / 'validation_prediction.png', dpi=130); plt.show()
        print('Next: 41 on the same dataset/conditions/history, then 42 for held-out scores.')
        ''') ,
    ]
    ssl = [
        md('''
        # Notebook 41 — L1 Pendulum: InfoNCE + Barlow Twins
        **Runtime:** Colab T4 GPU. **Prerequisites:** notebooks 39 and 40 (BT uses the NSV encoder).

        `infonce`: nb08/15 raw single-direction contrastive loss, direct encoder z2.
        No separate projector, L2 normalization, or nearby-negative masking.<br>
        `bt_temporal`: nb26 weighted 1..5 lag losses, BN/ReLU projector, D=3 and D=7.
        Both encoder and projector train; encoder starts from the matching notebook40 run.<br>
        `bt_same`: optional same-time augmented control; not an old climate experiment.

        InfoNCE saves `pre_fc` (32-D pooled features) and direct encoder `z` (2-D).
        Barlow saves encoder `h` (64-D) and projector `z` (D-D). Check feature names,
        not just dimensions, when interpreting readouts. Runs record source-cell hashes.
        ''') , train_intro,
        md('## Cell 1 — Drive and isolated paths'), cell(BOOTSTRAP),
        md('## Cell 2 — Embedded implementations'), module('l1_data.py'), module('l1_climate.py'), module('climate_reference_manifest.json'),
        md('## Cell 3 — Method-specific training configuration'), cell(train_config),
        cell('''
        METHODS = ['infonce', 'bt_temporal']
        BT_PROJECTION_DIMS = [3, 7]  # original MJO Barlow settings
        RUN_SAME_TIME_CONTROL = False
        # Only needed if several compatible NSV runs exist for a condition/seed.
        NSV_RUN_IDS = {}  # e.g. {('L1a', 42): 'nsv_s42_<hash>'}
        ''') ,
        md('## Cell 4 — Reproduce each method recipe'),
        cell('''
        RUNS = []
        for condition in CONDITIONS_TO_RUN:
            for seed in SEEDS:
                methods = METHODS + (['bt_same'] if RUN_SAME_TIME_CONTROL else [])
                for method in methods:
                    dims = [2] if method == 'infonce' else BT_PROJECTION_DIMS
                    for dim in dims:
                        config = climate_config(method, seed=seed, projection_dim=dim, run_tag=RUN_TAG)
                        path = run_training(L1_ROOT, DATASET_ID, condition, CACHE_DIR, config,
                                            warm_start_run_id=NSV_RUN_IDS.get((condition, seed)))
                        RUNS.append(path)
                        print('Completed:', path)
        ''') ,
        md('## Cell 5 — Objective curves and train-only collapse diagnostics'),
        cell('''
        for path in RUNS:
            meta = json.loads((path / 'run.json').read_text())
            history = pd.read_json(path / 'training_history.json')
            history.plot(x='epoch', y=['train', 'val'], title=path.name)
            plt.ylabel('Objective value (not comparable across methods)'); plt.show()
            bundle, _, _ = load_dataset(L1_ROOT, DATASET_ID, CONDITIONS[meta['condition']]['kind'])
            with np.load(path / 'features.npz') as f:
                train = bundle['splits'][f['index'][:, 0]] == 0
                for name in [key for key in ['pre_fc', 'h', 'z'] if key in f]:
                    z = f[name][train]
                    print(name, 'std min/median/max:', np.min(z.std(0)), np.median(z.std(0)), np.max(z.std(0)))
            display(pd.DataFrame(json.loads((path / 'id_train.json').read_text())['scales']))
        print('Next: notebook 42. Training loss alone does not establish physical state recovery.')
        ''') ,
    ]
    evaluation = [
        md('''
        # Notebook 42 — L1 Pendulum: Unified Held-out Evaluation
        **Runtime:** Colab CPU (no torch needed). **Prerequisites:** 39, then completed 40/41 runs.

        Each L1 condition compares all completed compatible runs. Missing methods are reported;
        they are never silently labelled as evaluated. PCA, scalers and probes fit **train** only;
        ridge hyperparameters use **validation** only. Test groups supply final scores and
        grouped bootstrap intervals. Intervals condition on each trained model; seeds stay
        separate. Avoid repeatedly selecting methods/hyperparameters on these test scores.

        Outputs: theta/velocity/(L) readouts, optional fixed-capacity nonlinear readouts,
        direct forecasts at ~T0/4, T0, 2T0, train-only ID diagnostics, latent plots and CSVs.
        Direct ridge forecasts do not constitute autonomous rollouts or prove a nonlinear
        latent is dynamically insufficient. The oracle-state persistence baseline is labelled
        because it receives the true current theta/velocity. Delay-PCA receives observed windows.
        ''') ,
        md('## Cell 1 — Drive and isolated paths'), cell(BOOTSTRAP),
        md('## Cell 2 — Embedded CPU implementations'), module('l1_data.py'), module('l1_evaluation.py'),
        md('## Cell 3 — Select dataset and compatible runs'),
        cell('''
        import pandas as pd
        from IPython.display import display, Image
        from l1_evaluation import EvalConfig, completed_runs, compare

        DATASET_ID = ''
        CONDITIONS_TO_EVALUATE = ['L1a']
        EVALUATION = EvalConfig(window=1, stride=4, bootstrap_repeats=300, nonlinear_probe=True,
                                protocol='climate_bsiso_v1', tsne=True)
        # Empty mapping uses all completed runs matching history/stride; never chooses "best test".
        # To compare a prespecified subset, map condition -> list of exact run directory names.
        SELECTED_RUN_IDS = {}
        DATASET_ID = resolve_dataset_id(DATASET_ID)
        ''') ,
        md('## Cell 4 — Fit probes/baselines and evaluate once on held-out groups'),
        cell('''
        RESULTS = {}
        for condition in CONDITIONS_TO_EVALUATE:
            paths = completed_runs(L1_ROOT, DATASET_ID, condition)
            if condition in SELECTED_RUN_IDS:
                requested = set(SELECTED_RUN_IDS[condition])
                found = {p.name for p in paths}
                if not requested <= found:
                    raise FileNotFoundError(f'Missing requested runs: {requested-found}')
                paths = [p for p in paths if p.name in requested]
            compatible, methods = [], set()
            for path in paths:
                meta = json.loads((path / 'run.json').read_text())
                training = meta['training']
                if (training['window'] == EVALUATION.window and training['eval_stride'] == EVALUATION.stride
                        and training.get('protocol', 'legacy') == EVALUATION.protocol):
                    compatible.append(path); methods.add(meta['method'])
                else:
                    print('Skipping incompatible history/stride/protocol:', path.name)
            missing = {'nsv', 'infonce', 'bt_temporal'} - methods
            if missing:
                print('Methods not yet available:', sorted(missing))
            output, scores = compare(L1_ROOT, DATASET_ID, condition, CACHE_DIR, compatible, EVALUATION)
            RESULTS[condition] = (output, scores)
            print(condition, 'outputs:', output)
        ''') ,
        md('## Cell 5 — Readouts, prediction curves and held-out trajectories'),
        cell('''
        for condition, (output, scores) in RESULTS.items():
            print(condition, 'CURRENT STATE READOUTS')
            columns = ['method', 'feature', 'dimension', 'training_seed', 'probe', 'variable',
                       'r2', 'rmse', 'rmse_ci_low', 'rmse_ci_high', 'test_groups']
            display(scores.loc[scores.horizon_frames == 0, columns].reset_index(drop=True))
            for name in ['latent_native_test.png', 'latent_test.png', 'latent_tsne_test.png',
                         'state_readout_test.png', 'forecast_test.png']:
                if (output / name).exists():
                    display(Image(filename=str(output / name)))
            print('All configurations and seeds remain separate in:', output / 'scorecard.csv')
        ''') ,
        md('## Cell 6 — Download reports for analysis'),
        cell('''
        from pathlib import Path
        from zipfile import ZipFile, ZIP_DEFLATED
        import json
        from google.colab import files

        assert RESULTS, "请先运行 notebook 42 的比较单元。"
        root = Path(L1_ROOT)
        folders = [root / "datasets" / DATASET_ID]

        # 收集本次评估，以及它实际使用的训练 runs
        for condition, (output, _) in RESULTS.items():
            output = Path(output)
            folders.append(output)
            evaluation = json.loads((output / "evaluation.json").read_text())
            for run in evaluation["runs"]:
                folders.append(
                    root / "runs" / DATASET_ID / condition / run["run_id"]
                )

        # 首轮分析只打包报告、配置、训练曲线数据和图片
        paths = sorted({
            p
            for folder in folders
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".json", ".csv", ".png"}
        })

        zip_path = Path("/content") / f"pendulum_results_{DATASET_ID}.zip"
        with ZipFile(zip_path, "w", ZIP_DEFLATED) as archive:
            for p in paths:
                archive.write(p, arcname=str(p.relative_to(root)))

        print("包含条件：", list(RESULTS))
        print(f"已打包 {len(paths)} 个文件：{zip_path}")
        files.download(str(zip_path))
        ''') ,
        md('''
        ## Interpretation checklist

        - L1a: assess angle recovery; velocity/future readouts diagnose information missing
          from a single position-only frame. A closed ring or ID=2 is not required.
        - L1b: compare the same initial conditions and split with L1a, at matched method settings.
        - L1c: length is continuous, constant per trajectory; compare visible and hidden rendering.
        - InfoNCE z is the direct encoder output, pre_fc is the pooled32 feature; BT h/z
          are encoder/projector; NSV z is the ID-selected refiner, z_d* are extra ablations.
        - Native coordinates, train-fitted PCA and descriptive t-SNE are distinct views.
          t-SNE is fitted to the displayed test subset only, never used by numerical probes.
        - Physical state ID ~2/3 is distinct from snapshot geometry: clean fixed-L/hidden-L
          images depend on theta, visible-L images on (theta,L). Noise/finite sampling
          can change estimated ID. No velocity channel is added to the single frame.
        - Figures display each completed run (including both BT dimensions), identified
          by dimension and run suffix. The CSV retains every representation and probe.
        - Initial pilot settings are not evidence that any method works. Report actual held-out
          results and seed variation before making scientific claims.
        ''') ,
    ]
    return {
        '39_pendulum_data.ipynb': (data, False),
        '40_pendulum_nsv.ipynb': (nsv, True),
        '41_pendulum_ssl_barlow.ipynb': (ssl, True),
        '42_pendulum_comparison.ipynb': (evaluation, False),
    }


def build(check=False):
    manifest_path = HERE/'climate_reference_manifest.json'
    manifest_text = json.dumps(reference_manifest(), indent=2, ensure_ascii=False)+'\n'
    if check:
        if not manifest_path.exists() or manifest_path.read_text() != manifest_text:
            raise SystemExit('Climate reference manifest out of sync; rebuild notebooks.')
    else:
        manifest_path.write_text(manifest_text)
    for name, (cells, gpu) in notebooks().items():
        metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                    'language_info': {'name': 'python', 'version': '3.11'},
                    'colab': {'name': name, 'provenance': []}}
        if gpu:
            metadata.update(accelerator='GPU')
            metadata['colab']['gpuType'] = 'T4'
        notebook = {'cells': cells, 'metadata': metadata, 'nbformat': 4, 'nbformat_minor': 5}
        serialized = json.dumps(notebook, indent=1, ensure_ascii=False)+'\n'
        target = HERE/name
        if check:
            if not target.exists() or target.read_text() != serialized:
                raise SystemExit(f'Notebook out of sync: {target}')
        else:
            target.write_text(serialized)
        print(('Checked' if check else 'Built'), name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    build(parser.parse_args().check)
