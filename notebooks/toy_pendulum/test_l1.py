"""Small correctness tests, not scientific training runs or benchmark results.

Run in an existing scientific Python environment:
    python -m unittest discover -s notebooks/toy_pendulum -p 'test_l1.py' -v
Torch tests skip when torch is absent. Never install torch into the project environment.
"""
import ast
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"ddcs_l1_mpl"))
import numpy as np

from l1_data import (SimulationConfig, CONDITIONS, create_dataset, load_dataset, make_bundle,
                     simulate, energy, numerical_checks, render, frame_cache, observation,
                     index_windows, sample_truth, group_split, file_sha)
from l1_evaluation import EvalConfig, ridge_probe, nonlinear_probe, metric_rows, compare

HAS_TORCH = importlib.util.find_spec("torch") is not None
HERE = Path(__file__).resolve().parent


class DataTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SimulationConfig(n_fixed=10, n_length_groups=10, trajectories_per_length=2,
                                    periods=3, frames_per_period=12)

    def test_small_angle_solution_and_energy(self):
        t = np.linspace(0, 4*np.pi, 101)
        gamma, amplitude = 0.02, 1e-3
        result = simulate([[amplitude, 0]], [1], t, damping=gamma)
        omega = np.sqrt(1-gamma**2/4)
        analytical = amplitude*np.exp(-gamma*t/2)*(np.cos(omega*t)+gamma/(2*omega)*np.sin(omega*t))
        self.assertLess(np.max(np.abs(result[0, :, 0]-analytical)), 2e-9)
        self.assertLess(np.diff(energy(result, [1])).max(), 1e-12)

    def test_group_splits_and_numerics(self):
        b = make_bundle(self.cfg, True)
        numerical_checks(b, self.cfg)
        sets = [set(b['groups'][b['splits'] == i]) for i in range(3)]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(len(np.unique(b['lengths'])), self.cfg.n_length_groups)
        np.testing.assert_array_equal(group_split(b['groups'], 123), group_split(b['groups'], 123))

    def test_length_visibility_and_noise_consistency(self):
        angle = np.array([0.3, 0.3])
        hidden = render(angle, [0.8, 1.2], visible_length=False)
        visible = render(angle, [0.8, 1.2], visible_length=True)
        np.testing.assert_array_equal(hidden[0], hidden[1])
        self.assertGreater(np.max(np.abs(visible[0]-visible[1])), 0.5)
        frames = np.tile(np.rint(255*hidden[:1]).astype('uint8')[None], (1, 10, 1, 1))
        first = observation(frames, 0, 4, 3, 0.05, 42)
        second = observation(frames, 0, 5, 3, 0.05, 42)
        np.testing.assert_array_equal(first[1:], second[:2])
        self.assertFalse(np.array_equal(first, observation(frames, 0, 4, 3, 0.05, 42, draw=1)))

    def test_dataset_reuse_checksum_and_shared_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_id = create_dataset(tmp, self.cfg)
            self.assertEqual(dataset_id, create_dataset(tmp, self.cfg))
            b, cfg, manifest = load_dataset(tmp, dataset_id, 'fixed')
            a = frame_cache(b, cfg, 'L1a', Path(tmp)/'cache', dataset_id)
            noisy = frame_cache(b, cfg, 'L1b_s005', Path(tmp)/'cache', dataset_id)
            self.assertEqual(a.filename, noisy.filename)
            ix = index_windows(b, window=3, future=24, stride=4)
            y = sample_truth(b, ix, 24)
            np.testing.assert_allclose(y[:, :2], b['states'][ix[:, 0], ix[:, 1]+24])
            self.assertLess(int(ix[:, 1].max()+24), b['states'].shape[1])
            p = Path(tmp)/'datasets'/dataset_id/'fixed.npz'
            with p.open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_dataset(tmp, dataset_id, 'fixed')

    def test_probe_does_not_fit_test_labels(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(60, 4))
        y = x[:, :2]*3 + 0.02*rng.normal(size=(60, 2))
        split = np.repeat([0, 1, 2], [36, 12, 12])
        p1, selected1 = ridge_probe(x, y, split)
        changed = y.copy(); changed[split == 2] = 1e6
        p2, selected2 = ridge_probe(x, changed, split)
        np.testing.assert_array_equal(p1, p2)
        self.assertEqual(selected1, selected2)
        small = EvalConfig(nonlinear_max_iter=5)
        nonlinear1, _ = nonlinear_probe(x, y, split, small)
        nonlinear2, _ = nonlinear_probe(x, changed, split, small)
        np.testing.assert_array_equal(nonlinear1, nonlinear2)
        metrics = metric_rows(y[split == 2], p1[split == 2], ['a', 'b'], np.repeat([1, 2, 3], 4), 50)
        self.assertEqual(metrics[0]['test_groups'], 3)

    def test_notebooks_valid_and_embedded_code_matches(self):
        import nbformat
        from build_notebooks import build
        build(check=True)
        for path in HERE.glob('*.ipynb'):
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
            for i, c in enumerate(nb.cells):
                if c.cell_type != 'code':
                    continue
                source = c.source
                if source.startswith('%%writefile '):
                    first, source = source.split('\n', 1)
                    self.assertEqual(source, (HERE/first.split()[-1]).read_text())
                ast.parse(source, filename=f'{path.name}:cell{i}')
                self.assertIsNone(c.execution_count)
                self.assertEqual(c.outputs, [])


@unittest.skipUnless(HAS_TORCH, 'torch unavailable; run torch checks in Colab')
class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except (ImportError, OSError) as exc:
            raise unittest.SkipTest(f'Existing torch installation is not usable: {exc}')
        torch.set_num_threads(2)

    def test_loss_correspondence_and_gradients(self):
        import torch
        from l1_learning import barlow_loss, infonce_loss, Representation, TrainConfig
        torch.manual_seed(42)
        a = torch.randn(12, 6, requires_grad=True)
        tr, t = torch.arange(12), torch.arange(12)
        aligned = infonce_loss(a, a, tr, t, 5)
        shuffled = infonce_loss(a, a.roll(1, 0), tr, t, 5)
        self.assertLess(aligned.item(), shuffled.item())
        # For nondegenerate identical views, zero off-diagonal penalty leaves ~zero loss.
        self.assertLess(barlow_loss(a, a, 0).item(), 1e-6)
        (aligned+barlow_loss(a, a.roll(1, 0))).backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        cfg = TrainConfig(window=3, hidden_dim=8, projection_dim=2)
        for predictive in [False, True]:
            model = Representation(cfg, predictive)
            h, y = model(torch.randn(4, 3, 64, 64))
            self.assertEqual(tuple(h.shape), (4, 8))
            self.assertEqual(tuple(y.shape), (4, 3, 64, 64) if predictive else (4, 2))
            y.square().mean().backward()

    def test_tiny_pipeline_all_objectives_and_evaluation(self):
        from l1_learning import TrainConfig, run_training
        cfg = SimulationConfig(n_fixed=10, n_length_groups=5, trajectories_per_length=2,
                               periods=3, frames_per_period=12)
        train = TrainConfig(window=3, lag=3, train_stride=8, eval_stride=4,
                            hidden_dim=8, batch_size=8, epochs=1, refine_epochs=1,
                            refine_dims=(2, 3), run_tag='unit-test-only')
        with tempfile.TemporaryDirectory(prefix='ddcs_l1_test_') as tmp:
            cache = Path(tmp)/'cache'
            dataset_id = create_dataset(tmp, cfg)
            paths = []
            for method in ['nsv', 'infonce', 'bt_temporal', 'bt_same']:
                path = run_training(tmp, dataset_id, 'L1a', cache, train, method, device='cpu')
                paths.append(path)
                self.assertTrue((path/'complete.json').exists())
                with np.load(path/'features.npz') as f:
                    self.assertTrue(np.isfinite(f['h']).all())
                    if method == 'nsv':
                        self.assertEqual(f['z_d3'].shape[1], 3)
                self.assertEqual(path, run_training(tmp, dataset_id, 'L1a', cache, train, method, device='cpu'))
            evaluation = EvalConfig(window=3, stride=4, pca_components=4, pca_fit_samples=40,
                                    bootstrap_repeats=20, nonlinear_probe=True, nonlinear_max_iter=5)
            out, score = compare(tmp, dataset_id, 'L1a', cache, paths, evaluation)
            self.assertEqual(set(score.method), {'nsv', 'infonce', 'bt_temporal', 'bt_same',
                                                'delay_pca', 'oracle_state_persistence'})
            self.assertTrue(np.isfinite(score.rmse).all())
            self.assertEqual(set(score.variable), {'theta', 'velocity'})
            self.assertTrue((out/'state_readout_test.png').exists())
            self.assertIn('mlp64', set(score.probe))
            cached_out, cached_score = compare(tmp, dataset_id, 'L1a', cache, paths, evaluation)
            self.assertEqual(out, cached_out)
            self.assertEqual(len(score), len(cached_score))
            # Keep only labelled test-fixture images for visual layout QA.
            qa = Path(tempfile.gettempdir())/'ddcs_l1_layout_qa'
            qa.mkdir(exist_ok=True)
            for name in ['latent_test.png', 'forecast_test.png', 'state_readout_test.png']:
                shutil.copy2(out/name, qa/name)
            print('Test-fixture plot QA:', qa)
            with self.assertRaisesRegex(ValueError, 'History'):
                compare(tmp, dataset_id, 'L1a', cache, paths, replace(evaluation, window=2))
            # Hidden-length + noisy observations exercise the second data branch and L probe.
            c_path = run_training(tmp, dataset_id, 'L1c_hidden', cache, train, 'bt_same', device='cpu')
            _, c_score = compare(tmp, dataset_id, 'L1c_hidden', cache, [c_path], evaluation)
            self.assertIn('length', set(c_score.variable))


if __name__ == '__main__':
    unittest.main()
