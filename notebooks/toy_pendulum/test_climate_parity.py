"""Independent numerical reference tests against the ORIGINAL climate notebook cells."""
import ast
import copy
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
HAS_TORCH = importlib.util.find_spec('torch') is not None


def reference_definitions(path, cell_index, names, adapt_decoder=False):
    import torch
    from torch import nn
    from torch.nn import functional as F
    nb = json.loads((HERE.parent/path).read_text())
    tree = ast.parse(''.join(nb['cells'][cell_index]['source']))
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError('Reference definition missing from notebook')
    tree = ast.Module(body=selected, type_ignores=[])
    if adapt_decoder:
        class ImageSizes(ast.NodeTransformer):
            def visit_Tuple(self, node):
                self.generic_visit(node)
                if all(isinstance(e, ast.Constant) for e in node.elts):
                    size = tuple(e.value for e in node.elts)
                    mapping = {(3, 6): (8, 8), (7, 12): (16, 16), (15, 25): (32, 32), (31, 51): (64, 64)}
                    if size in mapping:
                        return ast.copy_location(ast.Tuple(elts=[ast.Constant(x) for x in mapping[size]], ctx=ast.Load()), node)
                return node
        tree = ImageSizes().visit(tree)
    scope = dict(torch=torch, nn=nn, F=F, math=math, np=np)
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), scope)
    return scope


@unittest.skipUnless(HAS_TORCH, 'torch unavailable')
class ClimateParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def assert_forward_gradient_equal(self, actual, reference, x):
        import torch
        reference.load_state_dict(actual.state_dict())
        actual.eval(); reference.eval()
        a, b = x.detach().clone().requires_grad_(), x.detach().clone().requires_grad_()
        ya, yb = actual(a), reference(b)
        torch.testing.assert_close(ya, yb, atol=1e-7, rtol=1e-6)
        ya.square().mean().backward(); yb.square().mean().backward()
        torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=1e-6)
        for pa, pb in zip(actual.parameters(), reference.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad, atol=1e-7, rtol=1e-6)

    def test_encoders_decoder_match_notebook_after_documented_image_adapter(self):
        import torch
        from l1_climate import SSL2DEncoder, PredictiveEncoder, PredictiveDecoder
        ref = reference_definitions('extension_2d/08_ssl_temporal_2d.ipynb', 12, ['CNNEncoderNoL2'])
        a, b = SSL2DEncoder(2), ref['CNNEncoderNoL2'](2)
        b.conv1 = copy.deepcopy(a.conv1)  # ONLY channel count differs.
        self.assert_forward_gradient_equal(a, b, torch.randn(3, 1, 64, 64))
        ref = reference_definitions('nsv/18c_nsv_bsiso_stage1_lag10.ipynb', 6,
                                    ['EncoderBSISO', 'DecoderBSISO'], adapt_decoder=True)
        a, b = PredictiveEncoder(), ref['EncoderBSISO']()
        b.conv1 = copy.deepcopy(a.conv1)
        self.assert_forward_gradient_equal(a, b, torch.randn(3, 1, 64, 64))
        a, b = PredictiveDecoder(), ref['DecoderBSISO']()
        b.conv4 = copy.deepcopy(a.conv4)
        self.assert_forward_gradient_equal(a, b, torch.randn(3, 64))

    def test_projector_siren_and_dynamics_match_original(self):
        import torch
        from l1_climate import Projector, SirenAutoencoder, DynamicsMLP
        ref = reference_definitions('mjo/26_mjo_barlow_twins.ipynb', 8, ['Projector'])
        for dim in [3, 7]:
            torch.manual_seed(11); actual = Projector(64, dim)
            torch.manual_seed(11); original = ref['Projector'](64, dim)
            for key, tensor in actual.state_dict().items():
                torch.testing.assert_close(tensor, original.state_dict()[key], rtol=0, atol=0)
            self.assert_forward_gradient_equal(actual, original, torch.randn(8, 64))
        ref = reference_definitions('nsv/20_nsv_bsiso_refine_analysis.ipynb', 4,
                                    ['SirenLayer', 'SirenAutoencoder'])
        torch.manual_seed(11); actual = SirenAutoencoder(d_hat=2)
        torch.manual_seed(11); original = ref['SirenAutoencoder'](d_hat=2)
        for key, tensor in actual.state_dict().items():
            torch.testing.assert_close(tensor, original.state_dict()[key], rtol=0, atol=0)
        self.assert_forward_gradient_equal(actual, original, torch.randn(8, 64))
        ref = reference_definitions('nsv/20_nsv_bsiso_refine_analysis.ipynb', 16, ['DynamicsMLP'])
        self.assert_forward_gradient_equal(DynamicsMLP(2), ref['DynamicsMLP'](2), torch.randn(8, 2))

    def test_loss_and_gradient_parity_including_raw_norm_and_barlow_sample_std(self):
        import torch
        from l1_climate import infonce_loss_raw, barlow_cross_corr, barlow_loss, lag_weight, climate_config
        ref = reference_definitions('extension_2d/08_ssl_temporal_2d.ipynb', 12, ['InfoNCE_loss_raw'])
        a, b = torch.randn(9, 2), torch.randn(9, 2)
        x, y = a.clone().requires_grad_(), b.clone().requires_grad_()
        p, q = a.clone().requires_grad_(), b.clone().requires_grad_()
        actual, original = infonce_loss_raw(x, y), ref['InfoNCE_loss_raw'](p, q, 0.5)
        actual.backward(); original.backward()
        torch.testing.assert_close(actual, original)
        torch.testing.assert_close(x.grad, p.grad); torch.testing.assert_close(y.grad, q.grad)
        self.assertGreater(abs(float(infonce_loss_raw(3*a, 3*b)-infonce_loss_raw(a, b))), .01)
        ref = reference_definitions('mjo/26_mjo_barlow_twins.ipynb', 8, ['barlow_cross_corr', 'barlow_loss'])
        cfg = climate_config('bt_temporal')
        self.assertEqual(lag_weight(1, cfg), 1.0); self.assertEqual(lag_weight(5, cfg), .05)
        x, y = a.clone().requires_grad_(), b.clone().requires_grad_()
        p, q = a.clone().requires_grad_(), b.clone().requires_grad_()
        c = barlow_cross_corr(x, y); cr = ref['barlow_cross_corr'](p, q)
        actual = sum(barlow_loss(c, lag_weight(t, cfg), cfg.barlow_offdiag) for t in cfg.barlow_lags)
        original = sum(ref['barlow_loss'](cr, lag_weight(t, cfg), .005*lag_weight(t, cfg))[0] for t in cfg.barlow_lags)
        actual.backward(); original.backward()
        torch.testing.assert_close(c, cr)
        torch.testing.assert_close(actual, original)
        torch.testing.assert_close(x.grad, p.grad); torch.testing.assert_close(y.grad, q.grad)
        # nb26's sample std makes diagonal ~ (B-1)/B, NOT exactly one.
        aligned = barlow_cross_corr(a, a)
        torch.testing.assert_close(aligned.diag(), torch.full((2,), 8/9), atol=1e-4, rtol=1e-4)

    def test_sampling_offsets_boundaries_fixed_validation_and_method_roles(self):
        from l1_climate import climate_config, ClimateModel, PairDataset
        frames = np.zeros((3, 12, 64, 64), dtype='uint8')
        index = np.array([(tr, t) for tr in [0, 2] for t in range(12)])
        cfg = climate_config('infonce')
        ds = PairDataset(frames, index, 'L1a', 42, cfg, True)
        observed = set()
        for epoch in range(5):
            ds.epoch = epoch
            for i in range(len(ds)):
                tr, t, target = ds.pair_at(i)
                self.assertIn((tr, target), ds.allowed)
                self.assertTrue(0 < abs(target-t) <= 3)
                observed.add(target-t)
        self.assertEqual(observed, {-3, -2, -1, 1, 2, 3})
        val = PairDataset(frames, index, 'L1a', 42, cfg, False)
        pairs = [val.pair_at(i) for i in range(len(val))]
        val.epoch = 99
        self.assertEqual(pairs, [val.pair_at(i) for i in range(len(val))])
        with self.assertRaisesRegex(ValueError, 'neighbor'):
            PairDataset(frames, np.array([[0, 0], [2, 11]]), 'L1a', 42, cfg, True)
        self.assertFalse(hasattr(ClimateModel(cfg), 'projector'))
        self.assertEqual(cfg.checkpoint_selection, 'final_epoch')
        self.assertEqual(climate_config('bt_temporal').checkpoint_selection, 'training_objective')
        self.assertEqual(climate_config('nsv').checkpoint_selection, 'validation_objective')

    def test_tiny_replica_pipeline_warm_start_refinement_dynamics_and_protocol_isolation(self):
        import torch
        from l1_data import SimulationConfig, create_dataset
        from l1_climate import PROTOCOL, climate_config, run_training, resolve_warm_start
        from l1_evaluation import EvalConfig, compare, display_representations
        sim = SimulationConfig(n_fixed=10, n_length_groups=5, trajectories_per_length=2,
                               periods=3, frames_per_period=12)
        with tempfile.TemporaryDirectory(prefix='ddcs_climate_parity_') as tmp:
            root = Path(tmp); cache = root/'cache'; dataset_id = create_dataset(root, sim)
            shared = dict(epochs=1, batch_size=32, refine_epochs=1, dynamics_epochs=1,
                          refine_dims=(2,), id_backend='group_mle', id_scope='train',
                          lag=3, run_tag='unit-test-only')
            nsv = run_training(root, dataset_id, 'L1a', cache, climate_config('nsv', **shared), device='cpu')
            self.assertTrue((nsv/'dynamics_scores.json').exists())
            paths = [nsv]
            for method, dim in [('infonce', 2), ('bt_temporal', 3), ('bt_temporal', 7), ('bt_same', 3)]:
                cfg = climate_config(method, projection_dim=dim, barlow_lags=(1, 2), **shared)
                path = run_training(root, dataset_id, 'L1a', cache, cfg, device='cpu')
                paths.append(path)
                with np.load(path/'features.npz') as f:
                    self.assertEqual(f['z'].shape[1], dim)
                    if method == 'infonce':
                        self.assertIn('pre_fc', f.files); self.assertNotIn('h', f.files)
                        self.assertNotIn('z_unit', f.files)
                    else:
                        original = torch.load(nsv/'encoder_selected.pt', map_location='cpu')
                        initial = torch.load(path/'encoder_initial.pt', map_location='cpu')
                        for key in original:
                            torch.testing.assert_close(original[key], initial[key], rtol=0, atol=0)
                self.assertEqual(path, run_training(root, dataset_id, 'L1a', cache, cfg, device='cpu'))
            evaluation = EvalConfig(protocol=PROTOCOL, bootstrap_repeats=10, pca_components=4,
                                    pca_fit_samples=40, nonlinear_probe=False)
            out, scores = compare(root, dataset_id, 'L1a', cache, paths, evaluation)
            self.assertTrue(np.isfinite(scores.rmse).all())
            self.assertTrue((out/'latent_native_test.png').exists())
            self.assertEqual(set(scores.loc[(scores.method == 'bt_temporal') & (scores.feature == 'z'), 'dimension']), {3, 7})
            with self.assertRaisesRegex(ValueError, 'protocol'):
                compare(root, dataset_id, 'L1a', cache, paths, replace(evaluation, protocol='legacy'))
            # No implicit latest/best choice when warm-start lineage becomes ambiguous.
            shutil.copytree(nsv, nsv.with_name(nsv.name+'_duplicate'))
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                resolve_warm_start(root, dataset_id, 'L1a', climate_config('bt_temporal'))
            qa = Path(tempfile.gettempdir())/'ddcs_climate_replica_qa'
            qa.mkdir(exist_ok=True)
            for name in ['latent_test.png', 'latent_native_test.png', 'state_readout_test.png', 'forecast_test.png']:
                shutil.copy2(out/name, qa/name)
            print('Replica fixture plots:', qa)

    def test_native_and_tsne_views_preserve_separate_runs_and_original_features(self):
        from l1_evaluation import EvalConfig, plot_native_and_tsne
        rng = np.random.default_rng(123)
        representations = {
            'bt_run_a:z': (rng.normal(size=(180, 3)), 'bt_temporal', 'z', 42),
            'bt_run_b:z': (rng.normal(size=(180, 7)), 'bt_temporal', 'z', 42),
            'nsv_run:z': (rng.normal(size=(180, 1)), 'nsv', 'z', 42),
        }
        before = {k: v[0].copy() for k, v in representations.items()}
        truth = rng.normal(size=(180, 3)); splits = np.repeat([0, 1, 2], 60)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            plot_native_and_tsne(out, representations, truth, splits, EvalConfig(tsne=True))
            with np.load(out/'latent_native_coordinates.npz') as native, np.load(out/'latent_tsne_coordinates.npz') as tsne:
                idx = native['sample_indices']
                for key, value in representations.items():
                    np.testing.assert_array_equal(value[0], before[key])
                    np.testing.assert_array_equal(native[key][:, 0], before[key][idx, 0])
                    if value[0].shape[1] == 1:
                        np.testing.assert_array_equal(native[key][:, 1], 0)
                    else:
                        np.testing.assert_array_equal(native[key], before[key][idx, :2])
                    self.assertEqual(tsne[key].shape, (len(idx), 2))
                    self.assertTrue(np.isfinite(tsne[key]).all())
            qa = Path(tempfile.gettempdir())/'ddcs_climate_replica_qa'
            qa.mkdir(exist_ok=True)
            shutil.copy2(out/'latent_tsne_test.png', qa/'latent_tsne_test.png')


if __name__ == '__main__':
    unittest.main()
