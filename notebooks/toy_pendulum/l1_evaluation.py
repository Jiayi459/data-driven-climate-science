"""CPU evaluation: train-only probes/PCA, common direct forecasts, grouped uncertainty."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from l1_data import (CONDITIONS, digest, estimate_id, file_sha, frame_cache, index_windows,
                     json_write, load_dataset, observation, sample_truth)


@dataclass(frozen=True)
class EvalConfig:
    seed: int = 123
    window: int = 5
    stride: int = 4
    pca_components: int = 64
    pca_fit_samples: int = 3000
    bootstrap_repeats: int = 300
    nonlinear_probe: bool = True
    nonlinear_max_iter: int = 200


def ridge_probe(x, y, splits):
    """Fit all scalers on train; select ridge alpha using validation only."""
    train, val = splits == 0, splits == 1
    scaler = StandardScaler().fit(x[train])
    xx = scaler.transform(x)
    mean, scale = y[train].mean(0), np.maximum(y[train].std(0), 1e-8)
    yy = (y-mean)/scale
    best = None
    for alpha in (1e-4, 0.01, 1.0, 100.0, 10000.0):
        model = Ridge(alpha=alpha).fit(xx[train], yy[train])
        score = float(np.mean((model.predict(xx[val])-yy[val])**2))
        if best is None or score < best[0]:
            best = score, model, alpha
    return best[1].predict(xx)*scale+mean, {"alpha": best[2], "val_normalized_mse": best[0]}


def nonlinear_probe(x, y, splits, cfg):
    """Fixed small MLP; no internal random validation split across temporal samples."""
    train, val = splits == 0, splits == 1
    scaler = StandardScaler().fit(x[train])
    xx = scaler.transform(x)
    mean, scale = y[train].mean(0), np.maximum(y[train].std(0), 1e-8)
    yy = (y-mean)/scale
    # One fixed capacity, batch schedule and regularization for every representation.
    model = MLPRegressor(hidden_layer_sizes=(64,), activation="tanh", alpha=0.01,
                         max_iter=cfg.nonlinear_max_iter, batch_size=min(256, int(train.sum())),
                         learning_rate_init=1e-3, early_stopping=False, random_state=cfg.seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(xx[train], yy[train])
    info = {"val_normalized_mse": float(np.mean((model.predict(xx[val])-yy[val])**2)),
            "iterations": model.n_iter_,
            "convergence_warning": any(isinstance(w.message, ConvergenceWarning) for w in caught)}
    return model.predict(xx)*scale+mean, info


def metric_rows(y, predicted, columns, groups, repeats=300, seed=123):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    results = []
    for j, name in enumerate(columns):
        error = predicted[:, j]-y[:, j]
        group_mse = np.array([np.mean(error[groups == g]**2) for g in unique])
        boot = np.sqrt(rng.choice(group_mse, size=(repeats, len(unique)), replace=True).mean(1))
        denominator = np.sum((y[:, j]-y[:, j].mean())**2)
        results.append({"variable": name, "rmse": float(np.sqrt(np.mean(error**2))),
                        "mae": float(np.mean(np.abs(error))),
                        "r2": float(1-np.sum(error**2)/denominator) if denominator > 1e-12 else np.nan,
                        "rmse_ci_low": float(np.quantile(boot, 0.025)),
                        "rmse_ci_high": float(np.quantile(boot, 0.975)),
                        "test_groups": len(unique), "test_samples": len(y)})
    return results


def delay_pca(frames, bundle, sim, condition, index, cfg):
    """16x16 block-averaged delay images; fit PCA on train only, then transform all splits."""
    spec = CONDITIONS[condition]
    x = np.empty((len(index), cfg.window*16*16), dtype="float32")
    for i, (tr, t) in enumerate(index):
        clip = observation(frames, tr, t, cfg.window, spec["noise_std"], sim.seed)
        x[i] = clip.reshape(cfg.window, 16, 4, 16, 4).mean(axis=(2, 4)).ravel()
    train = np.flatnonzero(bundle["splits"][index[:, 0]] == 0)
    fit_ids = np.random.default_rng(cfg.seed).choice(train, min(len(train), cfg.pca_fit_samples), replace=False)
    n_components = min(cfg.pca_components, len(fit_ids)-1, x.shape[1])
    if n_components < 1:
        raise ValueError("Insufficient training samples for PCA.")
    pca = PCA(n_components, svd_solver="randomized", random_state=cfg.seed).fit(x[fit_ids])
    return pca.transform(x).astype("float32"), pca.explained_variance_ratio_.tolist()


def completed_runs(root, dataset_id, condition):
    base = Path(root)/"runs"/dataset_id/condition
    return sorted(p.parent for p in base.glob("*/complete.json"))


def compare(root, dataset_id, condition, cache_root, run_paths, cfg=EvalConfig()):
    if cfg.bootstrap_repeats < 1 or cfg.pca_fit_samples < 2:
        raise ValueError("Invalid evaluation sample counts.")
    spec = CONDITIONS[condition]
    bundle, sim, manifest = load_dataset(root, dataset_id, spec["kind"])
    index = index_windows(bundle, cfg.window, 2*sim.frames_per_period, cfg.stride)
    splits, groups = bundle["splits"][index[:, 0]], bundle["groups"][index[:, 0]]
    test = splits == 2
    frames = frame_cache(bundle, sim, condition, cache_root, dataset_id)
    representations, run_meta = {}, []
    for path in map(Path, run_paths):
        meta = json.loads((path/"run.json").read_text())
        complete = json.loads((path/"complete.json").read_text())
        if meta["dataset_id"] != dataset_id or meta["condition"] != condition or meta["condition_spec"] != spec:
            raise ValueError(f"Incompatible dataset/condition: {path}")
        if meta["training"]["window"] != cfg.window or meta["training"]["eval_stride"] != cfg.stride:
            raise ValueError(f"History/evaluation stride mismatch: {path}")
        if file_sha(path/"features.npz") != complete["features_sha256"]:
            raise ValueError(f"Feature checksum mismatch: {path}")
        with np.load(path/"features.npz", allow_pickle=False) as features:
            if not np.array_equal(features["index"], index):
                raise ValueError("Feature rows do not match the common evaluation index.")
            for key in features.files:
                if key == "index":
                    continue
                # Keep training seeds/configs separate; never select the best seed on test.
                label = f"{meta['run_id']}:{key}"
                representations[label] = (features[key].copy(), meta["method"], key, meta["training"]["seed"])
        run_meta.append({"run_id": meta["run_id"], "features_sha": complete["features_sha256"]})
    pca_features, variance = delay_pca(frames, bundle, sim, condition, index, cfg)
    representations["delay_pca"] = (pca_features, "delay_pca", "pca", None)
    identity = {"dataset_id": dataset_id, "condition": condition, "config": asdict(cfg),
                "runs": run_meta, "evaluation_source_sha": file_sha(__file__)}
    out = Path(root)/"analysis"/dataset_id/condition/digest(identity)
    if (out/"complete.json").exists():
        complete = json.loads((out/"complete.json").read_text())
        if file_sha(out/"scorecard.csv") != complete["scorecard_sha256"]:
            raise ValueError("Completed scorecard was modified; inspect the saved analysis.")
        return out, pd.read_csv(out/"scorecard.csv")
    out.mkdir(parents=True, exist_ok=True)
    json_write(out/"evaluation.json", {**identity, "pca_variance_ratio": variance,
               "bootstrap_unit": "length group" if spec["kind"] == "variable" else "trajectory",
               "forecast_type": "direct multi-horizon readout, NOT an autonomous rollout",
               "truth_use": "evaluation probes only; never representation training",
               "uncertainty": "conditional on trained model; training seeds reported separately"})
    truth = sample_truth(bundle, index)
    current_columns = [0, 1, 2] if spec["kind"] == "variable" else [0, 1]
    names = ["theta", "velocity", "length"]
    rows, selections, ids, current_predictions = [], {}, {}, {}
    horizons = sorted(set([0, int(round(sim.frames_per_period/4)), sim.frames_per_period, 2*sim.frames_per_period]))
    for label, (x, method, feature, seed) in representations.items():
        print("Evaluating", label, flush=True)
        if not np.isfinite(x).all():
            raise ValueError(f"Nonfinite representation: {label}")
        ids[label] = estimate_id(x[splits == 0], groups[splits == 0], seed=cfg.seed)
        for horizon in horizons:
            columns = current_columns if horizon == 0 else [0, 1]
            y = sample_truth(bundle, index, horizon)[:, columns]
            predicted, selected = ridge_probe(x, y, splits)
            key = f"{label}:h{horizon}:ridge"
            selections[key] = selected
            if horizon == 0:
                current_predictions[label] = predicted[:, :2]
            for result in metric_rows(y[test], predicted[test], [names[j] for j in columns],
                                      groups[test], cfg.bootstrap_repeats, cfg.seed):
                rows.append({"label": label, "method": method, "feature": feature,
                             "dimension": x.shape[1], "training_seed": seed, "probe": "ridge",
                             "horizon_frames": horizon, "horizon_periods": horizon/sim.frames_per_period,
                             **result})
        if cfg.nonlinear_probe:
            y = truth[:, current_columns]
            predicted, selected = nonlinear_probe(x, y, splits, cfg)
            selections[f"{label}:h0:mlp64"] = selected
            for result in metric_rows(y[test], predicted[test], [names[j] for j in current_columns],
                                      groups[test], cfg.bootstrap_repeats, cfg.seed):
                rows.append({"label": label, "method": method, "feature": feature,
                             "dimension": x.shape[1], "training_seed": seed, "probe": "mlp64",
                             "horizon_frames": 0, "horizon_periods": 0.0, **result})
    # Persistence is explicitly labelled as using the true current physical state.
    for horizon in horizons[1:]:
        future = sample_truth(bundle, index, horizon)[:, :2]
        for result in metric_rows(future[test], truth[test, :2], names[:2], groups[test],
                                  cfg.bootstrap_repeats, cfg.seed):
            rows.append({"label": "oracle_state_persistence", "method": "oracle_state_persistence",
                         "feature": "true_current_state", "dimension": 2, "training_seed": None,
                         "probe": "persistence", "horizon_frames": horizon,
                         "horizon_periods": horizon/sim.frames_per_period, **result})
    scorecard = pd.DataFrame(rows)
    scorecard.to_csv(out/"scorecard.csv", index=False)
    json_write(out/"probe_selection.json", selections)
    json_write(out/"id_train.json", ids)
    np.savez_compressed(out/"evaluation_truth.npz", index=index, states=truth, splits=splits, groups=groups)
    plot_results(out, representations, scorecard, truth, splits, groups, index, current_predictions)
    json_write(out/"complete.json", {"scorecard_sha256": file_sha(out/"scorecard.csv")})
    return out, scorecard


def plot_results(out, representations, scores, truth, splits, groups, index, current_predictions):
    import matplotlib.pyplot as plt
    # Each latent display projection is fitted on train, including when showing test.
    selected = []
    seen = set()
    for label, (x, method, feature, seed) in representations.items():
        if method not in seen and (feature in {"z", "z_d2", "z_d3", "pca"}):
            selected.append((label, x, method, feature))
            seen.add(method)
    colors = [(0, "angle"), (1, "angular velocity")]
    if truth[:, 2].std() > 1e-8:
        colors.append((2, "pendulum length"))
    fig, axes = plt.subplots(len(colors), len(selected),
                             figsize=(4*len(selected), 3.5*len(colors)), squeeze=False)
    test_idx = np.flatnonzero(splits == 2)[::3]
    for col, (label, x, method, feature) in enumerate(selected):
        n = min(2, x.shape[1])
        pca = PCA(n).fit(x[splits == 0])
        xy = pca.transform(x[test_idx])
        if n == 1:
            xy = np.column_stack((xy, np.zeros(len(xy))))
        for row, (j, name) in enumerate(colors):
            sc = axes[row, col].scatter(xy[:, 0], xy[:, 1], c=truth[test_idx, j], s=5, cmap="coolwarm")
            axes[row, col].set_title(f"{method} / {feature}\ncolored by {name}")
            axes[row, col].set_xlabel("train-fitted PC1")
            axes[row, col].set_ylabel("train-fitted PC2")
            fig.colorbar(sc, ax=axes[row, col])
    fig.tight_layout()
    fig.savefig(Path(out)/"latent_test.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, _, method, feature in selected:
        subset = scores[(scores.label == label) & (scores.probe == "ridge") & (scores.horizon_frames > 0)]
        for ax, name in zip(axes, ["theta", "velocity"]):
            part = subset[subset.variable == name].sort_values("horizon_frames")
            ax.plot(part.horizon_periods, part.rmse, "o-", label=f"{method}/{feature}")
            ax.fill_between(part.horizon_periods.to_numpy(), part.rmse_ci_low.to_numpy(),
                            part.rmse_ci_high.to_numpy(), alpha=0.12)
            ax.set(xlabel="forecast horizon / reference period", ylabel=f"{name} RMSE",
                   title="Direct forecasts; held-out trajectories")
    for ax, name in zip(axes, ["theta", "velocity"]):
        part = scores[(scores.method == "oracle_state_persistence") & (scores.variable == name)]
        ax.plot(part.horizon_periods, part.rmse, "k--", label="oracle state persistence")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out)/"forecast_test.png", dpi=140)
    plt.close(fig)

    # One held-out trajectory, ordered in time: no t-SNE trajectory interpolation.
    tr = int(index[np.flatnonzero(splits == 2)[0], 0])
    mask = index[:, 0] == tr
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for j, name in enumerate(["theta", "velocity"]):
        axes[j].plot(index[mask, 1], truth[mask, j], "k-", linewidth=2, label="truth")
        for label, _, method, feature in selected:
            axes[j].plot(index[mask, 1], current_predictions[label][mask, j], label=f"{method}/{feature}", alpha=0.7)
        axes[j].set(xlabel="frame", ylabel=name, title=f"Held-out trajectory {tr}: linear readout")
        axes[j].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out)/"state_readout_test.png", dpi=140)
    plt.close(fig)
