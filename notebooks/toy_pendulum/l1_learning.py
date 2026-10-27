"""Shared Colab training for predictive NSV, InfoNCE and two Barlow view protocols."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from l1_data import (CONDITIONS, digest, estimate_id, file_sha, frame_cache, index_windows,
                     json_write, load_dataset, observation)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    window: int = 5
    lag: int = 5
    train_stride: int = 2
    eval_stride: int = 4
    hidden_dim: int = 64
    projection_dim: int = 2
    batch_size: int = 128
    epochs: int = 30
    refine_epochs: int = 60
    refine_dims: tuple = (1, 2, 3, 4, 8)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    temperature: float = 0.1
    barlow_offdiag: float = 0.005
    view_noise_std: float = 0.01
    run_tag: str = "pilot"

    def validate(self):
        if min(self.window, self.lag, self.train_stride, self.eval_stride, self.hidden_dim,
               self.projection_dim, self.epochs, self.refine_epochs) < 1:
            raise ValueError("Counts/lags/dimensions must be positive.")
        if self.batch_size < 2 or self.temperature <= 0 or self.learning_rate <= 0:
            raise ValueError("Invalid batch size, temperature, or learning rate.")
        if self.view_noise_std < 0 or self.barlow_offdiag < 0:
            raise ValueError("Noise/penalties must be nonnegative.")
        if not self.refine_dims or min(self.refine_dims) < 1:
            raise ValueError("Specify positive NSV bottleneck dimensions.")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class WindowDataset(Dataset):
    """Physical state labels are deliberately absent from all training batches."""
    def __init__(self, frames, index, condition, data_seed, config, mode="embedding", augment=False):
        self.frames, self.index = frames, np.asarray(index)
        self.spec, self.seed, self.cfg = CONDITIONS[condition], data_seed, config
        self.mode, self.augment, self.epoch = mode, augment, 0

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        tr, t = map(int, self.index[i])
        x = observation(self.frames, tr, t, self.cfg.window, self.spec["noise_std"], self.seed, 0)
        if self.mode == "embedding":
            return torch.from_numpy(x)
        shift = 0 if self.mode == "same" else self.cfg.lag
        y = observation(self.frames, tr, t+shift, self.cfg.window,
                        self.spec["noise_std"], self.seed, 1)
        if self.augment and self.cfg.view_noise_std:
            rng = np.random.default_rng(np.random.SeedSequence([self.cfg.seed, self.epoch, int(i), 301]))
            x += self.cfg.view_noise_std*rng.standard_normal(x.shape).astype("float32")
            # Prediction targets are future observations, never oracle clean images.
            if self.mode != "predict":
                y += self.cfg.view_noise_std*rng.standard_normal(y.shape).astype("float32")
        return torch.from_numpy(x), torch.from_numpy(y), tr, t


class Encoder(nn.Module):
    def __init__(self, window=5, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(window, 16, 5, 2, 2), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(4, 64), nn.SiLU(),
            nn.Flatten(), nn.Linear(64*8*8, hidden_dim))

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, hidden_dim, window):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 64*8*8)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, window, 3, padding=1))

    def forward(self, h):
        # Linear output accommodates zero-mean observation noise outside [0,1].
        return self.net(self.fc(h).reshape(-1, 64, 8, 8))


class Representation(nn.Module):
    def __init__(self, cfg, predictive=False):
        super().__init__()
        self.encoder = Encoder(cfg.window, cfg.hidden_dim)
        self.head = Decoder(cfg.hidden_dim, cfg.window) if predictive else nn.Sequential(
            nn.Linear(cfg.hidden_dim, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, cfg.projection_dim))

    def forward(self, x):
        h = self.encoder(x)
        return h, self.head(h)


def infonce_loss(a, b, trajectory, ends, exclusion, temperature=0.1):
    """Symmetric cross-view InfoNCE; the same row is the sole designated positive."""
    if len(a) < 2:
        raise ValueError("InfoNCE needs at least two pairs.")
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    logits = a @ b.T / temperature
    near = (trajectory[:, None] == trajectory[None, :]) & ((ends[:, None]-ends[None, :]).abs() < exclusion)
    diagonal = torch.eye(len(a), dtype=torch.bool, device=a.device)
    logits = logits.masked_fill(near & ~diagonal, -torch.inf)
    target = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, target)+F.cross_entropy(logits.T, target))/2


def barlow_loss(a, b, offdiag=0.005):
    if len(a) < 2:
        raise ValueError("Cross-correlation needs at least two samples.")
    a = (a-a.mean(0))/torch.sqrt(a.var(0, unbiased=False)+1e-5)
    b = (b-b.mean(0))/torch.sqrt(b.var(0, unbiased=False)+1e-5)
    c = a.T @ b / len(a)
    diagonal = torch.diagonal(c)
    mask = ~torch.eye(c.shape[0], dtype=torch.bool, device=c.device)
    return (diagonal-1).square().sum()+offdiag*c[mask].square().sum()


def batch_loss(model, batch, method, cfg, device):
    x, y, tr, t = (v.to(device) for v in batch)
    _, za = model(x)
    if method == "nsv":
        return F.mse_loss(za, y)
    _, zb = model(y)
    if method == "infonce":
        return infonce_loss(za, zb, tr, t, cfg.window+cfg.lag, cfg.temperature)
    return barlow_loss(za, zb, cfg.barlow_offdiag)


def fit_representation(model, train_ds, val_ds, method, cfg, device, out):
    if min(len(train_ds), len(val_ds)) < 2:
        raise ValueError("Too few training/validation windows.")
    generator = torch.Generator().manual_seed(cfg.seed)
    train = DataLoader(train_ds, batch_size=min(cfg.batch_size, len(train_ds)), shuffle=True,
                       drop_last=True, num_workers=0, generator=generator)
    val = DataLoader(val_ds, batch_size=min(cfg.batch_size, len(val_ds)), shuffle=False,
                     drop_last=True, num_workers=0)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    best, best_state, history = math.inf, None, []
    start = time.time()
    for epoch in range(cfg.epochs):
        train_ds.epoch = epoch
        record = {"epoch": epoch+1}
        for name, loader in [("train", train), ("val", val)]:
            model.train(name == "train")
            total, count = 0.0, 0
            for batch in loader:
                with torch.set_grad_enabled(name == "train"):
                    loss = batch_loss(model, batch, method, cfg, device)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite {method} loss at epoch {epoch+1}")
                    if name == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                total += float(loss.detach())*len(batch[0])
                count += len(batch[0])
            record[name] = total/count
        if record["val"] < best:
            best = record["val"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, Path(out)/"model_best.pt")
        record["elapsed_seconds"] = time.time()-start
        history.append(record)
        json_write(Path(out)/"training_history.json", history)
        print(f"{method} epoch {epoch+1:3d}: train={record['train']:.5f}, val={record['val']:.5f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return history


def encode(model, ds, device, batch_size=128, predictive=False):
    hs, zs = [], []
    model.eval()
    with torch.no_grad():
        for x in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0):
            h = model.encoder(x.to(device))
            hs.append(h.cpu().numpy())
            if not predictive:
                zs.append(model.head(h).cpu().numpy())
    return np.concatenate(hs), None if predictive else np.concatenate(zs)


class SineLayer(nn.Module):
    def __init__(self, n_in, n_out, first=False):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        bound = 1/n_in if first else np.sqrt(6/n_in)/30
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(30*self.linear(x))


class Refiner(nn.Module):
    def __init__(self, h_train, dim):
        super().__init__()
        mean = h_train.mean(0)
        std = h_train.std(0).clip(min=max(float(h_train.std(0).max())*0.01, 1e-5))
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(std, dtype=torch.float32))
        n_in = h_train.shape[1]
        self.encoder = nn.Sequential(SineLayer(n_in, 128, first=True), SineLayer(128, 64), nn.Linear(64, dim))
        self.decoder = nn.Sequential(SineLayer(dim, 64, first=True), SineLayer(64, 128), nn.Linear(128, n_in))

    def forward(self, h):
        normalized = (h-self.mean)/self.scale
        z = self.encoder(normalized)
        return z, self.decoder(z), normalized


def refine(h, splits, dim, cfg, device, out):
    seed_all(cfg.seed+dim)
    model = Refiner(h[splits == 0], dim).to(device)
    loaders = {name: DataLoader(TensorDataset(torch.from_numpy(h[splits == split])),
                               batch_size=cfg.batch_size, shuffle=(split == 0), num_workers=0)
               for name, split in [("train", 0), ("val", 1)]}
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    history, best, best_state = [], math.inf, None
    for epoch in range(cfg.refine_epochs):
        row = {"epoch": epoch+1}
        for name, loader in loaders.items():
            model.train(name == "train")
            total, count = 0.0, 0
            for (x,) in loader:
                with torch.set_grad_enabled(name == "train"):
                    _, reconstructed, target = model(x.to(device))
                    loss = F.mse_loss(reconstructed, target)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite NSV refine loss.")
                    if name == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                total += float(loss.detach())*len(x)
                count += len(x)
            row[name] = total/count
        history.append(row)
        if row["val"] < best:
            best = row["val"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    torch.save(best_state, Path(out)/f"refiner_d{dim}.pt")
    json_write(Path(out)/f"refine_history_d{dim}.json", history)
    with torch.no_grad():
        z = np.concatenate([model(torch.from_numpy(x).to(device))[0].cpu().numpy()
                            for x in np.array_split(h, max(1, math.ceil(len(h)/cfg.batch_size)))])
    print(f"NSV refine d={dim}: best validation normalized-latent MSE={best:.5f}", flush=True)
    return z


def run_training(root, dataset_id, condition, cache_root, cfg, method, device=None):
    """One immutable run; test states never enter optimization or hyperparameter selection."""
    cfg.validate()
    if method not in {"nsv", "infonce", "bt_temporal", "bt_same"}:
        raise ValueError(method)
    spec = CONDITIONS[condition]
    bundle, sim, manifest = load_dataset(root, dataset_id, spec["kind"])
    identity = {"dataset_id": dataset_id, "condition": condition, "condition_spec": spec,
                "method": method, "training": asdict(cfg), "data_sha": manifest["data_sha256"][spec["kind"]],
                "learning_source_sha": file_sha(__file__)}
    run_id = f"{method}_s{cfg.seed}_{digest(identity)}"
    out = Path(root)/"runs"/dataset_id/condition/run_id
    if (out/"complete.json").exists():
        complete = json.loads((out/"complete.json").read_text())
        if file_sha(out/"features.npz") != complete["features_sha256"]:
            raise ValueError("Completed feature artifact was modified.")
        print("Reusing completed run:", out)
        return out
    if out.exists():
        raise FileExistsError(f"Incomplete run at {out}. Inspect it or change run_tag; no silent overwrite.")
    seed_all(cfg.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # Construct/validate windows before creating a run directory.
    train_index = index_windows(bundle, cfg.window, cfg.lag, cfg.train_stride)
    eval_index = index_windows(bundle, cfg.window, 2*sim.frames_per_period, cfg.eval_stride)
    frames = frame_cache(bundle, sim, condition, cache_root, dataset_id)
    mode = "predict" if method == "nsv" else ("same" if method == "bt_same" else "temporal")
    train_splits = bundle["splits"][train_index[:, 0]]
    train_ds = WindowDataset(frames, train_index[train_splits == 0], condition, sim.seed, cfg, mode, True)
    val_ds = WindowDataset(frames, train_index[train_splits == 1], condition, sim.seed, cfg, mode, True)
    if min(len(train_ds), len(val_ds)) < 2:
        raise ValueError("Insufficient windows per split.")
    out.mkdir(parents=True)
    runtime = {"python": platform.python_version(), "torch": str(torch.__version__),
               "numpy": np.__version__, "device": str(device), "cuda": torch.version.cuda}
    json_write(out/"run.json", {**identity, "run_id": run_id, "runtime": runtime,
                               "selection": "validation objective only; no test selection",
                               "temporal_window_overlap_fraction": max(0, cfg.window-cfg.lag)/cfg.window})
    model = Representation(cfg, predictive=(method == "nsv"))
    fit_representation(model, train_ds, val_ds, method, cfg, device, out)
    ds = WindowDataset(frames, eval_index, condition, sim.seed, cfg)
    h, z = encode(model, ds, device, cfg.batch_size, predictive=(method == "nsv"))
    splits = bundle["splits"][eval_index[:, 0]]
    groups = bundle["groups"][eval_index[:, 0]]
    features = {"index": eval_index, "h": h}
    diagnostic = estimate_id(h[splits == 0], groups[splits == 0], seed=cfg.seed)
    json_write(out/"id_train.json", diagnostic)
    if method == "nsv":
        for dim in cfg.refine_dims:
            features[f"z_d{dim}"] = refine(h, splits, dim, cfg, device, out)
    else:
        features["z"] = z
        if method == "infonce":
            features["z_unit"] = z/np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    np.savez_compressed(out/"features.npz", **features)
    json_write(out/"complete.json", {"run_id": run_id, "features_sha256": file_sha(out/"features.npz")})
    return out
