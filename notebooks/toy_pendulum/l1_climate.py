"""Climate-pipeline replica for single-frame pendulum images.

BSISO supplies the 2-D encoders/decoder, nb08/15 the raw InfoNCE objective,
nb26 the Barlow projector/lag objective, and nb20/23 the SIREN refinement.
See PIPELINE_PARITY.md for deliberate adaptations; legacy l1_learning is separate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
import json
import math
from pathlib import Path
import platform
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from l1_data import (CONDITIONS, digest, estimate_id, file_sha, frame_cache,
                     index_windows, json_write, load_dataset, observation)

PROTOCOL = "climate_bsiso_v1"


@dataclass(frozen=True)
class ClimateConfig:
    protocol: str = PROTOCOL
    method: str = "infonce"
    seed: int = 42
    window: int = 1
    train_stride: int = 1
    eval_stride: int = 4
    hidden_dim: int = 64
    projection_dim: int = 2
    lag: int = 10                     # NSV prediction/dynamics only
    positive_radius: int = 3          # InfoNCE: +/- radius, excluding zero
    barlow_lags: tuple = (1, 2, 3, 4, 5)
    barlow_offdiag: float = 0.005
    barlow_min_weight: float = 0.05
    temperature: float = 0.5
    view_noise_std: float = 0.0
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    min_learning_rate: float = 1e-5
    gradient_clip: float = 1.0
    checkpoint_selection: str = "final_epoch"
    warm_start: bool = False
    val_pairs: int = 1000
    refine_epochs: int = 500
    refine_batch_size: int = 64
    refine_learning_rate: float = 1e-4
    refine_min_learning_rate: float = 1e-6
    refine_dims: tuple = (2,)         # extra ablations, in addition to estimated d
    id_backend: str = "skdim"
    id_scope: str = "train_val"      # nb19/22; independent test remains excluded
    id_max_samples: int = 5000
    dynamics_epochs: int = 300
    run_tag: str = "climate_replica"

    def validate(self):
        if self.protocol != PROTOCOL or self.window != 1 or self.hidden_dim != 64:
            raise ValueError("Climate replica requires its explicit protocol, window=1, hidden_dim=64.")
        if self.method not in {"infonce", "nsv", "bt_temporal", "bt_same"}:
            raise ValueError(self.method)
        if min(self.train_stride, self.eval_stride, self.projection_dim, self.lag,
               self.positive_radius, self.epochs, self.refine_epochs, self.refine_batch_size,
               self.val_pairs, self.id_max_samples) < 1 or self.batch_size < 2:
            raise ValueError("Invalid sample counts or training dimensions.")
        if not self.barlow_lags or min(self.barlow_lags) < 1 or len(set(self.barlow_lags)) != len(self.barlow_lags):
            raise ValueError("Barlow lags must be distinct positive integers.")
        if self.temperature <= 0 or not 0 < self.barlow_min_weight <= 1:
            raise ValueError("Invalid temperature/lag weighting.")
        if min(self.learning_rate, self.refine_learning_rate) <= 0 or min(self.view_noise_std, self.weight_decay, self.barlow_offdiag, self.gradient_clip, self.dynamics_epochs) < 0:
            raise ValueError("Invalid optimizer or augmentation settings.")
        if any(d < 1 for d in self.refine_dims):
            raise ValueError("Refiner dimensions must be positive.")
        if self.id_backend not in {"skdim", "group_mle"} or self.id_scope not in {"train", "train_val"}:
            raise ValueError("Unknown ID protocol.")
        if self.checkpoint_selection not in {"final_epoch", "validation_objective", "training_objective"}:
            raise ValueError("Unknown checkpoint selection.")
        if self.warm_start and not self.method.startswith("bt_"):
            raise ValueError("Only Barlow uses the NSV warm start.")


def climate_config(method, **overrides):
    """Method-specific notebook defaults, not a common optimizer/architecture."""
    cfg = ClimateConfig(method=method)
    if method == "nsv":
        cfg = replace(cfg, epochs=100, gradient_clip=0.0, checkpoint_selection="validation_objective")
    elif method.startswith("bt_"):
        cfg = replace(cfg, projection_dim=3, batch_size=256, epochs=100, gradient_clip=0.0,
                      checkpoint_selection="training_objective", warm_start=True,
                      view_noise_std=0.01 if method == "bt_same" else 0.0)
    cfg = replace(cfg, **overrides)
    cfg.validate()
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SSL2DEncoder(nn.Module):
    """nb08: only input channels change from three climate fields to one image."""
    def __init__(self, embedding_dim=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16); self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32); self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, embedding_dim)
        initialize_bsiso(self)

    def pre_fc(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        return self.global_pool(F.relu(self.bn3(self.conv3(x)))).flatten(1)

    def forward(self, x):
        return self.fc(self.pre_fc(x))


def initialize_bsiso(model, decoder=False):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if decoder and layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.constant_(layer.weight, 1); nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, 0, 0.01); nn.init.constant_(layer.bias, 0)


class PredictiveEncoder(nn.Module):
    """nb18c BSISO encoder; input channel adaptation only."""
    def __init__(self, latent_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1, bias=False); self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=False); self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1, bias=False); self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1, bias=False); self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 4, 2, 1, bias=False); self.bn5 = nn.BatchNorm2d(128)
        self.gap = nn.AdaptiveAvgPool2d(1); self.fc = nn.Linear(128, latent_dim)
        initialize_bsiso(self)

    def forward(self, x):
        for i in range(1, 6):
            x = F.relu(getattr(self, f"bn{i}")(getattr(self, f"conv{i}")(x)))
        return self.fc(self.gap(x).flatten(1))


class PredictiveDecoder(nn.Module):
    """nb18c decoder; resize ladder adapted to 64x64, final channel to one."""
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128)
        self.conv1 = nn.Conv2d(128, 64, 3, padding=1, bias=False); self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1, bias=False); self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 1, 3, padding=1)
        initialize_bsiso(self, decoder=True)

    def forward(self, h):
        x = self.fc(h).view(-1, 128, 1, 1)
        for i, size in enumerate((8, 16, 32), 1):
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            x = F.relu(getattr(self, f"bn{i}")(getattr(self, f"conv{i}")(x)))
        return self.conv4(F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False))


class Projector(nn.Module):
    """Exact nb26/27 projector: 64 -> 128 -> 64 -> D, BN/ReLU."""
    def __init__(self, in_dim=64, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Linear(64, out_dim))

    def forward(self, x):
        return self.net(x)


class ClimateModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.method = cfg.method
        if cfg.method == "infonce":
            self.encoder = SSL2DEncoder(cfg.projection_dim)
        else:
            self.encoder = PredictiveEncoder(cfg.hidden_dim)
            if cfg.method == "nsv":
                self.decoder = PredictiveDecoder(cfg.hidden_dim)
            else:
                self.projector = Projector(cfg.hidden_dim, cfg.projection_dim)

    def forward(self, x):
        if self.method == "infonce":
            pooled = self.encoder.pre_fc(x)
            return pooled, self.encoder.fc(pooled)
        h = self.encoder(x)
        return h, self.decoder(h) if self.method == "nsv" else self.projector(h)


def infonce_loss_raw(z_a, z_b, temperature=0.5):
    """nb08/15: one direction, raw dot product, no nearby-negative mask."""
    logits = torch.matmul(z_a, z_b.T) / temperature
    return F.cross_entropy(logits, torch.arange(len(z_a), device=z_a.device))


def barlow_cross_corr(z_a, z_b, eps=1e-5):
    """Preserve nb26's sample std (B-1) and eps outside the square root."""
    if len(z_a) < 2:
        raise ValueError("Barlow correlation needs at least two paired samples.")
    a = (z_a-z_a.mean(0))/(z_a.std(0)+eps)
    b = (z_b-z_b.mean(0))/(z_b.std(0)+eps)
    return a.T @ b / len(a)


def barlow_loss(c, weight=1.0, offdiag=0.005):
    on = (torch.diagonal(c)-1).square().sum()
    off = (c-torch.diag(torch.diagonal(c))).square().sum()
    return weight*(on+offdiag*off)


def lag_weight(lag, cfg):
    lo, hi = min(cfg.barlow_lags), max(cfg.barlow_lags)
    return cfg.barlow_min_weight**((lag-lo)/(hi-lo)) if hi != lo else 1.0


class PairDataset(Dataset):
    """Time-only pairs; never inspect simulated theta, velocity, or length values."""
    def __init__(self, frames, index, condition, data_seed, cfg, training, lag=None):
        self.frames, self.index = frames, np.asarray(index)
        self.spec, self.seed, self.cfg = CONDITIONS[condition], data_seed, cfg
        self.training, self.lag, self.epoch = training, lag, 0
        self.allowed = {(int(tr), int(t)) for tr, t in self.index}
        if cfg.method == "infonce":
            self.neighbors = []
            for tr, t in self.index:
                candidates = [int(t+d) for d in range(-cfg.positive_radius, cfg.positive_radius+1)
                              if d and (int(tr), int(t+d)) in self.allowed]
                if not candidates:
                    raise ValueError("InfoNCE anchor has no temporal neighbor; reduce train_stride.")
                self.neighbors.append(candidates)
            if not training:
                rng = np.random.default_rng(cfg.seed)
                # Same validation sample limit as nb08/15, but reproducible pair RNG.
                chosen = rng.choice(len(self.index), min(cfg.val_pairs, len(self.index)), replace=False)
                self.fixed_pairs = [(int(i), int(rng.choice(self.neighbors[i]))) for i in chosen]

    def __len__(self):
        if self.cfg.method == "infonce" and not self.training:
            return len(self.fixed_pairs)
        return len(self.index)

    def pair_at(self, i):
        if self.cfg.method == "infonce":
            if not self.training:
                i, target = self.fixed_pairs[i]
            else:
                rng = np.random.default_rng(np.random.SeedSequence([self.cfg.seed, self.epoch, int(i), 751]))
                target = int(rng.choice(self.neighbors[i]))
        else:
            target = int(self.index[i, 1]) + int(self.lag)
        tr, t = map(int, self.index[i])
        return tr, t, target

    def __getitem__(self, i):
        tr, t, target = self.pair_at(i)
        x = observation(self.frames, tr, t, 1, self.spec["noise_std"], self.seed, 0)
        # Climate fields are fixed observations reused in either branch.
        draw = 1 if self.cfg.method == "bt_same" else 0
        y = observation(self.frames, tr, target, 1, self.spec["noise_std"], self.seed, draw)
        if self.cfg.view_noise_std:
            rng = np.random.default_rng(np.random.SeedSequence([self.cfg.seed, self.epoch, int(i), 752]))
            x += self.cfg.view_noise_std*rng.standard_normal(x.shape).astype("float32")
            if self.cfg.method != "nsv":
                y += self.cfg.view_noise_std*rng.standard_normal(y.shape).astype("float32")
        return torch.from_numpy(x), torch.from_numpy(y)


def make_loaders(frames, bundle, condition, sim, cfg):
    lags = cfg.barlow_lags if cfg.method == "bt_temporal" else (0 if cfg.method in {"infonce", "bt_same"} else cfg.lag,)
    loaders, counts = {}, {}
    for split in (0, 1):
        loaders[split] = {}
        for lag in lags:
            ix = index_windows(bundle, 1, lag, cfg.train_stride)
            ix = ix[bundle["splits"][ix[:, 0]] == split]
            ds = PairDataset(frames, ix, condition, sim.seed, cfg, split == 0, lag)
            if len(ds) < 2:
                raise ValueError("Need at least two pairs in each split/lag.")
            batch = min(cfg.batch_size, len(ds))
            # BT's projector BatchNorm1d cannot train on a singleton tail.
            drop = split == 0 and cfg.method.startswith("bt_")
            loaders[split][lag] = DataLoader(ds, batch_size=batch, shuffle=split == 0,
                                            drop_last=drop, num_workers=0)
            counts[f"{split}:{lag}"] = {"pairs": len(ds), "batches": len(loaders[split][lag]), "drop_last": drop}
    return loaders, counts


def loss_on_pair(model, pair, cfg, lag, device):
    x, y = (v.to(device) for v in pair)
    _, a = model(x)
    if cfg.method == "nsv":
        return F.mse_loss(a, y), None
    _, b = model(y)
    if cfg.method == "infonce":
        norms = torch.cat((a.detach().norm(dim=1), b.detach().norm(dim=1)))
        return infonce_loss_raw(a, b, cfg.temperature), norms
    c = barlow_cross_corr(a, b)
    weight = lag_weight(lag, cfg) if cfg.method == "bt_temporal" else 1.0
    return barlow_loss(c, weight, cfg.barlow_offdiag), c


def train_stage1(model, loaders, cfg, device, out):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs, eta_min=cfg.min_learning_rate)
    history, best, chosen, selected_epoch = [], math.inf, None, 0
    correlations = {lag: [] for lag in loaders[1]}
    fixed = {lag: next(iter(loader)) for lag, loader in loaders[1].items()}

    def record_correlations():
        if not cfg.method.startswith("bt_"):
            return
        model.eval()
        with torch.no_grad():
            for lag, pair in fixed.items():
                _, c = loss_on_pair(model, pair, cfg, lag, device)
                correlations[lag].append(c.cpu().numpy())

    record_correlations()
    for epoch in range(cfg.epochs):
        for loader in loaders[0].values():
            loader.dataset.epoch = epoch
        record = {"epoch": epoch+1, "learning_rate": optimizer.param_groups[0]["lr"]}
        model.train()
        iters = {lag: iter(loader) for lag, loader in loaders[0].items()}
        steps = min(len(loader) for loader in loaders[0].values())
        total_value, total_weight = 0.0, 0
        norm_parts, on_by_lag, off_by_lag = [], {}, {}
        for _ in range(steps):
            optimizer.zero_grad()
            terms, batch_n = [], 0
            for lag, iterator in iters.items():
                pair = next(iterator); batch_n = len(pair[0])
                loss, diagnostic = loss_on_pair(model, pair, cfg, lag, device)
                if cfg.method == "infonce":
                    norm_parts.append(diagnostic.cpu().numpy())
                elif cfg.method.startswith("bt_"):
                    c = diagnostic.detach()
                    on_by_lag[lag] = on_by_lag.get(lag, 0.0)+float((c.diag()-1).square().mean())
                    off = (c-torch.diag(c.diag())).square().sum()
                    off_by_lag[lag] = off_by_lag.get(lag, 0.0)+float(off)/max(1, len(c)*(len(c)-1))
                terms.append(loss)
            loss = torch.stack(terms).sum()  # nb26: all lag losses, ONE backward/step
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {cfg.method} loss at epoch {epoch+1}.")
            loss.backward()
            if cfg.gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            # nb08 averages batch losses; nb18c weights by sample count; BT batches are full.
            weight = batch_n if cfg.method == "nsv" else 1
            total_value += float(loss.detach())*weight; total_weight += weight
        record["train"] = total_value/total_weight
        if norm_parts:
            norms = np.concatenate(norm_parts)
            record.update(mean_norm=float(norms.mean()), std_norm=float(norms.std()), max_norm=float(norms.max()))
        for lag in on_by_lag:
            record[f"on_tau{lag}"] = on_by_lag[lag]/steps
            record[f"off_tau{lag}"] = off_by_lag[lag]/steps
        model.eval()
        with torch.no_grad():
            val = 0.0
            for lag, loader in loaders[1].items():
                values, weights = [], []
                for pair in loader:
                    if cfg.method.startswith("bt_") and len(pair[0]) < 2:
                        continue  # sample std is undefined for a singleton validation batch
                    loss, _ = loss_on_pair(model, pair, cfg, lag, device)
                    values.append(float(loss))
                    weights.append(len(pair[0]) if cfg.method == "nsv" else 1)
                val += float(np.average(values, weights=weights))
            record["val"] = val
        scheduler.step()
        if not math.isfinite(record["val"]):
            raise FloatingPointError("Nonfinite validation loss.")
        key = record["val"] if cfg.checkpoint_selection == "validation_objective" else record["train"]
        if cfg.checkpoint_selection == "final_epoch" or key < best:
            best, chosen, selected_epoch = key, copy.deepcopy(model.state_dict()), epoch+1
        history.append(record)
        record_correlations()
        json_write(out/"training_history.json", history)
        if epoch == 0 or (epoch+1) % 10 == 0 or epoch+1 == cfg.epochs:
            print(f"{cfg.method} epoch {epoch+1}: train={record['train']:.5f} val={record['val']:.5f}", flush=True)
    torch.save(model.state_dict(), out/"model_final.pt")
    model.load_state_dict(chosen); model.eval()
    torch.save(model.state_dict(), out/"model_selected.pt")
    torch.save(model.encoder.state_dict(), out/"encoder_selected.pt")
    json_write(out/"checkpoint_selection.json", {"policy": cfg.checkpoint_selection, "epoch": selected_epoch})
    if cfg.method.startswith("bt_"):
        np.savez_compressed(out/"bt_corr_history.npz", lags=np.array(list(correlations)),
                            **{f"C_tau{lag}": np.array(values) for lag, values in correlations.items()})


class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        bound = 1/in_features if is_first else math.sqrt(6/in_features)/omega_0
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        return torch.sin(self.omega_0*self.linear(x))


class SirenAutoencoder(nn.Module):
    """nb20/23 architecture and initialization; raw h, no standardization."""
    def __init__(self, z_dim=64, d_hat=2, omega_0=30.0):
        super().__init__()
        self.e1 = SirenLayer(z_dim, 128, omega_0, True)
        self.e2 = SirenLayer(128, 64, omega_0)
        self.e3 = SirenLayer(64, 32, omega_0)
        self.e4 = nn.Linear(32, d_hat)
        self.d1 = SirenLayer(d_hat, 32, omega_0, True)
        self.d2 = SirenLayer(32, 64, omega_0)
        self.d3 = SirenLayer(64, 128, omega_0)
        self.d4 = nn.Linear(128, z_dim)
        nn.init.kaiming_uniform_(self.e4.weight, nonlinearity="linear"); nn.init.zeros_(self.e4.bias)
        nn.init.kaiming_uniform_(self.d4.weight, nonlinearity="linear"); nn.init.zeros_(self.d4.bias)

    def encode(self, h):
        return self.e4(self.e3(self.e2(self.e1(h))))

    def decode(self, z):
        return self.d4(self.d3(self.d2(self.d1(z))))

    def forward(self, h):
        return self.decode(self.encode(h))


class DynamicsMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, d))

    def forward(self, x):
        return self.net(x)


def fit_array_regression(model, x, y, splits, epochs, batch_size, lr, eta_min, wd, device, out, name):
    """Train/val only; test features never enter the optimizer or checkpoint selection."""
    loaders = {s: DataLoader(TensorDataset(torch.from_numpy(x[splits == s]), torch.from_numpy(y[splits == s])),
                             batch_size=batch_size, shuffle=s == 0) for s in (0, 1)}
    if not all(len(loader.dataset) for loader in loaders.values()):
        raise ValueError("Missing train/validation regression samples.")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=eta_min)
    history, best, state = [], math.inf, None
    for epoch in range(epochs):
        row = {"epoch": epoch+1}
        for split, label in [(0, "train"), (1, "val")]:
            model.train(split == 0); total, count = 0.0, 0
            for a, b in loaders[split]:
                a, b = a.to(device), b.to(device)
                with torch.set_grad_enabled(split == 0):
                    loss = F.mse_loss(model(a), b)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite {name} loss.")
                    if split == 0:
                        opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss.detach())*len(a); count += len(a)
            row[label] = total/count
        scheduler.step(); history.append(row)
        if row["val"] < best:
            best, state = row["val"], copy.deepcopy(model.state_dict())
        if epoch == 0 or (epoch+1) % 25 == 0 or epoch+1 == epochs:
            json_write(out/f"{name}_history.json", history)
            print(f"{name} epoch {epoch+1}: train={row['train']:.6g} val={row['val']:.6g}", flush=True)
    model.load_state_dict(state); model.eval()
    torch.save(state, out/f"{name}.pt")
    json_write(out/f"{name}_history.json", history)
    return model


def encode_frames(model, frames, index, condition, sim, device, batch_size):
    h_parts, z_parts = [], []
    model.eval()
    spec = CONDITIONS[condition]
    with torch.no_grad():
        for chunk in np.array_split(index, max(1, math.ceil(len(index)/batch_size))):
            x = np.stack([observation(frames, tr, t, 1, spec["noise_std"], sim.seed, 0) for tr, t in chunk])
            h, z = model(torch.from_numpy(x).to(device))
            h_parts.append(h.cpu().numpy())
            if model.method != "nsv":
                z_parts.append(z.cpu().numpy())
    return np.concatenate(h_parts), np.concatenate(z_parts) if z_parts else None


def apply_numpy(model, x, device, encode=False, batch_size=512):
    model.eval()
    with torch.no_grad():
        return np.concatenate([(model.encode if encode else model)(torch.from_numpy(chunk).to(device)).cpu().numpy()
                               for chunk in np.array_split(x, max(1, math.ceil(len(x)/batch_size)))])


def climate_id(h, splits, groups, cfg):
    mask = splits == 0 if cfg.id_scope == "train" else splits != 2
    data = h[mask]
    if cfg.id_backend == "group_mle":
        diagnostic = estimate_id(data, groups[mask], seed=cfg.seed)
        values = [row["median_id"] for row in diagnostic["scales"] if row["median_id"] is not None]
        if not values:
            raise ValueError("Insufficient groups/samples for ID.")
        return {"d_hat": max(1, round(float(np.mean(values)))), "backend": "group_mle_ablation",
                "scope": cfg.id_scope, "diagnostic": diagnostic}
    import skdim
    # nb19/22 deduplication and fractional-k sweep, with a recorded sample cap.
    _, unique = np.unique(data, axis=0, return_index=True)
    data = data[np.sort(unique)]
    n_unique = len(data)
    rng = np.random.default_rng(cfg.seed)
    if len(data) > cfg.id_max_samples:
        data = data[np.sort(rng.choice(len(data), cfg.id_max_samples, replace=False))]
    if len(data) <= 20:
        raise ValueError("Too few unique points for the climate ID estimator; use a larger dataset.")
    ks = [max(int(len(data)*fraction), 3) for fraction in (.008, .010, .012, .014, .016)]
    values = [float(skdim.id.MLE(K=k).fit(data).dimension_) for k in ks]
    # Record an explicit-neighbor sweep too: useful for auditing the old K argument.
    explicit = [float(skdim.id.MLE().fit(data, n_neighbors=k).dimension_) for k in ks]
    tnn = float(skdim.id.TwoNN().fit(data).dimension_)
    lpca = float(skdim.id.lPCA().fit(data).dimension_)
    noise = rng.standard_normal(data.shape).astype("float32")
    shuffled = data.copy()
    for column in range(shuffled.shape[1]):
        rng.shuffle(shuffled[:, column])
    controls = {"noise_TwoNN": float(skdim.id.TwoNN().fit(noise).dimension_),
                "noise_MLE": float(skdim.id.MLE(K=ks[2]).fit(noise).dimension_),
                "shuffled_TwoNN": float(skdim.id.TwoNN().fit(shuffled).dimension_),
                "shuffled_MLE": float(skdim.id.MLE(K=ks[2]).fit(shuffled).dimension_)}
    values_all = values+explicit+[tnn, lpca]+list(controls.values())
    if not np.isfinite(values_all).all():
        raise ValueError("Nonfinite ID estimates; inspect the encoder before refinement.")
    return {"backend": "skdim_0.3.4", "scope": cfg.id_scope, "n_unique": n_unique,
            "n_sampled": len(data), "k_list": ks, "LB_by_k_legacy_call": values,
            "LB_by_explicit_n_neighbors": explicit, "LB_mean": float(np.mean(values)),
            "LB_std": float(np.std(values)), "d_hat": max(1, round(float(np.mean(values)))),
            "TwoNN": tnn, "lPCA": lpca, **controls,
            "dimension_selection": "round mean of original MLE(K=k).fit calls; not proof of physical state dimension"}


def resolve_warm_start(root, dataset_id, condition, cfg, requested=None):
    candidates = []
    base = Path(root)/"runs"/dataset_id/condition
    for path in sorted(base.glob("*/complete.json")):
        meta = json.loads((path.parent/"run.json").read_text())
        train = meta["training"]
        if (meta["method"] == "nsv" and train.get("protocol") == cfg.protocol
                and train["seed"] == cfg.seed and train["window"] == cfg.window
                and train["hidden_dim"] == cfg.hidden_dim
                and meta.get("learning_source_sha") == file_sha(__file__)):
            if requested is None or path.parent.name == requested:
                candidates.append(path.parent)
    if len(candidates) != 1:
        raise ValueError(f"Choose exactly one compatible NSV run for warm start; run 40 first. Candidates: {[p.name for p in candidates]}")
    path = candidates[0]
    complete = json.loads((path/"complete.json").read_text())
    sha = file_sha(path/"encoder_selected.pt")
    if sha != complete["encoder_sha256"]:
        raise ValueError("NSV encoder checkpoint checksum mismatch.")
    return path, {"run_id": path.name, "encoder_sha256": sha}


def run_training(root, dataset_id, condition, cache_root, cfg, device=None, warm_start_run_id=None):
    cfg.validate()
    if cfg.method == "nsv" and cfg.id_backend == "skdim":
        try:
            from importlib.metadata import version
            installed = version("scikit-dimension")
        except Exception as exc:
            raise ImportError("Run notebook 40's Colab-only scikit-dimension==0.3.4 install cell first.") from exc
        if installed != "0.3.4":
            raise ValueError(f"Expected scikit-dimension 0.3.4, found {installed}.")
    spec = CONDITIONS[condition]
    bundle, sim, manifest = load_dataset(root, dataset_id, spec["kind"])
    warm_path, warm_identity = None, None
    if cfg.warm_start:
        warm_path, warm_identity = resolve_warm_start(root, dataset_id, condition, cfg, warm_start_run_id)
    references = json.loads(Path(__file__).with_name("climate_reference_manifest.json").read_text())
    identity = {"dataset_id": dataset_id, "condition": condition, "condition_spec": spec,
                "method": cfg.method, "training": asdict(cfg), "warm_start": warm_identity,
                "data_sha": manifest["data_sha256"][spec["kind"]],
                "learning_source_sha": file_sha(__file__), "reference_manifest": references}
    run_id = f"{cfg.method}_s{cfg.seed}_{digest(identity)}"
    out = Path(root)/"runs"/dataset_id/condition/run_id
    if (out/"complete.json").exists():
        complete = json.loads((out/"complete.json").read_text())
        for name, key in [("features.npz", "features_sha256"), ("encoder_selected.pt", "encoder_sha256"),
                          ("model_selected.pt", "model_sha256")]:
            if file_sha(out/name) != complete[key]:
                raise ValueError(f"Completed artifact checksum mismatch: {name}")
        print("Reusing completed run:", out)
        return out
    if out.exists():
        raise FileExistsError(f"Incomplete run at {out}; inspect it or use a new run_tag.")
    seed_all(cfg.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    frames = frame_cache(bundle, sim, condition, cache_root, dataset_id)
    loaders, counts = make_loaders(frames, bundle, condition, sim, cfg)
    model = ClimateModel(cfg).to(device)
    if warm_path:
        model.encoder.load_state_dict(torch.load(warm_path/"encoder_selected.pt", map_location=device))
    out.mkdir(parents=True)
    torch.save(model.encoder.state_dict(), out/"encoder_initial.pt")
    json_write(out/"run.json", {**identity, "run_id": run_id, "pair_counts": counts,
               "initial_encoder_sha256": file_sha(out/"encoder_initial.pt"),
               "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                           "numpy": np.__version__, "device": str(device)},
               "feature_semantics": "InfoNCE pre_fc=32D pooled features, z=direct encoder output; BT h=64D, z=projector; NSV z=ID-selected refiner",
               "temporal_units": "frames (T0/50 by default), not calendar days"})
    train_stage1(model, loaders, cfg, device, out)
    index = index_windows(bundle, 1, 2*sim.frames_per_period, cfg.eval_stride)
    if cfg.method != "nsv":
        h, z = encode_frames(model, frames, index, condition, sim, device, cfg.batch_size)
        features = {"index": index, "pre_fc" if cfg.method == "infonce" else "h": h, "z": z}
        json_write(out/"id_train.json", estimate_id(z[bundle["splits"][index[:, 0]] == 0],
                                                   bundle["groups"][index[:, 0]][bundle["splits"][index[:, 0]] == 0], seed=cfg.seed))
    else:
        # Dense anchors preserve the original daily refiner/dynamics sampling.
        dense = index_windows(bundle, 1, cfg.lag, cfg.train_stride)
        h, _ = encode_frames(model, frames, dense, condition, sim, device, cfg.batch_size)
        splits, groups = bundle["splits"][dense[:, 0]], bundle["groups"][dense[:, 0]]
        diagnostic = estimate_id(h[splits == 0], groups[splits == 0], seed=cfg.seed)
        json_write(out/"id_train.json", diagnostic)
        selected = climate_id(h, splits, groups, cfg)
        json_write(out/"intrinsic_dim.json", selected)
        d_hat = selected["d_hat"]
        lookup = {tuple(row): i for i, row in enumerate(dense)}
        try:
            common = np.array([lookup[tuple(row)] for row in index])
        except KeyError as exc:
            raise ValueError("NSV train_stride/lag leaves gaps in the common evaluation index; use train_stride=1.") from exc
        features = {"index": index, "h": h[common]}
        dense_features = {"index": dense, "h": h}
        for dim in sorted(set((d_hat, *cfg.refine_dims))):
            seed_all(cfg.seed+dim)
            refiner = SirenAutoencoder(d_hat=dim)
            fit_array_regression(refiner, h, h, splits, cfg.refine_epochs, cfg.refine_batch_size,
                                 cfg.refine_learning_rate, cfg.refine_min_learning_rate, 0.0,
                                 device, out, f"refine_d{dim}")
            z = apply_numpy(refiner, h, device, encode=True)
            name = "z" if dim == d_hat else f"z_d{dim}"
            features[name] = z[common]; dense_features[name] = z
            if dim == d_hat and cfg.dynamics_epochs:
                pairs = [(i, lookup[(int(tr), int(t)+cfg.lag)]) for i, (tr, t) in enumerate(dense)
                         if (int(tr), int(t)+cfg.lag) in lookup]
                a, b = np.asarray(pairs).T
                seed_all(cfg.seed)
                dynamics = DynamicsMLP(dim)
                fit_array_regression(dynamics, z[a], z[b], splits[a], cfg.dynamics_epochs, 64,
                                     1e-3, 1e-5, 1e-4, device, out, "dynamics")
                prediction = apply_numpy(dynamics, z[a], device)
                scores = {str(split): {"mse": float(np.mean((prediction[splits[a] == split]-z[b][splits[a] == split])**2)),
                                      "persistence_mse": float(np.mean((z[a][splits[a] == split]-z[b][splits[a] == split])**2))}
                          for split in (1, 2)}
                json_write(out/"dynamics_scores.json", {"lag_frames": cfg.lag, "dimension": dim,
                           "forecast_type": "one-step latent MLP, not autonomous rollout", "splits": scores})
        np.savez_compressed(out/"dense_features.npz", **dense_features)
        json_write(out/"refinement_summary.json", {"selected_dimension": d_hat, "extra_dimensions": list(cfg.refine_dims),
                   "target": "raw unstandardized h64", "selection": "minimum validation h-reconstruction MSE"})
    np.savez_compressed(out/"features.npz", **features)
    json_write(out/"complete.json", {"run_id": run_id, "features_sha256": file_sha(out/"features.npz"),
               "encoder_sha256": file_sha(out/"encoder_selected.pt"), "model_sha256": file_sha(out/"model_selected.pt")})
    return out
