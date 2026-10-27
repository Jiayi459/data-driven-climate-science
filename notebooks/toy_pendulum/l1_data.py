"""L1 pendulum simulation, observation, split and provenance utilities (no torch)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "pendulum-l1-v1"
SPLITS = {"train": 0, "val": 1, "test": 2}


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 20261027
    n_fixed: int = 100
    n_length_groups: int = 60
    trajectories_per_length: int = 5
    periods: float = 8.0
    frames_per_period: int = 50
    substeps: int = 8
    g: float = 1.0
    reference_length: float = 1.0
    damping: float = 0.02
    length_min: float = 0.8
    length_max: float = 1.2
    image_size: int = 64

    def validate(self):
        if self.n_fixed < 5 or self.n_length_groups < 5:
            raise ValueError("Need at least five independent groups for train/val/test.")
        if self.trajectories_per_length < 1 or self.periods <= 0:
            raise ValueError("Invalid trajectory count/duration.")
        if self.frames_per_period < 10 or self.substeps < 1:
            raise ValueError("Insufficient sampling/integration resolution.")
        if self.g <= 0 or self.reference_length <= 0 or self.damping < 0:
            raise ValueError("Invalid physical parameters.")
        if not 0 < self.length_min < self.length_max <= 1.5 * self.reference_length:
            raise ValueError("Length interval must be positive and fit the fixed camera.")
        if self.image_size != 64:
            raise ValueError("The initial CNN/renderer protocol uses 64 x 64 images.")

    @property
    def period(self):
        return 2 * np.pi * np.sqrt(self.reference_length / self.g)

    @property
    def dt(self):
        return self.period / self.frames_per_period


def group_split(groups, seed):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 5:
        raise ValueError("Too few independent groups.")
    order = np.random.default_rng(seed).permutation(unique)
    n_val = max(1, int(round(0.2 * len(order))))
    n_test = n_val
    mapping = {int(g): 0 for g in order}
    mapping.update({int(g): 1 for g in order[-n_val-n_test:-n_test]})
    mapping.update({int(g): 2 for g in order[-n_test:]})
    return np.array([mapping[int(g)] for g in groups], dtype=np.int8)


def simulate(initial, lengths, times, g=1.0, damping=0.02, substeps=8):
    """Vectorized RK4, constant length per trajectory; no stochastic force in L1a-c."""
    state = np.asarray(initial, dtype=np.float64).copy()
    lengths = np.asarray(lengths, dtype=np.float64)
    times = np.asarray(times)
    if state.shape != (len(lengths), 2) or np.any(lengths <= 0):
        raise ValueError("Expected initial=(trajectories,2), positive lengths.")
    if substeps < 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid time grid or substeps.")
    out = np.empty((len(lengths), len(times), 2), dtype=np.float64)
    out[:, 0] = state

    def rhs(y):
        return np.stack((y[:, 1], -damping*y[:, 1] - g/lengths*np.sin(y[:, 0])), axis=1)

    for i, interval in enumerate(np.diff(times), start=1):
        h = interval / substeps
        for _ in range(substeps):
            k1 = rhs(state)
            k2 = rhs(state + h*k1/2)
            k3 = rhs(state + h*k2/2)
            k4 = rhs(state + h*k3)
            state += h*(k1 + 2*k2 + 2*k3 + k4)/6
        out[:, i] = state
    return out


def energy(states, lengths, g=1.0):
    """Mechanical energy for unit mass (constant L), not a harmonic approximation."""
    length = np.asarray(lengths)[:, None]
    return 0.5*length**2*states[..., 1]**2 + g*length*(1-np.cos(states[..., 0]))


def make_bundle(cfg, variable_length=False):
    cfg.validate()
    rng = np.random.default_rng(cfg.seed + (1000 if variable_length else 0))
    if variable_length:
        group_lengths = rng.uniform(cfg.length_min, cfg.length_max, cfg.n_length_groups)
        groups = np.repeat(np.arange(cfg.n_length_groups), cfg.trajectories_per_length)
        lengths = group_lengths[groups]
    else:
        groups = np.arange(cfg.n_fixed)
        lengths = np.full(cfg.n_fixed, cfg.reference_length)
    initial = rng.uniform([-1.0, -0.5], [1.0, 0.5], size=(len(groups), 2))
    # Avoid a dataset dominated by almost stationary initial conditions.
    weak = np.linalg.norm(initial, axis=1) < 0.15
    while weak.any():
        initial[weak] = rng.uniform([-1.0, -0.5], [1.0, 0.5], size=(int(weak.sum()), 2))
        weak = np.linalg.norm(initial, axis=1) < 0.15
    times = np.arange(int(round(cfg.periods*cfg.frames_per_period)) + 1)*cfg.dt
    return {
        "states": simulate(initial, lengths, times, cfg.g, cfg.damping, cfg.substeps),
        "times": times, "lengths": lengths, "groups": groups.astype(np.int32),
        "splits": group_split(groups, cfg.seed + 17),
    }


def numerical_checks(bundle, cfg):
    original = bundle["states"][:3]
    fine = simulate(original[:, 0], bundle["lengths"][:3], bundle["times"],
                    cfg.g, cfg.damping, cfg.substeps*2)
    err = float(np.max(np.abs(original-fine)))
    e = energy(bundle["states"], bundle["lengths"], cfg.g)
    max_rise = float(np.max(np.diff(e, axis=1)))
    if err > 1e-5 or max_rise > 1e-8:
        raise ValueError(f"Integrator check failed: convergence={err}, energy rise={max_rise}")
    for group in np.unique(bundle["groups"]):
        if len(np.unique(bundle["splits"][bundle["groups"] == group])) != 1:
            raise ValueError("A group crossed splits.")
    return {"rk4_step_halving_max_error": err, "max_energy_increment": max_rise,
            "n_trajectories": len(e), "frames_per_trajectory": e.shape[1]}


def create_dataset(root, cfg):
    """Small truth arrays on Drive; frames are rendered only into disposable Colab cache."""
    cfg.validate()
    identity = {"schema": SCHEMA, "simulation": asdict(cfg), "source_sha": file_sha(__file__)}
    dataset_id = digest(identity)
    out = Path(root)/"datasets"/dataset_id
    if (out/"manifest.json").exists():
        load_dataset(root, dataset_id, "fixed")  # Validate existing hashes, never overwrite.
        load_dataset(root, dataset_id, "variable")
        return dataset_id
    if out.exists():
        raise FileExistsError(f"Incomplete dataset exists: {out}; use a new seed or inspect it.")
    out.mkdir(parents=True)
    checks, hashes = {}, {}
    for name, variable in [("fixed", False), ("variable", True)]:
        bundle = make_bundle(cfg, variable)
        checks[name] = numerical_checks(bundle, cfg)
        np.savez_compressed(out/f"{name}.npz", **bundle)
        hashes[name] = file_sha(out/f"{name}.npz")
    json_write(out/"manifest.json", {**identity, "dataset_id": dataset_id,
                                    "checks": checks, "data_sha256": hashes})
    return dataset_id


def load_dataset(root, dataset_id, kind):
    if kind not in {"fixed", "variable"}:
        raise ValueError(kind)
    out = Path(root)/"datasets"/dataset_id
    manifest = json.loads((out/"manifest.json").read_text())
    if manifest["schema"] != SCHEMA or manifest["dataset_id"] != dataset_id:
        raise ValueError("Dataset identity/schema mismatch.")
    if manifest["source_sha"] != file_sha(__file__):
        raise ValueError("Data code differs from notebook 39; use the same notebook release.")
    if file_sha(out/f"{kind}.npz") != manifest["data_sha256"][kind]:
        raise ValueError("Dataset file checksum mismatch.")
    with np.load(out/f"{kind}.npz", allow_pickle=False) as data:
        bundle = {k: data[k] for k in data.files}
    return bundle, SimulationConfig(**manifest["simulation"]), manifest


CONDITIONS = {
    "L1a": {"kind": "fixed", "noise_std": 0.0, "visible_length": True},
    "L1b_s002": {"kind": "fixed", "noise_std": 0.02, "visible_length": True},
    "L1b_s005": {"kind": "fixed", "noise_std": 0.05, "visible_length": True},
    "L1c_visible": {"kind": "variable", "noise_std": 0.02, "visible_length": True},
    "L1c_hidden": {"kind": "variable", "noise_std": 0.02, "visible_length": False},
}


def render(theta, lengths, size=64, visible_length=True, reference_length=1.0):
    """Fixed camera, Gaussian anti-aliased rod and bob; returns (N,H,W) in [0,1]."""
    theta = np.atleast_1d(theta)
    lengths = np.broadcast_to(lengths, theta.shape)
    radius = 0.42*(size-1)*(lengths/reference_length if visible_length else np.ones_like(theta))
    px, py = (size-1)/2, 0.14*(size-1)
    dx, dy = radius*np.sin(theta), radius*np.cos(theta)
    yy, xx = np.mgrid[:size, :size]
    rx, ry = xx[None]-px, yy[None]-py
    a = np.clip((rx*dx[:, None, None]+ry*dy[:, None, None]) /
                radius[:, None, None]**2, 0, 1)
    d2 = (rx-a*dx[:, None, None])**2+(ry-a*dy[:, None, None])**2
    rod = 0.45*np.exp(-d2/(2*0.65**2))
    bob = np.exp(-((rx-dx[:, None, None])**2+(ry-dy[:, None, None])**2)/(2*1.65**2))
    pivot = 0.7*np.exp(-(rx**2+ry**2)/(2*1.1**2))
    return np.maximum(np.maximum(rod, bob), pivot).astype(np.float32)


def frame_cache(bundle, cfg, condition, cache_root, dataset_id):
    spec = CONDITIONS[condition]
    name = f"{dataset_id}_{spec['kind']}_{'visible' if spec['visible_length'] else 'hidden'}"
    path = Path(cache_root)/f"{name}.npy"
    shape = (*bundle["states"].shape[:2], cfg.image_size, cfg.image_size)
    if path.exists():
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if arr.shape != shape or arr.dtype != np.uint8:
            raise ValueError("Frame cache shape/dtype mismatch; remove this disposable cache.")
        return arr
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype="uint8", shape=shape)
    for tr in range(shape[0]):
        arr[tr] = np.rint(255*render(bundle["states"][tr, :, 0], bundle["lengths"][tr],
                                   cfg.image_size, spec["visible_length"], cfg.reference_length)).astype("uint8")
    arr.flush()
    del arr
    tmp.replace(path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def index_windows(bundle, window=5, future=5, stride=2):
    """Each row is (trajectory, final input frame); all windows remain in one trajectory."""
    if min(window, stride) < 1 or future < 0:
        raise ValueError("Invalid window/stride/future.")
    ends = np.arange(window-1, bundle["states"].shape[1]-future, stride)
    if not len(ends):
        raise ValueError("Trajectories are too short for the requested history/horizon.")
    n = len(bundle["states"])
    return np.column_stack((np.repeat(np.arange(n), len(ends)), np.tile(ends, n))).astype("int32")


def observation(frames, trajectory, end, window, noise_std, seed, draw=0):
    """Noise is keyed by frame: overlapping windows share the same observed pixels."""
    start = int(end)-window+1
    if start < 0 or end >= frames.shape[1]:
        raise IndexError("Window outside trajectory.")
    out = np.array(frames[int(trajectory), start:int(end)+1], dtype="float32")/255
    if noise_std:
        for j, t in enumerate(range(start, int(end)+1)):
            rng = np.random.default_rng(np.random.SeedSequence([seed, draw, int(trajectory), t]))
            out[j] += noise_std*rng.standard_normal(out[j].shape).astype("float32")
    # Do not clip or normalize individual frames: keep the stated zero-mean noise model.
    return out


def sample_truth(bundle, index, horizon=0):
    tr, t = index.T
    return np.column_stack((bundle["states"][tr, t+horizon], bundle["lengths"][tr]))


def estimate_id(features, groups, ks=(10, 20, 40), max_points=2000, seed=17):
    """Train-only kNN MLE diagnostic; exclude neighbors from the same trajectory/group.

    This is a finite-sample, scale-dependent estimate, not a guaranteed physical dimension.
    No per-coordinate whitening: it would amplify near-null latent directions.
    """
    from sklearn.neighbors import NearestNeighbors
    z = np.asarray(features, dtype="float64")
    groups = np.asarray(groups)
    if len(z) != len(groups) or z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError("Invalid ID input.")
    take = np.random.default_rng(seed).choice(len(z), min(len(z), max_points), replace=False)
    z, groups = z[take], groups[take]
    candidates = min(len(z), max(128, max(ks)*8))
    distances, neighbors = NearestNeighbors(n_neighbors=candidates).fit(z).kneighbors(z)
    rows = []
    for k in ks:
        estimates = []
        for i in range(len(z)):
            valid = (groups[neighbors[i]] != groups[i]) & (distances[i] > 1e-10)
            d = distances[i][valid][:k]
            if len(d) == k:
                denominator = np.log(d[-1]/d[:-1]).sum()
                if denominator > 1e-10:
                    estimates.append((k-2)/denominator)
        rows.append({"k": int(k), "n_valid": len(estimates),
                     "median_id": float(np.median(estimates)) if estimates else None,
                     "mean_id": float(np.mean(estimates)) if estimates else None})
    singular = np.linalg.svd(z-z.mean(0), compute_uv=False)**2
    pr = float(singular.sum()**2/(np.square(singular).sum()+1e-30))
    return {"n_sampled": len(z), "scales": rows, "participation_ratio": pr,
            "neighbor_exclusion": "same group", "used_split": "train"}
