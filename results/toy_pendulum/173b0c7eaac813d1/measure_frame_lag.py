"""Measure image changes at lags 1 and 5 using the archived simulator configuration.

This is a CPU data diagnostic, not a training run. Reconstructed states are kept
in memory; the original dataset/reports and source simulator are not modified.
Statistics use train trajectories only and common single-frame candidate anchors.
"""
from pathlib import Path
import hashlib
import io
import json
import os
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ddcs_l1_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "notebooks/toy_pendulum"))
from l1_data import SimulationConfig, make_bundle, render, file_sha


def summarize(values):
    a = np.asarray(values)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p05": float(np.quantile(a, .05)), "p95": float(np.quantile(a, .95)),
            "max": float(a.max())}


def main():
    source = HERE / "received/datasets/173b0c7eaac813d1/manifest.json"
    manifest = json.loads(source.read_text())
    assert file_sha(REPO / "notebooks/toy_pendulum/l1_data.py") == manifest["source_sha"]
    cfg = SimulationConfig(**manifest["simulation"])
    b = make_bundle(cfg, variable_length=False)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **b)
    reconstructed_sha = hashlib.sha256(buffer.getvalue()).hexdigest()
    byte_match = reconstructed_sha == manifest["data_sha256"]["fixed"]
    print("Reconstructed fixed.npz checksum matches archive:", byte_match, flush=True)

    train = np.flatnonzero(b["splits"] == 0)
    anchors = np.arange(0, len(b["times"]) - 5, 2)
    records = {lag: {"angle_change_degrees": [], "velocity_change": [],
                     "bob_displacement_pixels": [], "image_rmse": []} for lag in [1, 5]}
    for tr in train:
        theta = b["states"][tr, :, 0]
        v = b["states"][tr, :, 1]
        # Match cache quantization, but keep this trajectory in memory only.
        images = np.rint(255 * render(theta, b["lengths"][tr])).astype("uint8").astype("float32") / 255
        for lag, metrics in records.items():
            delta = theta[anchors + lag] - theta[anchors]
            metrics["angle_change_degrees"].append(np.rad2deg(np.abs(delta)))
            metrics["velocity_change"].append(np.abs(v[anchors + lag] - v[anchors]))
            radius = .42 * (cfg.image_size - 1) * b["lengths"][tr] / cfg.reference_length
            metrics["bob_displacement_pixels"].append(2 * radius * np.abs(np.sin(delta / 2)))
            metrics["image_rmse"].append(np.sqrt(np.mean((images[anchors + lag] - images[anchors])**2, axis=(1, 2))))

    stats = {}
    for lag, metrics in records.items():
        stats[lag] = {name: summarize(np.concatenate(values)) for name, values in metrics.items()}
        stats[lag].update(time_separation=lag * cfg.dt, reference_period_fraction=lag / cfg.frames_per_period,
                          small_angle_reference_phase_degrees=360 * lag / cfg.frames_per_period)
    out = HERE / "frame_lag_diagnostic"
    out.mkdir(exist_ok=True)
    document = {"dataset_id": manifest["dataset_id"], "simulation": manifest["simulation"],
                "source_sha_verified": True, "reconstructed_fixed_npz_sha256": reconstructed_sha,
                "original_fixed_npz_sha256": manifest["data_sha256"]["fixed"],
                "reconstructed_npz_bytes_match": byte_match,
                "scope": "train trajectories; same anchors for lag1/5; clean quantized images; no model training",
                "n_trajectories": len(train), "anchors_per_trajectory": len(anchors),
                "n_pairs_per_lag": len(train) * len(anchors),
                "anchor_start_stop_stride": [int(anchors[0]), int(anchors[-1]), 2],
                "pairwise_augmentation_noise_rms_reference": float(np.sqrt(2) * .01),
                "lag_statistics": stats}
    (out / "statistics.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({"pairs": document["n_pairs_per_lag"], "statistics": stats}, indent=2), flush=True)

    # Select examples by a specified training displacement quantile, not by model scores.
    displacement = np.concatenate(records[5]["bob_displacement_pixels"])
    fig, axes = plt.subplots(3, 4, figsize=(11, 8), squeeze=False)
    examples = []
    for row, q in enumerate([.05, .5, .95]):
        j = int(np.argmin(np.abs(displacement - np.quantile(displacement, q))))
        tr, t = int(train[j // len(anchors)]), int(anchors[j % len(anchors)])
        theta = b["states"][tr, :, 0]
        imgs = np.rint(255 * render(theta[[t, t + 1, t + 5]], b["lengths"][tr])).astype("uint8") / 255
        for col, label in enumerate(["t", "t + 1", "t + 5"]):
            axes[row, col].imshow(imgs[col], cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(f"{label}: angle {np.rad2deg(theta[[t, t+1, t+5][col]]):.1f}°", fontsize=10)
        axes[row, 3].imshow(np.abs(imgs[2] - imgs[0]), cmap="magma", vmin=0, vmax=1)
        axes[row, 3].set_title(f"|I(t+5) - I(t)|\nbob displacement {displacement[j]:.2f} px", fontsize=10)
        axes[row, 0].set_ylabel(f"Lag-5 P{int(100*q)} example\ntrajectory {tr}, frame {t}", fontsize=9)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
        examples.append({"displacement_quantile": q, "trajectory": tr, "anchor_frame": t,
                         "bob_displacement_pixels": float(displacement[j])})
    fig.suptitle("One image per sample: how much changes after 1 or 5 frames?", fontsize=14)
    fig.text(.5, .01, "L1a, fixed L. Examples selected from training pairs; each panel is one 64×64 image. "
             "Noise augmentation excluded.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .95))
    for suffix in ["png", "pdf"]:
        fig.savefig(out / f"lag_comparison.{suffix}", dpi=150)
    plt.close(fig)
    document["examples"] = examples
    (out / "statistics.json").write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
