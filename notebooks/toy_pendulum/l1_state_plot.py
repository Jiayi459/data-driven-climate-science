"""Plot saved true states colored by the SAME samples' raw latent coordinates.

CPU only; no training, probes, PCA or resampling. Can run independently of nb42:
    python l1_state_plot.py /path/to/analysis/<evaluation_id>
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np


def plot_state_space(evaluation_dir, source_sha=None):
    import matplotlib.pyplot as plt

    root = Path(evaluation_dir)
    truth_file = root / "evaluation_truth.npz"
    latent_file = root / "latent_coordinates.npz"
    for path in (truth_file, latent_file):
        if not path.is_file():
            raise FileNotFoundError(f"Required saved array is missing: {path}. Use the original Colab/Drive analysis folder.")
    with np.load(truth_file, allow_pickle=False) as f:
        states, index, splits = f["states"], f["index"], f["splits"]
    with np.load(latent_file, allow_pickle=False) as f:
        sample_indices = f["sample_indices"]
        selected_index = f["index"]
        # The baseline is not a learned latent coordinate; compare the four learned methods.
        latent = {key: f[key] for key in f.files
                  if key not in {"sample_indices", "index", "delay_pca"}}
    expected = np.flatnonzero(splits == 2)[::3]
    if not np.array_equal(sample_indices, expected):
        raise ValueError("Saved samples do not match latent_test.png's held-out [::3] selection.")
    if not np.array_equal(selected_index, index[sample_indices]):
        raise ValueError("Trajectory/end-frame indices do not match the true states.")
    xy = states[sample_indices, :2]
    if not len(xy) or not latent or not np.isfinite(xy).all():
        raise ValueError("No valid paired state/latent samples.")
    for label, z in latent.items():
        if z.shape != (len(xy), 2) or not np.isfinite(z).all():
            raise ValueError(f"Expected two saved finite coordinates for {label}.")

    names = {"bt_same": "BT same-time", "bt_temporal": "BT temporal",
             "infonce": "InfoNCE", "nsv": "NSV refiner"}
    fig, axes = plt.subplots(2, len(latent), figsize=(4.3*len(latent), 7.8),
                             sharex=True, sharey=True, squeeze=False, constrained_layout=True)
    columns = ["trajectory", "end_frame", "theta", "angular_velocity"]
    data = [selected_index, xy]
    for col, (label, z) in enumerate(latent.items()):
        run_id = label.rsplit(":", 1)[0]
        method = run_id.rsplit("_s", 1)[0]
        for row in range(2):
            ax = axes[row, col]
            points = ax.scatter(xy[:, 0], xy[:, 1], c=z[:, row], cmap="coolwarm", s=12)
            ax.set_title(f"{names.get(method, method)}\ncolored by z{row+1}")
            ax.set_xlabel(r"True $\theta$ (rad)")
            ax.set_ylabel(r"True $\dot{\theta}$ (rad / model time)")
            ax.axhline(0, color="0.8", lw=0.5, zorder=0)
            ax.axvline(0, color="0.8", lw=0.5, zorder=0)
            fig.colorbar(points, ax=ax, label=f"raw z{row+1}", shrink=0.85)
        columns.extend([f"{label}:z1", f"{label}:z2"])
        data.append(z)
    fig.suptitle(f"True phase space — the same {len(xy)} held-out windows\n"
                 "State at the final input frame; color scales are independent latent units", fontsize=13)
    output = root / "state_space_by_latent_test.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    with (root / "state_space_by_latent_test.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in np.column_stack(data):
            writer.writerow([int(row[0]), int(row[1]), *row[2:]])
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    metadata = {"samples": len(xy), "trajectories": len(np.unique(selected_index[:, 0])),
                "state_alignment": "true state at each window's final frame",
                "sample_selection": "exact saved latent_test.png sample indices; test [::3]",
                "coordinate_transform": "none", "color_transform": "none; independent colorbar per coordinate",
                "features": list(latent), "plot_source_sha256": source_sha or sha(Path(__file__)),
                "truth_sha256": sha(truth_file), "latent_coordinates_sha256": sha(latent_file)}
    (root / "state_space_by_latent_protocol.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path)
    print(plot_state_space(parser.parse_args().evaluation_dir))
