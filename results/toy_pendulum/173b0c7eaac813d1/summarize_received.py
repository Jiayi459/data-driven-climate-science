"""Summarize the received reports; no fitting, simulation, or model execution.

Run with an existing numpy/pandas/matplotlib environment. Source reports remain
unchanged; all tables/figures are derived from the archived CSV and JSON files.
"""
from pathlib import Path
import hashlib
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ddcs_l1_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "received"
ANALYSIS = SOURCE / "analysis/173b0c7eaac813d1/L1a/a70d291efb9ff88c"
RUNS = SOURCE / "runs/173b0c7eaac813d1/L1a"


def read(path):
    return json.loads(path.read_text())


def main():
    csv_path = ANALYSIS / "scorecard.csv"
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == read(ANALYSIS / "complete.json")["scorecard_sha256"]
    scores = pd.read_csv(csv_path)
    assert len(scores) == 146
    assert set(scores.test_groups) == {20}
    assert set(scores.test_samples) == {1500}
    assert not scores.duplicated(["label", "probe", "horizon_frames", "variable"]).any()

    state = scores[scores.horizon_frames == 0].copy()
    state.to_csv(HERE / "current_state_all.csv", index=False)
    forecasts = scores[scores.horizon_frames > 0].copy()
    forecasts.to_csv(HERE / "forecasts_all.csv", index=False)
    ids = []
    for label, result in read(ANALYSIS / "id_train.json").items():
        for scale in result["scales"]:
            ids.append({"label": label, "participation_ratio": result["participation_ratio"], **scale})
    pd.DataFrame(ids).to_csv(HERE / "id_summary.csv", index=False)

    training, refiners = [], []
    for folder in sorted(RUNS.iterdir()):
        meta = read(folder / "run.json")
        history = read(folder / "training_history.json")
        best = min(history, key=lambda row: row["val"])
        training.append({"method": meta["method"], "best_epoch": best["epoch"],
                         "best_val": best["val"], "train_at_best": best["train"],
                         "last_train": history[-1]["train"], "last_val": history[-1]["val"],
                         "stage1_seconds": history[-1]["elapsed_seconds"]})
        for path in sorted(folder.glob("refine_history_d*.json")):
            history = read(path)
            best = min(history, key=lambda row: row["val"])
            refiners.append({"dimension": int(path.stem.split("_d")[-1]),
                             "best_epoch": best["epoch"], "best_val_mse": best["val"],
                             "last_val_mse": history[-1]["val"], "last_train_mse": history[-1]["train"]})
    pd.DataFrame(training).to_csv(HERE / "training_summary.csv", index=False)
    pd.DataFrame(refiners).to_csv(HERE / "refinement_summary.csv", index=False)

    # All methods are shown in a fixed order, without selecting on test scores.
    methods = ["nsv", "bt_same", "bt_temporal", "infonce", "delay_pca"]
    names = ["NSV", "BT same", "BT temporal", "InfoNCE", "Delay PCA"]
    colors = ["#2166ac", "#d95f02"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8))
    for ax, features, labels, title in [
        (axes[0, 0], ["h", "h", "h", "h", "pca"], names, "A  64-D features: current-state ridge readout"),
        (axes[0, 1], ["z_d2", "z", "z", "z"], names[:4], "B  2-D outputs: current-state ridge readout"),
    ]:
        x = np.arange(len(labels))
        for j, variable in enumerate(["theta", "velocity"]):
            values = [float(state[(state.method == method) & (state.feature == feature) &
                                  (state.probe == "ridge") & (state.variable == variable)].r2.iloc[0])
                      for method, feature in zip(methods, features)]
            bars = ax.bar(x + (j - 0.5) * 0.36, values, width=0.36, color=colors[j], label=variable)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, max(value, 0) + 0.025,
                        f"{value:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x, labels, fontsize=9)
        ax.set(ylim=(-0.04, 1.22), ylabel="Test R²", title=title)
        ax.axhline(0, color="#888888", linewidth=0.7)
        ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=8)

    ax = axes[1, 0]
    for path in sorted(RUNS.glob("nsv*/refine_history_d*.json")):
        history = pd.DataFrame(read(path))
        dim = int(path.stem.split("_d")[-1])
        line, = ax.plot(history.epoch, history.val, label=f"d={dim}", linewidth=1.6)
        best = history.loc[history.val.idxmin()]
        ax.scatter([best.epoch], [best.val], marker="*", s=95, color=line.get_color(), zorder=4)
    ax.set(yscale="log", xlabel="Refinement epoch", ylabel="Validation normalized-latent MSE",
           title="C  NSV refinement: stars mark saved checkpoints")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.grid(alpha=0.15)

    ax = axes[1, 1]
    for method, name in zip(methods, names):
        feature = "pca" if method == "delay_pca" else "h"
        part = forecasts[(forecasts.method == method) & (forecasts.feature == feature) &
                         (forecasts.variable == "velocity")].sort_values("horizon_periods")
        ax.plot(part.horizon_periods, part.rmse, "o-", label=name, linewidth=1.6)
    ax.set(xlabel="Forecast horizon / reference period", ylabel="Angular-velocity RMSE",
           title="D  64-D features: direct ridge forecasts")
    ax.set_xticks([0.24, 1, 2])
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.15)
    fig.suptitle("L1a pilot: useful encoder features, weak low-dimensional outputs", fontsize=15)
    fig.text(0.5, 0.012, "One training seed (42); 20 held-out trajectories / 1,500 windows. "
             "Point estimates shown; see CSV for group-bootstrap intervals. No autonomous rollout.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=2.5, w_pad=2)
    for suffix in ["png", "pdf"]:
        fig.savefig(HERE / f"l1a_diagnostics.{suffix}", dpi=170)
    plt.close(fig)
    print("Validated 146 score rows; wrote 5 summary CSVs and diagnostic PNG/PDF.")


if __name__ == "__main__":
    main()
