import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from clustering.datasets import digits_data, two_dim_suite
from clustering.deep_brb import DeepCentroidClustering
from clustering.dpc import DensityPeaks
from clustering.hdbscan_star import HDBSCANStar
from clustering.lsc import LandmarkSpectral
from clustering.metrics import evaluate

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def run_2d():
    suite = two_dim_suite()
    methods = {
        "k-means (k given)": lambda X, k: KMeans(k, n_init=10, random_state=0).fit_predict(X),
        "DPC (auto centres)": lambda X, k: DensityPeaks(density="gaussian").fit_predict(X),
        "DPC (k given)": lambda X, k: DensityPeaks(n_clusters=k, density="gaussian").fit_predict(X),
        "LSC (k given)": lambda X, k: LandmarkSpectral(k, n_landmarks=100, r=5).fit_predict(X),
        "HDBSCAN* (auto)": lambda X, k: HDBSCANStar(15).fit_predict(X),
    }
    rows, preds = [], {}
    for dname, (X, y) in suite.items():
        k = len(np.unique(y))
        for mname, f in methods.items():
            lab = f(X, k)
            preds[(dname, mname)] = lab
            rows.append(dict(dataset=dname, method=mname, **evaluate(y, lab)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "results_2d.csv"), index=False)
    piv = df.pivot(index="dataset", columns="method", values="ARI")[list(methods)]
    print("\nAdjusted Rand Index (higher is better)\n", piv.round(2).to_string())
    piv.round(3).to_csv(os.path.join(OUT, "results_2d_ari_table.csv"))

    # figure: datasets x methods
    fig, axes = plt.subplots(len(suite), len(methods), figsize=(3 * len(methods), 2.6 * len(suite)))
    for i, (dname, (X, y)) in enumerate(suite.items()):
        for j, mname in enumerate(methods):
            ax = axes[i, j]
            lab = preds[(dname, mname)]
            noise = lab < 0
            ax.scatter(X[~noise, 0], X[~noise, 1], c=lab[~noise], s=4, cmap="tab10")
            ax.scatter(X[noise, 0], X[noise, 1], c="lightgrey", s=4)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(mname, fontsize=9)
            if j == 0:
                ax.set_ylabel(dname, fontsize=9)
            ax.text(0.02, 0.02, f"ARI={df[(df.dataset == dname) & (df.method == mname)].ARI.iloc[0]:.2f}",
                    transform=ax.transAxes, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_2d.png"), dpi=130)
    plt.close(fig)

    # DPC decision graph example
    X, y = suite["aniso"]
    d = DensityPeaks(density="gaussian").fit(X)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.scatter(d.rho_, d.delta_, s=6, c="grey")
    ax.scatter(d.rho_[d.centers_], d.delta_[d.centers_], s=40, c="tab:red", label="selected centres")
    ax.set_xlabel("local density rho"); ax.set_ylabel("delta"); ax.legend(fontsize=8)
    ax.set_title("DPC decision graph (aniso)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_dpc_decision_graph.png"), dpi=130)
    plt.close(fig)


def run_deep(seeds=5):
    X, y = digits_data()
    rows, curves = [], {}
    km = evaluate(y, KMeans(10, n_init=10, random_state=0).fit_predict(X))
    rows.append(dict(method="k-means on raw pixels", seed="-", **km))
    for mode, name in [("none", "AE + centroid DC (no reclustering)"),
                       ("recluster", "+ periodic reclustering"),
                       ("brb", "+ soft reset & reclustering (BRB-style)")]:
        for s in range(seeds):
            m = DeepCentroidClustering(10, mode=mode, random_state=s)
            m.fit(X, y_monitor=y)
            rows.append(dict(method=name, seed=s, **evaluate(y, m.labels_)))
            if s == 0:
                curves[name] = m.history_
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "results_deep_runs.csv"), index=False)
    g = df[df.seed != "-"].groupby("method")[["ACC", "NMI", "ARI"]].agg(["mean", "std"])
    g.to_csv(os.path.join(OUT, "results_deep_summary.csv"))
    print(f"\nDigits (1797 x 64), mean +- std over {seeds} seeds\n", g.round(3).to_string())
    print("\nk-means on raw pixels:", {k: round(float(v), 3) for k, v in km.items()})
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for name, h in curves.items():
        ax.plot(range(1, len(h) + 1), h, label=name)
    ax.set_xlabel("clustering epoch"); ax.set_ylabel("accuracy (seed 0)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_deep_curve.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["2d", "deep", "all"], default="all")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    if a.only in ("2d", "all"):
        run_2d()
    if a.only in ("deep", "all"):
        run_deep(a.seeds)
    print("\nSaved to", OUT)
