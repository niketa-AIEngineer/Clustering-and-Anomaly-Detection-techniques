import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import load_breast_cancer, make_blobs
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")
SEED = 42

# Data
def make_synthetic(seed=SEED, n_dense=500, n_sparse=300, n_anom=40):
    """Two clusters of DIFFERENT density + global and local anomalies (label 1 = anomaly)"""
    rng = np.random.RandomState(seed)
    dense = rng.normal([0, 0], 0.4, size=(n_dense, 2))
    sparse = rng.normal([5, 5], 1.2, size=(n_sparse, 2))
    glob = rng.uniform(-4, 9, size=(n_anom // 2, 2))                 # scattered global outliers
    local = rng.normal([0, 0], 0.4, size=(n_anom // 2, 2)) * 0 + \
        np.c_[rng.uniform(1.4, 2.2, n_anom // 2), rng.uniform(-0.3, 0.3, n_anom // 2)]  # near dense cluster
    X = np.vstack([dense, sparse, glob, local])
    y = np.r_[np.zeros(n_dense + n_sparse), np.ones(n_anom)]
    return X, y.astype(int)


def make_breast_cancer(seed=SEED, anomaly_frac=0.05):
    """Standard benchmark construction: benign = normal, small subsample of malignant = anomalies"""
    d = load_breast_cancer()
    X, y = d.data, 1 - d.target            # sklearn: 1=benign -> flip so 1 = malignant
    rng = np.random.RandomState(seed)
    normal_idx = np.where(y == 0)[0]
    mal_idx = np.where(y == 1)[0]
    n_anom = int(round(anomaly_frac * len(normal_idx) / (1 - anomaly_frac)))
    keep = np.r_[normal_idx, rng.choice(mal_idx, n_anom, replace=False)]
    return X[keep], y[keep]


# From-scratch implementations
def c_factor(n):
    """Average path length of an unsuccessful BST search (normaliser in iForest)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


class _ITree:
    def __init__(self, X, depth, max_depth, rng):
        self.size = len(X)
        self.depth = depth
        self.leaf = depth >= max_depth or len(X) <= 1
        if self.leaf:
            return
        spread = np.where(X.max(0) > X.min(0))[0]
        if len(spread) == 0:
            self.leaf = True
            return
        self.q = rng.choice(spread)                                   # random feature
        self.p = rng.uniform(X[:, self.q].min(), X[:, self.q].max())  # random split value
        mask = X[:, self.q] < self.p
        self.left = _ITree(X[mask], depth + 1, max_depth, rng)
        self.right = _ITree(X[~mask], depth + 1, max_depth, rng)

    def path_length(self, x):
        if self.leaf:
            return self.depth + c_factor(self.size)
        return (self.left if x[self.q] < self.p else self.right).path_length(x)


class IsolationForestScratch:
    """Liu, Ting & Zhou (2008). score = 2^(-E[h(x)] / c(psi)); near 1 => anomaly."""

    def __init__(self, n_trees=100, psi=256, seed=SEED):
        self.n_trees, self.psi, self.seed = n_trees, psi, seed

    def fit(self, X):
        rng = np.random.RandomState(self.seed)
        self.psi_ = min(self.psi, len(X))
        max_depth = int(np.ceil(np.log2(self.psi_)))
        self.trees_ = []
        for _ in range(self.n_trees):
            idx = rng.choice(len(X), self.psi_, replace=False)
            self.trees_.append(_ITree(X[idx], 0, max_depth, rng))
        return self

    def score(self, X):
        h = np.array([[t.path_length(x) for t in self.trees_] for x in X]).mean(1)
        return 2.0 ** (-h / c_factor(self.psi_))


class LOFScratch:
    """Breunig et al. (2000). LOF ~ 1 for inliers, >> 1 for local outliers (novelty mode)."""

    def __init__(self, k=20):
        self.k = k

    def fit(self, X):
        self.X_ = X
        self.nn_ = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        d, idx = self.nn_.kneighbors(X)
        d, idx = d[:, 1:], idx[:, 1:]                  # drop self
        self.kdist_ = d[:, -1]                         # k-distance of every training point
        self.lrd_ = self._lrd(d, idx)
        return self

    def _lrd(self, d, idx):
        reach = np.maximum(d, self.kdist_[idx])        # reach-dist_k(p, o) = max(k-dist(o), d(p, o))
        return 1.0 / (reach.mean(1) + 1e-12)

    def score(self, Xq, in_train=False):
        off = 1 if in_train else 0
        nn = NearestNeighbors(n_neighbors=self.k + off).fit(self.X_)
        d, idx = nn.kneighbors(Xq)
        d, idx = d[:, off:], idx[:, off:]
        lrd_q = self._lrd(d, idx)
        return (self.lrd_[idx].mean(1)) / lrd_q


class ECODScratch:
    """Li et al. (2022/2023). Parameter-free: aggregate negative-log tail probabilities per dimension."""

    def fit(self, X):
        self.Xs_ = np.sort(X, axis=0)
        self.n_ = len(X)
        self.skew_ = np.sign(((X - X.mean(0)) ** 3).mean(0))
        return self

    def score(self, X):
        n = self.n_
        left = np.stack([np.searchsorted(self.Xs_[:, j], X[:, j], side="right") for j in range(X.shape[1])], 1) / n
        right = np.stack([n - np.searchsorted(self.Xs_[:, j], X[:, j], side="left") for j in range(X.shape[1])], 1) / n
        left, right = np.clip(left, 1 / n, 1), np.clip(right, 1 / n, 1)
        o_left, o_right = -np.log(left), -np.log(right)
        o_auto = np.where(self.skew_ < 0, o_left, o_right)          # skewness-corrected tail
        return np.maximum.reduce([o_left.sum(1), o_right.sum(1), o_auto.sum(1)])


# 3. Detector wrappers: fit(train) -> score(test), higher = more anomalous
class KMeansDist:
    def __init__(self, k=2): self.k = k
    def fit(self, X): self.m = KMeans(n_clusters=self.k, n_init=10, random_state=SEED).fit(X); return self
    def score(self, X): return self.m.transform(X).min(1)


class GMMScore:
    def __init__(self, k=2): self.k = k
    def fit(self, X): self.m = GaussianMixture(n_components=self.k, covariance_type="full", random_state=SEED).fit(X); return self
    def score(self, X): return -self.m.score_samples(X)


class DBSCANScore:
    """DBSCAN labels noise as -1; we turn that into a continuous score = distance to nearest core point."""
    def __init__(self, eps=0.5, min_samples=10): self.eps, self.ms = eps, min_samples
    def fit(self, X):
        self.m = DBSCAN(eps=self.eps, min_samples=self.ms).fit(X)
        core = X[self.m.core_sample_indices_]
        self.nn = NearestNeighbors(n_neighbors=1).fit(core if len(core) else X)
        return self
    def score(self, X): return self.nn.kneighbors(X)[0].ravel()


class KNNDist:
    def __init__(self, k=10): self.k = k
    def fit(self, X): self.nn = NearestNeighbors(n_neighbors=self.k).fit(X); return self
    def score(self, X): return self.nn.kneighbors(X)[0][:, -1]


class LOFSk:
    def __init__(self, k=20): self.k = k
    def fit(self, X): self.m = LocalOutlierFactor(n_neighbors=self.k, novelty=True).fit(X); return self
    def score(self, X): return -self.m.score_samples(X)


class OCSVMScore:
    def __init__(self, nu=0.05, gamma="scale"): self.nu, self.g = nu, gamma
    def fit(self, X): self.m = OneClassSVM(nu=self.nu, gamma=self.g).fit(X); return self
    def score(self, X): return -self.m.decision_function(X)


class IFSk:
    def fit(self, X): self.m = IsolationForest(n_estimators=200, random_state=SEED).fit(X); return self
    def score(self, X): return -self.m.score_samples(X)


class AutoEncoderScore:
    def __init__(self, hidden=(16, 4, 16)): self.h = hidden
    def fit(self, X):
        self.m = MLPRegressor(hidden_layer_sizes=self.h, activation="relu", solver="adam", max_iter=600,
                              random_state=SEED, early_stopping=False).fit(X, X)
        return self
    def score(self, X): return ((self.m.predict(X) - X) ** 2).mean(1)


class SupervisedRF:
    """Classification framing: needs labelled anomalies (reference/upper bound)."""
    def fit(self, X, y): self.m = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED).fit(X, y); return self
    def score(self, X): return self.m.predict_proba(X)[:, 1]


class Wrap:
    """Adapter for scratch classes whose scoring method is `.score`."""
    def __init__(self, obj): self.o = obj
    def fit(self, X): self.o.fit(X); return self
    def score(self, X): return self.o.score(X)


# Evaluation
def evaluate(name, family, det, Xtr, ytr, Xte, yte, supervised=False):
    t0 = time.time()
    det.fit(Xtr, ytr) if supervised else det.fit(Xtr)
    s = det.score(Xte)
    dt = time.time() - t0
    thr = np.quantile(s, 1 - yte.mean())                 # flag the top (true contamination) fraction
    return dict(method=name, family=family,
                ROC_AUC=roc_auc_score(yte, s), AP=average_precision_score(yte, s),
                F1=f1_score(yte, (s >= thr).astype(int)), seconds=dt), s


def run_benchmark(X, y, tag, k_clusters=2, db_eps=0.5, db_ms=10, ocsvm_nu=0.05, split_seed=SEED):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=split_seed)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    zoo = [
        ("k-means distance", "clustering", KMeansDist(k_clusters), False),
        ("GMM neg-log-lik", "clustering", GMMScore(k_clusters), False),
        ("DBSCAN (core dist.)", "clustering", DBSCANScore(db_eps, db_ms), False),
        ("kNN distance", "distance", KNNDist(10), False),
        ("LOF (sklearn)", "density", LOFSk(20), False),
        ("LOF (scratch)", "density", Wrap(LOFScratch(20)), False),
        ("ECOD (scratch)", "statistical", Wrap(ECODScratch()), False),
        ("One-Class SVM", "classification", OCSVMScore(ocsvm_nu), False),
        ("Isolation Forest (sklearn)", "isolation", IFSk(), False),
        ("Isolation Forest (scratch)", "isolation", Wrap(IsolationForestScratch()), False),
        ("Autoencoder (MLP)", "reconstruction", AutoEncoderScore(), False),
        ("Random Forest (supervised)", "classification", SupervisedRF(), True),
    ]
    rows, scores = [], {}
    for name, fam, det, sup in zoo:
        r, s = evaluate(name, fam, det, Xtr, ytr, Xte, yte, supervised=sup)
        r["dataset"] = tag
        rows.append(r)
        scores[name] = s
    return pd.DataFrame(rows), scores, (Xtr, Xte, ytr, yte)


def breakdown_by_anomaly_type(n_seeds=10):
    """Synthetic data only: ROC-AUC against scattered (global) vs clustered (local) anomalies separately.
    Detectors are fitted and scored on all points (transductive), so values differ from the train/test table."""
    dets = {"ECOD": lambda: Wrap(ECODScratch()), "LOF": lambda: LOFSk(20), "GMM": lambda: GMMScore(2),
            "Isolation Forest": lambda: IFSk(), "One-Class SVM": lambda: OCSVMScore(0.05)}
    out = {}
    for sd in range(n_seeds):
        X, y = make_synthetic(seed=sd)
        kind = np.zeros(len(y), int); kind[-40:-20] = 1; kind[-20:] = 2      # 1 = global, 2 = clustered
        Xs = StandardScaler().fit_transform(X)
        for name, mk in dets.items():
            s = mk().fit(Xs).score(Xs)
            for k, lab in [(1, "global"), (2, "clustered")]:
                m = (kind == 0) | (kind == k)
                out.setdefault((name, lab), []).append(roc_auc_score((kind[m] == k).astype(int), s[m]))
    rows = {n: {lab: f"{np.mean(out[(n, lab)]):.3f} +- {np.std(out[(n, lab)]):.3f}" for lab in ("global", "clustered")}
            for n in dets}
    return pd.DataFrame(rows).T


# Figures
def plot_score_maps(Xraw, y, path="score_maps.png"):
    """Visualise how each family scores the 2-D plane (raw coordinates)."""
    models = [("k-means distance", KMeansDist(2)), ("GMM neg-log-lik", GMMScore(2)),
              ("DBSCAN (core dist.)", DBSCANScore(0.5, 10)), ("LOF", LOFSk(20)),
              ("One-Class SVM", OCSVMScore(0.05)), ("Isolation Forest", IFSk())]
    gx, gy = np.meshgrid(np.linspace(-5, 10, 200), np.linspace(-5, 10, 200))
    grid = np.c_[gx.ravel(), gy.ravel()]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (name, m) in zip(axes.ravel(), models):
        m.fit(Xraw)
        z = rankdata(m.score(grid)).reshape(gx.shape) / grid.shape[0]
        ax.contourf(gx, gy, z, levels=20, cmap="viridis_r")
        ax.scatter(*Xraw[y == 0].T, s=4, c="white", alpha=.6)
        ax.scatter(*Xraw[y == 1].T, s=22, c="red", marker="x")
        ax.set_title(name)
    fig.suptitle("Anomaly score maps (dark = more anomalous; red x = injected anomalies)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_roc(scores, yte, title, path):
    plt.figure(figsize=(6.5, 5.5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(yte, s)
        plt.plot(fpr, tpr, label=f"{name} ({roc_auc_score(yte, s):.3f})")
    plt.plot([0, 1], [0, 1], "k:", lw=.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(title); plt.legend(fontsize=6.5, loc="lower right"); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


# Main
if __name__ == "__main__":
    pd.set_option("display.width", 160)
    Xs, ys = make_synthetic()
    Xb, yb = make_breast_cancer()
    print(f"Synthetic: {Xs.shape}, anomalies={ys.sum()} ({ys.mean():.1%})")
    print(f"Breast cancer (malignant as anomalies): {Xb.shape}, anomalies={yb.sum()} ({yb.mean():.1%})")

    df_s, sc_s, (_, _, _, yte_s) = run_benchmark(Xs, ys, "synthetic-2D", k_clusters=2, db_eps=0.35, db_ms=8)
    df_b, sc_b, (_, _, _, yte_b) = run_benchmark(Xb, yb, "breast-cancer-30D", k_clusters=2, db_eps=3.0, db_ms=5, ocsvm_nu=0.05)
    res = pd.concat([df_s, df_b], ignore_index=True)
    res.to_csv("results.csv", index=False)

    for tag, df in res.groupby("dataset", sort=False):
        print(f"\n=== {tag} ===")
        print(df.drop(columns="dataset").round(3).to_string(index=False))

    # sanity: scratch implementations agree with sklearn
    for ds, sc in [("synthetic", sc_s), ("breast-cancer", sc_b)]:
        r1 = spearmanr(sc["Isolation Forest (scratch)"], sc["Isolation Forest (sklearn)"])[0]
        r2 = spearmanr(sc["LOF (scratch)"], sc["LOF (sklearn)"])[0]
        print(f"[{ds}] Spearman scratch-vs-sklearn: iForest={r1:.3f}  LOF={r2:.3f}")

    # ---- robustness: repeat with 10 different data draws / splits, report mean +- std ----
    if "--quick" not in __import__("sys").argv:
        reps = []
        for sd in range(10):
            Xs_, ys_ = make_synthetic(seed=sd)
            Xb_, yb_ = make_breast_cancer(seed=sd)
            reps.append(run_benchmark(Xs_, ys_, "synthetic-2D", 2, 0.35, 8, split_seed=sd)[0])
            reps.append(run_benchmark(Xb_, yb_, "breast-cancer-30D", 2, 3.0, 5, 0.05, split_seed=sd)[0])
        allr = pd.concat(reps)
        g = allr.groupby(["dataset", "method"], sort=False)[["ROC_AUC", "AP", "F1"]].agg(["mean", "std"]).round(3)
        g.to_csv("results_10seeds.csv")
        print("\n=== mean +- std over 10 seeds ===")
        print(g.to_string())

    bd = breakdown_by_anomaly_type()
    bd.to_csv("results_by_anomaly_type.csv")
    print("\n=== synthetic: ROC-AUC by anomaly type (mean +- std over 10 seeds) ===")
    print(bd.to_string())

    plot_score_maps(Xs, ys)
    plot_roc(sc_s, yte_s, "ROC - synthetic 2-D", "roc_curves.png")
    plot_roc(sc_b, yte_b, "ROC - breast cancer 30-D", "roc_curves_breast_cancer.png")
    print("\nSaved results.csv, score_maps.png, roc_curves.png, roc_curves_breast_cancer.png")
