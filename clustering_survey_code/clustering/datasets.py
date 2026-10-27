"""Benchmark data sets used by the demos (all generated locally)."""
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


def two_dim_suite(n=900, seed=0):
    """Classic 2-D shape benchmarks. Returns {name: (X, y)}."""
    rs = np.random.RandomState(seed)
    out = {}

    X, y = datasets.make_blobs(n_samples=n, centers=3, cluster_std=0.9,
                               random_state=seed)
    out["blobs"] = (X, y)

    X, y = datasets.make_moons(n_samples=n, noise=0.06, random_state=seed)
    out["moons"] = (X, y)

    X, y = datasets.make_circles(n_samples=n, factor=0.45, noise=0.05,
                                 random_state=seed)
    out["circles"] = (X, y)

    X, y = datasets.make_blobs(n_samples=n, centers=3, random_state=170)
    X = X @ np.array([[0.6, -0.64], [-0.4, 0.85]])
    out["aniso"] = (X, y)

    X, y = datasets.make_blobs(n_samples=[n // 2, n // 3, n - n // 2 - n // 3],
                               centers=[[0, 0], [6, 0], [3, 6]],
                               cluster_std=[0.4, 1.0, 1.6], random_state=seed)
    out["varied_density"] = (X, y)

    # two dense spirals-like arcs + a blob (non-convex mixture)
    t = rs.uniform(0, np.pi, n // 3)
    a = np.c_[np.cos(t), np.sin(t)] * 3 + rs.normal(0, 0.12, (n // 3, 2))
    b = np.c_[1 - np.cos(t), 0.5 - np.sin(t)] * 3 + rs.normal(0, 0.12, (n // 3, 2))
    c = rs.normal([9, 1], 0.6, (n - 2 * (n // 3), 2))
    X = np.vstack([a, b, c])
    y = np.r_[np.zeros(n // 3), np.ones(n // 3), 2 * np.ones(n - 2 * (n // 3))].astype(int)
    out["arcs_blob"] = (X, y)

    return {k: (StandardScaler().fit_transform(v[0]), v[1]) for k, v in out.items()}


def digits_data():
    """sklearn 8x8 digits (1797 x 64), scaled to [0, 1]."""
    X, y = datasets.load_digits(return_X_y=True)
    return X / 16.0, y
