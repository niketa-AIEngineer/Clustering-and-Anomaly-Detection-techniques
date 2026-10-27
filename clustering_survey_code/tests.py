import warnings

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score as ari

from clustering.deep_brb import DeepCentroidClustering
from clustering.dpc import DensityPeaks
from clustering.hdbscan_star import HDBSCANStar
from clustering.lsc import LandmarkSpectral

warnings.filterwarnings("ignore")


def test_hdbscan_matches_sklearn():
    X, _ = make_moons(600, noise=0.07, random_state=1)
    ours = HDBSCANStar(15).fit_predict(X)
    ref = HDBSCAN(min_cluster_size=15, copy=True).fit_predict(X)
    assert ari(ours, ref) > 0.98, ari(ours, ref)


def test_lsc_moons():
    X, y = make_moons(800, noise=0.06, random_state=0)
    assert ari(y, LandmarkSpectral(2, n_landmarks=80).fit_predict(X)) > 0.95


def test_dpc_blobs():
    X, y = make_blobs(600, centers=3, cluster_std=0.6, random_state=3)
    assert ari(y, DensityPeaks(n_clusters=3).fit_predict(X)) > 0.95


def test_deep_runs_and_improves_over_chance():
    X, y = make_blobs(400, centers=4, n_features=20, cluster_std=1.5, random_state=0)
    X = (X - X.min(0)) / (np.ptp(X, axis=0) + 1e-9)
    m = DeepCentroidClustering(4, mode="brb", pretrain_epochs=15, epochs=30,
                               recluster_every=10, hidden=(32,), latent=4).fit(X)
    assert ari(y, m.labels_) > 0.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
