"""Landmark-based Spectral Clustering (LSC) for large-scale data.

Chen & Cai, "Large Scale Spectral Clustering with Landmark-Based
Representation", AAAI 2011 (doi:10.1609/aaai.v25i1.7900); extended in Cai &
Chen, IEEE Trans. Cybernetics, 2015 (doi:10.1109/TCYB.2014.2358564).

This is the single-view anchor/landmark graph that recent multi-view methods
(e.g. SMMSC, IEEE TKDE 2025) build on.

Algorithm (LSC-K variant: k-means landmarks)
1. Pick p << n landmarks U with k-means.
2. For every point x_i keep its r nearest landmarks and set
       z_ij = K(x_i, u_j) / sum_{j' in N_r(i)} K(x_i, u_j'),
   with a Gaussian kernel -> sparse Z (n x p), rows sum to 1.
3. Degree d_i = z_i . (Z^T 1);   Z_hat = D^{-1/2} Z.
4. The top-k left singular vectors of Z_hat equal the top-k eigenvectors of
   the implied n x n normalised affinity Z_hat Z_hat^T (never formed).
5. Row-normalise the embedding and run k-means.
Cost: O(n p d) for the graph and O(n p k) for the SVD (linear in n).
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


class LandmarkSpectral:
    def __init__(self, n_clusters, n_landmarks=200, r=5, random_state=0):
        self.n_clusters = n_clusters
        self.n_landmarks = n_landmarks
        self.r = r
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        p = min(self.n_landmarks, n)
        km = KMeans(p, n_init=1, random_state=self.random_state).fit(X)
        U = km.cluster_centers_
        # squared distances to landmarks (n x p), keep r nearest per point
        d2 = ((X ** 2).sum(1)[:, None] - 2 * X @ U.T + (U ** 2).sum(1)[None, :])
        d2 = np.maximum(d2, 0)
        r = min(self.r, p)
        idx = np.argpartition(d2, r - 1, axis=1)[:, :r]
        rows = np.repeat(np.arange(n), r)
        dist2 = d2[np.arange(n)[:, None], idx]
        sigma2 = np.mean(dist2) + 1e-12  # Gaussian bandwidth
        w = np.exp(-dist2 / (2 * sigma2))
        w /= w.sum(axis=1, keepdims=True) + 1e-12
        Z = csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n, p))
        deg = Z @ np.asarray(Z.sum(axis=0)).ravel()  # d_i = z_i . (Z^T 1)
        Zh = csr_matrix(Z.multiply(1.0 / np.sqrt(deg + 1e-12)[:, None]))
        k = min(self.n_clusters, p - 1)
        U_svd, s, _ = svds(Zh, k=k, random_state=self.random_state)
        emb = normalize(U_svd[:, np.argsort(-s)])
        self.embedding_ = emb
        self.labels_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state).fit_predict(emb)
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_
