import numpy as np
from scipy.spatial.distance import cdist


class DensityPeaks:
    """Parameters
    n_clusters : int or None
        None -> centres chosen automatically with an elbow rule on
        gamma = rho*delta (largest ratio gap among the top ``max_clusters+1``
        points). An int picks the top-k gamma points.
    density : {"cutoff", "gaussian", "knn"}"""

    def __init__(self, n_clusters=None, density="gaussian", dc_percent=2.0,
                 k=10, max_clusters=15):
        self.n_clusters = n_clusters
        self.density = density
        self.dc_percent = dc_percent
        self.k = k
        self.max_clusters = max_clusters

    def _density(self, D):
        n = D.shape[0]
        iu = np.triu_indices(n, 1)
        if self.density == "knn":
            kk = min(self.k, n - 1)
            knn = np.sort(D, axis=1)[:, 1:kk + 1]
            return np.exp(-np.mean(knn ** 2, axis=1))
        dc = np.percentile(D[iu], self.dc_percent)
        self.dc_ = dc
        if self.density == "cutoff":
            return (D < dc).sum(axis=1) - 1.0
        return np.exp(-(D / dc) ** 2).sum(axis=1) - 1.0

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        D = cdist(X, X)
        rho = self._density(D)
        order = np.argsort(-rho, kind="stable")
        delta = np.zeros(n)
        nn_higher = np.full(n, -1)
        delta[order[0]] = D[order[0]].max()
        for r in range(1, n):
            i = order[r]
            higher = order[:r]
            j = higher[np.argmin(D[i, higher])]
            delta[i] = D[i, j]
            nn_higher[i] = j
        # decision-graph score
        rn = (rho - rho.min()) / (np.ptp(rho) + 1e-12)
        dn = (delta - delta.min()) / (np.ptp(delta) + 1e-12)
        gamma = rn * dn
        if self.n_clusters is None:
            # elbow rule: the largest ratio gap among the top-gamma points
            top = np.argsort(-gamma)[:min(self.max_clusters + 1, n)]
            g = gamma[top]
            if len(g) < 2:
                kk = 1
            else:
                ratio = g[:-1] / (g[1:] + 1e-12)
                kk = int(np.argmax(ratio)) + 1
            centres = top[:kk]
        else:
            centres = np.argsort(-gamma)[:self.n_clusters]
        labels = np.full(n, -1)
        labels[centres] = np.arange(len(centres))
        for i in order:
            if labels[i] == -1:
                labels[i] = labels[nn_higher[i]]
        self.labels_ = labels
        self.centers_ = centres
        self.rho_, self.delta_, self.gamma_ = rho, delta, gamma
        self.n_clusters_ = len(centres)
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_
