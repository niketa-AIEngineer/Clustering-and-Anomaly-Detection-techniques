import numpy as np
from sklearn.cluster import KMeans

from .metrics import clustering_accuracy


class _MLP:
    def __init__(self, sizes, rng):
        self.sizes = sizes
        self.W, self.b = [], []
        for i, o in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.normal(0, np.sqrt(2.0 / i), (i, o)))
            self.b.append(np.zeros(o))
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, x):
        hs, pre = [x], []
        h = x
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            a = h @ W + b
            pre.append(a)
            h = a if l == len(self.W) - 1 else np.maximum(a, 0)
            hs.append(h)
        return h, (hs, pre)

    def backward(self, g, cache):
        hs, pre = cache
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            if l != len(self.W) - 1:
                g = g * (pre[l] > 0)
            gW[l] = hs[l].T @ g
            gb[l] = g.sum(0)
            g = g @ self.W[l].T
        return gW, gb, g

    def adam(self, gW, gb, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for l in range(len(self.W)):
            for p, g, m, v in ((self.W, gW, self.mW, self.vW),
                               (self.b, gb, self.mb, self.vb)):
                m[l] = b1 * m[l] + (1 - b1) * g[l]
                v[l] = b2 * v[l] + (1 - b2) * g[l] ** 2
                mh = m[l] / (1 - b1 ** self.t)
                vh = v[l] / (1 - b2 ** self.t)
                p[l] -= lr * mh / (np.sqrt(vh) + eps)

    def soft_reset(self, lam, rng):
        """Shrink-and-perturb: theta <- lam*theta + (1-lam)*fresh_init."""
        for l, (i, o) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
            fresh = rng.normal(0, np.sqrt(2.0 / i), (i, o))
            self.W[l] = lam * self.W[l] + (1 - lam) * fresh
            self.b[l] = lam * self.b[l]
        # Adam statistics refer to the old weights -> restart them
        for l in range(len(self.W)):
            self.mW[l][:] = 0; self.vW[l][:] = 0
            self.mb[l][:] = 0; self.vb[l][:] = 0
        self.t = 0


class DeepCentroidClustering:
    def __init__(self, n_clusters, latent=10, hidden=(128, 64), mode="brb",
                 pretrain_epochs=60, epochs=100, recluster_every=20,
                 beta=1.0, lam=0.8, lr=1e-3, batch=128, random_state=0):
        assert mode in ("none", "recluster", "brb")
        self.n_clusters = n_clusters
        self.latent = latent
        self.hidden = tuple(hidden)
        self.mode = mode
        self.pretrain_epochs = pretrain_epochs
        self.epochs = epochs
        self.recluster_every = recluster_every
        self.beta = beta
        self.lam = lam
        self.lr = lr
        self.batch = batch
        self.random_state = random_state

    def _embed(self, X):
        return self.enc.forward(X)[0]

    def _kmeans(self, Z):
        km = KMeans(self.n_clusters, n_init=10, random_state=self.random_state).fit(Z)
        return km.cluster_centers_

    def _step(self, xb, C, w_cluster):
        z, ce = self.enc.forward(xb)
        rec, cd = self.dec.forward(z)
        B, d = xb.shape
        g_rec = 2.0 * (rec - xb) / (B * d)
        gWd, gbd, gz = self.dec.backward(g_rec, cd)
        if w_cluster > 0:
            a = np.argmin(((z[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
            gz = gz + w_cluster * 2.0 * (z - C[a]) / (B * z.shape[1])
        gWe, gbe, _ = self.enc.backward(gz, ce)
        self.dec.adam(gWd, gbd, self.lr)
        self.enc.adam(gWe, gbe, self.lr)

    def fit(self, X, y_monitor=None):
        X = np.asarray(X, dtype=float)
        rng = np.random.RandomState(self.random_state)
        d = X.shape[1]
        self.enc = _MLP([d, *self.hidden, self.latent], rng)
        self.dec = _MLP([self.latent, *self.hidden[::-1], d], rng)
        n = len(X)
        self.history_ = []

        def batches():
            perm = rng.permutation(n)
            for s in range(0, n, self.batch):
                yield X[perm[s:s + self.batch]]

        for _ in range(self.pretrain_epochs):  # 1) autoencoder pre-training
            for xb in batches():
                self._step(xb, None, 0.0)

        C = self._kmeans(self._embed(X))  # 2) initial clustering
        for ep in range(1, self.epochs + 1):
            if (self.mode != "none" and ep > 1
                    and (ep - 1) % self.recluster_every == 0):
                if self.mode == "brb":  # soft reset, then recluster
                    self.enc.soft_reset(self.lam, rng)
                C = self._kmeans(self._embed(X))
            for xb in batches():
                self._step(xb, C, self.beta)
            Z = self._embed(X)  # centroid update (closed form, per epoch)
            a = np.argmin(((Z[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
            for j in range(self.n_clusters):
                if np.any(a == j):
                    C[j] = Z[a == j].mean(0)
            if y_monitor is not None:
                self.history_.append(clustering_accuracy(y_monitor, a))
        self.centers_ = C
        Z = self._embed(X)
        self.embedding_ = Z
        self.labels_ = np.argmin(((Z[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        return self

    def fit_predict(self, X, y_monitor=None):
        return self.fit(X, y_monitor).labels_
