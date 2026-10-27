import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist


class HDBSCANStar:
    def __init__(self, min_cluster_size=15, min_samples=None):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def _single_linkage(self, X):
        n = len(X)
        ms = self.min_samples or self.min_cluster_size
        D = cdist(X, X)
        core = np.sort(D, axis=1)[:, min(ms, n) - 1]
        mr = np.maximum(D, np.maximum(core[:, None], core[None, :]))
        np.fill_diagonal(mr, 0)
        mst = minimum_spanning_tree(mr).tocoo()
        order = np.argsort(mst.data, kind="stable")
        a, b, w = mst.row[order], mst.col[order], mst.data[order]
        parent = np.arange(2 * n - 1)
        size = np.r_[np.ones(n, int), np.zeros(n - 1, int)]
        left = np.full(2 * n - 1, -1)
        right = np.full(2 * n - 1, -1)
        height = np.zeros(2 * n - 1)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        nxt = n
        for u, v, d in zip(a, b, w):
            ru, rv = find(u), find(v)
            left[nxt], right[nxt], height[nxt] = ru, rv, d
            size[nxt] = size[ru] + size[rv]
            parent[ru] = parent[rv] = nxt
            nxt += 1
        return left, right, height, size, 2 * n - 2

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        left, right, height, size, root = self._single_linkage(X)
        mcs = self.min_cluster_size
        lam = lambda d: 1.0 / d if d > 0 else np.inf

        # leaves of every dendrogram node (iterative)
        def leaves(node):
            out, stack = [], [node]
            while stack:
                v = stack.pop()
                if v < n:
                    out.append(v)
                else:
                    stack += [left[v], right[v]]
            return out

        clusters = {0: dict(node=root, birth=0.0, fallen=0.0, end_size=0,
                            end_lam=0.0, kids=[], parent=None)}
        stack = [(root, 0)]
        while stack:
            node, cid = stack.pop()
            C = clusters[cid]
            while True:
                if node < n:  
                    C["end_size"], C["end_lam"] = 1, C["birth"]
                    break
                l, r_ = left[node], right[node]
                lm = lam(height[node])
                sl, sr = size[l], size[r_]
                if sl >= mcs and sr >= mcs:  
                    C["end_size"], C["end_lam"] = size[node], lm
                    for ch in (l, r_):
                        nid = len(clusters)
                        clusters[nid] = dict(node=ch, birth=lm, fallen=0.0,
                                             end_size=0, end_lam=0.0, kids=[],
                                             parent=cid)
                        C["kids"].append(nid)
                        stack.append((ch, nid))
                    break
                if sl < mcs and sr < mcs:  # cluster dies, everything falls off
                    C["fallen"] += size[node] * (lm - C["birth"])
                    C["end_size"], C["end_lam"] = 0, lm
                    break
                small, big = (l, r_) if sl < mcs else (r_, l)
                C["fallen"] += size[small] * (lm - C["birth"])
                node = big

        def stability(cid):
            C = clusters[cid]
            end = C["end_size"] * (C["end_lam"] - C["birth"]) if np.isfinite(C["end_lam"]) else 0.0
            return C["fallen"] + end

        sel_stab, selected = {}, {}
        for cid in sorted(clusters, reverse=True):  # children have larger ids
            C = clusters[cid]
            s_self = stability(cid)
            if not C["kids"]:
                sel_stab[cid], selected[cid] = s_self, [cid]
                continue
            s_kids = sum(sel_stab[k] for k in C["kids"])
            if cid != 0 and s_self >= s_kids:
                sel_stab[cid], selected[cid] = s_self, [cid]
            else:
                sel_stab[cid] = s_kids
                selected[cid] = [c for k in C["kids"] for c in selected[k]]
        chosen = selected[0]
        if chosen == [0]:  # whole data set is one cluster -> all noise by default
            chosen = []
        labels = np.full(n, -1)
        for lab, cid in enumerate(chosen):
            labels[leaves(clusters[cid]["node"])] = lab
        self.labels_ = labels
        self.n_clusters_ = len(chosen)
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_
