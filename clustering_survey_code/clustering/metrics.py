"""External validity metrics: ARI, NMI and Hungarian-matched clustering accuracy."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def clustering_accuracy(y_true, y_pred):
    """Best one-to-one label matching accuracy (Hungarian algorithm)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)
    w = np.zeros((len(clusters), len(classes)), dtype=np.int64)
    for i, c in enumerate(clusters):
        for j, k in enumerate(classes):
            w[i, j] = np.sum((y_pred == c) & (y_true == k))
    r, c = linear_sum_assignment(-w)
    return w[r, c].sum() / len(y_true)


def evaluate(y_true, y_pred):
    return {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "ACC": clustering_accuracy(y_true, y_pred),
        "k": int(len(np.unique(y_pred[y_pred >= 0]))),
    }
