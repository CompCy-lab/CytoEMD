import meld
import numpy as np
from .distances import (
    euclidean,
    manhattan,
    chebyshev,
    minkowski,
    emd_samples
)
from sklearn.neighbors import NearestNeighbors


def compute_pair_emd(input, use_fast, bins):
    i, j = input[0]
    mat1, mat2 = input[1]
    n_var = mat1.shape[1]

    dis_vec = np.zeros((n_var,), dtype=mat1.dtype)
    bin_edges, min_flows = [], []
    for k in range(n_var):
        dis_vec[k], be, mf = emd_samples(mat1[:, k], mat2[:, k], use_fast=use_fast, bins=bins)
        bin_edges.append(be)
        min_flows.append(mf)
    return (i, j, dis_vec), bin_edges, min_flows


def label2index(ordered_lbls, lbls):
    lbl_indices = np.array([ordered_lbls.index(lbl) for lbl in lbls])
    return lbl_indices


def ce_score(y, y_pred):
    return -np.sum(y * np.log(y_pred), axis=1).mean()


def embed_to_cross_entropy(meld_model, embedding, lbls):
    """
    Compute the cross-entropy score given a distance matrix using MELD
    """
    pred_density = meld_model.fit_transform(embedding, lbls)
    pred_prob = meld.utils.normalize_densities(pred_density)
    # To one-hot vector
    lbl_indices = label2index(pred_prob.columns.tolist(), lbls)
    one_hot_label = np.zeros_like(pred_prob)
    one_hot_label[np.arange(lbl_indices.size), lbl_indices] = 1
    # compute cross entropy score
    ce = ce_score(one_hot_label, pred_prob)
    return ce


def get_distance_func(metric: str = 'manhattan'):
    if metric == 'manhattan':
        dis_fn = manhattan
    elif metric == 'euclidean':
        dis_fn = euclidean
    elif metric == 'chebyshev':
        dis_fn = chebyshev
    elif metric == 'minkowski':
        dis_fn = minkowski
    else:
        raise ValueError(f"metric: {metric} cannot be found, please check the input.")
    return dis_fn


def knn_purity(latent: np.array, label: np.array, n_neighbors=3):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(latent)
    indices = nbrs.kneighbors(latent, return_distance=False)[:, 1:]
    neighbors_labels = np.vectorize(lambda i: label[i])(indices)
    # pre cell purity scores
    scores = ((neighbors_labels - label.reshape(-1, 1)) == 0).mean(axis=1)
    res = [np.mean(scores[label == i]) for i in np.unique(label)]  # per cell-type purity
    # average over different cell types
    return np.mean(res)
