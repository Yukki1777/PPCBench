import numpy as np
from scipy.spatial.distance import cdist
import torch
import sys
sys.path.append('..')

def kabsch_numpy(P: np.ndarray, Q: np.ndarray):
    P = P.astype(np.float64)
    Q = Q.astype(np.float64)

    PC = np.mean(P, axis=0)
    QC = np.mean(Q, axis=0)

    UP = P - PC
    UQ = Q - QC

    C = UP.T @ UQ
    V, S, W = np.linalg.svd(C)

    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        V[:, -1] = -V[:, -1]

    R: np.ndarray = V @ W

    t = QC - PC @ R # (3,)

    return (UP @ R + QC).astype(np.float32), R.astype(np.float32), t.astype(np.float32)

def protein_surface_intersection(X, Y, sigma=25, gamma=10):
    """
    :param X: point cloud to be referenced, (N, 3)
    :param Y: point cloud to be tested whether it is outside the protein, (M, 3)
    :param sigma, gamma: parameter
    :return: (M,)
    """
    return (gamma + sigma * torch.logsumexp(-(Y.unsqueeze(1).repeat(1, X.shape[0], 1) - X)
                                            .pow(2).sum(dim=-1) / sigma, dim=1))

def compute_crmsd(X, Y, aligned=False):
    if not aligned:
        X_aligned, _, _ = kabsch_numpy(X, Y)
    else:
        X_aligned = X
    dist = np.sum((X_aligned - Y) ** 2, axis=-1)
    crmsd = np.sqrt(np.mean(dist))
    return float(crmsd)


def compute_irmsd(X, Y, seg, aligned=False, threshold=8.):
    X_re, X_li = X[seg == 0], X[seg == 1]
    Y_re, Y_li = Y[seg == 0], Y[seg == 1]
    dist = cdist(Y_re, Y_li)
    positive_re_idx, positive_li_idx = np.where(dist < threshold)
    positive_Y = np.concatenate((Y_re[positive_re_idx], Y_li[positive_li_idx]), axis=0)
    positive_X = np.concatenate((X_re[positive_re_idx], X_li[positive_li_idx]), axis=0)
    return float(compute_crmsd(positive_X, positive_Y, aligned))