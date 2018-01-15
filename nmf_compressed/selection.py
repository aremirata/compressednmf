"""
This implements two of the most popular selection algorithms used in
separable NMF namely XRAY (Fast Conical Hull Algorithms for
Near-Separable Non-negative Matrix Factorization) and SPA(The
Successive Projection Algorithm (SPA), an Algorithm with a Spatial
Constraint for the Automatic Search of Endmembers in Hyperspectral Data)
"""
import numpy as np
import scipy.sparse as sps
from ntf_cython.nmf import bpp as bpp


def compute_norm(A, p=None, axis=None):
    if sps.issparse(A):
        X = sps.linalg.norm(A, ord=p, axis=axis)
    else:
        X = np.linalg.norm(A, ord=p, axis=axis)
    return X


def xray(A, r):
    """
    This algorithm implements Algorithm 1 in "Fast Conical Hull Algorithms
    for Near-Separable Non-negative Matrix Factorization".
    Implements XRAY algorithm for the selection of extreme rays
    (extreme columns) in separable NMF. Returns the indices of the columns
    chosen by X-ray
    """
    cols = []
    R = A

    while len(cols) < r:
        # Find an extreme ray by looping until a column has been
        # chosen which was not previously selected
        while True:
            p = np.random.random((1, A.shape[0]))
            scores = compute_norm(np.dot(R.T, A), axis=0)
            scores = scores / (np.dot(p, A))
            scores[0, cols] = -1
            best_col = np.argmax(scores)

            if best_col in cols:
                continue
            else:
                cols.append(best_col)
                H = bpp(A[:, cols], A)
                R = A - np.dot(A[:, cols], H)
                break
    return cols


def SPA(A, r):
    """
    Based on this paper "The Successive Projection Algorithm (SPA), an
    Algorithm with a Spatial Constraint for the Automatic Search of Endmembers
    in Hyperspectral Data". Returns the indices of the columns chosen by SPA
    """
    colnorms = compute_norm(A, p=1, axis=0)
    A = A / colnorms
    cols = []
    m = A.shape[0]
    for _ in range(r):
        col_norms = np.linalg.norm(A, axis=0)
        col_norms[cols] = -1
        col_idx = np.argmax(col_norms)
        cols.append(col_idx)
        col = np.reshape(A[:, col_idx], (m, 1))
        A = np.dot((np.eye(m) - np.dot(col, col.T) / col_norms[col_idx]), A)
    return cols
