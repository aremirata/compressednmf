"""
Implementation of NMF using structured random compression and separable
approaches discussed in the paper "Compressed Nonnegative Matrix Factorization
Is Fast And Accurate"
"""

import numpy as np
from ntf_cython.random import algo42, algo44, rel_error
from nmf_compressed.compression import algo41, algo43, algo45
from nmf_compressed.compression import algo46, structured_compression
from nmf_compressed.compression import count_gauss
from nmf_compressed.selection import xray, SPA
from ntf_cython.nmf import bpp
from numpy.linalg import solve


def compression_left(A, q=1, r=100, eps=0.01, oversampling=10,
                     oversampling_factor=10, algo='algo44'):
    """
    Compute the projection matrix with orthonormal columns

    Parameters
    ----------
    A: numpy.array
       Input data
    q: integer, default: 1
       Exponent used in algo43 and algo44
    r: integer, default: 100
       Target rank
    eps: double, default:0.01
       Tolerance value used in algo42
    oversampling: integer, default: 10
       A parameter granting more freedom in the choice of Q
       used in structured_compression algorithm
    oversampling_factor: integer, default:10
       A parameter used in count_gauss compression algorithm
    algo: compression algorithm used
    """
    if algo == 'algo41':
        L = algo41(A, r)
    elif algo == 'algo42':
        L = algo42(A, eps, r)
    elif algo == 'algo43':
        L = algo43(A, q, r)
    elif algo == 'algo44':
        L = algo44(A, q, r)
    elif algo == 'algo45':
        L = algo45(A, r)
    elif algo == 'algo46':
        L = algo46(A, r)
    elif algo == 'structured_compression':
        L = structured_compression(A, q, r, oversampling)
    elif algo == 'count_gaussian':
        L, Z = count_gauss(A, r, oversampling_factor)
    else:
        L, _ = np.linalg.qr(A)

    return L


def compression_right(A, q=1, r=100, eps=0.01, oversampling=10,
                      oversampling_factor=10, algo='algo44'):
    """
    Compute the projection matrix with orthonormal columns

    Parameters
    ----------
    A: numpy.array
       Input data
    q: integer, default: 1
       Exponent used in algo43 and algo44
    r: integer, default: 100
       Target rank
    eps: double, default:0.01
       Tolerance value used in algo42
    oversampling: integer, default: 10
       A parameter granting more freedom in the choice of Q
       used in structured_compression algorithm
    oversampling_factor: integer, default:10
       A parameter used in count_gauss compression algorithm
    algo: compression algorithm used
    """
    if algo == 'algo41':
        R = algo41(A.T, r).T
    elif algo == 'algo42':
        R = algo42(A.T, eps, r).T
    elif algo == 'algo43':
        R = algo43(A.T, q, r).T
    elif algo == 'algo44':
        R = algo44(A.T, q, r).T
    elif algo == 'algo45':
        R = algo45(A.T, r).T
    elif algo == 'algo46':
        R = algo46(A.T, r).T
    elif algo == 'structured_compression':
        R = structured_compression(A.T, q, r, oversampling).T
    elif algo == 'count_gaussian':
        R, Zt = count_gauss(A, r, oversampling_factor).T
    else:
        R, _ = np.linalg.qr(A.T)

    return R

def admm_2a(A, algo=None, q=1, r=100, max_iter=1000,
                       eps=0.01, oversampling=10, oversampling_factor=20,
                       lam=1., phi=1., c=1., limit=7, random_state=None,
                       tol=0.7):
    """
    NMF using structured random compression via alternating direction method
    of multipliers(ADMM)

    Parameters
    ----------
    A: numpy.array
        Input data
    q: integer, default: 1
        Exponent used in algo43 and algo44
    r: integer, default: 1
        Target rank
    max_iter: integer, default:1000
        Maximum number of iterations for ADMM
    eps: double, default:0.01
        Tolerance value used in algo42
    oversampling: integer, default:10
        A parameter granting more freedom in the choice of Q used in
        structured_compression algorithm
    oversampling_factor: integer, default:10
        A parameter used in count_gauss compression algorithm
    algo: compression algorithm used
    random_state: integer
    """

    m, n = A.shape
    l = min(n, max(oversampling_factor, r+10))

    OmegaL = np.random.randn(n, l)
    H = np.dot(A, OmegaL)

    for j in range(0, limit):
        H = np.dot(A, np.dot(A.T, H))
    L = compression_left(H, algo, q, l, eps, oversampling,
                         oversampling_factor)
    LA = np.dot(L.T, A)

    OmegaR = np.random.randn(l, l)
    H = np.dot(OmegaR, LA)

    for j in range(0, limit):
        H = np.dot(np.dot(H, LA.T), LA)

    R = compression_left(H.T, algo, q, l, eps, oversampling,
                         oversampling_factor)
    R = R.T
    M = np.dot(LA, R.T)

    np.random.seed(random_state)
    U = np.random.rand(m, r)
    V = np.random.randn(r, n)
    Y = V.dot(R.T)

    Lam = np.zeros((m, r))
    Phi = np.zeros((r, n))
    I = np.eye(r)
    iter_ = 0

    relative_error = []

    while iter_ <= max_iter:
        X = solve((Y.dot(Y.T) + lam * I).T,
                  (M.dot(Y.T) + (lam * L.T.dot(U) - L.T.dot(Lam))).T).T
        Y = solve(X.T.dot(X) + phi * I,
                  X.T.dot(M) + phi * V.dot(R.T) - Phi.dot(R.T))
        U = np.maximum(L.dot(X) + Lam/lam, 1e-15)


def sep_nmf(A, q=1, r=50, max_iter=50, eps=0.01, oversampling=10,
            oversampling_factor=10, algo='algo42', selection='xray'):
    """
    Implementation of separable NMF
    """
    L = compression_left(A, r=r, algo=algo)
    A_compressed = L.T.dot(A)
    if selection == 'spa':
        cols = SPA(A_compressed, r)
    else:
        cols = xray(A_compressed, r)
    H = bpp(A_compressed[:, cols], A_compressed)
    W = A[:, cols]

    return W, H


def random_projected_sepnmf(A, q=1, r=50, max_iter=50, eps=0.01,
                            oversampling_factor=10, algo='algo42'):
    """
    NMF with Random Projections with initial factors from separable
    NMF approach based on bpp
    """
    L = compression_left(A, r=r, algo=algo)
    R = compression_right(A, r=r, algo=algo)
    A_ = L.T.dot(A)

    cols = xray(A_, r)
    H = bpp(A_[:cols], A_)
    W = A[:, cols]

    for _ in range(max_iter):
        H = bpp(L.T.dot(W), L.T.dot(A), H > 0)
        W = bpp(R.dot(H.T), R.dot(A.T), W.T > 0).T

    return W, H


def random_projected_bppnmf(A, q=1, r=50, max_iter=50, eps=0.01,
                            oversampling_factor=10, algo='algo42'):
    """
    NMF with Random Projections with random initial factors based on BPP
    """
    L = compression_left(A, r=r, algo=algo)
    R = compression_right(A, r=r, algo=algo)
    W = np.random.rand(A.shape[0], r)
    H = bpp(W, A)

    for _ in range(max_iter):
        H = bpp(L.T.dot(W), L.T.dot(A), H > 0)
        W = bpp(R.dot(H.T), R.dot(A.T), W.T > 0).T

    return W, H


def structured_randomized_bppnmf(A, q=1, r=50, max_iter=50, eps=0.01,
                                 oversampling=10, oversampling_factor=10,
                                 algo='algo42', random_state=2):

    """
    NMF with  structured compression based on BPP method. This is a mash-up
    algorithm of separable nmf based on structured compression and structured
    compression method for bpp.
    """

    L = compression_left(A, r=r, algo=algo)
    R = compression_right(A, r=r, algo=algo)
    A_ = L.T.dot(A)

    cols = xray(A_, r)
    H = bpp(A_[:,cols], A_)
    W = A_[:,cols]
    relative_error = []

    for _ in range(max_iter):
        H = bpp(W, A_, H>0)
        W = bpp(H.T, A_.T, W.T>0).T
        relative_error.append(rel_error(A, L.dot(W).dot(H)))

    return L.dot(W), H, relative_error

