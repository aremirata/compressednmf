"""
Implementation of NMF using structured random compression and separable
approaches discussed in the paper "Compressed Nonnegative Matrix Factorization
Is Fast And Accurate"
"""

import scipy
import sparseqr
import numpy as np
from ntf_cython.random import rel_error
from numpy.linalg import norm, solve
from nmf_compressed.compression import algo41, algo42, algo43, algo44, algo45
from nmf_compressed.compression import algo46, structured_compression
from nmf_compressed.compression import count_gauss
from nmf_compressed.selection import xray, SPA
from ntf_cython.nmf import bpp
from numpy.linalg import solve
from ntf_cython.nmf import nmf
import scipy.sparse as sps

def compression_left(A, q=1, r=100, eps=0.01, oversampling=10,
                     oversampling_factor=10, algo='algo44', sparse=False):
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
    if sps.isspmatrix(A):
        print("The matrix is sparse. We use block principal pivoting method")
        W, H, n_iter, relative_error = nmf(A, n_components=r, max_iter=max_iter, random_state=random_state)
    
    else:
        print("The matrix is dense. We use compressed block principal pivoting method")
        L = compression_left(A, r=r, algo=algo)
        A_ = L.T.dot(A)

        relative_error = []

        cols = xray(A_, r)

        H_ = bpp(A_[:,cols], A_)
        W_ = A_[:,cols]

        for _ in range(max_iter):
            H_ = bpp(W_, A_, H_>0)
            W_ = bpp(H_.T, A_.T, W_.T>0).T
            relative_error.append(rel_error(A, L.dot(W_).dot(H_)))
            
        W = L.dot(W_)
        H = H_

    return W, H, relative_error

