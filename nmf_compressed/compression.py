"""
Implementation of compression algorithms in "Finding Structure with Randomness:
Probabilistic Algorithms for Constructing Approximate Matrix Decompositions,
"Compressed Nonnegative Matrix Factorization Is Fast and Accurate",
"How to Fake Multiply by a Gaussian Matrix"
"""

import scipy
import numpy as np
from numpy.linalg import norm, solve
from collections import OrderedDict


def algo41(A, r):
    """
    This scheme computes an m x r orthonormal matrix Q
    whose range approximates the range of A.

    Paramters
    ---------
    A: numpy.array
    r: target rank for the compression

    Returns
    _______
    Q: orthonormal matrix Q whose range approximates the range of original
    matrix A
    """
    n = A.shape[1]
    W = np.random.rand(n, r)
    Y = A.dot(W)
    Q, R = np.linalg.qr(Y)

    return Q


def algo43(A, q=1, r=None):
    """
    This algorithm computes an m x l orthonormal matrix Q whose range
    approximates the range of A.

    Parameters
    ----------
    A: numpy.array
    r: target rank for the compression

    Returns
    -------
    Q: orthonormal matrix Q whose range approximates the range of original
    matrix A
    """
    n = A.shape[1]
    W = np.random.normal(size=(n, r))
    Y = ((A.dot(A.T))**q).dot(A).dot(W)
    Q, _ = np.linalg.qr(Y)

    return Q


def compute_complex_diagonal(n):
    """
    This computes an n x n diagonal matrix whose entries are independent
    random variables uniformly distributed on the complex unit circle.
    """
    a = np.random.uniform(size=(n,)) + 1j * np.random.uniform(size=(n,))
    return np.diag(a / np.linalg.norm(a))


def random_permutation_matrix(n, r=None):
    """
    This computes an n x r matrix that samples r coordinates from n uniformly
    at random; i.e.,its r columns are drawn randomly without replacement from
    the columns of the n x n identity matrix.
    """
    R = np.identity(n)
    np.random.shuffle(R)
    if r is not None:
        R = R[:, :r]
    return R


def compute_unaryDFT(n):
    """
    This computes an n x n unitary discrete Fourier transform (DFT)
    """
    q = np.tile(np.arange(0, n), n).reshape(n, n)
    F = (1/np.sqrt(n)) * np.exp((-2 * 1j * np.pi * q.T * q) / n)
    return F


def SRFT(n, r):
    """
    Computes an n x r matrix which is the subsampled random Fourier transform
    """
    D = compute_complex_diagonal(n)
    F = compute_unaryDFT(n)
    R = random_permutation_matrix(n, r)
    return np.sqrt(n / r) * np.linalg.multi_dot([D, F, R])


def algo45(A, r):
    """
    Computes an m x r orthonormal matrix Q whose range approximates the
    range of A
    """
    n = A.shape[1]
    W = SRFT(n, r)
    Y = A.dot(W)
    Q, _ = np.linalg.qr(Y)
    return Q


def givens_rotation(n, i, j, theta):
    """
    Computes the entries of the rotation matrix  where
    givens_rotation[i,j;theta] denotes a rotation on the set of complex
    numbers by the angle theta in the (i,j) coordinate plane
    """
    givens_rotation = np.identity(n)
    givens_rotation[i, i] = np.cos(theta)
    givens_rotation[j, i] = np.cos(theta)
    givens_rotation[i, j] = np.sin(theta)
    givens_rotation[j, i] = -np.sin(theta)
    return givens_rotation


def chain_random_givens_rotation(n, r=None):
    """
    Computes a chain of random givens_rotations
    """
    P = random_permutation_matrix(n, r)
    for i in range(n - 1):
        P = np.dot(P, givens_rotation(n, i, i+1, np.random.normal()))
    return P


def GSRFT(n, r, num_iteration):
    """
    Computes Gaussian subsampled random Fourier transform
    """
    params = OrderedDict()
    params['D' + str(num_iteration)] = compute_complex_diagonal(n)
    i = num_iteration - 1

    while (i >= 0) and (i <= num_iteration - 1):
        params['theta' + str(i)] = chain_random_givens_rotation(n)
        params['D' + str(i)] = compute_complex_diagonal(n)
        i = i - 1

    F = compute_unaryDFT(n)
    R = random_permutation_matrix(n, r)

    a = np.linalg.multi_dot(list(params.values())[:-1])
    return np.linalg.multi_dot([a, params['D0'], F, R])


def algo46(A, r, num_iteration=3):
    """
    Computes an m x r orthonormal matrix Q whose range approximates
    the range of A based on Gausssian subsampled random Fourier transform
    """
    n = A.shape[1]
    W = GSRFT(n, r, num_iteration)
    Y = A.dot(W)
    Q, _ = np.linalg.qr(Y)

    return Q


def structured_compression(A, q, r, oversampling):
    """
    This is based on the algorithm presented in Figure 1 on the paper
    "Compressed Nonnegative Matrix Factorization Is Fast and Accurate"
    which is structured random compression algorithm

    Parameters
    ----------
    A: np.array, input matrix
    q: exponent based denoting the number of power iterations
    r: target rank for the compression
    oversampling: a parameter granting more freedom in the choice of Q
    """

    n = A.shape[1]
    W = np.random.normal(size=(n, r + oversampling))
    Y = ((A.dot(A.T))**q).dot(A).dot(W)
    Q, _ = np.linalg.qr(Y)

    return Q


def count_gauss(A, rank, oversampling_factor):
    """
    Based on Algorithm 1 of the How to Fake Multiply by a Gaussian Matrix
    https://arxiv.org/pdf/1606.05732v1.pdf
    """

    d = A.shape[0]
    k = rank * oversampling_factor
    R = np.random.randint(k, size=d)
    C = np.arange(d)
    D = np.random.choice([-1, 1], size=d)
    S = scipy.sparse.csc_matrix((D, (R, C)), shape=(k, d))
    G = np.random.randn(rank, k)
    Q = S.T.dot(G.T)
    Z = G.dot(S.dot(A))

    return Q, Z
