import sparseqr
import scipy

def algo41_sparse(A, r):
    """
    This scheme computes an m x r orthonormal matrix Q
    whose range approximates the range of A.

    Paramters
    ---------
    A: sparse matrix
    r: target rank for the compression

    Returns
    _______
    Q: orthonormal matrix Q whose range approximates the range of original
    matrix A
    """
    n = A.shape[1]
    W = scipy.sparse.rand(n, r)
    Y = A.dot(W)
    Q, R, E, rank = sparseqr.qr(Y)
    
    return Q