import numpy as np
from .base import e, tau


def lu_decomposition(a):
    """Calculate matrices U and L from a given matrix A (:a).

    :param a the matrix that should be decomposed into L and U.

    :return
        L: decomposed lower triangular matrix, such that L.U = A.
        U: decomposed upper triangular matrix, such that L.U = A.

    :raise
        ValueError if matrix is not LU decomposable.
    """
    for i in range(a.shape[0]):
        if a[i, i] == 0:
            raise ValueError('Matrix isn\'t LU decomposable.')

    l, u, _ = _lu_decomposition(a, a.shape[0] - 1)
    return l, u


def _lu_decomposition(a0, k):
    n = a0.shape[0]

    _l, _u, ak_minus_1 = (np.identity(n), a0, a0) if k == 1 else _lu_decomposition(a0, k - 1)

    t_dot_e_t = np.dot(tau(ak_minus_1[:, k - 1], k), e(k - 1, n).T)

    gk = np.identity(n) - t_dot_e_t
    _l += t_dot_e_t
    del t_dot_e_t

    ak = np.dot(gk, ak_minus_1)

    return _l, np.dot(gk, _u), ak
