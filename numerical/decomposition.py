import numpy as np
from .base import e, tau


def lu_decomposition(a):
    """Calculate matrices U and L from a given matrix A (:a).

    This function will raise a ValueError if matrix A isn't LU decomposable.

    :param a the matrix that should be decomposed into L and U.

    :return
        L: decomposed lower triangular matrix, such that L.U = A.
        U: decomposed upper triangular matrix, such that L.U = A.
    """
    return _lu_decomposition(a, a.shape[0] - 1)


def _lu_decomposition(a0, k):
    """Recursive call of :lu_decomposition function.

    :param a0: the matrix A (it's also the parameter :a in the :lu_decomposition).
    :param k: which level of reduction is the algorithm dealing. Resolves recursion.
    :return: L, U and the Ak matrix.
    """
    _l, ak_minus_1 = (np.identity(a0.shape[0]), a0) if k == 1 else _lu_decomposition(a0, k - 1)
    tau_matrix = np.dot(tau(ak_minus_1[:, k - 1], k), e(k - 1, a0.shape[0]).T)

    return _l + tau_matrix, np.dot(np.identity(a0.shape[0]) - tau_matrix, ak_minus_1)
