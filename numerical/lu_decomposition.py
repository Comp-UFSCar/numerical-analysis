import numpy as np

debugging = False


def e(i, d):
    """Create the i-th axis of the canonical base of the R^d.
    """
    _e = np.zeros((d, 1))
    _e[i, 0] = 1
    return _e


def tau(x, k):
    """Calculate tau_i^k = x_i/x_{k-1}.
    """
    t = x / x[k - 1]
    t[:k] = 0
    return t.reshape((x.shape[0], 1))


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

    t_dot_e = np.dot(tau(ak_minus_1[:, k - 1], k), e(k - 1, n).T)

    gk = np.identity(n) - t_dot_e
    _l += t_dot_e

    ak = np.dot(gk, ak_minus_1)

    if debugging:
        print('G_%i:' % k)
        print(gk)
        print('A_%i:' % k)
        print(ak)

    return _l, np.dot(gk, _u), ak
