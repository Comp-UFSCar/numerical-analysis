import numpy as np


def e(i, d):
    """Create the i-th axis of the canonical base of the R^d.
    """
    _e = np.zeros((d, 1))
    _e[i, 0] = 1
    return _e


def tau(x, k):
    """Calculate tau_i^k = x_i/x_{k-1}.

    :raise
        ValueError if matrix is not LU decomposable.
    """
    if x[k - 1] == 0:
        raise ValueError('Matrix isn\'t LU decomposable.')

    t = x / x[k - 1]
    t[:k] = 0
    return t.reshape((x.shape[0], 1))
