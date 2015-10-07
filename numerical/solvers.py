import numpy as np

from .decomposition import lu_decomposition


def forward_elimination(l, y):
    """Perform forward elimination over a Lz = y linear system.

    :param l: a lower triangular matrix retrieved from a method such as LU Decomposition.
    :param y: the result of the linear system.
    :return: z, where z = Ux.
    """
    z = y.copy()

    for k in range(y.shape[0]):
        z[k] -= np.sum(z[:k] * l[k, :k])

    return z / np.diag(l).reshape(z.shape)


def backward_elimination(u, z):
    x = z.copy()

    for k in range(z.shape[0] - 1, -1, -1):
        x[k] -= np.sum(x[k + 1:] * u[k, k + 1:])

    return x / np.diag(u).reshape(x.shape)


class LinearSolver:
    def __init__(self, a, y):
        self._a = a
        self._y = y

    def solve(self):
        l, u = lu_decomposition(self._a)

        z = forward_elimination(l, self._y)
        return backward_elimination(u, z)
