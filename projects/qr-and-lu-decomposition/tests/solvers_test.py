from unittest import TestCase

from nose_parameterized import parameterized

import numpy as np
from numpy.testing import assert_array_almost_equal

from numerical import solvers


class LinearSolverTest(TestCase):
    @parameterized.expand([
        (np.identity(2), [1, 2], [1, 2]),
        ([[3, 7], [5, 2]], [2, 1], [3 / 29, 7 / 29]),
    ])
    def test_matrices(self, a, y, expected):
        a = np.array(a)
        y = np.array(y)

        actual = solvers.LinearSolver(a, y).solve()
        assert_array_almost_equal(expected, actual)


class ForwardEliminationTest(TestCase):
    @parameterized.expand([
        (np.identity(2), [1, 2], [1, 2]),
        ([[1, 0], [5 / 3, 1]], [2, 1], [2, -7 / 3]),
    ])
    def test_sample_matrix(self, l, y, expected):
        l = np.array(l)
        y = np.array(y)

        actual = solvers.forward_elimination(l, y)
        assert_array_almost_equal(expected, actual)


class BackwardEliminationTest(TestCase):
    @parameterized.expand([
        (np.identity(2), [1, 2], [1, 2]),
        ([[3, 7], [0, -58 / 6]], [2, -7 / 3], [3 / 29, 7 / 29]),
    ])
    def test_sample_matrix(self, u, z, expected):
        u = np.array(u)
        z = np.array(z)

        actual = solvers.backward_elimination(u, z)
        assert_array_almost_equal(expected, actual)
