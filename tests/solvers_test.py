import numpy as np
from numpy.testing import assert_array_equal
from unittest import TestCase
from numerical import solvers


class LinearSolverTest(TestCase):
    def test_identity(self):
        a = np.array([
            [1, 0],
            [0, 1]
        ])
        y = np.array([1, 2])
        expected = [1, 2]

        actual = solvers.LinearSolver(a, y).solve()
        assert_array_equal(expected, actual)

    def test_sample_matrix(self):
        a = np.array([
            [2, 1],
            [3, 2]
        ])

        y = np.array([7, 14])
        expected = [0, 7]

        actual = solvers.LinearSolver(a, y).solve()
        assert_array_equal(expected, actual)


class ForwardEliminationTest(TestCase):
    def test_sample_matrix(self):
        sample_l = np.array([[2, 0], [3, 4]])
        sample_y = np.array([1, 2])

        result = solvers.forward_elimination(sample_l, sample_y)

        self.assertIsNotNone(result)
