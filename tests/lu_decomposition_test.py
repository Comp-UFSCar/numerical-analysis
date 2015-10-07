import numpy as np
from unittest import TestCase
from numerical.decomposition import lu_decomposition


class LUDecompositionTest(TestCase):
    def test_raise_value_error(self):
        a = np.zeros((4, 4))

        with self.assertRaises(ValueError):
            lu_decomposition(a)

    def test_simple_matrix(self):
        a = np.array([
            [1, 2],
            [3, 4]
        ])

        l, u = lu_decomposition(a)

        # Assert that L.U = A.
        np.array_equal(np.dot(l, u), a)

    def test_decompose_zarowsky_example_4_4(self):
        a = np.array([
            [1, 2, 3, 4],
            [-1, 1, 2, 1],
            [0, 2, 1, 3],
            [0, 0, 1, 1]
        ])

        l, u = lu_decomposition(a)

        # Assert that L.U = A.
        np.array_equal(np.dot(l, u), a)
