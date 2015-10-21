from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from numerical.decomposition import lu_decomposition, qr_decomposition


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


class QRDecompositionTest(TestCase):
    def test_raise_value_error(self):
        a = (100 * np.random.rand(2, 4)).astype(int)

        with self.assertRaises(ValueError):
            qr_decomposition(a)

    def test_decompose_zarowsky_example_4_6(self):
        a = np.array([4, 3, 0]).reshape((3, 1))
        expected_h1 = np.array([-5, 0, 0]).reshape((3, 1))

        q, r = qr_decomposition(a)

        assert_array_almost_equal(expected_h1, r)
        assert_array_almost_equal(np.dot(q, r), a)