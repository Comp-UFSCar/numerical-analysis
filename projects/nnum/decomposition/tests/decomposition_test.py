from unittest import TestCase
from nose_parameterized import parameterized

import numpy as np
from numpy.testing import assert_array_almost_equal

from ..decomposition import lu_decomposition, qr_decomposition


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

    @parameterized.expand([
        ([[1, 2],
          [0, 4]],),
        ([[1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]],),
        ([[1, 2, 10],
          [2, 0, 4],
          [10, 4, 10]],),
        ([[2, 0, 0],
          [1, 2, 3],
          [0, 1, 2]],)
    ])
    def test_defined_matrices(self, a):
        a = np.array(a).astype(float)

        q, r = qr_decomposition(a)

        # Assert Q is orthogonal.
        assert_array_almost_equal(np.dot(q, q.T), np.identity(q.shape[0]), err_msg='Q is not orthogonal')

        # Assert A = QR
        actual = np.dot(q, r)
        assert_array_almost_equal(a, actual, decimal=14)

    def test_random_matrices(self):
        for _ in range(100):
            n = np.random.randint(1, 100)
            m = np.random.randint(n, 100)
            a = 100 * np.random.rand(m, n)

            q, r = qr_decomposition(a)

            # Assert Q is orthogonal.
            assert_array_almost_equal(np.dot(q, q.T), np.identity(q.shape[0]), err_msg='Q is not orthogonal')

            # Assert A = QR
            actual = np.dot(q, r)
            assert_array_almost_equal(a, actual, decimal=11)
