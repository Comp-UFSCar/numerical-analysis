import numpy as np
from unittest import TestCase
from numerical import lu_decomposition as lu


class EFunctionTest(TestCase):
    def test_create_any_canonical_vector(self):
        actual = lu.e(3, 10)

        self.assertEqual(actual.shape, (10, 1))
        self.assertEqual(actual[3], 1)


class TauFunctionTest(TestCase):
    def test_identity(self):
        identity = np.identity(4)
        expected = [0, 0, 0, 0]

        for i in range(0, 4):
            actual = lu.tau(identity[i, :], i + 1)
            np.array_equal(expected, actual)

    def test_random_array(self):
        k, n = 2, 10
        sample = np.random.rand(n)
        xk_1 = sample[k - 1]

        expected = sample / xk_1
        expected[:k] = 0

        actual = lu.tau(sample, k)

        np.array_equal(expected, actual)


class LUDecompositionTest(TestCase):
    def test_raise_value_error(self):
        a = np.zeros((4, 4))

        with self.assertRaises(ValueError):
            lu.lu_decomposition(a)

    def test_simple_matrix(self):
        a = np.array([
            [1, 2],
            [3, 4]
        ])

        l, u = lu.lu_decomposition(a)

        # Assert that L.U = A.
        np.array_equal(np.dot(l, u), a)

    def test_decompose_zarowsky_example_4_4(self):
        a = np.array([
            [1, 2, 3, 4],
            [-1, 1, 2, 1],
            [0, 2, 1, 3],
            [0, 0, 1, 1]
        ])

        l, u = lu.lu_decomposition(a)

        # Assert that L.U = A.
        np.array_equal(np.dot(l, u), a)
