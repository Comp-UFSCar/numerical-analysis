import numpy as np
from unittest import TestCase
from ..base import e, tau


class EFunctionTest(TestCase):
    def test_create_any_canonical_vector(self):
        actual = e(3, 10)

        self.assertEqual(actual.shape, (10, 1))
        self.assertEqual(actual[3], 1)


class TauFunctionTest(TestCase):
    def test_identity(self):
        identity = np.identity(4)
        expected = [0, 0, 0, 0]

        for i in range(0, 4):
            actual = tau(identity[i, :], i + 1)
            np.array_equal(expected, actual)

    def test_random_array(self):
        k, n = 2, 10
        sample = np.random.rand(n)
        xk_1 = sample[k - 1]

        expected = sample / xk_1
        expected[:k] = 0

        actual = tau(sample, k)

        np.array_equal(expected, actual)
