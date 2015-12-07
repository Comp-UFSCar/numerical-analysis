from unittest import TestCase

import numpy as np

from projects.nnum.interpolation import lagrangian


class LagrangianTest(TestCase):
    def test_class_example_6_1(self):
        X = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
        ])

        i = lagrangian.LagrangianInterpolator(2)
        i.fit(X)

        actual = i.predict(.5)

        self.assertGreater(actual, X[0, 1])
        self.assertLess(actual, X[1, 1])
        self.assertLess(actual, X[2, 1])
