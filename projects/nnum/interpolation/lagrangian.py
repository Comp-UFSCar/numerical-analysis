import numpy as np

from . import base


class LagrangianInterpolation(base.Interpolation):
    def l(self, i, t):
        """Calculate the discreet Dirac Delta.

        :param i:
            The index L_i. References the current landmark.

        :param t:
            The point of interest for prediction.

        :return:
            :float: the value L_i(t).
        """
        l = 1
        ti = self.X[i, 0]

        for j in range(0, i):
            tj = self.X[j, 0]
            l *= (t - tj) / (ti - tj)

        for j in range(i+1, self.degree):
            tj = self.X[j, 0]
            l *= (t - tj) / (ti - tj)

        return l

    def predict(self, t):
        """Interpolate the function f described by the data X to predict f(t).

        :param t:
            The point of interest, where f(t) should be predicted.

        :return:
            :float: the interpolated value for f(t).
        """
        # Find closest :degree points to t.
        indices = np.argsort(self.X[:, 0] - t)[:self.degree]

        # The points found are ordered by distance to :t (e.g., 5, 1, 0, 3).
        # Reorder them by their indices (e.g.: 0, 1, 3, 5).
        indices = np.sort(indices)

        Y = self.X[indices, 1]

        pred = 0
        for i in range(self.degree):
            pred += Y[i] * self.l(i, t)

        return pred
