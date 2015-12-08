import numpy as np
from math import ceil
from . import base


class LagrangianInterpolator(base.Interpolator):
    def __init__(self, degree, fitting_profile='local'):
        """Lagrangian Integrator.

        :param degree: int
            The degree of the polynomial form.

        :param fitting_profile: str (default='local')
            The fitting profile of the interpolation.

            If 'local', will interpolate using the closest points to t.
            If 'sparse', will use the most sparse collection of points possible.

        """
        super().__init__(degree)

        self.fitting_profile = fitting_profile

    @property
    def fitting_profile(self):
        return self.__fitting_profile

    @fitting_profile.setter
    def fitting_profile(self, value):
        if value not in ('local', 'sparse'):
            raise ValueError('Cannot set fitting_profile property to %s' % str(value))

        self.__fitting_profile = value

    def l(self, i, t, indices):
        """Calculate the discreet Dirac Delta.

        :param i:
            The index L_i. References the current landmark.

        :param t:
            The point of interest for prediction.

        :return:
            :float: the value L_i(t).
        """
        l = 1

        for j in indices:
            if i == j:
                continue

            l *= ((t - self.X[j]) / (self.X[i] - self.X[j]))[0]

        return l

    def predict(self, t):
        """Interpolate the function f described by the data X to predict f(t).

        :param t:
            The point of interest, where f(t) should be predicted.

        :return:
            :float: the interpolated value for f(t).
        """
        if self.fitting_profile == 'local':
            # Select the "degree+1" closest points to t.
            indices = np.argsort(np.abs(self.X - t), axis=0)[:self.degree + 1]

        else:
            # Select the "degree + 1" most sparse points.
            indices = [i * self.X.shape[0] // self.degree + self.X.shape[0]
                       // (2 * self.degree) for i in range(self.degree)]

            # Shift points so the extremes of the interval are included.
            indices = [v - self.X.shape[0] // (2 * self.degree) for i, v in
                       enumerate(indices[:len(indices) // 2 + 1])] \
                      + [v + (self.X.shape[0] - 1) // (2 * self.degree) for i, v in
                         enumerate(indices[len(indices) // 2:])]

        # The points found are ordered by distance to :t (e.g., 5, 1, 0, 3).
        # Reorder them by their indices (e.g.: 0, 1, 3, 5).
        indices = np.sort(indices).flatten()
        return sum((self.y[i] * self.l(i, t, indices) for i in indices))
